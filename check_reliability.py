"""Audit fixed Experiment 1 models without changing submission artifacts."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from run_experiment import BASE_DIR, clean_headers, load_data
from run_stemming_comparison import make_pipeline


OUT = BASE_DIR / "reliability_results"
SEEDS = (42, 7, 21, 73, 101)


def canonical(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def duplicate_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    raw_train = train["text"].map(canonical)
    raw_test = test["text"].map(canonical)
    cleaned_train = train["text"].map(clean_headers).map(canonical)
    cleaned_test = test["text"].map(clean_headers).map(canonical)
    labels_by_text = train.groupby(cleaned_train)["target"].nunique()
    train_idx, val_idx = train_test_split(
        np.arange(len(train)), test_size=0.2, random_state=42, stratify=train["target"]
    )
    return {
        "definition": "exact string after lowercasing and whitespace collapse; cleaned version first removes configured header lines",
        "train_rows": len(train),
        "test_rows": len(test),
        "raw_duplicate_train_rows": int(raw_train.duplicated(keep=False).sum()),
        "cleaned_duplicate_train_rows": int(cleaned_train.duplicated(keep=False).sum()),
        "cleaned_distinct_texts_with_conflicting_labels": int((labels_by_text > 1).sum()),
        "raw_test_rows_matching_train": int(raw_test.isin(set(raw_train)).sum()),
        "cleaned_test_rows_matching_train": int(cleaned_test.isin(set(cleaned_train)).sum()),
        "seed42_validation_rows_matching_training_cleaned_text": int(
            cleaned_train.iloc[val_idx].isin(set(cleaned_train.iloc[train_idx])).sum()
        ),
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train, test = load_data()
    duplicates = duplicate_audit(train, test)
    cleaned_train = train["text"].map(clean_headers)
    cleaned_test = test["text"].map(clean_headers)
    canonical_train = cleaned_train.map(canonical)
    submitted = pd.read_csv(BASE_DIR / "predictions_stemming.csv", header=None).iloc[:, 0].to_numpy()
    if len(submitted) != len(test):
        raise ValueError("Submitted prediction count does not match the unlabeled test set")

    results = []
    stem_test_predictions = []
    for seed in SEEDS:
        fit_idx, val_idx = train_test_split(
            np.arange(len(train)), test_size=0.2, random_state=seed, stratify=train["target"]
        )
        overlap = canonical_train.iloc[val_idx].isin(set(canonical_train.iloc[fit_idx])).to_numpy()
        for stemming in (False, True):
            model = make_pipeline(stop_words=True, stemming=stemming)
            started = time.perf_counter()
            model.fit(cleaned_train.iloc[fit_idx], train["target"].iloc[fit_idx])
            fit_seconds = time.perf_counter() - started
            pred = model.predict(cleaned_train.iloc[val_idx])
            y_val = train["target"].iloc[val_idx].to_numpy()
            row = {
                "seed": seed,
                "stemming": stemming,
                "training_rows": len(fit_idx),
                "validation_rows": len(val_idx),
                "accuracy": accuracy_score(y_val, pred),
                "macro_f1": f1_score(y_val, pred, average="macro"),
                "overlapping_validation_rows": int(overlap.sum()),
                "accuracy_without_exact_overlap": accuracy_score(y_val[~overlap], pred[~overlap]),
                "macro_f1_without_exact_overlap": f1_score(y_val[~overlap], pred[~overlap], average="macro"),
                "fit_seconds": fit_seconds,
            }
            if stemming:
                test_pred = model.predict(cleaned_test)
                stem_test_predictions.append(test_pred)
                row["agreement_with_full_train_submission"] = float(np.mean(test_pred == submitted))
            else:
                row["agreement_with_full_train_submission"] = None
            results.append(row)
            print(
                f"seed={seed} stemming={stemming} accuracy={row['accuracy']:.4f} "
                f"macro-F1={row['macro_f1']:.4f}",
                flush=True,
            )

    frame = pd.DataFrame(results)
    original = json.loads((BASE_DIR / "stemming_results" / "best_config.json").read_text(encoding="utf-8"))
    seed42 = frame.loc[(frame["seed"] == 42) & frame["stemming"]].iloc[0]
    if abs(seed42["macro_f1"] - original["validation_macro_f1"]) > 1e-10:
        raise ValueError("Seed 42 failed to reproduce the original selected-model score")

    paired = frame.pivot(index="seed", columns="stemming", values=["macro_f1", "accuracy"])
    macro_delta = paired["macro_f1"][True] - paired["macro_f1"][False]
    accuracy_delta = paired["accuracy"][True] - paired["accuracy"][False]
    predictions = np.stack(stem_test_predictions)
    consensus = []
    for col in predictions.T:
        _, counts = np.unique(col, return_counts=True)
        consensus.append(int(counts.max()))
    consensus = np.asarray(consensus)
    stem_rows = frame.loc[frame["stemming"]]
    plain_rows = frame.loc[~frame["stemming"]]
    summary = {
        "seeds": list(SEEDS),
        "split": "five stratified 80/20 holdouts; same fixed parameters and paired indices for both models",
        "stemmed_validation_accuracy_mean": float(stem_rows["accuracy"].mean()),
        "stemmed_validation_accuracy_sample_std": float(stem_rows["accuracy"].std(ddof=1)),
        "stemmed_validation_accuracy_min": float(stem_rows["accuracy"].min()),
        "stemmed_validation_accuracy_max": float(stem_rows["accuracy"].max()),
        "stemmed_validation_macro_f1_mean": float(stem_rows["macro_f1"].mean()),
        "stemmed_validation_macro_f1_sample_std": float(stem_rows["macro_f1"].std(ddof=1)),
        "stemmed_validation_macro_f1_min": float(stem_rows["macro_f1"].min()),
        "stemmed_validation_macro_f1_max": float(stem_rows["macro_f1"].max()),
        "stemmed_macro_f1_without_exact_overlap_mean": float(stem_rows["macro_f1_without_exact_overlap"].mean()),
        "stemmed_seed42_macro_f1_without_exact_overlap": float(stem_rows.loc[stem_rows["seed"] == 42, "macro_f1_without_exact_overlap"].iloc[0]),
        "stemmed_seed42_accuracy_without_exact_overlap": float(stem_rows.loc[stem_rows["seed"] == 42, "accuracy_without_exact_overlap"].iloc[0]),
        "plain_validation_macro_f1_mean": float(plain_rows["macro_f1"].mean()),
        "stem_minus_plain_macro_f1_mean": float(macro_delta.mean()),
        "stem_minus_plain_macro_f1_positive_splits": int((macro_delta > 0).sum()),
        "stem_minus_plain_accuracy_mean": float(accuracy_delta.mean()),
        "test_prediction_all_five_agree_rows": int((consensus == len(SEEDS)).sum()),
        "test_prediction_at_least_four_agree_rows": int((consensus >= len(SEEDS) - 1).sum()),
        "test_prediction_total_rows": len(test),
        "mean_agreement_with_full_train_submission": float(stem_rows["agreement_with_full_train_submission"].mean()),
        "note": "Prediction agreement is stability only, not unlabeled test accuracy; repeated holdouts overlap and are not independent test sets.",
    }
    frame.to_csv(OUT / "split_scores.csv", index=False)
    pd.DataFrame({
        "seed": list(SEEDS),
        "macro_f1_stem_minus_plain": macro_delta.reindex(SEEDS).to_numpy(),
        "accuracy_stem_minus_plain": accuracy_delta.reindex(SEEDS).to_numpy(),
    }).to_csv(OUT / "paired_differences.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "duplicate_audit.json").write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(duplicates, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
