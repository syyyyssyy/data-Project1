from __future__ import annotations

import argparse
import json
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "experiment_results" / "baseline"


@dataclass(frozen=True)
class Experiment:
    name: str
    text_variant: str
    vectorizer_params: dict
    model_name: str
    model_params: dict


HEADER_PREFIXES = (
    "from:",
    "organization:",
    "lines:",
    "nntp-posting-host:",
    "distribution:",
    "reply-to:",
    "article-i.d.:",
    "sender:",
    "writes:",
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = BASE_DIR / "data" / "train_data.csv"
    test_path = BASE_DIR / "data" / "test_data_unlabeled.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)
    return train_df, test_df


def clean_headers(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if any(lower.startswith(prefix) for prefix in HEADER_PREFIXES):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def normalize_text(text: str) -> str:
    text = clean_headers(text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def transform_texts(texts: pd.Series, variant: str) -> pd.Series:
    if variant == "raw":
        return texts
    if variant == "clean_headers":
        return texts.map(clean_headers)
    if variant == "normalized":
        return texts.map(normalize_text)
    raise ValueError(f"Unknown text variant: {variant}")


def make_model(model_name: str, params: dict):
    if model_name == "MultinomialNB":
        return MultinomialNB(**params)
    if model_name == "LogisticRegression":
        return LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
    if model_name == "LinearSVC":
        return LinearSVC(random_state=RANDOM_STATE, dual="auto", **params)
    raise ValueError(f"Unknown model: {model_name}")


def build_experiments(mode: str) -> list[Experiment]:
    if mode == "quick":
        vectorizer_grid = [
            (
                "tfidf_20k_unigram",
                {
                    "max_features": 20000,
                    "ngram_range": (1, 1),
                    "min_df": 2,
                    "sublinear_tf": True,
                },
            )
        ]
        model_grid = [
            ("MultinomialNB", [{"alpha": 0.5}, {"alpha": 1.0}]),
            ("LogisticRegression", [{"C": 1.0}, {"C": 3.0}]),
            ("LinearSVC", [{"C": 0.5}, {"C": 1.0}]),
        ]
        text_variants = ["raw"]
    else:
        vectorizer_grid = [
            (
                "tfidf_20k_unigram",
                {
                    "max_features": 20000,
                    "ngram_range": (1, 1),
                    "min_df": 2,
                    "sublinear_tf": True,
                },
            ),
            (
                "tfidf_50k_unigram",
                {
                    "max_features": 50000,
                    "ngram_range": (1, 1),
                    "min_df": 2,
                    "sublinear_tf": True,
                },
            ),
            (
                "tfidf_50k_bigram",
                {
                    "max_features": 50000,
                    "ngram_range": (1, 2),
                    "min_df": 2,
                    "sublinear_tf": True,
                },
            ),
            (
                "tfidf_50k_bigram_stop",
                {
                    "max_features": 50000,
                    "ngram_range": (1, 2),
                    "min_df": 2,
                    "sublinear_tf": True,
                    "stop_words": "english",
                },
            ),
        ]
        model_grid = [
            ("MultinomialNB", [{"alpha": 0.1}, {"alpha": 0.5}, {"alpha": 1.0}]),
            ("LogisticRegression", [{"C": 0.3}, {"C": 1.0}, {"C": 3.0}, {"C": 10.0}]),
            ("LinearSVC", [{"C": 0.1}, {"C": 0.5}, {"C": 1.0}, {"C": 3.0}]),
        ]
        text_variants = ["raw", "clean_headers"]

    experiments: list[Experiment] = []
    for text_variant in text_variants:
        for vectorizer_name, vectorizer_params in vectorizer_grid:
            for model_name, params_list in model_grid:
                for params in params_list:
                    name = (
                        f"{text_variant}__{vectorizer_name}__"
                        f"{model_name}__"
                        + "_".join(f"{k}-{v}" for k, v in params.items())
                    )
                    experiments.append(
                        Experiment(
                            name=name,
                            text_variant=text_variant,
                            vectorizer_params=vectorizer_params,
                            model_name=model_name,
                            model_params=params,
                        )
                    )
    return experiments


def evaluate_experiment(
    exp: Experiment,
    train_texts: pd.Series,
    val_texts: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
) -> tuple[dict, Pipeline, np.ndarray]:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, **exp.vectorizer_params)),
            ("clf", make_model(exp.model_name, exp.model_params)),
        ]
    )

    started = time.perf_counter()
    pipeline.fit(transform_texts(train_texts, exp.text_variant), y_train)
    fit_seconds = time.perf_counter() - started

    pred = pipeline.predict(transform_texts(val_texts, exp.text_variant))
    row = {
        "name": exp.name,
        "text_variant": exp.text_variant,
        "model": exp.model_name,
        "model_params": json.dumps(exp.model_params, sort_keys=True),
        "vectorizer_params": json.dumps(exp.vectorizer_params, sort_keys=True),
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(y_val, pred, average="macro"),
        "weighted_f1": f1_score(y_val, pred, average="weighted"),
        "fit_seconds": fit_seconds,
    }
    return row, pipeline, pred


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="quick runs a small sanity grid; full runs the report grid.",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)
    train_df, test_df = load_data()

    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["target"],
    )
    labels = sorted(train_df["target"].unique())
    experiments = build_experiments(args.mode)

    metadata = {
        "random_state": RANDOM_STATE,
        "mode": args.mode,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_split_rows": int(len(train_part)),
        "validation_rows": int(len(val_part)),
        "label_counts": {
            str(k): int(v) for k, v in train_df["target"].value_counts().sort_index().items()
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    save_json(RESULT_DIR / "metadata.json", metadata)

    rows: list[dict] = []
    best_score = -1.0
    best_exp: Experiment | None = None
    best_pipeline: Pipeline | None = None
    best_pred: np.ndarray | None = None

    for index, exp in enumerate(experiments, start=1):
        print(f"[{index:02d}/{len(experiments):02d}] {exp.name}", flush=True)
        row, pipeline, pred = evaluate_experiment(
            exp,
            train_part["text"],
            val_part["text"],
            train_part["target"],
            val_part["target"],
        )
        rows.append(row)
        print(
            f"    accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f} "
            f"fit_seconds={row['fit_seconds']:.1f}",
            flush=True,
        )
        if row["macro_f1"] > best_score:
            best_score = row["macro_f1"]
            best_exp = exp
            best_pipeline = pipeline
            best_pred = pred

    results_df = pd.DataFrame(rows).sort_values(
        by=["macro_f1", "accuracy"], ascending=False
    )
    results_df.to_csv(RESULT_DIR / "validation_results.csv", index=False)

    assert best_exp is not None and best_pipeline is not None and best_pred is not None
    best_report = classification_report(
        val_part["target"],
        best_pred,
        labels=labels,
        digits=4,
        zero_division=0,
    )
    (RESULT_DIR / "best_validation_report.txt").write_text(best_report, encoding="utf-8")
    pd.DataFrame(
        confusion_matrix(val_part["target"], best_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(RESULT_DIR / "best_confusion_matrix.csv")

    best_config = {
        "name": best_exp.name,
        "text_variant": best_exp.text_variant,
        "model": best_exp.model_name,
        "model_params": best_exp.model_params,
        "vectorizer_params": best_exp.vectorizer_params,
        "validation_macro_f1": float(best_score),
    }
    save_json(RESULT_DIR / "best_config.json", best_config)

    final_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, **best_exp.vectorizer_params)),
            ("clf", make_model(best_exp.model_name, best_exp.model_params)),
        ]
    )
    final_pipeline.fit(transform_texts(train_df["text"], best_exp.text_variant), train_df["target"])
    test_predictions = final_pipeline.predict(transform_texts(test_df["text"], best_exp.text_variant))
    pd.DataFrame(test_predictions).to_csv(
        RESULT_DIR / "predictions.csv",
        index=False,
        header=False,
    )

    print("\nBest configuration:")
    print(json.dumps(best_config, ensure_ascii=False, indent=2))
    print(f"Saved results to {RESULT_DIR}")
    print(f"Saved baseline predictions to {RESULT_DIR / 'predictions.csv'}")


if __name__ == "__main__":
    main()
