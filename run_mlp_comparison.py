"""Small, fixed MLP comparison on the Experiment 1 validation split."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from run_experiment import BASE_DIR, RANDOM_STATE, clean_headers, load_data
from run_stemming_comparison import StemmedTokenizer, vectorizer_params


OUTPUT_DIR = BASE_DIR / "mlp_results"
CANDIDATES = (
    {"hidden_layer_sizes": (50,), "alpha": 0.0001},
    {"hidden_layer_sizes": (100,), "alpha": 0.0001},
    {"hidden_layer_sizes": (100,), "alpha": 0.01},
)
COMMON_PARAMS = {
    "solver": "adam",
    "batch_size": 256,
    "max_iter": 50,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "tol": 0.001,
    "random_state": RANDOM_STATE,
}


def make_vectorizer() -> TfidfVectorizer:
    params = vectorizer_params(stop_words=True)
    params.pop("stop_words")
    return TfidfVectorizer(
        lowercase=True,
        tokenizer=StemmedTokenizer(remove_stop_words=True),
        token_pattern=None,
        **params,
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_df, test_df = load_data()
    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["target"],
    )

    print("Fitting TF-IDF on the training split only", flush=True)
    started_features = time.perf_counter()
    vectorizer = make_vectorizer()
    x_train = vectorizer.fit_transform(train_part["text"].map(clean_headers))
    x_val = vectorizer.transform(val_part["text"].map(clean_headers))
    feature_seconds = time.perf_counter() - started_features
    print(f"TF-IDF shape={x_train.shape}, seconds={feature_seconds:.1f}", flush=True)

    reference = LinearSVC(C=0.5, random_state=RANDOM_STATE, dual="auto")
    reference_started = time.perf_counter()
    reference.fit(x_train, train_part["target"])
    reference_seconds = time.perf_counter() - reference_started
    reference_train = reference.predict(x_train)
    reference_val = reference.predict(x_val)
    svm_reference = {
        "model": "LinearSVC(C=0.5)",
        "train_accuracy": accuracy_score(train_part["target"], reference_train),
        "train_macro_f1": f1_score(train_part["target"], reference_train, average="macro"),
        "validation_accuracy": accuracy_score(val_part["target"], reference_val),
        "validation_macro_f1": f1_score(val_part["target"], reference_val, average="macro"),
        "fit_seconds": reference_seconds,
    }
    write_json(OUTPUT_DIR / "svm_reference.json", svm_reference)
    previous_svm = json.loads((BASE_DIR / "stemming_results" / "best_config.json").read_text(encoding="utf-8"))
    if abs(svm_reference["validation_macro_f1"] - previous_svm["validation_macro_f1"]) > 1e-10:
        raise ValueError("SVM reference did not reproduce the previous validation score")
    print(f"SVM reference reproduced macro-F1={svm_reference['validation_macro_f1']:.4f}", flush=True)

    rows = []
    predictions = {}
    for params in CANDIDATES:
        name = f"MLP_{params['hidden_layer_sizes'][0]}_alpha_{params['alpha']}"
        print(f"Training {name}", flush=True)
        classifier = MLPClassifier(**COMMON_PARAMS, **params)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x_train, train_part["target"])
        fit_seconds = time.perf_counter() - started
        pred = classifier.predict(x_val)
        train_pred = classifier.predict(x_train)
        warning_messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
        row = {
            "name": name,
            "hidden_layer_sizes": str(params["hidden_layer_sizes"]),
            "alpha": params["alpha"],
            "max_iter": COMMON_PARAMS["max_iter"],
            "n_iter": int(classifier.n_iter_),
            "stopped_early": bool(classifier.n_iter_ < COMMON_PARAMS["max_iter"]),
            "internal_best_accuracy": float(classifier.best_validation_score_),
            "final_train_loss": float(classifier.loss_),
            "convergence_warning": " | ".join(warning_messages),
            "accuracy": accuracy_score(val_part["target"], pred),
            "macro_f1": f1_score(val_part["target"], pred, average="macro"),
            "weighted_f1": f1_score(val_part["target"], pred, average="weighted"),
            "train_accuracy": accuracy_score(train_part["target"], train_pred),
            "train_macro_f1": f1_score(train_part["target"], train_pred, average="macro"),
            "fit_seconds": fit_seconds,
        }
        rows.append(row)
        predictions[name] = pred
        print(
            f"  accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, "
            f"epochs={row['n_iter']}, seconds={fit_seconds:.1f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    results.to_csv(OUTPUT_DIR / "mlp_validation_results.csv", index=False)
    best = results.iloc[0]
    selected_params = next(
        params for params in CANDIDATES if f"MLP_{params['hidden_layer_sizes'][0]}_alpha_{params['alpha']}" == best["name"]
    )
    labels = sorted(train_df["target"].unique())
    best_pred = predictions[best["name"]]
    (OUTPUT_DIR / "best_mlp_validation_report.txt").write_text(
        classification_report(val_part["target"], best_pred, labels=labels, digits=4, zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(
        confusion_matrix(val_part["target"], best_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(OUTPUT_DIR / "best_mlp_confusion_matrix.csv")

    previous = json.loads((BASE_DIR / "stemming_results" / "best_config.json").read_text(encoding="utf-8"))
    best_previous = float(previous["validation_macro_f1"])
    selected = float(best["macro_f1"]) > best_previous
    summary = {
        "purpose": "Fixed MLP candidates compared on the existing Experiment 1 split",
        "random_state": RANDOM_STATE,
        "train_split_rows": len(train_part),
        "validation_rows": len(val_part),
        "feature_method": "clean_headers + Snowball English + English stop words + TF-IDF",
        "vectorizer_params": vectorizer_params(stop_words=True),
        "feature_count": x_train.shape[1],
        "feature_fit_seconds": feature_seconds,
        "common_mlp_params": COMMON_PARAMS,
        "candidate_count": len(CANDIDATES),
        "best_mlp": best["name"],
        "best_mlp_macro_f1": float(best["macro_f1"]),
        "previous_best_macro_f1": best_previous,
        "mlp_selected": selected,
        "selected_model": best["name"] if selected else previous["name"],
    }

    if selected:
        print("MLP improved macro-F1; refitting on all labeled data", flush=True)
        full_vectorizer = make_vectorizer()
        x_full = full_vectorizer.fit_transform(train_df["text"].map(clean_headers))
        x_test = full_vectorizer.transform(test_df["text"].map(clean_headers))
        final_model = MLPClassifier(**COMMON_PARAMS, **selected_params)
        final_model.fit(x_full, train_df["target"])
        pd.DataFrame(final_model.predict(x_test)).to_csv(
            BASE_DIR / "predictions_mlp.csv", index=False, header=False
        )
        summary["final_prediction_file"] = "predictions_mlp.csv"
    else:
        summary["final_prediction_file"] = "predictions_stemming.csv"

    write_json(OUTPUT_DIR / "summary.json", summary)
    print(f"Selected: {summary['selected_model']}", flush=True)
    print(f"Final prediction file: {summary['final_prediction_file']}", flush=True)


if __name__ == "__main__":
    main()
