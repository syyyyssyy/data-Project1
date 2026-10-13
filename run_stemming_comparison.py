"""Controlled Snowball stemming comparison on the existing Experiment 1 split."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from run_experiment import BASE_DIR, RANDOM_STATE, RESULT_DIR, clean_headers, load_data
from third_party.snowball_english import EnglishStemmer


STEM_RESULT_DIR = BASE_DIR / "stemming_results"


class StemmedTokenizer:
    """Use sklearn's default token boundaries; remove stop words before stemming."""

    def __init__(self, remove_stop_words: bool):
        self.remove_stop_words = remove_stop_words
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.stemmer = EnglishStemmer()

    @lru_cache(maxsize=100000)
    def stem(self, word: str) -> str:
        return self.stemmer.stemWord(word)

    def __call__(self, text: str) -> list[str]:
        words = self.token_pattern.findall(text)
        if self.remove_stop_words:
            words = [word for word in words if word not in ENGLISH_STOP_WORDS]
        return [self.stem(word) for word in words]


def vectorizer_params(stop_words: bool) -> dict:
    params = {
        "max_features": 50000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
    }
    if stop_words:
        params["stop_words"] = "english"
    return params


def make_pipeline(stop_words: bool, stemming: bool) -> Pipeline:
    params = vectorizer_params(stop_words)
    if stemming:
        params.pop("stop_words", None)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            tokenizer=StemmedTokenizer(stop_words),
            token_pattern=None,
            **params,
        )
    else:
        vectorizer = TfidfVectorizer(lowercase=True, **params)
    return Pipeline(
        [("tfidf", vectorizer), ("clf", LinearSVC(C=0.5, random_state=RANDOM_STATE, dual="auto"))]
    )


def name_for(stop_words: bool, stemming: bool) -> str:
    feature = "tfidf_50k_bigram_stop" if stop_words else "tfidf_50k_bigram"
    suffix = "__snowball_stemming" if stemming else ""
    return f"clean_headers__{feature}__LinearSVC__C-0.5{suffix}"


def save_json(path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    train_df, test_df = load_data()
    train_part, val_part = train_test_split(
        train_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train_df["target"],
    )
    train_texts = train_part["text"].map(clean_headers)
    val_texts = val_part["text"].map(clean_headers)
    STEM_RESULT_DIR.mkdir(exist_ok=True)

    previous = pd.read_csv(RESULT_DIR / "validation_results.csv")
    previous = previous[~previous["name"].str.endswith("__snowball_stemming")].copy()
    if len(previous) != 88:
        raise ValueError("Expected the original 88 full-mode results. Run run_experiment.py --mode full first.")

    rows = []
    predictions = {}
    for stop_words in (False, True):
        for stemming in (False, True):
            name = name_for(stop_words, stemming)
            print(f"Evaluating {name}", flush=True)
            pipeline = make_pipeline(stop_words, stemming)
            started = time.perf_counter()
            pipeline.fit(train_texts, train_part["target"])
            fit_seconds = time.perf_counter() - started
            pred = pipeline.predict(val_texts)
            row = {
                "name": name,
                "text_variant": "clean_headers",
                "stemming": stemming,
                "model": "LinearSVC",
                "model_params": json.dumps({"C": 0.5}),
                "vectorizer_params": json.dumps(vectorizer_params(stop_words), sort_keys=True),
                "accuracy": accuracy_score(val_part["target"], pred),
                "macro_f1": f1_score(val_part["target"], pred, average="macro"),
                "weighted_f1": f1_score(val_part["target"], pred, average="weighted"),
                "fit_seconds": fit_seconds,
            }
            rows.append(row)
            predictions[name] = pred
            print(f"  accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}", flush=True)

            if not stemming:
                prior = previous.loc[previous["name"] == name]
                if len(prior) != 1 or abs(row["macro_f1"] - float(prior.iloc[0]["macro_f1"])) > 1e-10:
                    raise ValueError(f"Original control result did not reproduce: {name}")

    comparison = pd.DataFrame(rows)
    stem_rows = comparison.loc[comparison["stemming"]].copy()
    previous["stemming"] = False
    all_results = pd.concat([previous, stem_rows], ignore_index=True)
    all_results = all_results.sort_values(["macro_f1", "accuracy"], ascending=False)
    best = all_results.iloc[0]
    best_name = best["name"]
    if best_name not in predictions:
        raise ValueError("The best result lies outside the controlled comparison.")

    stop_words = "_bigram_stop__" in best_name
    stemming = bool(best["stemming"])
    best_config = {
        "name": best_name,
        "text_variant": "clean_headers",
        "stemming": stemming,
        "stemmer": "Snowball English 3.1.1" if stemming else None,
        "model": "LinearSVC",
        "model_params": {"C": 0.5},
        "vectorizer_params": vectorizer_params(stop_words),
        "validation_macro_f1": float(best["macro_f1"]),
    }

    # Finish all validation comparisons before using every labeled row for the submission model.
    final_model = make_pipeline(stop_words, stemming)
    final_model.fit(train_df["text"].map(clean_headers), train_df["target"])
    test_predictions = final_model.predict(test_df["text"].map(clean_headers))

    comparison.to_csv(STEM_RESULT_DIR / "stemming_comparison.csv", index=False)
    all_results.to_csv(STEM_RESULT_DIR / "validation_results.csv", index=False)
    save_json(STEM_RESULT_DIR / "best_config.json", best_config)
    labels = sorted(train_df["target"].unique())
    best_pred = predictions[best_name]
    (STEM_RESULT_DIR / "best_validation_report.txt").write_text(
        classification_report(val_part["target"], best_pred, labels=labels, digits=4, zero_division=0),
        encoding="utf-8",
    )
    pd.DataFrame(
        confusion_matrix(val_part["target"], best_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(STEM_RESULT_DIR / "best_confusion_matrix.csv")
    metadata_path = RESULT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["mode"] = "full + controlled Snowball stemming comparison"
    metadata["experiment_count"] = len(all_results)
    metadata["stemming_library"] = "snowballstemmer 3.1.1, English algorithm (vendored)"
    save_json(STEM_RESULT_DIR / "metadata.json", metadata)
    pd.DataFrame(test_predictions).to_csv(BASE_DIR / "predictions_stemming.csv", index=False, header=False)
    print(f"Selected {best_name}: macro-F1={best['macro_f1']:.4f}")
    print(f"Saved {len(all_results)} results and {len(test_predictions)} final predictions")


if __name__ == "__main__":
    main()
