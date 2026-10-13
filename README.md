# Experiment 1: Text Classification

This repository contains the data, code, validation results, and prediction files for a 10-class English text classification experiment. The final test set has no labels, so all accuracy and F1 scores below are **validation** results.

## Environment

Python 3.10 was used. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The Snowball English stemmer used by the final model is included under `third_party/snowball_english/` with its license.

## Reproduce the experiment

Run these commands from the repository root, in this order:

```bash
python run_experiment.py --mode full
python run_stemming_comparison.py
python run_mlp_comparison.py
python check_reliability.py
```

1. `run_experiment.py --mode full` uses a stratified 80/20 split with seed 42. It compares 88 combinations of text treatment, TF-IDF features, and three classifiers. It writes `results/` and the initial `predictions.csv`.
2. `run_stemming_comparison.py` reads the 88 saved results, verifies the two unchanged controls, and compares two additional stemming configurations. It selects the best validation configuration, retrains it on all 7,368 labeled rows, and writes `stemming_results/` and **`predictions_stemming.csv`**, the final submission file. There is no separate final-model script because this step already trains and exports the selected model.
3. `run_mlp_comparison.py` compares three MLP configurations using the selected stemmed features. Its saved `mlp_results/summary.json` records that none exceeded the selected linear SVM, so `predictions_stemming.csv` remains the final prediction file.
4. `check_reliability.py` evaluates fixed configurations over five stratified splits and checks exact text duplicates. It writes `reliability_results/` without changing the submitted predictions.

Do not run the optional `--mode quick` before the stemming comparison: it replaces the full 88-row results that the comparison expects.

## Data and outputs

| Path | Purpose |
| --- | --- |
| `train_data.csv` | 7,368 labeled rows with `text,target` columns |
| `test_data_unlabeled.csv` | 2,457 unlabeled rows with a `text` column |
| `results/` | 88-run validation table, chosen configuration, per-class report, confusion matrix, and metadata |
| `stemming_results/` | Stemming comparison, combined 90-configuration results, chosen configuration, and validation details |
| `mlp_results/` | Three MLP comparisons and selection summary |
| `reliability_results/` | Five-split scores, paired differences, and duplicate audit |
| `predictions.csv` | Earlier non-stemmed model predictions; one label per row, no header |
| `predictions_stemming.csv` | **Final selected model predictions**; one label per row, no header |

The selected model removes specified email header lines, applies English stop-word filtering and Snowball stemming, builds unigram/bigram TF-IDF features (`max_features=50000`, `min_df=2`, `sublinear_tf=True`), and trains `LinearSVC(C=0.5)`. Its seed-42 validation accuracy is 0.9484 and macro-F1 is 0.9488. True performance on `test_data_unlabeled.csv` cannot be measured locally because that file has no target labels.
