"""Stage 2: load the dataset, keep the official fixed splits, and save a summary.

Dataset: TweetEval, "offensive" subset (English tweets from SemEval-2019 Task 6,
OffensEval / OLID). Labels: 0 = not offensive ("clean"), 1 = offensive.

Why we do NOT make our own split:
    TweetEval ships fixed train / validation / test splits. Using them means every
    model (rule baseline, tf-idf, word2vec+FFN, DistilBERT, quantized DistilBERT)
    is scored on exactly the same 'held-out' test data (R3), and nobody can say we
    picked a convenient split.

What this script does:
    1. Downloads the splits once and saves local copies in data/*.csv, so every
       later script reads identical rows, even offline. (data/ is git-ignored:
       we don't redistribute the tweets; the script re-downloads them.)
    2. Computes a fingerprint (hash) of each split. benchmark.py checks it later
       to prove all models used the same test set.
    3. Writes results/data_summary.json: the real sizes and label counts for the
       data slide.

Run from the repo root:
    python -m src.data            (use cached data/*.csv if present)
    python -m src.data --refresh  (force a fresh download)
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared constants: other stages import these, so there is one source of truth.
# ---------------------------------------------------------------------------
SEED = 42
LABEL_NAMES = {0: "clean", 1: "offensive"}   # UC4 wording: "clean / offensive"
SPLITS = ("train", "validation", "test")

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

# Tried in order. The first is the current home of TweetEval on the Hub; the
# second is the old short name, kept as a fallback.
DATASET_CANDIDATES = [
    ("cardiffnlp/tweet_eval", "offensive"),
    ("tweet_eval", "offensive"),
]


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # torch is not needed for the classical stages
        pass


def download_splits() -> tuple[dict, str]:
    """Download the three splits from the Hugging Face Hub as DataFrames."""
    from datasets import load_dataset  # imported here: only needed for download

    errors = []
    for name, config in DATASET_CANDIDATES:
        try:
            ds = load_dataset(name, config)
            splits = {
                split: ds[split].to_pandas()[["text", "label"]].reset_index(drop=True)
                for split in SPLITS
            }
            return splits, f"{name}/{config}"
        except Exception as exc:  # try the next candidate, remember why it failed
            errors.append(f"{name}/{config}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not download the dataset from any source:\n  " + "\n  ".join(errors)
        + "\nCheck your internet connection, then run again."
    )


def load_splits(refresh: bool = False) -> dict:
    """Return {'train': df, 'validation': df, 'test': df}, each with text + label.

    Every later stage calls this function, so they all see the same rows.
    """
    paths = {split: DATA_DIR / f"{split}.csv" for split in SPLITS}

    if refresh or not all(p.exists() for p in paths.values()):
        splits, source = download_splits()
        DATA_DIR.mkdir(exist_ok=True)
        for split, df in splits.items():
            df.to_csv(paths[split], index=False, encoding="utf-8")
        (DATA_DIR / "SOURCE.txt").write_text(source, encoding="utf-8")
        print(f"Downloaded from {source} and saved to {DATA_DIR}/")

    # keep_default_na=False: otherwise a tweet that is literally "NA" or "null"
    # would be read back as a missing value.
    return {
        split: pd.read_csv(paths[split], keep_default_na=False, encoding="utf-8")
        for split in SPLITS
    }


def split_fingerprint(df: pd.DataFrame) -> str:
    """Short SHA-256 hash of (text, label) rows. Same rows -> same fingerprint."""
    h = hashlib.sha256()
    for text, label in zip(df["text"], df["label"]):
        h.update(f"{text}\t{label}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def summarize(splits: dict) -> dict:
    """Facts for the data slide. Every number here is computed, never typed."""
    summary = {"label_names": LABEL_NAMES, "splits": {}}
    for split, df in splits.items():
        counts = df["label"].value_counts().sort_index()
        n = len(df)
        summary["splits"][split] = {
            "rows": n,
            "clean": int(counts.get(0, 0)),
            "offensive": int(counts.get(1, 0)),
            "offensive_pct": round(100 * counts.get(1, 0) / n, 1),
            "mean_words": round(df["text"].str.split().str.len().mean(), 1),
            "fingerprint": split_fingerprint(df),
        }

    # Leakage check: identical tweet text in both train and test.
    overlap = set(splits["train"]["text"]) & set(splits["test"]["text"])
    summary["train_test_exact_duplicates"] = len(overlap)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: data and splits")
    parser.add_argument("--refresh", action="store_true", help="force re-download")
    args = parser.parse_args()

    set_seed()
    splits = load_splits(refresh=args.refresh)

    # Sanity checks: fail loudly now rather than produce wrong numbers later.
    for split, df in splits.items():
        assert list(df.columns) == ["text", "label"], f"{split}: unexpected columns"
        assert set(df["label"].unique()) <= {0, 1}, f"{split}: unexpected labels"
        assert (df["text"].str.len() > 0).all(), f"{split}: empty text found"

    summary = summarize(splits)
    source_file = DATA_DIR / "SOURCE.txt"
    summary["source"] = source_file.read_text(encoding="utf-8") if source_file.exists() else "unknown"

    print("\n=== Splits ===")
    print(f"{'split':<11}{'rows':>7}{'clean':>8}{'offens.':>9}{'off.%':>7}{'words':>7}  fingerprint")
    for split, s in summary["splits"].items():
        print(f"{split:<11}{s['rows']:>7}{s['clean']:>8}{s['offensive']:>9}"
              f"{s['offensive_pct']:>7}{s['mean_words']:>7}  {s['fingerprint']}")
    print(f"\nExact train/test duplicate texts: {summary['train_test_exact_duplicates']}")
    print(f"Source: {summary['source']}")

    print("\n=== Two training examples per class ===")
    train = splits["train"]
    for label, name in LABEL_NAMES.items():
        for text in train[train["label"] == label]["text"].head(2):
            print(f"[{name}] {text[:110]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "data_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
