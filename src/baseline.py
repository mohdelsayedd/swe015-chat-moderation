"""Stage 3c: classical baseline = regex normalization + tf-idf + logistic regression.

Area B (tf-idf, UC4-B3). This is the "classical baseline" rung of the model ladder.

How it works, in three lines:
  1. normalize() cleans the text (src/preprocess.py).
  2. TfidfVectorizer turns each message into a sparse vector: one number per word
     or word pair, high when the word is frequent in this message but rare overall.
  3. LogisticRegression learns one weight per word; a positive weight pushes the
     message towards "offensive". Output is a probability, which we show as confidence.

What we tune (on VALIDATION only): C, the inverse regularization strength
(small C = simpler model, large C = fits the training data more closely).
class_weight="balanced" because only about a third of the tweets are offensive.

Run from the repo root:
    python -m src.baseline
Output: models/tfidf_lr.joblib and results/val_tfidf_lr.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.data import LABEL_NAMES, RESULTS_DIR, SEED, load_splits, set_seed
from src.preprocess import normalize

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "tfidf_lr.joblib"
GRID_C = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def make_pipeline(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize,      # our regex normalizer runs first
            ngram_range=(1, 2),          # single words and word pairs ("shut up")
            min_df=2,                    # ignore n-grams seen in only one tweet
            sublinear_tf=True,           # 1 + log(tf): repeating a word 10x != 10x the signal
        )),
        ("clf", LogisticRegression(
            C=c,
            class_weight="balanced",     # classes are imbalanced (about 1/3 offensive)
            max_iter=2000,
            random_state=SEED,
        )),
    ])


class TfidfBaseline:
    """Thin wrapper so every model in the project has the same interface."""

    name = "tf-idf + logistic regression"

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict_proba(self, texts) -> list[float]:
        """Probability that each message is offensive."""
        return self.pipeline.predict_proba(list(texts))[:, 1].tolist()

    def predict(self, texts) -> list[int]:
        return [int(p >= 0.5) for p in self.predict_proba(texts)]

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "TfidfBaseline":
        return cls(joblib.load(path))


def main() -> None:
    set_seed()
    splits = load_splits()
    train, val = splits["train"], splits["validation"]

    # Grid search on VALIDATION (test split untouched until Stage 7).
    print("     C   val macro-F1")
    best = None
    for c in GRID_C:
        pipe = make_pipeline(c).fit(train["text"], train["label"])
        f1 = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
        print(f"{c:>6}   {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, c, pipe)

    f1, c, pipe = best
    print(f"\nChosen on validation: C={c} (val macro-F1 {f1:.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    preds = pipe.predict(val["text"])
    print(classification_report(val["label"], preds, target_names=list(LABEL_NAMES.values()), digits=4))
    print("Confusion matrix on validation (rows = true, cols = predicted) [clean, offensive]:")
    print(confusion_matrix(val["label"], preds))

    # Most influential n-grams: good for explaining the model (and its biases).
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    print("\nTop 15 n-grams pushing towards OFFENSIVE:", ", ".join(vocab[order[-15:][::-1]]))
    print("Top 15 n-grams pushing towards CLEAN:    ", ", ".join(vocab[order[:15]]))
    print(f"Vocabulary size: {len(vocab)} n-grams")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "val_tfidf_lr.json").write_text(json.dumps({
        "model": TfidfBaseline.name, "chosen": {"C": c},
        "val_macro_f1": round(f1, 4), "split_used_for_tuning": "validation",
        "vocabulary_size": int(len(vocab)),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH} and {RESULTS_DIR / 'val_tfidf_lr.json'}")


if __name__ == "__main__":
    main()
