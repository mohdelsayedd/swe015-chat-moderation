"""Stage 3b: edit-distance lexicon matcher, the rule baseline (Area B, UC4-B2).

Idea: keep a list (lexicon) of words that signal an offensive message. A message is
flagged if any of its words is within a small edit distance of a lexicon word, so
misspellings like "stuupid" or "idiott" still match "stupid" / "idiot".

Where the lexicon comes from (decision D2):
    We build it from the TRAINING split only. For every word we compare how often
    it appears in offensive vs. clean training tweets (a smoothed log-odds ratio)
    and keep the most offensive-leaning words. No outside word list is needed,
    and the validation/test data never influence it.

Design choice, not tuned: max edit distance d = 1 (one typo allowed).
    Live game chat is full of typos ("idoit", "stupd"); tolerating them is the
    whole point of this matcher. Tweets in our data are mostly spelled correctly,
    so on validation d = 0 can score about the same or slightly higher. We print
    both so the trade-off is visible, and keep d = 1 for the reason above.

Only non-words get typo tolerance (classic "non-word spelling correction"):
    A token that is a normal word in the training vocabulary ("aunt", "shut",
    "sick") is NOT a typo, so it must match a lexicon word exactly. Only unknown
    tokens ("idoit", "stupd") are compared by edit distance. Without this rule,
    "aunt" would match "cunt" and "shut" would match "shit".

What we tune (on VALIDATION only, Hard Rule 2):
    - lexicon size K: how many top words we keep

Run from the repo root:
    python -m src.lexicon
Output: models/lexicon.json and results/val_lexicon.json
"""

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from src.data import LABEL_NAMES, RESULTS_DIR, load_splits, set_seed
from src.preprocess import tokenize

MODEL_DIR = Path("models")
LEXICON_PATH = MODEL_DIR / "lexicon.json"

MIN_COUNT = 5            # ignore words seen in fewer than 5 training tweets (too noisy)
GRID_K = [50, 100, 200, 400]
MAX_DIST = 1             # design choice, see module docstring
MIN_LEN_FOR_TYPO = 4     # words shorter than this must match exactly ("ass" vs "as")
KNOWN_MIN_COUNT = 2      # a token seen in >= 2 training tweets counts as a real word


# ---------------------------------------------------------------------------
# Minimum edit distance, classic dynamic programming, with one extension:
# swapping two neighbouring letters ("idoit" -> "idiot") counts as ONE edit.
# This variant is called Damerau-Levenshtein (optimal string alignment).
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str, max_dist: int = 99) -> int:
    """Minimum number of edits to turn a into b.

    Allowed edits: insert a letter, delete a letter, substitute a letter, or swap
    two neighbouring letters. We fill the DP table one row at a time:
    row[j] = distance between a[:i] and b[:j]. If two rows in a row are entirely
    above max_dist, the final answer must be too, so we stop early (speed).
    """
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev2 = None                                   # row i-2 (needed for swaps)
    prev = list(range(len(b) + 1))                 # row i-1
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,              # delete a[i-1]
                         cur[j - 1] + 1,           # insert b[j-1]
                         prev[j - 1] + cost)       # substitute (free if equal)
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                cur[j] = min(cur[j], prev2[j - 2] + 1)   # swap two neighbours
        if min(cur) > max_dist and min(prev) > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[-1]


# ---------------------------------------------------------------------------
# Building the lexicon from training data
# ---------------------------------------------------------------------------
def build_ranked_words(texts, labels) -> list[tuple[str, float]]:
    """Rank words by how strongly they lean towards the offensive class.

    For each word w we count the tweets that contain it, per class, and compute
        score(w) = log( (off_w + 1) / (n_off + 2) ) - log( (clean_w + 1) / (n_clean + 2) )
    i.e. the log of how much more frequent w is in offensive tweets. The "+1"
    (add-one smoothing) stops a word seen only once from getting an infinite score.
    """
    off_counts, clean_counts = Counter(), Counter()
    n_off = n_clean = 0
    for text, label in zip(texts, labels):
        words = set(tokenize(text))           # set: count each tweet once per word
        if label == 1:
            off_counts.update(words)
            n_off += 1
        else:
            clean_counts.update(words)
            n_clean += 1

    ranked = []
    for word in set(off_counts) | set(clean_counts):
        if off_counts[word] + clean_counts[word] < MIN_COUNT:
            continue
        score = (math.log((off_counts[word] + 1) / (n_off + 2))
                 - math.log((clean_counts[word] + 1) / (n_clean + 2)))
        if score > 0:                          # keep offensive-leaning words only
            ranked.append((word, round(score, 4)))
    ranked.sort(key=lambda pair: -pair[1])
    return ranked


def build_known_words(texts) -> list[str]:
    """Training vocabulary: tokens that appear in at least KNOWN_MIN_COUNT tweets."""
    counts = Counter()
    for text in texts:
        counts.update(set(tokenize(text)))
    return sorted(w for w, c in counts.items() if c >= KNOWN_MIN_COUNT)


class LexiconMatcher:
    """Rule baseline. predict_proba() returns 1.0 (match) or 0.0 (no match)."""

    name = "Rule baseline (edit-distance lexicon)"

    def __init__(self, words: list[str], max_dist: int, known_words=()):
        self.words = list(words)
        self.exact = set(self.words)
        self.max_dist = max_dist
        self.known = set(known_words)   # real words: never treated as typos
        self._closest = lru_cache(maxsize=100_000)(self._closest_uncached)

    def _closest_uncached(self, token: str):
        """Return the lexicon word that `token` matches, or None (cached per token)."""
        if token in self.exact:
            return token
        if self.max_dist == 0 or len(token) < MIN_LEN_FOR_TYPO or token in self.known:
            return None
        for word in self.words:
            if len(word) >= MIN_LEN_FOR_TYPO and edit_distance(token, word, self.max_dist) <= self.max_dist:
                return word
        return None

    def matches(self, text: str) -> list[tuple[str, str]]:
        """(token in message, lexicon word it matched). Shown in the app."""
        hits = []
        for token in tokenize(text):
            word = self._closest(token)
            if word is not None:
                hits.append((token, word))
        return hits

    def predict_proba(self, texts) -> list[float]:
        return [1.0 if self.matches(t) else 0.0 for t in texts]

    def predict(self, texts) -> list[int]:
        return [int(p >= 0.5) for p in self.predict_proba(texts)]

    def save(self, path: Path = LEXICON_PATH) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"max_dist": self.max_dist, "words": self.words,
                                    "known_words": sorted(self.known)}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = LEXICON_PATH) -> "LexiconMatcher":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["words"], data["max_dist"], data.get("known_words", ()))


def main() -> None:
    set_seed()
    splits = load_splits()
    train, val = splits["train"], splits["validation"]

    ranked = build_ranked_words(train["text"], train["label"])
    known = build_known_words(train["text"])
    print(f"Offensive-leaning words found in training data: {len(ranked)}")
    print("Top 25:", ", ".join(w for w, _ in ranked[:25]))

    # Grid search over K on VALIDATION. The test split is not touched here.
    # The d=0 column is only a comparison (exact match, no typo tolerance).
    print("\n   K   val macro-F1 (d=1, used)   (d=0, comparison)")
    best = None
    comparison = {}
    for k in GRID_K:
        words = [w for w, _ in ranked[:k]]
        f1_d1 = f1_score(val["label"], LexiconMatcher(words, MAX_DIST, known).predict(val["text"]), average="macro")
        f1_d0 = f1_score(val["label"], LexiconMatcher(words, 0).predict(val["text"]), average="macro")
        comparison[k] = {"d1": round(f1_d1, 4), "d0": round(f1_d0, 4)}
        print(f"{k:>5}   {f1_d1:.4f}                     {f1_d0:.4f}")
        if best is None or f1_d1 > best[0]:
            best = (f1_d1, k)

    f1, k = best
    d = MAX_DIST
    print(f"\nChosen on validation: K={k} (d={d} fixed by design), val macro-F1 {f1:.4f}")
    matcher = LexiconMatcher([w for w, _ in ranked[:k]], d, known)
    matcher.save()

    preds = matcher.predict(val["text"])
    print(classification_report(val["label"], preds, target_names=list(LABEL_NAMES.values()), digits=4))

    print("Example fuzzy matches (token -> lexicon word) on validation:")
    shown = 0
    for text in val["text"]:
        for token, word in matcher.matches(text):
            if token != word and shown < 8:
                print(f"  {token} -> {word}")
                shown += 1

    print("\nTypo-tolerance demo on made-up chat lines:")
    for line in ["ur such an idoit lol", "my aunt is sick", "shut up and play", "stuuupid n00b"]:
        print(f"  {line!r:28} -> {matcher.matches(line) or 'no match'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "val_lexicon.json").write_text(json.dumps({
        "model": matcher.name, "chosen": {"K": k, "max_dist": d},
        "val_macro_f1": round(f1, 4), "split_used_for_tuning": "validation",
        "grid_val_macro_f1": comparison,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {LEXICON_PATH} and {RESULTS_DIR / 'val_lexicon.json'}")


if __name__ == "__main__":
    main()
