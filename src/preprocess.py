"""Stage 3a: regex normalization of noisy chat text (Area B, UC4-B1).

Chat messages are noisy: "y0u 1d10t", "stuuupiiid", "#LoserAlert", "&amp;".
Normalizing them first means the lexicon, tf-idf and word2vec all see the same
clean words, so "1d10t" and "idiot" count as the same word.

Every step below is a plain regular expression or a lookup table, nothing learned.
All models except DistilBERT use normalize(); DistilBERT gets the raw text,
because its own WordPiece tokenizer was pretrained on raw text.

Try it:
    python -m src.preprocess
"""

import html
import re

# Turkish characters -> closest ASCII letter (UC4-B1 asks for this; our data is
# English, but chat users mix languages, and the rule is cheap).
TURKISH_MAP = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u",
})

# Leetspeak: digits/symbols that people use in place of letters.
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
    "@": "a", "$": "s", "!": "i",
})

# --- Compiled regular expressions (compiled once, reused for every message) ---
RE_URL = re.compile(r"https?://\S+|www\.\S+|\burl\b", re.IGNORECASE)
RE_MENTION = re.compile(r"@user\b|@\w+", re.IGNORECASE)   # TweetEval writes "@user"
RE_HASHTAG = re.compile(r"#(\w+)")
# A "leet word": contains at least one letter AND at least one leet character,
# e.g. "1d10t", "sh!t", "a$$". Pure numbers like "2026" are left alone.
RE_LEET_WORD = re.compile(r"\b(?=[\w@$!]*[a-z])(?=[\w@$!]*[013457@$!])[\w@$!]+")
RE_REPEAT = re.compile(r"(\w)\1{2,}")                      # 3+ same chars in a row
RE_BANG = re.compile(r"!+(?![a-z])")                       # "!" NOT followed by a letter = punctuation
RE_NON_TEXT = re.compile(r"[^a-z0-9\s']")                  # drop emoji & punctuation
RE_SPACES = re.compile(r"\s+")
RE_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def normalize(text: str) -> str:
    """Return a cleaned, lower-case version of one chat message."""
    text = html.unescape(text)                  # "&amp;" -> "&"
    text = text.translate(TURKISH_MAP)          # do this before lower(): "İ".lower() is 2 chars
    text = text.lower()
    text = RE_URL.sub(" ", text)                # links carry no offensive meaning
    text = RE_MENTION.sub(" ", text)            # "@user" is in almost every tweet
    text = RE_HASHTAG.sub(r"\1", text)          # "#loser" -> "loser"
    text = RE_BANG.sub(" ", text)               # "idiot!!!" -> "idiot", but "sh!t" keeps its "!"
    text = RE_LEET_WORD.sub(lambda m: m.group(0).translate(LEET_MAP), text)  # "1d10t" -> "idiot"
    text = RE_REPEAT.sub(r"\1\1", text)         # "stuuupiiid" -> "stuupiid" (2 kept: "good")
    text = RE_NON_TEXT.sub(" ", text)
    return RE_SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """normalize() + split into words. Used by the lexicon and word2vec."""
    return RE_TOKEN.findall(normalize(text))


if __name__ == "__main__":
    # Demo for the preprocessing slide and for the viva: before -> after.
    examples = [
        "@user You are such an 1D10T!!! #LoserAlert",
        "stuuupiiiid game, u r sh!t at it",
        "Check this https://t.co/abc &amp; laugh URL",
        "Bu oyun çok güzel, İyi şanslar",
        "Good luck have fun :) 2026",
    ]
    for ex in examples:
        print(f"IN : {ex}\nOUT: {normalize(ex)}\nTOK: {tokenize(ex)}\n")
