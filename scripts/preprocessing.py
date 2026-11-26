"""
Shared text preprocessing and data loading for Group Gamma (seed 44).

One source of truth for how review text is cleaned, so that training data,
notebooks, and the Streamlit app all see text in exactly the same form.

    scripts/clean_reviews_script_final.py   builds data/group_cleaned_seed44.csv with these functions
    notebooks/*.ipynb                       load_cleaned(), split_frames(), TOKEN_PATTERN
    app.py                                  preprocess() on every review a user types

Usage from the repository root (or after adding it to sys.path):

    from scripts.preprocessing import load_cleaned, split_frames, preprocess, TOKEN_PATTERN
"""

from __future__ import annotations

import html
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pandas as pd
from spellchecker import SpellChecker


# ============================================================
# SHARED CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

SEED = 44
CLEANED_FILENAME = f"group_cleaned_seed{SEED}.csv"

# Validation share *of the training split*, used by the deep model for its
# training/validation loss curves. The 20% test split is never touched.
VAL_SIZE = 0.10

# Token pattern for CountVectorizer / TfidfVectorizer that keeps contractions
# whole ("can't", "don't") instead of scikit-learn's default r"(?u)\b\w\w+\b",
# which splits "can't" into "can" + "t" and then drops the "t", so the
# negation is lost. Tokens still need 2+ characters, like the default.
TOKEN_PATTERN = r"(?u)\b\w[\w']*\w\b"
DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"   # scikit-learn's default, for ablations


# ============================================================
# CORRECTION SETTINGS
# ============================================================

MAX_EDIT_DISTANCE = 1
MIN_WORD_LENGTH = 5
MIN_SUGGESTION_FREQUENCY = 100
MIN_FREQUENCY_RATIO = 20
REQUIRE_UNIQUE_CANDIDATE = True


# ============================================================
# MANUAL CORRECTIONS
# ============================================================

# Manual corrections always take priority.
# Mapping a word to itself preserves the original spelling.

MANUAL_CORRECTIONS = {
    "theyre": "they're",
    "doesn": "doesn't",
    "publix": "publix",
    "coulnt": "couldn't",
    "stevia": "stevia",

    # Standardized contractions
    "cant": "can't",
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "havent": "haven't",
    "hasnt": "hasn't",
    "hadnt": "hadn't",
    "theyve": "they've",
    "youre": "you're",
    "youve": "you've",
    "im": "i'm",
    "ive": "i've",
    "thats": "that's",
    "theres": "there's"
}


# ============================================================
# PROTECTED VOCABULARY
# ============================================================

PROTECTED = {
    "no", "not", "never", "none", "nothing",
    "neither", "nor", "cannot",
    "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "couldn't",
    "shouldn't", "isn't", "aren't",
    "wasn't", "weren't",
    "bad", "good", "terrible", "excellent",
    "awful", "great", "love", "hate",
    "recommend", "disappointed", "disappointing",
    "positive", "negative", "best", "worst",
    "better", "worse", "poor", "perfect",
    "amazing", "horrible", "delicious",
    "disgusting", "favorite", "favourite"
}

CUSTOM_WORDS = {
    "amazon", "kindle", "keurig", "nespresso",
    "starbucks", "nutella", "gatorade",
    "gluten", "decaf", "organic",
    "vegan", "nonfat", "sugarfree",
    "wifi", "ebook", "publix", "stevia"
}

# Match words while preserving punctuation and spacing.
WORD_RE = re.compile(
    r"[^\W\d_]+(?:['’][^\W\d_]+)*",
    re.UNICODE
)

# URLs and bare domains are skipped by the corrector so links are not altered.
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|net|org|edu|gov)\b\S*",
    re.IGNORECASE
)

QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'", "´": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
})


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_review(value):
    """
    Decode HTML entities, remove HTML tags,
    and normalize whitespace.

    Kept identical to the M2 notebooks' clean_review() so that duplicate
    detection (and therefore the row set and split) matches M2.
    """
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_quotes(text):
    """Replace typographic quotes/apostrophes with ASCII ones (don’t -> don't)."""
    return text.translate(QUOTE_TRANSLATION)


def restore_case(original, replacement):
    """
    Preserve capitalization of the original word.
    """
    if original.isupper():
        return replacement.upper()

    if original.istitle():
        return replacement.capitalize()

    return replacement


# ============================================================
# SPELLCHECKER INITIALIZATION
# ============================================================

spell = SpellChecker(
    language="en",
    distance=MAX_EDIT_DISTANCE,
    case_sensitive=False
)

spell.word_frequency.load_words(CUSTOM_WORDS)


# ============================================================
# CONSERVATIVE SPELLING CORRECTION
# ============================================================

@lru_cache(maxsize=100000)
def get_conservative_correction(word):
    """
    Return corrected word and correction status.

    Unknown or ambiguous words are preserved.
    """

    normalized = word.lower().replace("’", "'")

    # 1. Apply manual corrections first.
    if normalized in MANUAL_CORRECTIONS:

        replacement = MANUAL_CORRECTIONS[normalized]

        if replacement == normalized:
            return word, "manual_preserved"

        return (
            restore_case(word, replacement),
            "manual_corrected"
        )

    # 2. Preserve existing contractions.
    if "'" in normalized:
        return word, "protected_contraction"

    # 3. Preserve short words (this also protects most abbreviations).
    if len(normalized) < MIN_WORD_LENGTH:
        return word, "protected_short"

    # 4. Preserve sentiment words and brand names.
    if normalized in PROTECTED or normalized in CUSTOM_WORDS:
        return word, "protected_vocabulary"

    # 5. Preserve recognized English words.
    if not spell.unknown([normalized]):
        return word, "known_word"

    # 6. Generate spelling candidates.
    candidates = spell.candidates(normalized)

    if not candidates:
        return word, "unknown_no_candidate"

    candidates = {
        candidate
        for candidate in candidates
        if candidate != normalized
        and candidate not in PROTECTED
        and candidate.isalpha()
    }

    if not candidates:
        return word, "unknown_no_candidate"

    # Reject ambiguous suggestions.
    if REQUIRE_UNIQUE_CANDIDATE and len(candidates) != 1:
        return word, "ambiguous"

    best = max(
        candidates,
        key=lambda candidate: spell.word_frequency[candidate]
    )

    best_frequency = spell.word_frequency[best]
    original_frequency = spell.word_frequency[normalized]

    if best_frequency < MIN_SUGGESTION_FREQUENCY:
        return word, "low_frequency"

    frequency_ratio = (
        best_frequency / max(original_frequency, 1)
    )

    if frequency_ratio < MIN_FREQUENCY_RATIO:
        return word, "low_confidence"

    return (
        restore_case(word, best),
        "dictionary_corrected"
    )


# ============================================================
# MANUAL CONTRACTION NORMALIZATION
# ============================================================

def normalize_contractions(text):
    """
    Ensure all manually specified contractions are standardized.

    Examples:
        cant -> can't
        dont -> don't
        theyre -> they're
    """

    def replace_match(match):

        original = match.group(0)
        normalized = original.lower().replace("’", "'")

        if normalized in MANUAL_CORRECTIONS:

            replacement = MANUAL_CORRECTIONS[normalized]

            return restore_case(original, replacement)

        return original

    return WORD_RE.sub(replace_match, text)


# ============================================================
# REMOVE COMMAS AFTER NEGATIONS
# ============================================================

NEGATIONS = [
    "can't", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "couldn't",
    "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "haven't",
    "hasn't", "hadn't"
]

NEGATION_COMMA_RE = re.compile(
    r"\b(" +
    "|".join(re.escape(word) for word in NEGATIONS) +
    r"),(?=\s|$)",
    flags=re.IGNORECASE
)


def normalize_negation_commas(text):
    """
    Remove commas immediately following negations.

    Examples:
        "I can't, believe it" -> "I can't believe it"
        "I don't, like it" -> "I don't like it"
        "I won't, buy it" -> "I won't buy it"
    """

    return NEGATION_COMMA_RE.sub(r"\1", text)


# ============================================================
# CORRECT INDIVIDUAL REVIEW
# ============================================================

def correct_review(text, row_index, audit, counts):
    """
    Correct individual words while preserving sentence structure,
    punctuation, and any URLs.
    """

    def replace_match(match):

        original = match.group(0)

        corrected, status = get_conservative_correction(
            original
        )

        counts[status] += 1

        if corrected != original:

            audit.append({
                "row_index": row_index,
                "original_word": original,
                "corrected_word": corrected,
                "status": status
            })

        return corrected

    # Correct only the text between URLs; URLs are copied through unchanged.
    pieces = []
    last = 0

    for url in URL_RE.finditer(text):
        pieces.append(WORD_RE.sub(replace_match, text[last:url.start()]))
        pieces.append(url.group(0))
        counts["protected_url"] += 1
        last = url.end()

    pieces.append(WORD_RE.sub(replace_match, text[last:]))

    return "".join(pieces)


def correct_text(text, row_index=-1, audit=None, counts=None):
    """
    Apply every word-level correction to text that is already HTML-cleaned.

    Order: quotes -> spelling/manual corrections -> contractions -> negation commas.
    Returns (corrected_text, negation_commas_removed). Pass audit/counts to
    collect statistics (the cleaning script does); leave them out otherwise.
    """

    audit = [] if audit is None else audit
    counts = Counter() if counts is None else counts

    # Typographic apostrophes -> ASCII so "don’t" == "don't".
    after = normalize_quotes(text)

    # Dictionary and manual corrections.
    after = correct_review(after, row_index, audit, counts)

    # Final contraction standardization.
    after = normalize_contractions(after)

    # Remove commas immediately after negations.
    after, removed = NEGATION_COMMA_RE.subn(r"\1", after)

    # Safety check.
    if not after.strip():
        after = text
        counts["rejected_empty"] += 1

    return after, removed


def preprocess(raw_text):
    """
    Full preprocessing for ONE new review, exactly as the training text was built.

    Use this in the app (and anywhere else a model sees new text):
        model.predict_proba([preprocess(user_text)])
    """
    return correct_text(clean_review(raw_text))[0]


# ============================================================
# LOADING THE SHARED CLEANED DATASET
# ============================================================

def load_cleaned(data_dir=None):
    """
    Read data/group_cleaned_seed44.csv (built by clean_reviews_script_final.py).

    keep_default_na=False keeps reviews such as "NA" or "None" as text and
    empty summaries as "" instead of NaN.
    """

    path = Path(data_dir or DEFAULT_DATA_DIR) / CLEANED_FILENAME

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/clean_reviews_script_final.py` "
            "from the repository root first (see data/README.md)."
        )

    df = pd.read_csv(path, keep_default_na=False)

    expected = {"Text", "Sentiment", "split", "is_val"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing columns {sorted(missing)}. It was made by an "
            "older version of the cleaning script; run it again."
        )

    df["is_val"] = df["is_val"].astype(str).str.lower().eq("true")
    return df


def make_vectorizer(kind="tfidf", token_pattern=TOKEN_PATTERN, **overrides):
    """
    The team's feature settings from notebooks/feature_engineering.ipynb.

        kind="bow"    CountVectorizer, unigrams + bigrams, min_df=2, 50,000 terms
        kind="tfidf"  same vocabulary rules + sublinear tf, L2-normalized rows

    Fit it on training rows only (split_frames(df)["fit"] for the deep model).
    Any keyword overrides a default, e.g. make_vectorizer("tfidf", max_features=20000).
    """

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    settings = dict(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        lowercase=True,
        token_pattern=token_pattern,
    )

    if kind == "bow":
        settings.update(overrides)
        return CountVectorizer(**settings)

    if kind == "tfidf":
        settings["sublinear_tf"] = True
        settings.update(overrides)
        return TfidfVectorizer(**settings)

    raise ValueError(f"kind must be 'bow' or 'tfidf', got {kind!r}")


def split_frames(df):
    """
    The shared splits as dataframes.

        train  all training rows (classical models tune with CV on these)
        fit    training rows minus the validation slice (deep model trains on these)
        val    the validation slice of train (deep model's validation loss)
        test   the shared 20% held-out test set (final evaluation for every model)
    """

    train = df[df["split"] == "train"]
    return {
        "train": train,
        "fit": train[~train["is_val"]],
        "val": train[train["is_val"]],
        "test": df[df["split"] == "test"],
    }
