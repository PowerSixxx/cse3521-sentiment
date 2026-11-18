"""
Amazon Review Sampling and Conservative Dictionary Cleaning (Group Gamma, seed 44)

Run from the repository root:
    python scripts/clean_reviews_script_final.py
    python scripts/clean_reviews_script_final.py --data-dir path/to/folder

Install:
    pip install -r requirements.txt      (needs pandas, numpy, scikit-learn, pyspellchecker)

Input (in --data-dir, default: <repo>/data):
    Reviews.csv               full Kaggle dataset; sampled with seed 44, or
    group_sample_seed44.csv   the team's existing 50,000-row sample (used if Reviews.csv is absent)

Pipeline:
 1. Sample 50,000 reviews using seed 44 (identical to the instructor's sampling code)
 2. Save the untouched sample
 3. Remove missing, neutral (3-star), empty, and duplicate reviews
 4. Clean HTML entities/tags and whitespace (Text and Summary)
 5. Create binary Sentiment labels (1-2 = negative, 4-5 = positive)
 6. Assign the shared stratified 80/20 train/test split (seed 44)
 7. Normalize typographic quotes (’ -> ') so contractions are consistent
 8. Apply manual spelling/contraction corrections
 9. Apply conservative dictionary corrections (URLs are never touched)
10. Remove commas immediately following negations
11. Validate corrections and probe non-standard language
12. Save cleaned reviews, audit CSV, and JSON report

Design decisions (see scripts/PREPROCESSING_updated.md for the full rationale):
  * Punctuation is RETAINED. Classical vectorizers drop it at tokenization time anyway,
    while the deep model (and any contraction-aware tokenizer) can use "!", "?", and the
    apostrophe in "can't". Removing it here would be irreversible for every model.
  * No separate abbreviation/slang normalizer. Words shorter than 5 letters are never
    auto-corrected, which already protects most abbreviations (u, thx, pls, omg, lol).
    The report includes a probe that counts these forms so the decision is backed by data.
  * Correction stays conservative and context-free (unique candidate, edit distance 1,
    frequency checks). A 1,000-review pilot showed context-aware correction gave a
    negligible benefit for a much higher cost and risk of changing meaning.

Rows 1-6 reproduce the filtering in notebooks/classical_models.ipynb exactly, so the
row set and the train/test split match the M2 baseline results.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from spellchecker import SpellChecker


# ============================================================
# CONFIGURATION
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

SEED = 44
SAMPLE_SIZE = 50000
TEST_SIZE = 0.20

MAX_EDIT_DISTANCE = 1
MIN_WORD_LENGTH = 5
MIN_SUGGESTION_FREQUENCY = 100
MIN_FREQUENCY_RATIO = 20
REQUIRE_UNIQUE_CANDIDATE = True

PROGRESS_INTERVAL = 5000


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

# Common abbreviations / internet shorthand. Not corrected; only counted in the
# report so the "no abbreviation normalizer" decision is backed by evidence.
ABBREVIATION_PROBE = {
    "u", "ur", "r", "thx", "thanx", "pls", "plz", "omg", "lol", "btw",
    "imo", "imho", "tbh", "idk", "def", "prob", "awsome", "gr8", "luv",
    "w/", "w/o", "b/c", "bc", "smh", "fav", "fave", "yum", "yummm", "sooo",
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

# Letters repeated 3+ times ("sooooo", "yummmm") mark elongated, emphatic words.
ELONGATION_RE = re.compile(r"\b\w*([a-z])\1{2,}\w*\b", re.IGNORECASE)

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

    Kept identical to clean_review() in the notebooks so that duplicate
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


# ============================================================
# VALIDATION
# ============================================================

def validate_corrections(df):
    """
    Check whether uncorrected standalone contractions remain
    in the final Text column.
    """

    patterns = {
        "cant": r"\bcant\b",
        "dont": r"\bdont\b",
        "doesnt": r"\bdoesnt\b",
        "couldnt": r"\bcouldnt\b",
        "theyre": r"\btheyre\b"
    }

    results = {}

    for word, pattern in patterns.items():

        count = int(
            df["Text"].str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            ).sum()
        )

        results[word] = count

    return results


def probe_nonstandard_language(texts):
    """
    Measure how common abbreviations and elongated words are.

    Nothing is changed; the numbers justify leaving them as-is.
    """

    abbreviation_tokens = Counter()
    elongated_tokens = Counter()
    total_tokens = 0
    reviews_with_any = 0

    for text in texts:
        lowered = URL_RE.sub(" ", text.lower())
        tokens = re.findall(r"[a-z0-9/]+", lowered)
        total_tokens += len(tokens)

        abbr = [t for t in tokens if t in ABBREVIATION_PROBE]
        elong = [m.group(0) for m in ELONGATION_RE.finditer(lowered)]

        abbreviation_tokens.update(abbr)
        elongated_tokens.update(elong)

        if abbr or elong:
            reviews_with_any += 1

    def share(n):
        return round(n / max(total_tokens, 1) * 100, 4)

    return {
        "total_tokens": total_tokens,
        "abbreviation_tokens": sum(abbreviation_tokens.values()),
        "abbreviation_token_share_pct": share(sum(abbreviation_tokens.values())),
        "elongated_tokens": sum(elongated_tokens.values()),
        "elongated_token_share_pct": share(sum(elongated_tokens.values())),
        "reviews_with_any": reviews_with_any,
        "reviews_with_any_share_pct": round(
            reviews_with_any / max(len(texts), 1) * 100, 2
        ),
        "top_abbreviations": dict(abbreviation_tokens.most_common(15)),
        "top_elongations": dict(elongated_tokens.most_common(15)),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample and clean the Group Gamma Amazon review dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Folder holding Reviews.csv (or group_sample_seed44.csv). "
             "Outputs are written here too. Default: <repo>/data",
    )
    return parser.parse_args()


def load_sample(input_file, sample_file):
    """
    Return (sample dataframe, original dataset row count).

    Samples from Reviews.csv when available; otherwise reuses the team's
    saved sample, which was produced by the same seed-44 sampling code.
    """

    if input_file.is_file():

        print("Loading original dataset...", flush=True)

        raw = pd.read_csv(input_file)
        original_rows = len(raw)

        if original_rows < SAMPLE_SIZE:
            raise ValueError(
                f"Dataset contains {original_rows:,} rows. "
                f"Cannot sample {SAMPLE_SIZE:,} without replacement."
            )

        print(
            f"Sampling {SAMPLE_SIZE:,} reviews with seed {SEED}...",
            flush=True
        )

        # Same call as the instructor's script: df.sample(n=50000, random_state=44)
        df = raw.sample(
            n=SAMPLE_SIZE,
            random_state=SEED,
            replace=False
        ).copy()

        # Preserve original source row index.
        df["source_row_index"] = df.index
        df = df.reset_index(drop=True)

        # Save the untouched random sample.
        df.to_csv(sample_file, index=False)
        print(f"Saved raw sample: {sample_file}")

        return df, original_rows

    if sample_file.is_file():

        print(
            f"{input_file.name} not found; using existing sample "
            f"{sample_file.name}",
            flush=True
        )

        return pd.read_csv(sample_file), None

    raise FileNotFoundError(
        f"Neither {input_file} nor {sample_file} exists. "
        "See data/README.md for download instructions."
    )


def main():

    args = parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    input_file = data_dir / "Reviews.csv"
    sample_file = data_dir / f"group_sample_seed{SEED}.csv"
    output_file = data_dir / f"group_cleaned_seed{SEED}.csv"
    audit_file = data_dir / f"group_audit_seed{SEED}.csv"
    report_file = data_dir / f"group_report_seed{SEED}.json"

    start = time.perf_counter()

    # --------------------------------------------------------
    # 1-2. LOAD AND SAMPLE (BEFORE ANY CLEANING)
    # --------------------------------------------------------

    df, original_rows = load_sample(input_file, sample_file)
    sampled_rows = len(df)

    score_counts = {
        int(k): int(v)
        for k, v in df["Score"].value_counts().sort_index().items()
    }
    expected_counts = {1: 4721, 2: 2690, 3: 3803, 4: 7043, 5: 31743}
    sample_matches_team = score_counts == expected_counts

    if not sample_matches_team:
        print(
            "WARNING: Score counts differ from the team sample "
            f"{expected_counts}; got {score_counts}",
            flush=True
        )

    # --------------------------------------------------------
    # 3. REMOVE MISSING DATA
    # --------------------------------------------------------

    df = df.dropna(
        subset=["Text", "Score"]
    ).copy()

    after_missing = len(df)

    # --------------------------------------------------------
    # 4. REMOVE NEUTRAL REVIEWS
    # --------------------------------------------------------

    df = df[
        df["Score"].isin([1, 2, 4, 5])
    ].copy()

    after_neutral = len(df)

    # --------------------------------------------------------
    # 5. CLEAN HTML AND WHITESPACE
    # --------------------------------------------------------

    df["Text"] = df["Text"].map(clean_review)

    # Summary is a secondary input; clean it the same way (no spell correction).
    df["Summary"] = (
        df["Summary"].fillna("").map(clean_review).map(normalize_quotes)
    )

    # --------------------------------------------------------
    # 6. REMOVE EMPTY REVIEWS
    # --------------------------------------------------------

    df = df[
        df["Text"].str.len() > 0
    ].copy()

    after_empty = len(df)

    # --------------------------------------------------------
    # 7. REMOVE DUPLICATES
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=["Text"]
    ).reset_index(drop=True)

    after_duplicates = len(df)

    # --------------------------------------------------------
    # 8. CREATE SENTIMENT LABELS
    # --------------------------------------------------------

    df["Sentiment"] = np.where(
        df["Score"] <= 2,
        "negative",
        "positive"
    )

    # --------------------------------------------------------
    # 9. SHARED TRAIN/TEST SPLIT
    # --------------------------------------------------------

    # Same arguments as the notebooks, so every model uses the same 20% test set.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["Sentiment"]
    )

    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # --------------------------------------------------------
    # 10. PRESERVE TEXT BEFORE CORRECTION
    # --------------------------------------------------------

    df["Text_before_correction"] = df["Text"]

    # --------------------------------------------------------
    # 11. APPLY QUOTE, SPELLING, AND CONTRACTION CORRECTIONS
    # --------------------------------------------------------

    print(
        f"Cleaning {len(df):,} sampled reviews...",
        flush=True
    )

    counts = Counter()
    audit = []
    commas_removed = 0
    quotes_normalized = 0

    texts = df["Text"].tolist()

    for idx, before in enumerate(texts):

        # Typographic apostrophes -> ASCII so "don’t" == "don't".
        after = normalize_quotes(before)

        if after != before:
            quotes_normalized += 1

        # Dictionary and manual corrections.
        after = correct_review(
            after,
            idx,
            audit,
            counts
        )

        # Final contraction standardization.
        after = normalize_contractions(after)

        # Remove commas immediately after negations.
        after, removed = NEGATION_COMMA_RE.subn(
            r"\1",
            after
        )

        commas_removed += removed

        # Safety check.
        if not after.strip():
            after = before
            counts["rejected_empty"] += 1

        texts[idx] = after

        if (idx + 1) % PROGRESS_INTERVAL == 0:

            print(
                f"Processed {idx + 1:,}/{len(df):,}",
                flush=True
            )

    df["Text"] = texts

    # --------------------------------------------------------
    # 12. VALIDATE CONTRACTIONS, PROBE NON-STANDARD LANGUAGE
    # --------------------------------------------------------

    validation = validate_corrections(df)

    print("\nRemaining uncorrected contractions:")

    for word, count in validation.items():
        print(f"  {word}: {count}")

    nonstandard = probe_nonstandard_language(df["Text"].tolist())

    print(
        "\nNon-standard language: "
        f"{nonstandard['abbreviation_token_share_pct']}% abbreviation tokens, "
        f"{nonstandard['elongated_token_share_pct']}% elongated tokens"
    )

    # --------------------------------------------------------
    # 13. SAVE CLEANED DATASET
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # 14. SAVE CORRECTION AUDIT
    # --------------------------------------------------------

    audit_df = pd.DataFrame(
        audit,
        columns=[
            "row_index",
            "original_word",
            "corrected_word",
            "status"
        ]
    )

    audit_df.to_csv(
        audit_file,
        index=False
    )

    # --------------------------------------------------------
    # 15. GENERATE JSON REPORT
    # --------------------------------------------------------

    report = {
        "seed": SEED,
        "sample_size_requested": SAMPLE_SIZE,
        "sampling_method": "random_without_replacement",
        "sample_matches_team_score_counts": sample_matches_team,
        "sample_score_counts": score_counts,

        "original_dataset_rows": original_rows,
        "sampled_rows": sampled_rows,

        "rows_removed_missing":
            sampled_rows - after_missing,

        "rows_removed_neutral":
            after_missing - after_neutral,

        "rows_removed_empty":
            after_neutral - after_empty,

        "rows_removed_duplicates":
            after_empty - after_duplicates,

        "final_rows": len(df),

        "sentiment_distribution": {
            str(k): int(v)
            for k, v in
            df["Sentiment"].value_counts().items()
        },

        "split": {
            "method": "stratified_train_test_split",
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "counts": {
                split: {
                    str(k): int(v)
                    for k, v in group["Sentiment"].value_counts().items()
                }
                for split, group in df.groupby("split")
            },
        },

        "punctuation": "retained",
        "correction_method": "conservative_dictionary",
        "correction_settings": {
            "max_edit_distance": MAX_EDIT_DISTANCE,
            "min_word_length": MIN_WORD_LENGTH,
            "min_suggestion_frequency": MIN_SUGGESTION_FREQUENCY,
            "min_frequency_ratio": MIN_FREQUENCY_RATIO,
            "require_unique_candidate": REQUIRE_UNIQUE_CANDIDATE,
        },

        "manual_corrections": sum(
            1 for entry in audit
            if entry["status"] == "manual_corrected"
        ),

        "dictionary_corrections": sum(
            1 for entry in audit
            if entry["status"] == "dictionary_corrected"
        ),

        "total_words_corrected": len(audit),

        "reviews_with_quotes_normalized": quotes_normalized,

        "negation_commas_removed": commas_removed,

        "reviews_modified": int(
            (
                df["Text"] !=
                df["Text_before_correction"]
            ).sum()
        ),

        "correction_status_counts": dict(counts),

        "remaining_uncorrected_contractions": validation,

        "nonstandard_language_probe": nonstandard,

        "average_words_before": round(
            float(
                df["Text_before_correction"]
                .str.split()
                .str.len()
                .mean()
            ),
            2
        ) if len(df) else 0,

        "average_words_after": round(
            float(
                df["Text"]
                .str.split()
                .str.len()
                .mean()
            ),
            2
        ) if len(df) else 0,

        "elapsed_seconds": round(
            time.perf_counter() - start,
            2
        )
    }

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 16. PRINT SUMMARY
    # --------------------------------------------------------

    print("\nProcessing complete!")

    if original_rows is not None:
        print(f"Original rows: {original_rows:,}")
    print(f"Sampled rows: {sampled_rows:,}")
    print(f"Final cleaned rows: {len(df):,}")
    print(
        f"Train / test: {int((df['split'] == 'train').sum()):,} / "
        f"{int((df['split'] == 'test').sum()):,}"
    )

    print(
        f"Manual corrections: "
        f"{report['manual_corrections']:,}"
    )

    print(
        f"Dictionary corrections: "
        f"{report['dictionary_corrections']:,}"
    )

    print(
        f"Negation commas removed: "
        f"{commas_removed:,}"
    )

    print(
        f"Reviews modified: "
        f"{report['reviews_modified']:,}"
    )

    print(
        f"Elapsed time: "
        f"{report['elapsed_seconds']} seconds"
    )

    print("\nOutput files:")

    for path in [
        sample_file,
        output_file,
        audit_file,
        report_file
    ]:
        print(path)

    print("\nFirst three cleaned reviews:")

    print(
        df[["Score", "Sentiment", "split", "Text"]].head(3)
    )


if __name__ == "__main__":
    main()
