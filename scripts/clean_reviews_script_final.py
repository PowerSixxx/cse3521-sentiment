
"""
Amazon Review Sampling and Conservative Dictionary Cleaning

Run:
    py sample_and_clean_reviews.py

Install:
    pip install pandas numpy pyspellchecker

Pipeline:
1. Sample 50,000 reviews using seed 44
2. Save the original sample
3. Remove missing, neutral, empty, and duplicate reviews
4. Clean HTML and whitespace
5. Apply manual spelling corrections
6. Apply conservative dictionary corrections
7. Standardize contractions
8. Remove commas immediately following negations
9. Validate corrections
10. Save cleaned reviews, audit CSV, and JSON report

Only the randomly selected sample is cleaned.
"""

from __future__ import annotations

import html
import json
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from spellchecker import SpellChecker


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("C:/Users/andre/Downloads/3521PData")
BASE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = BASE_DIR / "Reviews.csv"

SEED = 44
SAMPLE_SIZE = 50000

SAMPLE_FILE = BASE_DIR / f"group_sample_seed{SEED}.csv"
OUTPUT_FILE = BASE_DIR / f"group_cleaned_seed{SEED}.csv"
AUDIT_FILE = BASE_DIR / f"group_audit_seed{SEED}.csv"
REPORT_FILE = BASE_DIR / f"group_report_seed{SEED}.json"

MAX_EDIT_DISTANCE = 1
MIN_WORD_LENGTH = 5
MIN_SUGGESTION_FREQUENCY = 100
MIN_FREQUENCY_RATIO = 20
REQUIRE_UNIQUE_CANDIDATE = True

PROGRESS_INTERVAL = 1000


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


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_review(value):
    """
    Decode HTML entities, remove HTML tags,
    and normalize whitespace.
    """
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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

    # 3. Preserve short words.
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
    Correct individual words while preserving sentence structure.
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

    return WORD_RE.sub(replace_match, text)


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


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    start = time.perf_counter()

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # --------------------------------------------------------
    # 1. LOAD ORIGINAL DATASET
    # --------------------------------------------------------

    print("Loading original dataset...", flush=True)

    raw = pd.read_csv(INPUT_FILE)

    original_rows = len(raw)

    if original_rows < SAMPLE_SIZE:
        raise ValueError(
            f"Dataset contains {original_rows:,} rows. "
            f"Cannot sample {SAMPLE_SIZE:,} without replacement."
        )

    # --------------------------------------------------------
    # 2. RANDOM SAMPLE BEFORE CLEANING
    # --------------------------------------------------------

    print(
        f"Sampling {SAMPLE_SIZE:,} reviews with seed {SEED}...",
        flush=True
    )

    df = raw.sample(
        n=SAMPLE_SIZE,
        random_state=SEED,
        replace=False
    ).copy()

    # Preserve original source row index.
    df["source_row_index"] = df.index

    df = df.reset_index(drop=True)

    sampled_rows = len(df)

    # Save the untouched random sample.
    df.to_csv(
        SAMPLE_FILE,
        index=False
    )

    print(f"Saved raw sample: {SAMPLE_FILE}")

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
    # 9. PRESERVE ORIGINAL CLEANED TEXT
    # --------------------------------------------------------

    df["Text_before_correction"] = df["Text"]

    # --------------------------------------------------------
    # 10. APPLY SPELLING AND CONTRACTION CORRECTIONS
    # --------------------------------------------------------

    print(
        f"Cleaning {len(df):,} sampled reviews...",
        flush=True
    )

    counts = Counter()
    audit = []
    commas_removed = 0

    for idx in range(len(df)):

        before = df.at[idx, "Text"]

        # Dictionary and manual corrections.
        after = correct_review(
            before,
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

        df.at[idx, "Text"] = after

        if (idx + 1) % PROGRESS_INTERVAL == 0:

            print(
                f"Processed {idx + 1:,}/{len(df):,}",
                flush=True
            )

    # --------------------------------------------------------
    # 11. VALIDATE CONTRACTIONS
    # --------------------------------------------------------

    validation = validate_corrections(df)

    print("\nRemaining uncorrected contractions:")

    for word, count in validation.items():
        print(f"  {word}: {count}")

    # --------------------------------------------------------
    # 12. SAVE CLEANED DATASET
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 13. SAVE CORRECTION AUDIT
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
        AUDIT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 14. GENERATE JSON REPORT
    # --------------------------------------------------------

    report = {
        "seed": SEED,
        "sample_size_requested": SAMPLE_SIZE,
        "sampling_method": "random_without_replacement",

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

        "correction_method": "conservative_dictionary",

        "manual_corrections": sum(
            1 for entry in audit
            if entry["status"] == "manual_corrected"
        ),

        "dictionary_corrections": sum(
            1 for entry in audit
            if entry["status"] == "dictionary_corrected"
        ),

        "total_words_corrected": len(audit),

        "negation_commas_removed": commas_removed,

        "reviews_modified": int(
            (
                df["Text"] !=
                df["Text_before_correction"]
            ).sum()
        ),

        "correction_status_counts": dict(counts),

        "remaining_uncorrected_contractions": validation,

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

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 15. PRINT SUMMARY
    # --------------------------------------------------------

    print("\nProcessing complete!")

    print(f"Original rows: {original_rows:,}")
    print(f"Sampled rows: {sampled_rows:,}")
    print(f"Final cleaned rows: {len(df):,}")

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
        SAMPLE_FILE,
        OUTPUT_FILE,
        AUDIT_FILE,
        REPORT_FILE
    ]:
        print(path)

    print("\nFirst three cleaned reviews:")

    print(
        df[["Score", "Sentiment", "Text"]].head(3)
    )


if __name__ == "__main__":
    main()
