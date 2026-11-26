"""Evaluate every saved model on the shared test set and write models/model_metrics.json.

The Streamlit app's "Model performance" tab reads that JSON file, so rerun this
script whenever a model in models/ is retrained:

    python scripts/export_metrics.py

Which test set is used:
  * data/group_cleaned_seed44.csv (output of scripts/clean_reviews_script_final.py),
    using its `split == "test"` rows, if that file exists;
  * otherwise data/group_sample_seed44.csv with the same filtering, deduplication and
    stratified 80/20 split (seed 44) as notebooks/classical_models.ipynb.

Training time cannot be measured here (the models are already trained), so the
`training_seconds` values are copied over from the existing JSON. Edit them by hand
from the training notebook output.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ffnn_model import load_model

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUT_PATH = MODELS_DIR / "model_metrics.json"
SEED = 44

# Display name -> model file. Keep in sync with MODELS in app.py.
MODELS = {
    "Multinomial Naive Bayes (BoW)": "naive_bayes_bow.joblib",
    "Logistic Regression (TF-IDF)": "logreg_tfidf.joblib",
    "Feedforward Neural Network (TF-IDF)": "ffnn_complete.joblib",
}


def clean_review(value) -> str:
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_test_set():
    cleaned = DATA_DIR / "group_cleaned_seed44.csv"
    if cleaned.is_file():
        df = pd.read_csv(cleaned)
        test = df[df["split"] == "test"]
        train = df[df["split"] == "train"]
        return train["Sentiment"], test["Text"].astype(str), test["Sentiment"], cleaned.name

    # Same steps as notebooks/classical_models.ipynb (sections 2 and 3).
    df = pd.read_csv(DATA_DIR / "group_sample_seed44.csv")
    df = df.dropna(subset=["Text", "Score"])
    df = df[df["Score"] != 3].copy()
    df["Text"] = df["Text"].map(clean_review)
    df = df[df["Text"].str.len() > 0]
    df = df.drop_duplicates(subset="Text")
    df["Sentiment"] = df["Score"].map(lambda s: "negative" if s <= 2 else "positive")
    _, X_test, y_train, y_test = train_test_split(
        df["Text"], df["Sentiment"], test_size=0.20, random_state=SEED, stratify=df["Sentiment"]
    )
    return y_train, X_test, y_test, "group_sample_seed44.csv (notebook split)"


def row(truth, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(truth, pred, average="macro", zero_division=0)
    cm = confusion_matrix(truth, pred, labels=["negative", "positive"])
    return {
        "accuracy": round(accuracy_score(truth, pred), 4),
        "precision_macro": round(p, 4),
        "recall_macro": round(r, 4),
        "f1_macro": round(f, 4),
        "confusion_matrix": cm.tolist(),  # rows = actual, columns = predicted
    }


def main():
    old = json.loads(OUT_PATH.read_text()) if OUT_PATH.is_file() else {}
    old_seconds = {m["name"]: m.get("training_seconds") for m in old.get("models", [])}

    y_train, X_test, y_test, source = load_test_set()
    majority = y_train.mode()[0]
    results = [{"name": "Majority-class baseline", **row(y_test, [majority] * len(y_test))}]

    for name, filename in MODELS.items():
        path = MODELS_DIR / filename
        if not path.is_file():
            print(f"skip {name}: {path} not found")
            continue
        model = load_model(path)
        results.append({"name": name, **row(y_test, model.predict(X_test))})
        print(f"evaluated {name}")

    for r in results:
        r["training_seconds"] = old_seconds.get(r["name"])

    OUT_PATH.write_text(json.dumps(
        {"test_source": source, "test_size": int(len(y_test)), "models": results}, indent=2
    ) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({len(y_test)} test reviews)")


if __name__ == "__main__":
    main()
