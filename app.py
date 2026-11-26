"""Streamlit app: live sentiment predictions for Amazon Fine Food reviews.

Run locally from the repository root:
    streamlit run app.py

Three tabs:
  1. Live prediction   - paste a review, every model predicts side by side
  2. Model performance - test-set metrics and confusion matrices (models/model_metrics.json)
  3. About the data    - dataset, labels, and how the models were built
"""
from pathlib import Path
import html
import json
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "report" / "figures"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Every model the app can show. To add a model, save its sklearn Pipeline
# (vectorizer + classifier, with predict_proba) into models/ with joblib, add one
# line here, then rerun `python scripts/export_metrics.py` to update the metrics tab.
MODELS = {
    "Multinomial Naive Bayes (BoW)": "naive_bayes_bow.joblib",
    "Logistic Regression (TF-IDF)": "logreg_tfidf.joblib",
}

EXAMPLES = {
    "Positive": "Best coffee I've had in years. Rich, smooth, and very fresh. Will order again!",
    "Negative": "Total waste of money. The chips arrived stale and tasted awful. I threw away the whole bag.",
    "Mixed": "The flavor is great, but the package arrived crushed and half of the cookies were broken.",
    "Negation": "I wouldn't recommend these. Not sweet, not crunchy, and not worth the price.",
}


# ---------------------------------------------------------------- preprocessing
def basic_clean(value: str) -> str:
    """Unescape HTML, drop tags, squash spaces (same as clean_review in the notebooks)."""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_cleaner():
    """Use the team's cleaning script if it is available, so a new review gets the
    same treatment as the training data (quotes, spelling, contractions, negation
    commas). If the script or pyspellchecker is missing, fall back to basic_clean."""
    try:
        from scripts import clean_reviews_script_final as cr

        def full_clean(value: str) -> str:
            text = cr.normalize_quotes(cr.clean_review(value))
            text = cr.correct_review(text, 0, [], Counter())
            text = cr.normalize_contractions(text)
            return cr.normalize_negation_commas(text)

        full_clean("test dont")  # fail here, not in the middle of a prediction
        return full_clean, "team cleaning script (scripts/clean_reviews_script_final.py)"
    except Exception:
        return basic_clean, "basic HTML/whitespace cleaning"


# ---------------------------------------------------------------- models
@st.cache_resource
def load_models():
    """Load every model file once and keep it in memory between user clicks."""
    loaded, missing = {}, []
    for name, filename in MODELS.items():
        path = MODELS_DIR / filename
        if path.is_file():
            loaded[name] = joblib.load(path)
        else:
            missing.append(filename)
    return loaded, missing


@st.cache_data
def load_metrics():
    if METRICS_PATH.is_file():
        return json.loads(METRICS_PATH.read_text())
    return None


def word_contributions(model, text: str, top_n: int = 5):
    """Which words in this review push the prediction toward positive / negative.

    Naive Bayes:          push = count(word) x [log P(word|pos) - log P(word|neg)]
    Logistic Regression:  push = tfidf(word) x weight(word)
    Returns None for models that are neither (e.g. a neural network).
    """
    vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
    classes = list(classifier.classes_)
    if hasattr(classifier, "feature_log_prob_"):
        weights = (classifier.feature_log_prob_[classes.index("positive")]
                   - classifier.feature_log_prob_[classes.index("negative")])
    elif hasattr(classifier, "coef_") and classifier.coef_.shape[0] == 1:
        # Binary LR: coef_ points toward classes_[1].
        weights = classifier.coef_[0] * (1 if classes[1] == "positive" else -1)
    else:
        return None

    x = vectorizer.transform([text])
    idx = x.nonzero()[1]
    if len(idx) == 0:
        return [], []
    scores = np.asarray(x[0, idx].todense()).ravel() * weights[idx]
    terms = vectorizer.get_feature_names_out()[idx]
    order = np.argsort(scores)
    neg = [(terms[i], scores[i]) for i in order[:top_n] if scores[i] < 0]
    pos = [(terms[i], scores[i]) for i in order[::-1][:top_n] if scores[i] > 0]
    return pos, neg


def set_example(text: str):
    st.session_state["review"] = text


# ---------------------------------------------------------------- page layout
st.set_page_config(page_title="Food Review Sentiment", page_icon="🍪", layout="wide")

models, missing = load_models()
clean, cleaner_name = load_cleaner()
metrics = load_metrics()

# A review can be passed in the URL: ...?review=These+cookies+were+stale
if "review" not in st.session_state and "review" in st.query_params:
    st.session_state["review"] = st.query_params["review"]

with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts whether an Amazon food review is **positive** or "
        "**negative**. Models were trained on a 50,000-review sample "
        "(seed 44) of the Amazon Fine Food Reviews dataset. "
        "1–2 stars = negative, 4–5 stars = positive."
    )
    st.subheader("Models loaded")
    for name in models:
        st.write(f"✅ {name}")
    for filename in missing:
        st.write(f"⚠️ missing `models/{filename}`")
    st.caption(f"Input cleaning: {cleaner_name}")
    st.caption("CSE 3521 · Autumn 2026 · Group Gamma")

st.title("🍪 Food Review Sentiment")

tab_predict, tab_perf, tab_about = st.tabs(
    ["🔮 Live prediction", "📊 Model performance", "ℹ️ About the data"]
)

# ---------------------------------------------------------------- tab 1
with tab_predict:
    st.write("Paste a food review and compare predictions from our trained models side by side.")

    st.write("Try an example:")
    example_cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(example_cols, EXAMPLES.items()):
        col.button(label, on_click=set_example, args=(text,), width="stretch")

    review = st.text_area("Review text", key="review", height=150,
                          placeholder="e.g. These cookies were stale and tasted like cardboard.")

    if not models:
        st.error("No model files found in models/. Run the training notebook first.")
    else:
        clicked = st.button("Predict sentiment", type="primary")
        text = clean(review) if review else ""

        if (clicked or review) and not text:
            st.info("Type or paste a review above to get a prediction.")
        elif text:
            result_cols = st.columns(len(models))
            for col, (name, model) in zip(result_cols, models.items()):
                probs = model.predict_proba([text])[0]
                classes = list(model.classes_)
                label = classes[int(np.argmax(probs))]
                p_pos = probs[classes.index("positive")]

                with col:
                    st.subheader(name)
                    if label == "positive":
                        st.success(f"👍 Positive  ·  {p_pos:.1%} confident")
                    else:
                        st.error(f"👎 Negative  ·  {1 - p_pos:.1%} confident")
                    st.progress(float(p_pos), text=f"P(positive) = {p_pos:.3f}")

                    contrib = word_contributions(model, text)
                    if contrib is not None:
                        pos, neg = contrib
                        with st.expander("Why? Words that drove this prediction"):
                            left, right = st.columns(2)
                            left.markdown("**Toward positive**")
                            for term, s in pos:
                                left.write(f"`{term}`  +{s:.2f}")
                            right.markdown("**Toward negative**")
                            for term, s in neg:
                                right.write(f"`{term}`  {s:.2f}")

            if len(models) > 1:
                labels = {
                    name: m.classes_[int(np.argmax(m.predict_proba([text])[0]))]
                    for name, m in models.items()
                }
                if len(set(labels.values())) > 1:
                    st.warning("The models disagree on this review. Mixed or "
                               "sarcastic reviews are where our models struggle most.")

            with st.expander("Text the models actually saw (after cleaning)"):
                st.code(text, language=None, wrap_lines=True)

# ---------------------------------------------------------------- tab 2
with tab_perf:
    if metrics is None:
        st.info("No metrics yet. Run `python scripts/export_metrics.py` to create "
                "`models/model_metrics.json`.")
    else:
        st.write(
            f"All models are scored on the same held-out test set "
            f"({metrics['test_size']:,} reviews, 20% of the data). "
            "Precision, recall and F1 are macro averages, so the smaller "
            "negative class counts as much as the positive class."
        )
        rows = pd.DataFrame([{
            "Model": m["name"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision_macro"],
            "Recall": m["recall_macro"],
            "F1 (macro)": m["f1_macro"],
            "Training time (s)": "—" if m.get("training_seconds") is None else f"{m['training_seconds']:.2f}",
        } for m in metrics["models"]])
        st.dataframe(
            rows.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1 (macro)"]}
                              )
                .highlight_max(subset=["F1 (macro)"], color="#c8e6c9"),
            hide_index=True, width="stretch",
        )
        best = max(metrics["models"], key=lambda m: m["f1_macro"])
        st.success(f"Best macro F1: **{best['name']}** ({best['f1_macro']:.4f}).")
        st.caption("Training time includes hyperparameter search (GridSearchCV).")

        st.subheader("Confusion matrices")
        st.caption("Rows = actual label, columns = predicted label.")
        shown = [m for m in metrics["models"] if m["name"] != "Majority-class baseline"]
        cm_cols = st.columns(max(len(shown), 1))
        for col, m in zip(cm_cols, shown):
            cm = pd.DataFrame(m["confusion_matrix"],
                              index=["actual negative", "actual positive"],
                              columns=["pred. negative", "pred. positive"])
            col.markdown(f"**{m['name']}**")
            col.dataframe(cm.style.background_gradient(cmap="Blues", axis=None), width="stretch")

# ---------------------------------------------------------------- tab 3
with tab_about:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            """
**Dataset.** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
(568,454 reviews). We use a 50,000-review sample drawn with seed 44.

**Labels.** 1–2 stars → negative, 4–5 stars → positive. 3-star reviews are
left out. After removing duplicates, about 84% of reviews are positive, so a
model that always says "positive" already gets 84% accuracy. That is why we
compare models by **macro F1**.

**Models.**
- *Multinomial Naive Bayes* on Bag-of-Words counts (unigrams + bigrams, 50k vocabulary).
- *Logistic Regression* on TF-IDF weights (unigrams + bigrams, 50k vocabulary, balanced class weights).

**How a prediction works.** Your review is cleaned the same way as the
training data, turned into word counts / TF-IDF weights by each model's own
vectorizer, and the classifier returns P(positive).

**Known weak spots.** Sarcasm, reviews that praise the product but complain
about shipping, and long reviews that tell a story before giving an opinion.
            """
        )
    with right:
        fig = FIGURES_DIR / "score_distribution_seed44.png"
        if fig.is_file():
            st.image(str(fig), caption="Star ratings in our 50,000-review sample")
