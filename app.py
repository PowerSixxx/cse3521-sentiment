"""Streamlit app: live sentiment predictions for Amazon Fine Food reviews.

Run locally from the repository root:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# The exact cleaning used to build the training data (HTML, quotes, spelling,
# contractions, negation commas), so live input matches what the models saw.
from scripts.preprocessing import preprocess

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

# Every model the app can show. To add a model (e.g. Logistic Regression at M3),
# save its sklearn Pipeline with joblib into models/ and add one line here.
MODELS = {
    "Naive Bayes (Bag of Words)": "naive_bayes_bow.joblib",
    # "Logistic Regression (TF-IDF)": "logreg_tfidf.joblib",
}

EXAMPLES = {
    "Positive": "Best coffee I've had in years. Rich, smooth, and very fresh. Will order again!",
    "Negative": "Total waste of money. The chips arrived stale and tasted awful. I threw away the whole bag.",
    "Mixed": "The flavor is great, but the package arrived crushed and half of the cookies were broken.",
}


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


def word_contributions(model, text: str, top_n: int = 5):
    """For Naive Bayes: which words in this review push toward positive / negative.

    Each word's push = (count in review) x (log P(word|positive) - log P(word|negative)).
    Returns None for models that are not a vectorizer + Naive Bayes pipeline.
    """
    vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
    if not hasattr(classifier, "feature_log_prob_"):
        return None
    counts = vectorizer.transform([text])
    classes = list(classifier.classes_)
    diff = (classifier.feature_log_prob_[classes.index("positive")]
            - classifier.feature_log_prob_[classes.index("negative")])
    idx = counts.nonzero()[1]
    if len(idx) == 0:
        return [], []
    scores = np.asarray(counts[0, idx].todense()).ravel() * diff[idx]
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
        st.write(f" {name}")
    for filename in missing:
        st.write(f"⚠️ missing `models/{filename}`")
    st.caption("CSE 3521 · Autumn 2026 · Group Gamma")

st.title("🍪 Food Review Sentiment")
st.write("Paste a food review and compare predictions from our trained models.")

st.write("Try an example:")
example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES.items()):
    col.button(label, on_click=set_example, args=(text,), use_container_width=True)

review = st.text_area("Review text", key="review", height=150,
                      placeholder="e.g. These cookies were stale and tasted like cardboard.")

if not models:
    st.error("No model files found in models/. Run the training notebook first.")
    st.stop()

if st.button("Predict sentiment", type="primary") or review:
    text = preprocess(review)
    if not text:
        st.info("Type or paste a review above to get a prediction.")
        st.stop()

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
                    st.caption("Score = word count × log-probability ratio "
                               "between the positive and negative class.")
