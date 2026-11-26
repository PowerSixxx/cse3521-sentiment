# AI-Powered Text Intelligence Platform

**Sentiment Analysis of Amazon Fine Food Reviews: From Data to Deployment**
CSE 3521: Survey of Artificial Intelligence I, Autumn 2026, The Ohio State University

**Group:** Gamma · **Data seed:** 44
**Live app:** https://cse3521-sentiment.streamlit.app

## Team

| Name | OSU email | Primary role |
| --- | --- | --- |
| Andrew Kobasko | kobasko.1@osu.edu | Project Lead + ML Engineer (deep) |
| Tarun Shivakumar | shivakumar.15@osu.edu | Data Engineer + Analyst / Writer |
| Henry Benner | benner.162@osu.edu | ML Engineer (classical) |
| Baowen Liu | liu.11884@osu.edu | App Engineer |

## About the project

We build a sentiment classifier for Amazon Fine Food Reviews and serve it through a Streamlit web app. A user pastes a review and gets a positive or negative prediction from at least two of our trained models, shown side by side.

The work has four layers:

1. **Feature engineering:** at least two text representations (Bag of Words and TF-IDF, both built).
2. **Classical models:** at least two tuned classical classifiers (Multinomial Naive Bayes trained; Logistic Regression planned for M3).
3. **Deep / modern model:** at least one neural model compared against the classical baselines (a feedforward network, with DistilBERT as a stretch goal).
4. **Deployed app:** a Streamlit Community Cloud app that returns live predictions.

Every model is evaluated on a held-out 20% test set with accuracy, precision, recall, F1-score and a confusion matrix.

[Group Project PDF](./group_project.pdf)

[M1 Project Charter (PDF)](./report/M1_Project_Charter.pdf)

## Dataset

We use a 50,000-review sample of the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset (568,454 reviews), drawn with seed 44. We label 1–2-star reviews as negative and 4–5-star reviews as positive, and exclude 3-star reviews from the binary classification task. The CSV is **not** committed to this repository. See [`data/README.md`](data/README.md) for how to download the data and generate the sample.

## Repository structure

```
.
├── README.md          Project overview, team, live app URL, setup
├── data/              Dataset download and sampling instructions (CSV files are git-ignored)
├── scripts/           Data cleaning script + shared preprocessing used by notebooks and app
├── notebooks/         Jupyter notebooks for EDA, feature engineering, model training
├── models/            Saved model files used by the app
├── app.py             Streamlit app entry point
├── requirements.txt   Python dependencies with pinned versions
└── report/            Milestone reports, final report PDF and slides
```

## Setup

Requires Python 3.11 or 3.12.

```bash
git clone https://github.com/PowerSixxx/cse3521-sentiment.git
cd cse3521-sentiment

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then follow [`data/README.md`](data/README.md) to download `Reviews.csv` into `data/`.

### Run the app locally

The trained models are already in `models/`, so the app runs without the dataset:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 and loads the trained pipelines in `models/`, so it runs without the dataset. It has three tabs:

| Tab | What it shows |
|---|---|
| Live prediction | Paste a review and see the predictions of Naive Bayes, Logistic Regression and the FFNN side by side, with confidence (and, for the linear models, the words that drove each prediction) |
| Model performance | Test-set accuracy, macro precision, recall, F1, training time and a confusion matrix per model (read from `models/model_metrics.json`) |
| About the data | Dataset, label scheme, and how the models work |

After retraining a model, refresh the metrics the app shows with:

```bash
python scripts/export_metrics.py
```

To add a new model to the app, save it in `models/` as a scikit-learn `Pipeline` (vectorizer + classifier with `predict_proba`) or as a PyTorch FFNN bundle that `scripts/ffnn_model.py` can load. Then add one line to `MODELS` in `app.py` and in `scripts/export_metrics.py`.

### Reproduce our results

Run every command from the repository root. Each step reads only the outputs of the steps before it.

**1. Get the data.** Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) into `data/` ([`data/README.md`](data/README.md) has the steps and a Kaggle CLI command). Data files are git-ignored and are never committed.

**2. Sample and clean.**

```bash
python scripts/clean_reviews_script_final.py
```

This draws our 50,000-review sample with seed 44 (`df.sample(n=50000, random_state=44)`, the instructor's call), cleans it, and writes:

| File | Contents |
|---|---|
| `data/group_sample_seed44.csv` | The untouched 50,000-row sample |
| `data/group_cleaned_seed44.csv` | **The only file models read.** Cleaned `Text`, `Sentiment` label, shared `split` (train/test), `is_val` flag |
| `data/group_audit_seed44.csv` | Every word the spelling/contraction step changed |
| `data/group_report_seed44.json` | Row counts at each step, split sizes, correction statistics |

**Check that your data matches ours.** The script prints a warning unless the sample's Score counts for 1–5 stars are exactly 4,721 / 2,690 / 3,803 / 7,043 / 31,743. `group_report_seed44.json` then records the cleaned row count and the train/test sizes, which must match the numbers in our reports.

**3. Train and evaluate.** Open the notebooks (`jupyter notebook notebooks/`) and use **Run All** in this order:

| Order | Notebook | What it does | Writes |
|---|---|---|---|
| 1 | `notebooks/feature_engineering.ipynb` | Builds Bag of Words and TF-IDF, top features, feature-space figures, tokenizer ablation | `report/figures/m3_tfidf_*.png`, `report/M3_tokenizer_ablation.md` |
| 2 | `notebooks/classical_models.ipynb` | Majority baseline, Naive Bayes (BoW) and Logistic Regression (TF-IDF), tuned by cross-validation on train only, scored on the shared test set | `models/naive_bayes_bow.joblib`, `models/logreg_tfidf.joblib`, confusion matrix, results Markdown |

Every model reads `group_cleaned_seed44.csv` through `scripts/preprocessing.py` (`load_cleaned()` and `split_frames()`), so all of them use the same rows and the same 20% test set:

| Split | Rows | Used for |
|---|---|---|
| `train` | `split == "train"` (80%) | Fitting and cross-validated tuning of classical models |
| `fit` / `val` | train with `is_val` false / true (90% / 10% of train) | Deep model: train on `fit`, validation loss curves on `val`; tokenizer ablation |
| `test` | `split == "test"` (20%) | Final scores for every model, used once |

All randomness uses seed 44 (sampling, split, validation slice, cross-validation folds, model `random_state`). Training times in our tables were measured on a laptop CPU and will vary by machine.

**4. Serve.** The models saved in `models/` are complete scikit-learn pipelines (vectorizer + classifier). `app.py` loads them and cleans every typed review with `preprocess()` from `scripts/preprocessing.py`, the same function that built the training text, so live predictions see text in the same form as the training data.

## Milestones

| Milestone | Due | Deliverable |
|---|---|---|
| M1: Project Charter & Scope | Tue Nov 10 | Charter PDF + this repository |
| M2: Initial Solution | Wed Nov 18 | Feature pipeline, first classical model results |
| M3: Refined Solution & App Prototype | Sun Nov 22 | All models compared, Streamlit app running |
| M4: Final Report & Live Demo | Sun Nov 29 | Final report, slides, live demo |

## Contributing

- Commit small and often from your own account so individual contributions show in the history.
- Work on a branch (`git checkout -b yourname/feature`) and open a pull request into `main`.
- Never commit data files (`*.csv`), API keys, or `kaggle.json`.

## AI assistance disclosure

We used AI assistance in this project; every team member reviewed this work, and all analysis and interpretation are our own.
