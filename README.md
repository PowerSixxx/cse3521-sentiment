# AI-Powered Text Intelligence Platform

**Sentiment Analysis of Amazon Fine Food Reviews: From Data to Deployment**
CSE 3521: Survey of Artificial Intelligence I, Autumn 2026, The Ohio State University

**Group:** Gamma · **Data seed:** 44
**Live app:** [STREAMLIT URL, added at M3]

## Team

| Name | OSU email | Primary role |
|---|---|---|
| [Name 1] | [name.#]@osu.edu | Project Lead |
| [Name 2] | [name.#]@osu.edu | Data Engineer |
| [Name 3] | [name.#]@osu.edu | ML Engineer (classical models) |
| [Name 4] | [name.#]@osu.edu | ML Engineer (deep model) |
| [Name 5] | [name.#]@osu.edu | App Engineer |
| [Name 6] | [name.#]@osu.edu | Analyst / Writer |

## About the project

We build a sentiment classifier for Amazon Fine Food Reviews and serve it through a Streamlit web app. A user pastes a review and gets a positive or negative prediction from at least two of our trained models, shown side by side.

The work has four layers:

1. **Feature engineering:** at least two text representations (Bag of Words and TF-IDF are planned).
2. **Classical models:** at least two tuned classical classifiers (Multinomial Naive Bayes and Logistic Regression are planned).
3. **Deep / modern model:** at least one neural model compared against the classical baselines (a feedforward network, with DistilBERT as a stretch goal).
4. **Deployed app:** a Streamlit Community Cloud app that returns live predictions.

Every model is evaluated on a held-out 20% test set with accuracy, precision, recall, F1-score and a confusion matrix.

[Group Project PDF](./group_project.pdf)

## Dataset

We use a 50,000-review sample of the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset (568,454 reviews), drawn with our group seed. The CSV is **not** committed to this repository. See [`data/README.md`](data/README.md) for how to download the data and generate the sample.

## Repository structure

```
.
├── README.md          Project overview, team, live app URL, setup
├── data/              Dataset download and sampling instructions (CSV files are git-ignored)
├── notebooks/         Jupyter notebooks for EDA, feature engineering, model training
├── models/            Saved model files used by the app
├── app.py             Streamlit app entry point (added at M3)
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

Then follow [`data/README.md`](data/README.md) to place `group_sample_seed44.csv` in `data/`.

### Run the notebooks

```bash
jupyter notebook notebooks/
```

### Run the app locally (from M3)

```bash
streamlit run app.py
```

### Reproduce our results

[To be completed at M3: the order to run the notebooks, how to retrain each model, and where the trained files are saved.]

## Milestones

| Milestone | Due | Deliverable |
|---|---|---|
| M1: Project Charter & Scope | Tue Nov 10 | Charter PDF + this repository |
| M2: Initial Solution | Sun Nov 15 | Feature pipeline, first classical model results |
| M3: Refined Solution & App Prototype | Sun Nov 22 | All models compared, Streamlit app running |
| M4: Final Report & Live Demo | Sun Nov 29 | Final report, slides, live demo |

## Contributing

- Commit small and often from your own account so individual contributions show in the history.
- Work on a branch (`git checkout -b yourname/feature`) and open a pull request into `main`.
- Never commit data files (`*.csv`), API keys, or `kaggle.json`.

## AI assistance disclosure

[One sentence for the final report, for example: "We used [tool] for implementation assistance; all analysis, interpretation and writing are our own."]
