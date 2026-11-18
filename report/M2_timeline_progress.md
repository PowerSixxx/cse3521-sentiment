# Milestone 2: Updated Timeline and Individual Progress Log (Group Gamma)

## 4. Updated Timeline

| Week | Milestone | Who does what |
| --- | --- | --- |
| Nov 11–18 | **M2** (extended to Wed Nov 18) | Henry: cleaning, deduplication, split, majority baseline, Naive Bayes on BoW. Baowen: shared seed-44 sample, Streamlit app skeleton, TF-IDF and feature figures. Tarun: error analysis and M2 report. Andrew: review the revised timeline. |
| Nov 19–22 | **M3** (Sun Nov 22) | Henry: Logistic Regression on TF-IDF, compare with Naive Bayes. Andrew: build and train the FFNN, plot loss curves. Baowen: add the second model to the app and deploy to Streamlit Community Cloud. Tarun: reproducibility steps in the README. Andrew and Tarun: model comparison table and M3 report. |
| Nov 23–29 | **M4** (Sun Nov 29) | Baowen: final deployment. Tarun: final report. Andrew: slides. All: contribution paragraphs and proofreading. |
| Nov 30–Dec 2 | Live demo | All: rehearse the 8-minute demo and Q&A. |

**Scope changes and rationale**

- **TF-IDF moved from Tarun to Baowen.** The app skeleton was finished early, so Baowen built the second feature representation to keep M2 on schedule.
- **The FFNN moved from M2 to M3.** It uses the TF-IDF features, which were only ready on Nov 17. DistilBERT remains a stretch goal.
- **Negation handling added to M3.** Error analysis showed that the default tokenizer turns "can't beat" into "can beat", losing the negation. In M3 we will test a tokenizer that keeps contractions.

## 5. Individual Progress Log

**Baowen Liu (App Engineer).** Since M1, I generated the shared seed-44 sample and built the Streamlit app skeleton (`app.py`), which loads our saved Naive Bayes model and predicts the sentiment of a pasted review. I also built the TF-IDF representation in `notebooks/feature_engineering.ipynb`, with its vocabulary size, top-10 terms for each class, and two feature figures. Next, I will add the second model to the app and deploy it to Streamlit Community Cloud.

**Henry Benner (ML Engineer, classical).** Since M1, Henry built `notebooks/classical_models.ipynb`: data cleaning and deduplication, the stratified 80/20 split, a majority-class baseline, and a Multinomial Naive Bayes model on Bag of Words tuned by cross-validation (α = 0.5). Naive Bayes reaches 91.8% test accuracy against 84.0% for the baseline. Next, he will train and tune Logistic Regression on TF-IDF and compare it with Naive Bayes.

**Tarun Shivakumar (Data Engineer + Analyst / Writer).** Since M1, Tarun prepared and uploaded the final M1 charter PDF. For M2 he is writing the error analysis and assembling the report. Next, he will document the reproducibility steps in the README for M3.

**Andrew Kobasko (Project Lead + ML Engineer, deep).** Since M1, Andrew coordinated the M2 plan and reviewed the revised timeline. Next, he will build and train the feedforward neural network on TF-IDF features and plot its training and validation loss curves for M3.
