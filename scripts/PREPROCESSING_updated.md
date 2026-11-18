---
output:
  pdf_document: default
  html_document: default
---
# Amazon Review Dataset Preprocessing

Run preprocessing **before training any model**. All models must use the cleaned CSV, **not** the original `Reviews.csv` or the raw sample. Generated data files stay local and are not committed to GitHub.

## 1. Run the script

Put `Reviews.csv` (or the team's `group_sample_seed44.csv`) in the repository's `data/` folder (see `data/README.md`), then run from the repository root:

```bash
pip install -r requirements.txt
python scripts/clean_reviews_script_final.py
```

To use a different folder: `python scripts/clean_reviews_script_final.py --data-dir path/to/folder`.

The script samples 50,000 reviews with seed **44** (the same call as the instructor's code) and checks the Score counts against the team sample (4,721 / 2,690 / 3,803 / 7,043 / 31,743). It then removes missing, neutral (3-star), empty, and duplicate-text reviews, so the cleaned dataset has fewer than 50,000 rows.

## 2. Cleaning steps

| Step | What it does | Why |
| --- | --- | --- |
| HTML and whitespace | Decodes entities (`&amp;` to `&`), strips tags (`<br />`), collapses whitespace in `Text` and `Summary` | Markup is not language and would become fake tokens |
| Filtering and deduplication | Drops missing, 3-star, empty, and duplicate `Text` rows | Binary task, and duplicates could fall in both train and test |
| Labels | 1–2 stars = `negative`, 4–5 stars = `positive` | Our binary sentiment scheme from M1 |
| Shared split | `split` column, stratified 80/20, seed 44 | Every model is tested on the same 20% test set |
| Quote normalization | `don’t` to `don't` | Otherwise the same contraction is two different tokens |
| Contractions | `dont` to `don't`, `cant` to `can't`, and others | Keeps negations as one consistent token |
| Conservative spelling | Corrects only unambiguous, high-frequency, edit-distance-1 fixes for words of 5+ letters; URLs, brands, and sentiment words are never changed | Fixes clear typos without changing meaning |
| Negation commas | `don't, like` to `don't like` | Keeps the negation next to the word it negates |

Filtering, deduplication, and the split use exactly the same code as `notebooks/classical_models.ipynb`, so the row set and test set match our M2 baseline results.

## 3. Design decisions

**Punctuation is kept.** We do not remove punctuation in the cleaned text. The classical models' vectorizers (`CountVectorizer`, `TfidfVectorizer`) already ignore it when they split text into tokens, so removing it would not change the M2 results. The deep model, and any tokenizer that keeps contractions, can use the apostrophe in `can't` and emphasis such as `!!!`. Removing punctuation here could not be undone by any later model.

**No separate abbreviation or slang normalizer.** Words shorter than 5 letters are never auto-corrected, which already protects most shorthand (`u`, `thx`, `pls`, `omg`, `lol`). In our checks, non-standard forms were rare, and most unknown words were nouns such as brand and product names, which carry little sentiment. Adjectives carry the most sentiment, and those are mostly spelled correctly. The script still measures this: the `nonstandard_language_probe` field in the JSON report gives the share of abbreviation and elongated (`sooo`) tokens, so the report can cite real numbers.

**Spelling correction stays conservative and context-free.** We tested a context-aware corrector on a 1,000-review pilot and the gain was negligible. It was also slower and more likely to change a correct but unusual word, which would change the meaning of the review. The simple corrector only acts when there is exactly one strong candidate, and every change is logged in the audit CSV.

## 4. Output files

| File | Purpose |
| --- | --- |
| `data/group_sample_seed44.csv` | Original 50,000-row sample, before cleaning |
| `data/group_cleaned_seed44.csv` | **Use this for all models**: filtered reviews, corrected text, labels, and split |
| `data/group_audit_seed44.csv` | Log of every accepted word correction |
| `data/group_report_seed44.json` | Row counts at each step, split counts, correction counts, validation, and the non-standard language probe |

### Cleaned dataframe columns

The cleaned CSV keeps the original review columns and adds or updates the following:

| Column | Description |
| --- | --- |
| `Text` | Final corrected review text; **model input** |
| `Summary` | Review title, HTML-cleaned (not spell-corrected) |
| `Text_before_correction` | Review text after HTML cleaning, before quote/spelling/contraction corrections |
| `Sentiment` | Binary target derived from `Score`: 1–2 stars = negative, 4–5 stars = positive |
| `split` | `train` or `test`; the shared stratified 80/20 split (seed 44) |
| `source_row_index` | Row index in the full `Reviews.csv` (only when sampled from `Reviews.csv`) |
| `Score` | Original star rating (1–5); **do not use as a predictor** |

### Loading it in a notebook

```python
df = pd.read_csv(ROOT / 'data' / 'group_cleaned_seed44.csv')
train, test = df[df['split'] == 'train'], df[df['split'] == 'test']
X_train, y_train = train['Text'], train['Sentiment']
X_test, y_test = test['Text'], test['Sentiment']
```
