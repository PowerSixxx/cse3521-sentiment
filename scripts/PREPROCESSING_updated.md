---
output:
  pdf_document: default
  html_document: default
---
# Amazon Review Dataset Preprocessing

Run preprocessing **before training any model**. All models must use the new cleaned CSV, **not** the original `Reviews.csv` or the raw sample. Generated data files are kept locally and are not committed to GitHub.

## 1. Set the dataset path

Download the original Amazon Fine Food Reviews dataset. Open `sample_and_clean_reviews.py` and **change the input path** to where you saved `Reviews.csv`:

```python
# CHANGE THIS to your local unzipped folder for the project data. Include the folder path only, do not include path of the actual file.
BASE_DIR = Path("C:/Users/andre/Downloads/3521PData")
```

The script first samples 50,000 reviews with seed **44**, then removes neutral (3-star), missing/empty, and duplicate-text reviews. It conservatively corrects spelling and contractions (e.g., `cant` → `can't`), preserves selected brand names, and removes commas immediately after specified negations. The cleaned dataset therefore contains fewer than 50,000 rows.

## 2. Output files

| File | Purpose |
| --- | --- |
| `data/group_sample_seed44.csv` | Original 50,000-row sample, before cleaning |
| `data/group_cleaned_seed44.csv` | **Use this for all models**: filtered reviews and corrected text |
| `data/group_audit_seed44.csv` | Log of accepted word corrections |
| `data/group_report_seed44.json` | Counts and validation statistics |

### Cleaned dataframe columns

The cleaned CSV retains the original review columns and adds/updates the following:

| Column | Description |
| --- | --- |
| `Text` | Final corrected review text; **model input** |
| `Text_before_correction` | Original review text before spelling/contraction corrections |
| `Sentiment` | Binary target derived from `Score`: 1–2 stars = negative, 4–5 stars = positive |
| `Score` | Original star rating (1–5); **do not use as a predictor** |


