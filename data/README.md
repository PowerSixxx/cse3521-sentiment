# Data

Data files are **not** stored in this repository. Every `*.csv` in this folder is ignored by git (see `.gitignore`). Each team member generates the same sample locally with the steps below.

## 1. Download the raw dataset (one time)

1. Go to <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews> and sign in.
2. Click **Download** and unzip the archive.
3. Move `Reviews.csv` (about 287 MB, 568,454 rows) into this `data/` folder.

Alternatively, with the Kaggle CLI and an API token in `~/.kaggle/kaggle.json`:

```bash
kaggle datasets download -d snap/amazon-fine-food-reviews -p data/ --unzip
```

## 2. Generate our group sample

Our group seed is **44** (Gamma). Use this seed for the whole project and never re-sample with a different one.

Run this from the repository root. It is the instructor's sampling script, with the paths pointed at this folder instead of `~/Downloads`:

```python
import os
import pandas as pd

seed = 44                                  # our group seed
reviews_path = os.path.join("data", "Reviews.csv")

df = pd.read_csv(reviews_path)
sample = df.sample(n=50000, random_state=seed)

out_path = os.path.join("data", f"group_sample_seed{seed}.csv")
sample.to_csv(out_path, index=False)

print(f"Saved {len(sample)} rows to {out_path}")
print(sample[["Score", "Text"]].head(3))
```

The result is `data/group_sample_seed44.csv`, which is the only dataset all notebooks and models should read.

**Sanity check:** the sample has 50,000 rows, and its Score counts for 1 to 5 should be exactly 4,721 / 2,690 / 3,803 / 7,043 / 31,743. If your counts differ, your sample is not the team's sample.

## Columns

| Column | Description |
|---|---|
| `Id` | Row id |
| `ProductId` | Amazon product id (ASIN) |
| `UserId` | Reviewer id |
| `ProfileName` | Reviewer display name |
| `HelpfulnessNumerator` | Users who found the review helpful |
| `HelpfulnessDenominator` | Users who voted on helpfulness |
| `Score` | Star rating, 1 to 5 (our prediction target) |
| `Time` | Unix timestamp |
| `Summary` | Review title |
| `Text` | Review body |

## Known data quality notes

- **Class imbalance:** about 78% of reviews are 4 or 5 stars.
- **Duplicates:** the same review text often appears under several products (one review posted to multiple product variants). Deduplicate on `Text` before the train/test split so the same review cannot land in both sets.
- **Missing values:** a few rows have an empty `Summary` or `ProfileName`.
- **HTML:** some `Text` values contain tags such as `<br />` that should be stripped during cleaning.
