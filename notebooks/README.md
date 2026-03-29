# CAFA 6 Notebooks

## Running Locally

```bash
bash setup.sh
source .venv/bin/activate
jupyter lab notebooks/01_EDA.ipynb
```

## Running on Google Colab (GPU)

Click any link below to open directly in Colab:

| Notebook | Colab Link | GPU Needed? |
|----------|------------|-------------|
| 01 - EDA | [Open in Colab](https://colab.research.google.com/github/AyushSharma173/CafaChallenge/blob/main/notebooks/01_EDA.ipynb) | No |
| 02 - Baseline | [Open in Colab](https://colab.research.google.com/github/AyushSharma173/CafaChallenge/blob/main/notebooks/02_Baseline.ipynb) | No |
| 03 - ESM-2 Embeddings | [Open in Colab](https://colab.research.google.com/github/AyushSharma173/CafaChallenge/blob/main/notebooks/03_ESM2_Embeddings.ipynb) | Yes (T4 min, A100 recommended) |
| 04 - Train Models | [Open in Colab](https://colab.research.google.com/github/AyushSharma173/CafaChallenge/blob/main/notebooks/04_Train_Models.ipynb) | No (uses precomputed embeddings) |
| 05 - Submission | [Open in Colab](https://colab.research.google.com/github/AyushSharma173/CafaChallenge/blob/main/notebooks/05_Submission.ipynb) | No |

### Colab Setup Steps

1. Click a Colab link above
2. **Runtime > Change runtime type > T4 GPU** (or A100 with Colab Pro)
3. Run the first cell — it clones the repo, installs deps, and downloads competition data
4. When prompted, upload your `kaggle.json` (get it from https://www.kaggle.com/settings > API > Create New Token)
5. Run the remaining cells

### Recommended Order

1. `01_EDA` — explore the dataset
2. `02_Baseline` — frequency baseline to validate the pipeline
3. `03_ESM2_Embeddings` — extract protein embeddings (GPU required, ~2-6 hrs)
4. `04_Train_Models` — train LightGBM and MLP on embeddings
5. `05_Submission` — generate and validate Kaggle submission
