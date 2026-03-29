# CAFA 6 - Protein Function Prediction

Kaggle competition: Predict Gene Ontology (GO) terms for proteins from amino acid sequences.

**Competition**: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
**Prize**: $50,000 | **Deadline**: ~Feb 2026 | **Metric**: Fmax (protein-centric F1)

---

## Quick Start

```bash
# 1. Setup environment
bash setup.sh

# 2. Activate
source .venv/bin/activate

# 3. Download data (needs Kaggle API key in ~/.kaggle/kaggle.json)
bash scripts/download_data.sh

# 4. Start exploring
jupyter lab notebooks/01_EDA.ipynb
```

## Competition Overview

**Task**: Given a protein's amino acid sequence, predict which GO terms (biological functions) apply to it.

**Three ontologies** (scored separately, then averaged):
- **BPO** (Biological Process): What role does the protein play? (e.g., DNA repair, signal transduction)
- **MFO** (Molecular Function): What does it do biochemically? (e.g., binding, catalysis)
- **CCO** (Cellular Component): Where is it located? (e.g., nucleus, membrane)

**Evaluation**: **Fmax** = maximum F1 score across confidence thresholds. Protein-centric (averaged across proteins, not terms). Scores computed per ontology then averaged.

**Key rules**:
- Predict GO terms with confidence scores for ALL proteins in `testsuperset.fasta`
- GO propagation is applied automatically: parent terms inherit max child confidence
- The test set is a secret subset of the superset -- predict everything

## Dataset

| File | Description |
|------|-------------|
| `train_sequences.fasta` | Training protein sequences (FASTA format) |
| `testsuperset.fasta` | Test proteins to predict |
| `train_terms.tsv` | Ground truth: `EntryID, term, aspect` |
| `go-basic.obo` | Gene Ontology structure (terms + hierarchy) |
| `train_taxonomy.tsv` | Protein-to-species mapping |
| `IA.txt` | Information accretion weights per GO term |

## Project Structure

```
notebooks/          Jupyter notebooks (EDA -> Baseline -> Embeddings -> Training -> Submission)
src/cafa6/          Reusable Python modules
scripts/            CLI scripts for data download, embeddings, training, submission
configs/            YAML experiment configs
data/raw/           Downloaded competition data
data/embeddings/    ESM-2 embeddings (HDF5)
models/             Saved model checkpoints
submissions/        Generated submission CSVs
```

## Experiment Workflow

### Phase 1: EDA & Baselines (local, no GPU)
```bash
jupyter lab notebooks/01_EDA.ipynb      # Explore data
jupyter lab notebooks/02_Baseline.ipynb  # Frequency baseline (~0.3 Fmax)
```

### Phase 2: Generate Embeddings (GPU required)
```bash
# Option A: SSH into cloud GPU (fastest)
python scripts/generate_embeddings.py --device cuda --batch-size 8

# Option B: Use notebook on Colab/Kaggle
jupyter lab notebooks/03_ESM2_Embeddings.ipynb

# Option C: Apple Silicon (slower but works)
python scripts/generate_embeddings.py --device mps --batch-size 2
```

### Phase 3: Train Models (local, no GPU needed)
```bash
# LightGBM (fast, ~2-3 hours on CPU)
python scripts/train.py --model-type lightgbm

# MLP (uses MPS on Mac)
python scripts/train.py --model-type mlp

# Or use notebook for interactive exploration
jupyter lab notebooks/04_Train_Models.ipynb
```

### Phase 4: Generate Submission
```bash
python scripts/submit.py --model-type lightgbm
# Or: jupyter lab notebooks/05_Submission.ipynb
```

## GPU Compute Guide

**You need a GPU only for ESM-2 embedding extraction** (~2-6 hours).
Model training on precomputed embeddings runs fine on CPU/M3.

| Option | GPU | Cost | Best For |
|--------|-----|------|----------|
| **Lambda Labs** | A100 | $1.29/hr | SSH, fastest, reliable |
| **RunPod** | A100 | $1.39/hr | SSH, good alternative |
| Colab Pro+ | A100 | $49.99/mo | Notebook workflow |
| Kaggle | P100 | Free (30hr/wk) | Budget option |

**Recommended**: Rent an A100 via SSH for ~3 hours (~$5 total). Generate embeddings, download HDF5 files, do everything else locally.

**SSH workflow**:
```bash
# On rented GPU
git clone <your-repo> && cd CAF6
pip install -e .
python scripts/generate_embeddings.py --device cuda --batch-size 8
# Download: scp user@host:CAF6/data/embeddings/*.h5 ./data/embeddings/
```

## Key Approaches

1. **Frequency baseline**: Predict training-set term frequency for all proteins (~0.3 Fmax)
2. **ESM-2 + LightGBM**: One GBM classifier per GO term on ESM-2 embeddings (~0.5-0.6 Fmax)
3. **ESM-2 + MLP**: Multi-output neural network (~0.5-0.6 Fmax)
4. **Advanced** (not yet implemented):
   - GO graph neural networks (ProtBoost approach)
   - Ensemble/stacking of multiple models
   - Taxonomy-aware features
   - Larger ESM-2 models (3B, 15B)
   - Fine-tuning ESM-2 with LoRA

## Modules Reference

| Module | Key Functions |
|--------|--------------|
| `data_loader` | `load_fasta()`, `load_train_terms()`, `build_label_matrix()`, `create_cv_split()` |
| `go_utils` | `load_go_graph()`, `propagate_annotations()`, `propagate_scores()`, `get_ancestors()` |
| `metrics` | `compute_fmax()`, `compute_smin()`, `evaluate_per_ontology()` |
| `embeddings` | `load_esm_model()`, `extract_embeddings()`, `load_embeddings()` |
| `models` | `NaiveFrequency`, `LightGBMMultilabel`, `MLPMultilabel` |
| `submission` | `generate_submission()`, `validate_submission()`, `predictions_from_matrices()` |
| `config` | `Config.from_yaml()` |
