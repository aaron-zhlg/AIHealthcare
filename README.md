# AIHealthcare

AI-driven brain network analysis for autism research using the ABIDE dataset. This project builds a pipeline from resting-state fMRI ROI time series to graph neural network (GNN) classification, interpretability, literature grounding, and in-silico intervention.

---

## Research Goal

Use **functional brain connectivity** patterns to:

1. **Classify** ASD vs. typically developing controls
2. **Identify** key brain regions and connections driving the classification
3. **Ground** findings in existing literature via LLM retrieval
4. **Simulate** virtual node/edge interventions to explore whether network states shift toward a control-like profile

> This is a **research pipeline**, not a clinical diagnostic tool.

---

## Roadmap

```text
Resting-state fMRI ROI time series
       ↓
Functional connectivity (Pearson) → brain graph
       ↓
GNN classification baseline (ASD vs control)
       ↓
Identify key brain regions (GNN interpretability)
       ↓
LLM literature retrieval for evidence
       ↓
Virtual node modulation → test network state shift
```

### Status

| Step | Description | Status |
|------|-------------|--------|
| **1. Data & preprocessing** | Download ABIDE `rois_ho`, align phenotypic CSV with local `.1D` files, compute Pearson FC matrices | ✅ Done |
| **2. GNN classification baseline** | Simple GCN on 111×111 FC graphs; train/val split; accuracy, AUC, F1 | ✅ Done ([results](experiments/baseline_gcn_v1/README.md)) |
| **2b. Cross-site validation** | Leave-one-site-out (LOSO-CV) across 20 acquisition sites | ✅ Done ([results](experiments/loso_cv_gcn_v1/README.md)) |
| **3. Key brain regions** | Integrated gradients on any `SimpleGCN` checkpoint; ROI ranking + chord plot | 🔲 Code ready (`./scripts/run_explain.sh`) |
| **4. LLM literature retrieval** | Query whether identified regions are implicated in ASD | 🔲 Planned |
| **5. Virtual intervention** | Perturb node/edge features in-silico; re-run GNN; observe prediction shift | 🔲 Planned |
| **Future** | Swap FC algorithms (Spearman, partial correlation, dynamic FC) from `.1D` time series | 🔲 Planned |

### Design decisions

- **Start with Pearson FC** — simple, standard in ABIDE literature, already precomputed in `processed/fc/`. Once the pipeline works, recompute FC from `.1D` with alternative methods for comparison.
- **884 aligned subjects** — phenotypic CSV and local `.1D` files intersected; subjects with excessive head motion (`func_mean_fd ≥ 0.2`) or missing files excluded.
- **111 ROIs** — Harvard-Oxford atlas; each subject → one 111×111 FC matrix → one brain graph.

---

## Results

All metrics are taken at the **final training epoch**. Nothing is selected using the set
it is scored on.

| Experiment | Validation | Accuracy | AUC | F1 |
|------------|------------|----------|-----|-----|
| Baseline GCN v1 | Random 80/20 split (N=177) | 64.4% | **0.677** | 0.690 |
| LOSO-CV GCN v1 | Leave-one-site-out, 20 sites | 59.3% ± 11.2% | **0.623 ± 0.118** | 0.600 ± 0.133 |

Largest held-out site (NYU, N=171): AUC **0.682**. Three sites fall below chance, so
cross-site generalization remains the open problem.

Full reports: [baseline](experiments/baseline_gcn_v1/README.md) ·
[LOSO-CV](experiments/loso_cv_gcn_v1/README.md)

> **Revised 2026-08-27.** Earlier revisions reported 0.688 and 0.707 by taking the epoch
> with the highest score on the evaluation set. That selects a model using the data it is
> scored on, so those numbers were optimistic; on LOSO the bias was worth about 0.07 AUC.
> The pipeline now reports the final epoch, and the tables above are the corrected values.

---

## Automated Experiments

`autoresearch/` runs the search for a better model: an agent proposes one change per
trial, runs it on a throwaway branch, and the branch survives only if it clears a gate.

```bash
./autoresearch/run_trial.sh --name dropout03 --note "less regularization" -- --dropout 0.3
```

| Stage | Evaluation | Cost | Gate |
|-------|------------|------|------|
| `screen` | Random 80/20 × 3 seeds | ~1 min | AUC ≥ 0.63 |
| `loso-subset` | LOSO, 5 largest sites | ~2 min | AUC ≥ 0.67 |
| `loso-full` | LOSO, all 20 sites | ~6 min | AUC ≥ 0.66 |

See [autoresearch/README.md](autoresearch/README.md) for the workflow and
[autoresearch/program.md](autoresearch/program.md) for the agent's rules.

---

## Current Data

| Item | Count | Location |
|------|-------|----------|
| Aligned subjects | 884 (408 ASD + 476 control) | `data/abide/processed/manifest.csv` |
| ROI time series | 884 `.1D` files | `data/abide/raw/Outputs/cpac/filt_global/rois_ho/` |
| FC matrices (Pearson) | 884 `.npy` files (111×111) | `data/abide/processed/fc/` |
| Labels | `0`=ASD, `1`=control | `data/abide/processed/labels.npy` |

See [data/abide/processed/README.md](data/abide/processed/README.md) for field definitions and usage examples.

---

## Project Structure

```text
AIHealthcare/
├── README.md                          ← this file
├── pyproject.toml / uv.lock           ← uv environment (Python 3.12, numpy)
├── data/abide/
│   ├── README.md                      ← raw ABIDE download documentation
│   ├── phenotypic/                    ← labels & metadata CSV
│   ├── raw/                           ← ROI time series (.1D), gitignored
│   ├── processed/                     ← aligned manifest, FC matrices, docs
│   └── scripts/
│       ├── download_abide_preproc.py  ← download ROI time series from S3
│       ├── build_aligned_dataset.py   ← align CSV + .1D, compute FC
│       └── download_func_minimal.py   ← optional 4D fMRI download (~220 GB)
├── aihealthcare/                     ← GCN model, train, eval, loso_cv, explain
├── autoresearch/                     ← automated gated experiment loop
├── experiments/
│   ├── baseline_gcn_v1/              ← frozen baseline config + results
│   └── loso_cv_gcn_v1/               ← LOSO-CV config + results
├── scripts/
│   ├── run_gcn.sh                    ← train + eval orchestration
│   ├── run_loso_cv.sh                ← leave-one-site-out CV
│   ├── run_explain.sh                ← attribute a checkpoint and write figures
│   ├── train_gnn_baseline.py
│   └── eval_gnn_baseline.py
└── doc/
    └── public_fmri_neuroimaging_datasets.md
```

---

## Quick Start

```bash
# Install dependencies
uv sync

# Regenerate aligned dataset + FC matrices
uv run python data/abide/scripts/build_aligned_dataset.py

# Train GCN baseline (uses MPS on Apple Silicon if available)
./scripts/run_gcn.sh

# Train only
./scripts/run_gcn.sh --train-only

# Evaluate saved checkpoint on validation split
./scripts/run_gcn.sh --eval-only

# Custom hyperparameters
./scripts/run_gcn.sh --epochs 50 --batch-size 32

# Force CPU
./scripts/run_gcn.sh --cpu

# Attribute any trained SimpleGCN; figures go in that experiment's folder
./scripts/run_explain.sh --checkpoint outputs/gcn_baseline/final_model.pt \
    --output-dir experiments/baseline_gcn_v1/figures
```

---

## Next Step

**Run Step 3 on published checkpoints**, then use the ROI list for literature retrieval (Step 4).

---

## References

- ABIDE Preprocessed: http://preprocessed-connectomes-project.org/abide/
- Di Martino et al. (2014). *The autism brain imaging data exchange.* Molecular Psychiatry.
