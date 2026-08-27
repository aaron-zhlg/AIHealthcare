# Baseline GCN v1 — ASD vs Control Classification

Frozen reference experiment for the AIHealthcare pipeline. Use this as the comparison point for all future improvements.

> **Git tag:** `baseline-gcn-v1`  
> **Evaluated:** 2026-08-27  
> **Device:** Apple Silicon MPS

---

## Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 62.7% |
| **AUC** | **0.688** |
| **F1** | 0.680 |

Validation set: **177 subjects** (82 ASD, 95 control), stratified 80/20 split, `seed=42`.

**Conclusion:** The pipeline works and learns signal above chance (AUC > 0.5). Performance is moderate and consistent with typical ABIDE connectivity classification results (~60–70%). Not suitable for clinical use.

---

## Confusion Matrix

Rows = true label, columns = predicted label (`0`=ASD, `1`=control):

```text
              Pred ASD    Pred Control
True ASD        41            41
True Control    25            70
```

| Class | Support | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| ASD | 82 | 50.0% | 62.1% | 0.55 |
| Control | 95 | 73.7% | 63.1% | 0.68 |

The model is **better at identifying controls** than ASD (ASD recall = 50%).

---

## Methods

### Data

- **Dataset:** ABIDE Preprocessed (`rois_ho`, C-PAC, `filt_global`)
- **Subjects:** 884 aligned (408 ASD + 476 control)
- **Input:** 111×111 Pearson functional connectivity matrix per subject
- **Atlas:** Harvard-Oxford (111 ROIs)

### Model

- **Architecture:** 2-layer GCN (`SimpleGCN`) with global mean pooling
- **Node features:** FC matrix rows (connectivity profile per ROI)
- **Adjacency:** \|FC\| with symmetric normalization + self-loops
- **Output:** Binary classification (ASD vs control)

### Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Hidden dim | 64 |
| Dropout | 0.5 |
| Optimizer | Adam (weight decay 1e-4) |
| Loss | CrossEntropyLoss |

### Evaluation

- **Split:** StratifiedShuffleSplit, 80% train / 20% val
- **Seed:** 42 (fixed; see `outputs/gcn_baseline/split.json` after training)
- **Metrics:** Accuracy, AUC, F1

---

## Reproduce

```bash
uv sync
./scripts/run_gcn.sh
```

Or checkout the frozen code:

```bash
git checkout baseline-gcn-v1
./scripts/run_gcn.sh
```

---

## Artifacts

| File | Location | In Git? |
|------|----------|---------|
| Config | `experiments/baseline_gcn_v1/run_config.json` | ✅ |
| Results | `experiments/baseline_gcn_v1/results.json` | ✅ |
| Checkpoint | `outputs/gcn_baseline/best_model.pt` | ❌ local |
| Predictions | `outputs/gcn_baseline/predictions_val.csv` | ❌ local |
| Train log | `outputs/gcn_baseline/train.log` | ❌ local |
| Eval log | `outputs/gcn_baseline/eval.log` | ❌ local |

---

## Limitations

1. **Random 80/20 split only** — not leave-one-site-out (LOSO-CV). Results may be optimistic due to 20 acquisition sites.
2. **Single seed** — no multi-seed averaging; variance unknown.
3. **Default hyperparameters** — no systematic tuning.
4. **Simple GCN** — no attention, no site covariates, no class balancing.
5. **Research only** — not a diagnostic tool.

**Before publication:** add LOSO-CV and report mean ± std across folds.

---

## Paper-Ready Text (draft)

> We established a baseline using a two-layer Graph Convolutional Network on Pearson functional connectivity matrices (111 ROIs, Harvard-Oxford atlas). Subjects (N=884) were split 80/20 for training and validation (stratified, seed=42). On the held-out validation set (N=177), the model achieved 62.7% accuracy, 0.688 AUC, and 0.680 F1. ASD recall was 50.0% and control recall was 73.7%, indicating a bias toward predicting the control class.

---

## Comparison Target

All future experiments should beat:

```text
Baseline GCN v1:  AUC = 0.688,  Accuracy = 62.7%
```

Improve on `experiment/improvements` branch; do not overwrite this experiment directory.
