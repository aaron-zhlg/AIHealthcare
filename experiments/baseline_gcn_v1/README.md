# Baseline GCN v1 — ASD vs Control Classification

Frozen reference experiment for the AIHealthcare pipeline. Use this as the comparison point for all future improvements.

> **Git tag:** `baseline-gcn-v1`  
> **Evaluated:** 2026-08-27  
> **Model selection:** final epoch — no selection on the evaluation set

---

## Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | 64.4% |
| **AUC** | **0.677** |
| **F1** | 0.690 |

Validation set: **177 subjects** (82 ASD, 95 control), stratified 80/20 split, `seed=42`.

**Conclusion:** The pipeline works and learns signal above chance (AUC > 0.5). Performance is moderate and consistent with typical ABIDE connectivity classification results (~60–70%). Not suitable for clinical use.

> **Revised 2026-08-27.** An earlier version of this report gave AUC 0.688 by taking the
> epoch with the highest validation AUC. Because that epoch is chosen using the same set
> it is scored on, the number was optimistic. The metrics above come from the final
> epoch. For reference, the best epoch reached AUC 0.695 — the gap between 0.695 and
> 0.677 is the size of the bias that selection introduced.

---

## Confusion Matrix

Rows = true label, columns = predicted label (`0`=ASD, `1`=control):

```text
              Pred ASD    Pred Control
True ASD        44            38
True Control    25            70
```

| Class | Support | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| ASD | 82 | 53.7% | 63.8% | 0.58 |
| Control | 95 | 73.7% | 64.8% | 0.69 |

The model is **better at identifying controls** than ASD (ASD recall = 53.7%).

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
- **Model selection:** the final epoch is reported. `best_model.pt` is still written, but
  the epoch it captures was chosen using the validation set, so its metrics are a
  diagnostic only and must not be reported.

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
| Checkpoint (reported) | `outputs/gcn_baseline/final_model.pt` | ❌ local |
| Checkpoint (diagnostic) | `outputs/gcn_baseline/best_model.pt` | ❌ local |
| Predictions | `outputs/gcn_baseline/predictions_val.csv` | ❌ local |
| Train log | `outputs/gcn_baseline/train.log` | ❌ local |
| Eval log | `outputs/gcn_baseline/eval.log` | ❌ local |

---

## Limitations

1. **Random 80/20 split only** — see [LOSO-CV results](../loso_cv_gcn_v1/README.md) for cross-site validation (AUC 0.623 ± 0.118).
2. **Single seed** — no multi-seed averaging. Across seeds 42/1337/2026 the same config gives AUC 0.628 ± 0.036, so this single-seed number sits at the optimistic end of its own spread.
3. **No model selection** — the final epoch is reported because there is no third split to select on. A proper inner validation split, carved out of the training data, would likely do better than either endpoint.
4. **Default hyperparameters** — no systematic tuning; see `autoresearch/` for the search.
5. **Simple GCN** — no attention, no site covariates, no class balancing.
6. **Research only** — not a diagnostic tool.

**Cross-site validation:** completed in [loso_cv_gcn_v1](../loso_cv_gcn_v1/README.md) (mean AUC 0.623 ± 0.118).

---

## Paper-Ready Text (draft)

> We established a baseline using a two-layer Graph Convolutional Network on Pearson functional connectivity matrices (111 ROIs, Harvard-Oxford atlas). Subjects (N=884) were split 80/20 for training and validation (stratified, seed=42). Metrics are reported at the final training epoch, with no model selection on the evaluation set. On the held-out validation set (N=177), the model achieved 64.4% accuracy, 0.677 AUC, and 0.690 F1. ASD recall was 53.7% and control recall was 73.7%, indicating a bias toward predicting the control class.

---

## Comparison Target

All future experiments should beat:

```text
Baseline GCN v1:  AUC = 0.677,  Accuracy = 64.4%   (random split, seed 42)
```

Improve on `experiment/improvements` branch; do not overwrite this experiment directory.
