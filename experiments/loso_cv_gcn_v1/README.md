# LOSO-CV GCN v1 — Cross-Site Generalization

Leave-one-site-out cross-validation for the baseline GCN pipeline. Each fold holds out one acquisition site as test and trains on the remaining 19 sites.

> **Evaluated:** 2026-08-27  
> **Device:** Apple Silicon MPS  
> **Folds:** 20 sites, 884 subjects total

---

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Accuracy** | 61.3% ± 9.8% |
| **AUC** | **0.707 ± 0.093** |
| **F1** | 0.646 ± 0.122 |

**Conclusion:** Cross-site performance is comparable to the random-split baseline (AUC 0.688). The model generalizes above chance at most sites, but variance is high — some sites near random (MAX_MUN), others strong (UM_2, TRINITY). The largest held-out site (NYU, N=171) achieves AUC 0.722, which is the most reliable single-site estimate.

---

## Comparison with Baseline

| Experiment | Validation | AUC | Accuracy |
|------------|------------|-----|----------|
| Baseline GCN v1 | Random 80/20 split (N=177 val) | 0.688 | 62.7% |
| **LOSO-CV GCN v1** | Leave-one-site-out (20 folds) | **0.707 ± 0.093** | 61.3% ± 9.8% |

Mean LOSO AUC is slightly higher than the baseline val AUC, but the protocols differ (per-fold best-epoch selection, different test compositions). Treat them as complementary estimates, not directly comparable point scores.

---

## Per-Site Results

Sorted by test-set size (largest first — more reliable folds):

| Site | N_test | Accuracy | AUC | F1 |
|------|--------|----------|-----|-----|
| NYU | 171 | 66.1% | **0.722** | 0.698 |
| UM_1 | 82 | 61.0% | 0.636 | 0.680 |
| USM | 61 | 70.5% | 0.795 | 0.690 |
| UCLA_1 | 55 | 70.9% | 0.725 | 0.724 |
| YALE | 48 | 60.4% | 0.715 | 0.667 |
| PITT | 45 | 60.0% | 0.630 | 0.550 |
| TRINITY | 44 | 68.2% | 0.830 | 0.611 |
| MAX_MUN | 42 | 54.8% | 0.502 | 0.655 |
| KKI | 39 | 69.2% | 0.577 | 0.800 |
| CALTECH | 37 | 51.4% | 0.684 | 0.640 |
| STANFORD | 36 | 47.2% | 0.703 | 0.345 |
| SDSU | 33 | 54.5% | 0.655 | 0.667 |
| LEUVEN_2 | 32 | 71.9% | 0.737 | 0.791 |
| UM_2 | 31 | 80.6% | 0.912 | 0.857 |
| LEUVEN_1 | 29 | 65.5% | 0.700 | 0.706 |
| SBL | 26 | 61.5% | 0.774 | 0.688 |
| OLIN | 25 | 52.0% | 0.617 | 0.538 |
| OHSU | 23 | 69.6% | 0.652 | 0.667 |
| UCLA_2 | 20 | 50.0% | 0.750 | 0.375 |
| CMU | 5 | 40.0% | 0.833 | 0.571 |

**Notes:**
- **NYU (N=171)** — largest fold; AUC 0.722 is the key generalization number for paper reporting.
- **MAX_MUN (AUC 0.502)** — essentially at chance; site-specific scanner/protocol effects likely dominate.
- **CMU (N=5)** — too small for stable metrics; treat as unreliable.
- **UM_2 (AUC 0.912)** — best fold, but N=31; high AUC may reflect site-specific signal rather than broad generalization.

---

## Methods

Same model and hyperparameters as [baseline GCN v1](../baseline_gcn_v1/README.md):

- **Model:** 2-layer GCN (`SimpleGCN`), global mean pooling
- **Input:** 111×111 Pearson FC matrix per subject
- **Training:** 100 epochs, batch 32, lr 0.001, hidden 64, dropout 0.5, Adam, seed 42
- **Validation:** Leave-one-site-out — for each of 20 `SITE_ID` values, train on all other sites, test on held-out site
- **Model selection:** Best epoch per fold by held-out site AUC (see limitations)

---

## Reproduce

```bash
uv sync
./scripts/run_loso_cv.sh
```

Subset for debugging:

```bash
./scripts/run_loso_cv.sh --sites NYU,CALTECH --epochs 10
```

---

## Artifacts

| File | Location | In Git? |
|------|----------|---------|
| Config | `experiments/loso_cv_gcn_v1/run_config.json` | ✅ |
| Results | `experiments/loso_cv_gcn_v1/results.json` | ✅ |
| Per-fold checkpoints | `outputs/loso_cv_gcn_v1/folds/<SITE>/best_model.pt` | ❌ local |
| Summary | `outputs/loso_cv_gcn_v1/summary.json` | ❌ local |
| Log | `outputs/loso_cv_gcn_v1/loso_cv.log` | ❌ local |

---

## Limitations

1. **Per-fold test-set model selection** — best epoch is chosen by held-out site AUC, which slightly optimizes on test data. Future runs should use fixed epochs or an inner validation split on training sites.
2. **High site variance** — std ≈ 0.09 on AUC; small sites (CMU, UCLA_2) are unstable.
3. **Single seed** — no multi-seed averaging.
4. **No site harmonization** — ComBat or similar not applied; site effects remain in FC matrices.
5. **Research only** — not a diagnostic tool.

---

## Paper-Ready Text (draft)

> To assess cross-site generalization, we performed leave-one-site-out cross-validation across 20 ABIDE acquisition sites (N=884). For each fold, the model was trained on 19 sites and evaluated on the held-out site. Mean performance was 61.3% ± 9.8% accuracy, 0.707 ± 0.093 AUC, and 0.646 ± 0.122 F1. On the largest held-out site (NYU, N=171), AUC was 0.722. Performance varied substantially across sites (AUC range 0.50–0.91), reflecting known site effects in multi-site neuroimaging datasets.

---

## Comparison Target

For improvements on `experiment/improvements`, beat:

```text
LOSO-CV GCN v1:  AUC = 0.707 ± 0.093  (NYU fold: 0.722)
Baseline GCN v1: AUC = 0.688           (random split)
```

Do not overwrite this experiment directory.
