# LOSO-CV GCN v1 — Cross-Site Generalization

Leave-one-site-out cross-validation for the baseline GCN pipeline. Each fold holds out one acquisition site as test and trains on the remaining 19 sites.

> **Evaluated:** 2026-08-27  
> **Folds:** 20 sites, 884 subjects total  
> **Model selection:** final epoch — no selection on the held-out site

---

## Summary

| Metric | Mean ± Std |
|--------|------------|
| **Accuracy** | 59.3% ± 11.2% |
| **AUC** | **0.623 ± 0.118** |
| **F1** | 0.600 ± 0.133 |

**Conclusion:** Cross-site generalization is weak. The mean AUC of 0.623 is above chance
but well below the 0.677 obtained on a random split, which is the expected direction:
holding out an entire site removes the scanner and protocol the model was tuned to.
Variance across sites is large (AUC 0.40 to 0.93), and three sites land below chance.

> **Revised 2026-08-27.** An earlier version of this report gave AUC 0.707 ± 0.093 by
> taking, for each fold, the epoch with the highest AUC on that fold's held-out site.
> That selects a model using the very data it is scored on, once per fold, so the number
> was optimistic. Rerunning the identical configuration and reporting the final epoch
> gives 0.623 ± 0.118. The mean best-epoch AUC was 0.692, so **selection was worth about
> 0.07 AUC** — larger than most of the improvements one would hope to publish.

---

## Comparison

| Experiment | Validation | AUC |
|------------|------------|-----|
| Baseline GCN v1 | Random 80/20 split (N=177 val) | 0.677 |
| **LOSO-CV GCN v1** | Leave-one-site-out (20 folds) | **0.623 ± 0.118** |

The 0.054 drop from random split to leave-one-site-out is the cost of site effects, and
is the number to quote when claiming generalization.

---

## Per-Site Results

Sorted by test-set size (largest first — more reliable folds). The final column is what
best-epoch selection *would* have reported; it is shown to document the bias, not to be
cited.

| Site | N_test | Accuracy | AUC | F1 | (best-epoch AUC) |
|------|--------|----------|-----|-----|------------------|
| NYU | 171 | 56.7% | **0.682** | 0.565 | 0.713 |
| UM_1 | 82 | 45.1% | 0.475 | 0.416 | 0.658 |
| USM | 61 | 65.6% | 0.728 | 0.488 | 0.779 |
| UCLA_1 | 55 | 65.5% | 0.704 | 0.655 | 0.718 |
| YALE | 48 | 62.5% | 0.638 | 0.710 | 0.696 |
| PITT | 45 | 68.9% | 0.632 | 0.731 | 0.654 |
| TRINITY | 44 | 56.8% | 0.625 | 0.558 | 0.791 |
| MAX_MUN | 42 | 42.9% | 0.398 | 0.478 | 0.537 |
| KKI | 39 | 46.2% | 0.506 | 0.533 | 0.531 |
| CALTECH | 37 | 73.0% | 0.623 | 0.706 | 0.670 |
| STANFORD | 36 | 58.3% | 0.601 | 0.516 | 0.693 |
| SDSU | 33 | 63.6% | 0.556 | 0.750 | 0.619 |
| LEUVEN_2 | 32 | 68.8% | 0.696 | 0.762 | 0.717 |
| UM_2 | 31 | 87.1% | **0.925** | 0.905 | 0.965 |
| LEUVEN_1 | 29 | 62.1% | 0.671 | 0.667 | 0.710 |
| SBL | 26 | 57.7% | 0.768 | 0.645 | 0.774 |
| OLIN | 25 | 40.0% | 0.429 | 0.483 | 0.597 |
| OHSU | 23 | 60.9% | 0.553 | 0.571 | 0.614 |
| UCLA_2 | 20 | 45.0% | 0.573 | 0.353 | 0.740 |
| CMU | 5 | 60.0% | 0.667 | 0.500 | 0.667 |

**Notes:**

- **NYU (N=171)** — by far the largest fold, AUC 0.682. This is the most trustworthy
  single-site estimate and the one to quote alongside the mean.
- **Below chance at three sites** — MAX_MUN (0.398), OLIN (0.429) and UM_1 (0.475) score
  worse than random. Whatever the model learns from the other 19 sites actively
  misleads it there.
- **UM_2 (0.925)** — the best fold, but N=31. A single site scoring this far above the
  rest more likely reflects that site being easy than the model being good.
- **CMU (N=5)** — too small for a stable AUC; effectively noise.
- **Small folds are unreliable in both directions**, which is why the ±0.118 spread
  should not be read as a confidence interval on the mean.

---

## Methods

Same model and hyperparameters as [baseline GCN v1](../baseline_gcn_v1/README.md):

- **Model:** 2-layer GCN (`SimpleGCN`), global mean pooling
- **Input:** 111×111 Pearson FC matrix per subject
- **Training:** 100 epochs, batch 32, lr 0.001, hidden 64, dropout 0.5, Adam, seed 42
- **Validation:** Leave-one-site-out — for each of 20 `SITE_ID` values, train on all
  other sites, test on the held-out site
- **Model selection:** none. The final epoch is reported for every fold.

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
| Attribution figures | `experiments/loso_cv_gcn_v1/figures/` | ✅ |
| Per-fold checkpoints | `outputs/loso_cv_gcn_v1/folds/<SITE>/final_model.pt` | ❌ local |
| Summary | `outputs/loso_cv_gcn_v1/summary.json` | ❌ local |
| Log | `outputs/loso_cv_gcn_v1/loso_cv.log` | ❌ local |

---

## Limitations

1. **No model selection at all** — reporting the final epoch is honest but crude. The
   right fix is an inner validation split carved out of the 19 training sites, which
   would allow early stopping without touching the held-out site.
2. **Single seed** — each fold is trained once; per-fold variance is unknown and the
   ±0.118 reflects differences between sites, not run-to-run noise.
3. **Small folds dominate the spread** — five sites have fewer than 30 subjects. A
   subject-weighted mean would be a more stable summary than the unweighted one.
4. **No site harmonization** — ComBat or similar is not applied, so site effects remain
   in the FC matrices. Given three below-chance folds, this is the most promising place
   to improve.
5. **Research only** — not a diagnostic tool.

---

## Paper-Ready Text (draft)

> To assess cross-site generalization, we performed leave-one-site-out cross-validation
> across 20 ABIDE acquisition sites (N=884). For each fold, the model was trained on 19
> sites and evaluated on the held-out site, with metrics taken at the final training
> epoch so that no model selection used the held-out data. Mean performance was
> 59.3% ± 11.2% accuracy, 0.623 ± 0.118 AUC, and 0.600 ± 0.133 F1. On the largest
> held-out site (NYU, N=171), AUC was 0.682. Performance varied substantially across
> sites (AUC 0.40–0.93), with three sites falling below chance, reflecting known site
> effects in multi-site neuroimaging datasets.

---

## Comparison Target

For improvements on `experiment/improvements`, beat:

```text
LOSO-CV GCN v1:  AUC = 0.623 ± 0.118  (NYU fold: 0.682)
Baseline GCN v1: AUC = 0.677           (random split)
```

Do not overwrite this experiment directory.
