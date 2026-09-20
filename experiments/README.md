# Experiments

Each experiment gets a **folder** here with frozen config and results (safe to cite in papers).

| Folder | Description | Git tag |
|--------|-------------|---------|
| `baseline_gcn_v1/` | 2-layer GCN, random 80/20 split | `baseline-gcn-v1` |
| `loso_cv_gcn_v1/` | Leave-one-site-out CV (20 sites) | — |
| `iter1_adding-a-residual-skip-h-h-f-h-w_v1/` | LOSO residual skip; final-epoch AUC 0.661 | — |
| `iter2_zero-initializing-the-dim-changi_v1/` | LOSO zero-init dim-changing skip; final-epoch AUC 0.661 | — |
| `iter6_fisher-z-transform-arctanh-with-_v1/` | LOSO Fisher-z node features; final-epoch AUC 0.690 | — |
| `iter14_per-site-rank-quantile-normaliza_v1/` | LOSO per-site rank/quantile edge map; final-epoch AUC 0.692 | — |
| `iter15_a-train-fold-only-coral-style-se_v1/` | LOSO CORAL-style site alignment; final-epoch AUC 0.692 | — |

## Convention

```text
experiments/<experiment_name>/
├── README.md        # methods, metrics, interpretation
├── run_config.json  # hyperparameters & data settings
├── results.json     # metrics, confusion matrix
└── figures/         # GNN attribution plots from ./scripts/run_explain.sh

outputs/<experiment_name>/   # local only (gitignored)
├── final_model.pt   # the reported model
├── best_model.pt    # diagnostic only; selected on the evaluation set
├── train.log
└── eval.log
```

## Reporting rule

Report the **final epoch**. Selecting the epoch that scores best on the evaluation set
uses that set twice, once to choose and once to report, which inflates the result. On
this dataset the inflation was worth about 0.07 AUC under LOSO.

## Branches vs folders

| Use | For |
|-----|-----|
| **Git branch** | Code changes (e.g. `experiment/loso-cv`) |
| **`experiments/` folder** | Frozen results & config for paper |
| **`outputs/` folder** | Model checkpoints & logs (local) |

When an experiment is done: commit results to its folder on `main`, tag if it's a reference baseline.
