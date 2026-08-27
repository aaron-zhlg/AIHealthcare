# Experiments

Each experiment gets a **folder** here with frozen config and results (safe to cite in papers).

| Folder | Description | Git tag |
|--------|-------------|---------|
| `baseline_gcn_v1/` | 2-layer GCN, random 80/20 split | `baseline-gcn-v1` |
| `loso_cv_gcn_v1/` | *(planned)* Leave-one-site-out CV | — |

## Convention

```text
experiments/<experiment_name>/
├── README.md        # methods, metrics, interpretation
├── run_config.json  # hyperparameters & data settings
└── results.json     # metrics, confusion matrix

outputs/<experiment_name>/   # local only (gitignored)
├── best_model.pt
├── train.log
└── eval.log
```

## Branches vs folders

| Use | For |
|-----|-----|
| **Git branch** | Code changes (e.g. `experiment/loso-cv`) |
| **`experiments/` folder** | Frozen results & config for paper |
| **`outputs/` folder** | Model checkpoints & logs (local) |

When an experiment is done: commit results to its folder on `main`, tag if it's a reference baseline.
