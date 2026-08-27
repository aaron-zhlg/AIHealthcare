# Autoresearch Program

Instructions for an AI agent running experiments on this repository. Read this file
completely before making any change.

---

## 1. Objective

Improve ASD vs. healthy-control classification on ABIDE resting-state functional
connectivity, measured by **cross-site generalization** (leave-one-site-out AUC).

**Compare against the measured baseline, not the published one.** The published results
(`baseline_gcn_v1` AUC 0.688, `loso_cv_gcn_v1` AUC 0.707) selected the best epoch by
held-out AUC, which is optimistic. Scoring the same default config through this
framework with final-epoch metrics gives the honest numbers below, and those are what
the gates use.

| Stage | Measured baseline AUC | Published (best-epoch) |
|-------|----------------------|------------------------|
| `screen` | 0.628 ± 0.036 | 0.688 |
| `loso-subset` | 0.659 ± 0.071 | — |
| `loso-full` | 0.631 ± 0.108 | 0.707 |

Full numbers live in `autoresearch/gates.json` under `measured_baseline`.

---

## 2. Non-negotiable rules

1. **Never commit to `main`.** Every trial runs on its own branch.
2. **Never modify `experiments/`.** Those directories are frozen published results.
   New results go in a new directory only after a `loso-full` trial passes.
3. **Never modify `data/`.** The dataset is fixed at 884 subjects, 111 ROIs.
4. **Change one thing at a time.** A trial that varies four hyperparameters at once
   teaches nothing about which one mattered.
5. **Gate on final-epoch metrics.** `autoresearch/trial.py` reports these by design;
   do not add best-epoch selection on the test fold, which leaks.
6. **Do not tune against `loso-full`.** It is the confirmation stage, not a search
   signal. Repeatedly sweeping on it overfits the only honest estimate available.

---

## 3. Evaluation stages

Trials are cheap-to-expensive. Do not skip ahead.

| Stage | What it runs | Cost | Gate |
|-------|--------------|------|------|
| `screen` | Random 80/20 split × 3 seeds | ~1 min | mean AUC ≥ 0.63 |
| `loso-subset` | LOSO on NYU, UM_1, USM, UCLA_1, YALE | ~2 min | mean AUC ≥ 0.67 |
| `loso-full` | LOSO on all 20 sites | ~6 min | mean AUC ≥ 0.66 |

Thresholds live in `autoresearch/gates.json`. `trial.py` exits `0` on PASS and `3` on
FAIL, so `run_trial.sh` can branch on the result.

Runtimes assume Apple Silicon MPS with the FC matrices cached in memory. They are short
enough that there is no excuse for skipping a stage.

---

## 4. Branch policy

```
main                            stable code + published results
experiment/autoresearch         this framework
experiment/trial-<slug>         one trial (created and destroyed by run_trial.sh)
```

`run_trial.sh` implements the policy:

1. Require a clean working tree.
2. Branch from `main`: `experiment/trial-<slug>`.
3. Apply the change, run the trial.
4. **PASS** → commit code + `result.json`, push the branch, report it for review.
5. **FAIL** → record in the ledger, return to `main`, delete the branch.

Failed trials leave no remote branch, but they are never lost: every run appends to
`outputs/autoresearch/ledger.jsonl`, which is gitignored and therefore survives branch
switches. **Read the ledger before proposing a new config** so the same dead end is not
explored twice.

---

## 5. Search space

Hyperparameters exposed by `trial.py` (no code change needed):

| Flag | Default | Sensible range |
|------|---------|----------------|
| `--epochs` | 100 | 30–200 |
| `--batch-size` | 32 | 8–64 |
| `--lr` | 1e-3 | 1e-4 – 5e-3 |
| `--hidden-dim` | 64 | 32–256 |
| `--dropout` | 0.5 | 0.2–0.7 |
| `--weight-decay` | 1e-4 | 1e-5 – 1e-2 |

Structural changes require editing `aihealthcare/gcn.py` on the trial branch. `trial.py`
imports whatever `SimpleGCN` the branch defines, so no framework change is needed.

Ideas worth testing, roughly in order of expected value:

1. **Class weighting** — ASD recall was only 50% in the baseline; the model favours the
   majority control class.
2. **Site harmonization** (e.g. ComBat) — site effects are the main obstacle, and
   per-site AUC ranges from 0.50 to 0.91.
3. **Edge sparsification** — threshold weak correlations instead of using a dense
   111×111 adjacency.
4. **Fisher z-transform** of correlations before use as node features.
5. **Attention pooling** instead of global mean pooling.
6. **Deeper or wider GCN**, with the caveat that 884 subjects is a small dataset.

---

## 6. Trial checklist

Before running:

- [ ] Read `outputs/autoresearch/ledger.jsonl`; confirm this config is new.
- [ ] State a one-line hypothesis and pass it via `--note`.
- [ ] Confirm exactly one thing differs from the best known config.

After running:

- [ ] Record the outcome, including failures, with the observed margin.
- [ ] On FAIL, say what the result rules out — that is the useful output.
- [ ] On PASS at `screen`, promote to `loso-subset`; on PASS there, promote to
      `loso-full`.
- [ ] On PASS at `loso-full`, create `experiments/<name>_v1/` with `run_config.json`,
      `results.json`, and a `README.md` following the format of
      `experiments/loso_cv_gcn_v1/README.md`, then open the branch for review.

---

## 7. Commands

```bash
# Single trial, screening stage
./autoresearch/run_trial.sh --name dropout03 --stage screen -- --dropout 0.3

# Promote a promising config
./autoresearch/run_trial.sh --name dropout03 --stage loso-subset -- --dropout 0.3

# Run the trial directly, without branch management
uv run python -m autoresearch.trial --name dropout03 --stage screen --dropout 0.3

# Review history
cat outputs/autoresearch/ledger.jsonl | jq -r '[.name,.stage,.summary.auc_mean,.verdict.passed] | @tsv'
```

---

## 8. Reporting

When reporting to the user, lead with the outcome: what was tried, what the number was,
and whether it beat the reference. Include the margin, not just PASS/FAIL. Never claim
an improvement from a `screen` result alone — it is a filter, not evidence.
