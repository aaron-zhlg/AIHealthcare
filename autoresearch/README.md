# Autoresearch

An automated experiment loop for the neuroasd GCN pipeline. An AI agent proposes a
change, runs it on a throwaway branch, and the branch survives only if the result beats
a predefined gate.

`program.md` is written for the agent. This file is written for you.

---

## Idea

Manual hyperparameter tuning is slow and easy to fool yourself with. This framework
makes three things mechanical:

1. **Cheap before expensive.** A full leave-one-site-out sweep takes hours, so every
   idea is first screened on a random split in minutes.
2. **Honest scoring.** Gating uses final-epoch metrics, so no model is selected by
   peeking at the test fold.
3. **Nothing is lost, nothing is cluttered.** Failed trials are deleted as branches but
   recorded in a local ledger, so the same dead end is not explored twice.

---

## Layout

```text
autoresearch/
├── program.md      instructions the agent reads before each trial
├── gates.json      pass thresholds per stage
├── trial.py        runs one config, scores it, writes result.json
├── run_trial.sh    branch lifecycle: create, run, push or delete
└── results/        result.json of trials that passed (committed on trial branches)

outputs/autoresearch/          local only, gitignored
├── ledger.jsonl               append-only record of every trial ever run
└── trials/<stage>__<name>/    per-trial result.json
```

The ledger is gitignored, which means it persists across branch switches and gives the
agent memory that is independent of git history.

---

## Stages

| Stage | Evaluation | Cost | Measured baseline | Gate |
|-------|------------|------|-------------------|------|
| `screen` | Random 80/20 split × 3 seeds | ~1 min | 0.628 | AUC ≥ 0.63 |
| `loso-subset` | LOSO on the 5 largest sites | ~2 min | 0.659 | AUC ≥ 0.67 |
| `loso-full` | LOSO on all 20 sites | ~6 min | 0.631 | AUC ≥ 0.66 |

Only a `loso-full` pass justifies a new directory under `experiments/`.

This framework is what surfaced the best-epoch selection bias in the original results:
rescoring the same config with final-epoch metrics moved LOSO from 0.707 to 0.62, and
`experiments/` has since been corrected to match.

---

## Usage

Run a trial with branch management:

```bash
./autoresearch/run_trial.sh --name dropout03 --note "less regularization" -- --dropout 0.3
```

Run a trial directly, without touching git:

```bash
uv run python -m autoresearch.trial --name dropout03 --stage screen --dropout 0.3
```

Establish honest reference points before reading small margins:

```bash
./autoresearch/run_trial.sh --name baseline --note "calibrate gates" --keep-on-fail --
```

Review history:

```bash
jq -r '[.name, .stage, .summary.auc_mean, .verdict.passed] | @tsv' \
  outputs/autoresearch/ledger.jsonl
```

---

## Branch policy

```text
main                       stable code and published results
experiment/autoresearch    this framework
experiment/trial-<slug>    one trial, created and usually deleted by run_trial.sh
```

`run_trial.sh` refuses to start with a dirty working tree, branches from `main`, and on
failure returns you to where you started and deletes the branch. On success it commits
the change plus the result and pushes the branch for review. It never merges to `main`
on its own.

---

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Gate cleared; branch pushed |
| 3 | Gate not cleared; branch discarded |
| 64 | Bad arguments |
| 65 | Dirty working tree, or the branch name is taken |
| other | The trial itself crashed; the branch is left in place for debugging |

---

## Adjusting the gates

Edit `autoresearch/gates.json`. Raise a threshold once a better configuration has been
promoted, so later trials must beat the new state of the art rather than the original
baseline.
