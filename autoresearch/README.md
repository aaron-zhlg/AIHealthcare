# Autoresearch

The lab and the PI for the neuroasd GCN pipeline.

`trial.py` + `gates.json` run one honest trial. `loop/` is a multi-agent
driver that keeps proposing one code change, linting it, measuring it, and
iterating. Built on [orchestra](https://github.com/aaron-zhlg/orchestra).

`program.md` is written for the agent. This file is written for you.

---

## Idea

Manual hyperparameter tuning is slow and easy to fool yourself with. This
framework makes four things mechanical:

1. **Cheap before expensive.** A full leave-one-site-out sweep takes hours, so
   every idea is first screened on a random split in minutes.
2. **Honest scoring.** Gating uses final-epoch metrics, so no model is selected
   by peeking at the test fold.
3. **Nothing is lost, nothing is cluttered.** Failed trials are deleted as
   branches but recorded in a local ledger, so the same dead end is not
   explored twice.
4. **Write → lint → measure → insight.** Agents do not share a chat. Each
   worker is a new instance; the next coder only sees the last insight.

---

## Layout

```text
autoresearch/
├── program.md      instructions the agent reads before each trial
├── gates.json      pass thresholds per stage
├── trial.py        runs one config, scores it, writes result.json
├── run_trial.sh    branch lifecycle: create, run, push or delete
├── results/        result.json of trials that passed (committed on trial branches)
└── loop/
    ├── orchestrator.py    lead + CLI
    ├── coder.py
    ├── linter.py
    ├── experimenter.py
    ├── workspace.py       session state and next_role()
    ├── protocol.py        final-epoch / leak guards
    ├── promote.py         review branch + score-first PR
    └── check_loop.py      no-LLM checks of the state machine

outputs/autoresearch/          local only, gitignored
├── ledger.jsonl               append-only record of every trial ever run
├── trials/<stage>__<name>/    per-trial result.json
└── loop/
    ├── workspace.json         multi-agent session (survives Ctrl-C)
    └── logs/
```

The ledger is gitignored, which means it persists across branch switches and
gives the agent memory that is independent of git history.

---

## Stages

| Stage | Evaluation | Cost | Measured baseline | Gate |
|-------|------------|------|-------------------|------|
| `screen` | Random 80/20 split × 3 seeds | ~1 min | 0.628 | AUC ≥ 0.66 |
| `loso-subset` | LOSO on the 5 largest sites | ~2 min | 0.659 | AUC ≥ 0.72 |
| `loso-full` | LOSO on all 20 sites | ~6 min | 0.631 | AUC ≥ 0.6895 |

Only a `loso-full` pass justifies a new directory under `experiments/`. A
`screen` PASS is only a filter.

This framework is what surfaced the best-epoch selection bias in the original
results: rescoring the same config with final-epoch metrics moved LOSO from
0.707 to 0.62, and `experiments/` has since been corrected to match.

---

## Multi-agent loop

Each worker is a **new instance** with an empty context window. The lead
dispatches **one** of them per round. They do not reuse chat history.

Cross-round memory is a file, not a conversation:
`outputs/autoresearch/loop/workspace.json`. It holds the hypothesis, files
touched, lint verdict, required stage, and the last insight (`ruled_out`,
`next_code_change`, final-epoch AUC). Restarting the process continues from
that file. `--fresh` wipes it.

That split exists so the coder cannot “remember” a best-epoch number, and so a
failed tool loop cannot quietly train a half-written edit.

| Agent | File | Job |
|-------|------|-----|
| Lead | `loop/orchestrator.py` (`GNNLead`) | Plan → dispatch one worker → evaluate → synthesize. Forces role order; does not edit or train. |
| Coder | `loop/coder.py` | One mechanism per turn, under `neuroasd/` or `trial.py`. Must read the last insight before editing. |
| Linter | `loop/linter.py` | Mechanical PASS/FAIL. The model cannot override the tool. |
| Experimenter | `loop/experimenter.py` | Run the workspace-required stage via `trial.py`, then write an insight the next coder can act on. |

The lead’s planner and evaluator still call an LLM, but `next_role()` in
`workspace.py` is the real scheduler. If the planner asks for the wrong worker,
the lead replaces the assignment.

```text
idle
  → coder writes one mechanism
  → coder tool-loop FAIL  → do not train; coder again
  → linter
       syntax, no best-epoch leak,
       change visible to trial.py (SimpleGCN + train_one_epoch)
  → lint FAIL  → coder, with the lint report as insight
  → experimenter runs one required stage
  → screen FAIL   → revert the diff; coder (new idea)
  → screen PASS   → same code, loso-subset   (coder frozen)
  → subset PASS   → same code, loso-full     (coder frozen)
  → full PASS     → review PR; keep the code; coder stacks the next mechanism
  → loso-full accuracy ≥ 0.80 → stop
```

Writable: `neuroasd/*.py` and `autoresearch/trial.py`. Not writable:
`gates.json`, `data/`, `experiments/`, `autoresearch/loop/`, `medresearch/`,
`.env`.

The scored path is `autoresearch.trial.run_fold`, which imports `SimpleGCN` and
`train_one_epoch`. An edit that only lives in `train.py` / `loso_cv.py` is a
measurement gap: lint FAILs it, because a trial would not see the change.

While status is `needs_loso_subset` or `needs_loso_full`, the current diff is
frozen. A FAIL archives the diff under `outputs/autoresearch/loop/graveyard/`
and restores coder-writable files to the last `loso-full` winner (or HEAD if
there is none). The rejected mechanism is appended to `workspace.ruled_out`
and injected into every later coder briefing. Each experimenter instance may
call `run_trial` once.

A protocol-clean `loso-full` PASS opens a review PR (scores first), snapshots
the winning files as the new baseline, and **keeps searching** — the next
coder stacks one new mechanism on that win. It does not wait for merge and
does not raise `gates.json`. The loop stops when `loso-full` accuracy_mean
reaches `--target-acc` (default 0.80) or on Ctrl-C. A later full run that
does not beat the last win's AUC is treated as a FAIL and reverted.

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

Unbounded multi-agent search (needs `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`):

```bash
set -a; source .env; set +a
uv run python -m autoresearch.loop.check_loop     # no LLM, no training
uv run python -m autoresearch.loop                # until 80% accuracy or Ctrl-C
uv run python -m autoresearch.loop --max-rounds 2 # local smoke
uv run python -m autoresearch.loop --fresh        # wipe workspace and restart
```

Omit `--max-rounds` (or pass `0`) for the unbounded loop. Each round is an LLM
call; after lint PASS it is also a real training job. Ctrl-C still writes a
session report. The same command, without `--fresh`, resumes from
`workspace.json`.

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
experiment/trial-<slug>    one trial, opened for review after a win
```

`run_trial.sh` refuses to start with a dirty working tree, branches from
`main`, and on failure returns you to where you started and deletes the
branch. On success it commits the change plus the result, pushes the branch,
and opens a review PR whose description starts with the scores. The agent
loop does the same after a `loso-full` PASS. Neither path merges to `main`
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

Edit `autoresearch/gates.json`. Raise a threshold once a better configuration
has been promoted, so later trials must beat the new state of the art rather
than the original baseline.
