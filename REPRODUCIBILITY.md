# CLEF: reproducibility package

## Run

    pip install -r requirements.txt
    ./run_all.sh            # or run the five scripts in the order listed there

Runtime is about one minute on a laptop (single-threaded Python).

## What each script produces

| Script | Produces | Used in main.tex as |
|---|---|---|
| `verify_core.py` | console check | reproduces the n=200, seed=42 benchmark; tests Propositions 1-3 on ~300 random blocks |
| `scale_experiments.py` | `tab_scale_workloads.tex`, `tab_scale_structural.tex`, `tab_toy_cases.tex`, `tab_guard.tex`, `clef_numbers.tex`, `fig_scale_workloads.png`, `fig_scale_sweep.png`, `results_*.csv`, `verification.txt` | Tables 2-5, Figures 1-2, numbers in the text |
| `feedback_experiment.py` | `tab_feedback.tex`, `feedback_numbers.tex`, `fig_feedback.png` | feedback-loop table and figure |
| `paper_figures.py` | `algo_*.png`, `figure_numbers.tex` | illustrative figures for Algorithms A-G and Cases 6-8 |
| `audit_numbers.py` | console check | verifies every number written by hand in main.tex |

`verification.txt` records, for every run, whether Propositions 1-3 held.

## Definitions implemented (must match the paper)

- Home shard S(tx): shard of the first written resource (first read if none), shard = resource mod k.
- Cross-shard message: a read whose last earlier writer has a different home shard, counted per read.
- Communication depth: cross-shard messages on the longest read-after-write path.
- Hotness: fraction of the last W transactions that access a resource.
- Algorithm D: capacity rule (1.2 x mean block load). Algorithm E: descent rule on mapping-induced load.

## Superseded files - do not use

Earlier drafts use conflicting definitions (home shard = smallest written ID, per-access hotness,
an O(n^2) Algorithm C, the order-sensitive Algorithm D, the original Algorithm E) and will give
different numbers: `shared.py`, `clef_algorithms.py`, `algo_A_*.py` ... `algo_G_*.py`,
`cases_6_7_8.py`, `algo_worst_cases.py`, `ordering_algorithms.py`, `shard_sim.py`,
`clef_architecture.tex`, `shard_algorithm.tex`.
The exploratory `testCase_algorithms/` folder uses the same home-shard definition and remains
valid as history; its Algorithm 2 result ("near-zero improvement") is now explained exactly by
Proposition 1 (the improvement is provably zero).
