#!/usr/bin/env bash
# Regenerates every number, table and figure used by main.tex (about 1 minute).
set -e
python verify_core.py          # reproduces the n=200 benchmark, tests Propositions 1-3
python scale_experiments.py    # n=10,000 workloads, sweep, structural cases, guard
python feedback_experiment.py  # 60-block feedback loop
python paper_figures.py        # illustrative figures A-G and Cases 6-8
python audit_numbers.py        # checks every hard-coded number in main.tex
cat verification.txt
