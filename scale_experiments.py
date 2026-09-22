"""
scale_experiments.py
====================
Production-scale evaluation of CLEF (n = 10,000) plus a block-size sweep,
structural cases at scale, and the toy-case summary table.

Run:     python scale_experiments.py
Outputs: results_*.csv, tab_*.tex, clef_numbers.tex, verification.txt,
         fig_scale_workloads.png, fig_scale_sweep.png

Every run checks Propositions 1-3 and the reported counts of those checks
are written to verification.txt.
"""

import csv
import math
import statistics as st
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clef_core import (
    NUM_SHARDS, WORKLOADS, CASE_NAMES, algorithm_C, algorithm_D, algorithm_F,
    algorithm_G, case6_closed_form, generate_block, metrics, order_shard_grouped,
    rmw_lower_bound, scaled_case, toy_case,
)

N_PROD = 10_000
SEEDS = [1, 2, 3, 4, 5]
SWEEP_N = [200, 500, 1_000, 2_000, 5_000, 10_000]
SWEEP_WL = ["DEX_BURSTY", "MIXED"]
QUADRATIC_CASES = {6: 1_000, 8: 1_000}      # k grows with n in these cases

COLORS = {"Naive": "#D85A30", "Model A (Alg. C)": "#888780",
          "Shard-grouped": "#1D9E75", "CLEF": "#534AB7"}
checks = defaultdict(lambda: [0, 0])          # name -> [passed, total]


def check(name, ok):
    checks[name][1] += 1
    checks[name][0] += int(bool(ok))


def timed(fn, *a, repeat=1):
    best, out = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn(*a)
        best = min(best, time.perf_counter() - t)
    return out, best * 1000.0


def evaluate_block(b, workload, repeat=1):
    """All strategies on one block, with proposition checks."""
    lb, lb_r = rmw_lower_bound(b)
    m_naive = metrics(b)
    a_naive, _ = algorithm_D(b)
    g_naive = algorithm_F(b, a_naive, m=m_naive)["G"]

    order_a, t_a = timed(algorithm_C, b, repeat=repeat)
    m_a = metrics(order_a)
    order_sg, t_sg = timed(order_shard_grouped, b, repeat=repeat)
    m_sg = metrics(order_sg)
    plan, t_g = timed(algorithm_G, b, repeat=repeat)
    m_g = plan["metrics"]
    plan_gd = algorithm_G(b, guard=True)
    m_gd = plan_gd["metrics"]
    check("Guard never worse than arrival (msgs)", m_gd["msgs"] <= m_naive["msgs"])

    # Proposition 1: conflict-preserving reordering changes nothing
    check("P1 Model A == naive (msgs, rounds, frontier)",
          (m_a["msgs"], m_a["rounds"], m_a["frontier"])
          == (m_naive["msgs"], m_naive["rounds"], m_naive["frontier"]))
    # Proposition 2: lower bound for every order; tight on pure-RMW workloads
    for m in (m_naive, m_a, m_sg, m_g):
        check("P2 msgs >= LB and rounds >= LB_rounds",
              m["msgs"] >= lb and m["rounds"] >= lb_r)
    if workload.startswith("DEX"):
        check("P2 shard-grouped attains LB on DEX", m_sg["msgs"] == lb)
    # Proposition 3: shard grouping bounds depth by k-1
    check("P3 shard-grouped rounds <= k-1", m_sg["rounds"] <= NUM_SHARDS - 1)
    check("CLEF msgs == shard-grouped msgs", m_g["msgs"] == m_sg["msgs"])

    return {
        "workload": workload, "n": len(b), "lb": lb, "lb_rounds": lb_r,
        "naive_msgs": m_naive["msgs"], "naive_rounds": m_naive["rounds"],
        "naive_f0": m_naive["F0"] / len(b), "naive_G": g_naive,
        "a_msgs": m_a["msgs"], "a_rounds": m_a["rounds"], "a_ms": t_a,
        "sg_msgs": m_sg["msgs"], "sg_rounds": m_sg["rounds"], "sg_ms": t_sg,
        "clef_msgs": m_g["msgs"], "clef_rounds": m_g["rounds"],
        "clef_f0": m_g["F0"] / len(b), "clef_G": plan["goodness"]["G"],
        "clef_ms": t_g, "n_hot": plan["n_hot"], "n_extreme": plan["n_extreme"],
        "guard_msgs": m_gd["msgs"], "guard_rounds": m_gd["rounds"],
        "guard_used": int(plan_gd["guarded"]),
        "n_avoidable": plan["n_avoidable"], "n_relocated": plan["n_relocated"],
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean(rows, key):
    return st.mean(r[key] for r in rows)


def sd(rows, key):
    vals = [r[key] for r in rows]
    return st.pstdev(vals) if len(vals) > 1 else 0.0


def fmt_int(x):
    return f"{x:,.0f}".replace(",", "{,}")


def pm(rows, key, dec=0):
    m, s = mean(rows, key), sd(rows, key)
    if dec == 0:
        return f"{fmt_int(m)}\\,$\\pm$\\,{fmt_int(s)}"
    return f"{m:.{dec}f}\\,$\\pm$\\,{s:.{dec}f}"


# ─────────────────────────────────────────────────────────────────────────
# Experiment 1: five workloads at n = 10,000
# ─────────────────────────────────────────────────────────────────────────

def experiment_workloads():
    rows = []
    for w in WORKLOADS:
        for s in SEEDS:
            rows.append(evaluate_block(generate_block(N_PROD, w, s), w))
            print(f"  [workloads] {w:<11} seed {s} done")
    write_csv("results_workloads.csv", rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Experiment 2: block-size sweep
# ─────────────────────────────────────────────────────────────────────────

def experiment_sweep():
    rows = []
    for w in SWEEP_WL:
        for n in SWEEP_N:
            for s in SEEDS:
                rows.append(evaluate_block(generate_block(n, w, s), w, repeat=3))
            print(f"  [sweep] {w:<11} n={n} done")
    write_csv("results_sweep.csv", rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Experiment 3: structural cases at scale, and the toy-case table
# ─────────────────────────────────────────────────────────────────────────

def structural_row(c, b):
    lb, lb_r = rmw_lower_bound(b)
    m0 = metrics(b)
    plan = algorithm_G(b)
    m1 = plan["metrics"]
    plan_gd = algorithm_G(b, guard=True)
    a0, _ = algorithm_D(b)
    return {"case": c, "n": len(b), "lb": lb,
            "naive_msgs": m0["msgs"], "clef_msgs": m1["msgs"],
            "naive_rounds": m0["rounds"], "clef_rounds": m1["rounds"],
            "naive_f0": m0["F0"], "clef_f0": m1["F0"],
            "naive_G": algorithm_F(b, a0, m=m0)["G"],
            "clef_G": plan["goodness"]["G"],
            "n_avoidable": plan["n_avoidable"], "n_relocated": plan["n_relocated"],
            "guard_msgs": plan_gd["metrics"]["msgs"],
            "guard_rounds": plan_gd["metrics"]["rounds"],
            "guard_used": int(plan_gd["guarded"])}


def experiment_structural():
    rows = []
    for c in range(9):
        n = QUADRATIC_CASES.get(c, N_PROD)
        rows.append(structural_row(c, scaled_case(c, n)))
        print(f"  [structural] case {c} n={n} done")
    write_csv("results_structural.csv", rows)
    toy = [structural_row(c, toy_case(c)) for c in range(9)]
    write_csv("results_toy.csv", toy)
    return rows, toy


# ─────────────────────────────────────────────────────────────────────────
# LaTeX tables and macros
# ─────────────────────────────────────────────────────────────────────────

def arrow(a, b, dec=None):
    if dec is None:
        return f"{fmt_int(a)} $\\to$ {fmt_int(b)}"
    return f"{a:.{dec}f} $\\to$ {b:.{dec}f}"


def table_workloads(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["workload"]].append(r)
    lines = [
        "\\begin{table}[ht]", "\\centering\\small",
        "\\caption{Production-scale results ($n=10{,}000$, $k=4$ shards, "
        f"mean $\\pm$ s.d.\\ over {len(SEEDS)} seeds). "
        "LB is the lower bound of Proposition~\\ref{prop:lb}; "
        "the Model~A scheduler reproduced the naive metrics exactly in every run "
        "(Proposition~\\ref{prop:invariance}).}",
        "\\label{tab:scale}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Workload & Naive msgs & CLEF msgs & $\\Delta$ & LB & Naive rds & "
        "CLEF rds & CLEF time (ms) \\\\", "\\midrule",
    ]
    for w in WORKLOADS:
        rs = by[w]
        red = 100 * (1 - mean(rs, "clef_msgs") / mean(rs, "naive_msgs"))
        delta = f"$-${red:.1f}\\%" if red >= 0 else f"$+${-red:.1f}\\%"
        name = w.replace("_", "\\_")
        lines.append(
            f"{name} & {pm(rs, 'naive_msgs')} & {pm(rs, 'clef_msgs')} & {delta} & "
            f"{fmt_int(mean(rs, 'lb'))} & {pm(rs, 'naive_rounds')} & "
            f"{pm(rs, 'clef_rounds', 1)} & {pm(rs, 'clef_ms', 1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open("tab_scale_workloads.tex", "w").write("\n".join(lines) + "\n")


def table_structural(rows):
    lines = [
        "\\begin{table}[ht]", "\\centering\\small",
        "\\caption{Structural cases at scale. Cases~0--5 and~7 use "
        "$n=10{,}000$ (parametric or tiled); Cases~6 and~8 use $n=1{,}000$ "
        "because their read sets grow with $n$. For Case~6 at $n=10{,}000$ the "
        "closed form gives " + fmt_int(case6_closed_form(N_PROD)[0]) +
        " naive messages and " + fmt_int(case6_closed_form(N_PROD)[1]) +
        " rounds.}",
        "\\label{tab:structural}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{clrrrr}", "\\toprule",
        "Case & Structure & $n$ & Msgs (naive $\\to$ CLEF) & LB & "
        "Rounds (naive $\\to$ CLEF) \\\\", "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r['case']} & {CASE_NAMES[r['case']]} & {fmt_int(r['n'])} & "
                     f"{arrow(r['naive_msgs'], r['clef_msgs'])} & {fmt_int(r['lb'])} & "
                     f"{arrow(r['naive_rounds'], r['clef_rounds'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open("tab_scale_structural.tex", "w").write("\n".join(lines) + "\n")


def table_toy(rows):
    lines = [
        "\\begin{table}[ht]", "\\centering\\small",
        "\\caption{The nine structural cases at their illustrative sizes "
        "(naive order $\\to$ CLEF pipeline). Generated by "
        "\\texttt{scale\\_experiments.py}.}",
        "\\label{tab:results}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{clcccccc}", "\\toprule",
        "Case & Structure & $n$ & Msgs & Rounds & $|F_0|$ & LB & "
        "$\\mathcal{G}(B)$ \\\\", "\\midrule",
    ]
    for r in rows:
        if r["case"] == 6:
            lines.append("\\midrule")
        lines.append(
            f"{r['case']} & {CASE_NAMES[r['case']]} & {r['n']} & "
            f"{arrow(r['naive_msgs'], r['clef_msgs'])} & "
            f"{arrow(r['naive_rounds'], r['clef_rounds'])} & "
            f"{arrow(r['naive_f0'], r['clef_f0'])} & {r['lb']} & "
            f"{arrow(r['naive_G'], r['clef_G'], 2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open("tab_toy_cases.tex", "w").write("\n".join(lines) + "\n")


def table_guard(rows, structural):
    by = defaultdict(list)
    for r in rows:
        by[r["workload"]].append(r)
    lines = [
        "\\begin{table}[ht]", "\\centering\\small",
        "\\caption{Effect of the guard, which commits the arrival order "
        "whenever it has fewer cross-shard messages (ties broken by rounds). "
        f"Workloads: $n=10{{,}}000$, mean over {len(SEEDS)} seeds; the last "
        "column counts blocks in which the guard kept the arrival order.}",
        "\\label{tab:guard}",
        "\\begin{tabular}{lrrrrc}", "\\toprule",
        "Block & \\clef{} msgs & +guard msgs & \\clef{} rds & +guard rds & "
        "arrival kept \\\\", "\\midrule",
    ]
    for w in WORKLOADS:
        rs = by[w]
        lines.append(f"{w.replace('_', chr(92) + '_')} & {fmt_int(mean(rs, 'clef_msgs'))} & "
                     f"{fmt_int(mean(rs, 'guard_msgs'))} & "
                     f"{mean(rs, 'clef_rounds'):.1f} & {mean(rs, 'guard_rounds'):.1f} & "
                     f"{sum(r['guard_used'] for r in rs)}/{len(rs)} \\\\")
    r8 = [r for r in structural if r["case"] == 8][0]
    lines.append("\\midrule")
    lines.append(f"Case 8 ($n={fmt_int(r8['n'])}$) & {fmt_int(r8['clef_msgs'])} & "
                 f"{fmt_int(r8['guard_msgs'])} & {r8['clef_rounds']} & "
                 f"{r8['guard_rounds']} & {r8['guard_used']}/1 \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open("tab_guard.tex", "w").write("\n".join(lines) + "\n")


def fit_exponent(sweep, key):
    """Least-squares slope of log(time) vs log(n) using per-n means."""
    by = defaultdict(list)
    for r in sweep:
        by[r["n"]].append(r[key])
    xs = np.log([n for n in sorted(by)])
    ys = np.log([st.mean(by[n]) for n in sorted(by)])
    return float(np.polyfit(xs, ys, 1)[0])


def write_macros(rows, sweep, structural):
    by = defaultdict(list)
    for r in rows:
        by[r["workload"]].append(r)

    def red(w):
        return 100 * (1 - mean(by[w], "clef_msgs") / mean(by[w], "naive_msgs"))

    tags = {"P2P": "PP", "DEX_AVG": "DA", "DEX_BURSTY": "DB",
            "NFT": "NF", "MIXED": "MX"}
    out = ["% Auto-generated by scale_experiments.py -- do not edit by hand."]
    out.append(f"\\newcommand{{\\ScaleN}}{{{fmt_int(N_PROD)}}}")
    out.append(f"\\newcommand{{\\NumSeeds}}{{{len(SEEDS)}}}")
    for w, t in tags.items():
        out.append(f"\\newcommand{{\\{t}naive}}{{{fmt_int(mean(by[w], 'naive_msgs'))}}}")
        out.append(f"\\newcommand{{\\{t}clef}}{{{fmt_int(mean(by[w], 'clef_msgs'))}}}")
        out.append(f"\\newcommand{{\\{t}lb}}{{{fmt_int(mean(by[w], 'lb'))}}}")
        out.append(f"\\newcommand{{\\{t}red}}{{{abs(red(w)):.1f}}}")
        out.append(f"\\newcommand{{\\{t}naiverounds}}"
                   f"{{{fmt_int(mean(by[w], 'naive_rounds'))}}}")
    out.append(f"\\newcommand{{\\PipelineMs}}"
               f"{{{max(mean(by[w], 'clef_ms') for w in WORKLOADS):.0f}}}")
    out.append(f"\\newcommand{{\\ExpClef}}{{{fit_exponent(sweep, 'clef_ms'):.2f}}}")
    out.append(f"\\newcommand{{\\ExpAlgC}}{{{fit_exponent(sweep, 'a_ms'):.2f}}}")
    total_runs = len(rows) + len(sweep)
    out.append(f"\\newcommand{{\\NumRuns}}{{{total_runs}}}")
    nf = by["NFT"]
    out.append(f"\\newcommand{{\\NFguard}}{{{fmt_int(mean(nf, 'guard_msgs'))}}}")
    out.append(f"\\newcommand{{\\NFguardrounds}}{{{mean(nf, 'guard_rounds'):.1f}}}")
    out.append(f"\\newcommand{{\\NFguardused}}{{{sum(r['guard_used'] for r in nf)}}}")
    out.append(f"\\newcommand{{\\GuardUsedTotal}}"
               f"{{{sum(r['guard_used'] for r in rows)}}}")
    open("clef_numbers.tex", "w").write("\n".join(out) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────

def style(ax):
    ax.set_facecolor("#F8F7F2")
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def figure_workloads(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["workload"]].append(r)
    strategies = [("Naive", "naive"), ("Model A (Alg. C)", "a"),
                  ("Shard-grouped", "sg"), ("CLEF", "clef")]
    x = np.arange(len(WORKLOADS))
    wid = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle(f"Production scale: n = {N_PROD:,} transactions, k = {NUM_SHARDS} "
                 f"shards, {len(SEEDS)} seeds (error bars = s.d.)",
                 fontsize=13, fontweight="bold")
    for panel, (metric, title) in enumerate([("msgs", "Cross-shard messages"),
                                             ("rounds", "Communication rounds")]):
        ax = axes[panel]
        for k, (label, key) in enumerate(strategies):
            means = [max(mean(by[w], f"{key}_{metric}"), 0.5) for w in WORKLOADS]
            sds = [sd(by[w], f"{key}_{metric}") for w in WORKLOADS]
            ax.bar(x + (k - 1.5) * wid, means, wid, yerr=sds, capsize=2,
                   label=label, color=COLORS[label], alpha=0.9, edgecolor="white")
        if metric == "msgs":
            idx = [i for i, w in enumerate(WORKLOADS) if mean(by[w], "lb") > 0]
            ax.scatter([x[i] + 1.5 * wid for i in idx],
                       [mean(by[WORKLOADS[i]], "lb") for i in idx],
                       marker="_", s=320, linewidths=2.5, color="black",
                       zorder=5, label="Lower bound (Prop. 2, where > 0)")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace("_", "\n") for w in WORKLOADS], fontsize=9)
        ax.set_title(f"({'ab'[panel]}) {title} (log scale)", fontsize=11,
                     fontweight="bold")
        ax.set_ylim(top=ax.get_ylim()[1] * 8)
        ax.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.95)
        style(ax)
    ax = axes[2]
    g_naive = [mean(by[w], "naive_G") for w in WORKLOADS]
    g_clef = [mean(by[w], "clef_G") for w in WORKLOADS]
    ax.bar(x - 0.18, g_naive, 0.36, color=COLORS["Naive"], label="Naive",
           alpha=0.9, edgecolor="white")
    ax.bar(x + 0.18, g_clef, 0.36, color=COLORS["CLEF"], label="CLEF",
           alpha=0.9, edgecolor="white")
    for i in range(len(WORKLOADS)):
        ax.text(x[i] + 0.18, g_clef[i] + 0.01, f"{g_clef[i]:.2f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", "\n") for w in WORKLOADS], fontsize=9)
    ax.set_title("(c) Block goodness G(B)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("fig_scale_workloads.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_sweep(sweep):
    by = defaultdict(list)
    for r in sweep:
        by[(r["workload"], r["n"])].append(r)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    fig.suptitle("Block-size sweep (n = 200 ... 10,000, 5 seeds per point)",
                 fontsize=13, fontweight="bold")
    ns = np.array(SWEEP_N)
    for col, w in enumerate(SWEEP_WL):
        ax = axes[0, col]
        for label, key, mk in [("Naive", "naive_msgs", "o--"),
                               ("CLEF", "clef_msgs", "s-"),
                               ("Lower bound (Prop. 2)", "lb", "k:")]:
            ys = [mean(by[(w, n)], key) for n in SWEEP_N]
            color = COLORS.get(label, "black")
            ax.plot(ns, ys, mk.replace("k", ""), color=color, label=label, markersize=6,
                    linewidth=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Block size n")
        ax.set_ylabel("Cross-shard messages")
        ax.set_title(f"({'ab'[col]}) {w}: messages vs n", fontweight="bold")
        ax.legend(fontsize=8)
        style(ax)
    ax = axes[1, 0]
    for w, ls in zip(SWEEP_WL, ["-", "--"]):
        ax.plot(ns, [mean(by[(w, n)], "naive_rounds") for n in SWEEP_N], "o" + ls,
                color=COLORS["Naive"], label=f"Naive, {w}")
        ax.plot(ns, [mean(by[(w, n)], "clef_rounds") for n in SWEEP_N], "s" + ls,
                color=COLORS["CLEF"], label=f"CLEF, {w}")
    ax.axhline(NUM_SHARDS - 1, color="black", linestyle=":", linewidth=1,
               label="k − 1 bound (Prop. 3)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block size n")
    ax.set_ylabel("Communication rounds")
    ax.set_title("(c) Rounds vs n", fontweight="bold")
    ax.legend(fontsize=8)
    style(ax)
    ax = axes[1, 1]
    for label, key, mk in [("CLEF pipeline (Alg. G)", "clef_ms", "s-"),
                           ("Alg. C alone", "a_ms", "^-"),
                           ("Shard-grouped", "sg_ms", "o-")]:
        ys = [st.mean(r[key] for w in SWEEP_WL for r in by[(w, n)]) for n in SWEEP_N]
        color = {"clef_ms": COLORS["CLEF"], "a_ms": COLORS["Model A (Alg. C)"],
                 "sg_ms": COLORS["Shard-grouped"]}[key]
        ax.plot(ns, ys, mk, color=color, label=label, markersize=6, linewidth=1.8)
    ref = [st.mean(r["clef_ms"] for w in SWEEP_WL for r in by[(w, SWEEP_N[-1])])]
    c = ref[0] / (SWEEP_N[-1] * math.log2(SWEEP_N[-1]))
    ax.plot(ns, c * ns * np.log2(ns), "k:", label="c · n log n (reference)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block size n")
    ax.set_ylabel("Wall-clock time (ms, Python)")
    ax.set_title(f"(d) Scheduling time; fitted exponent "
                 f"{fit_exponent(sweep, 'clef_ms'):.2f}", fontweight="bold")
    ax.legend(fontsize=8)
    style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("fig_scale_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print("Experiment 1: workloads at n = 10,000")
    rows = experiment_workloads()
    print("Experiment 2: block-size sweep")
    sweep = experiment_sweep()
    print("Experiment 3: structural cases")
    structural, toy = experiment_structural()

    table_workloads(rows)
    table_structural(structural)
    table_toy(toy)
    table_guard(rows, structural)
    write_macros(rows, sweep, structural)
    figure_workloads(rows)
    figure_sweep(sweep)

    with open("verification.txt", "w") as f:
        for name, (ok, total) in checks.items():
            line = f"{name:<48} {ok}/{total}"
            f.write(line + "\n")
            print(line)
    print(f"Done in {time.time() - t0:.0f} s")
