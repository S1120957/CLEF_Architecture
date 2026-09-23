import random
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sortedcontainers import SortedList

from clef_core import (
    NUM_ACCOUNTS, SENDER_CUM, Tx, _sample, algorithm_D, algorithm_E, algorithm_E_original,
    coefficient_of_variation, make_mixed, mapping_loads,
)

S_OBJ, W = 16, 4
N_BLOCK, N_BLOCKS = 2_000, 60
PHASES = [(0, 20, "MIXED"), (20, 40, "MIXED + hotspot"), (40, 60, "MIXED")]
SEEDS = [1, 2, 3, 4, 5]
HOT_SHARD, HOT_FRACTION = 5, 0.5
SLACK, DELTA, MAX_MOVES = 1.2, 0.20, 4

POLICIES = ["Static affinity", "Balance only", "Loop, E original",
            "Loop, original rule, corrected input", "Loop, E corrected"]
COLORS = dict(zip(POLICIES, ["#D85A30", "#BA7517", "#888780", "#1D9E75",
                             "#534AB7"]))


def home16(tx):
    return (tx.writes[0] if tx.writes else tx.reads[0]) % S_OBJ


def make_hotspot(i, rng):
    nft = HOT_SHARD + S_OBJ * rng.randint(0, (1000 - HOT_SHARD) // S_OBJ)
    sender = _sample(SENDER_CUM, rng)
    receiver = rng.randint(0, NUM_ACCOUNTS - 1)
    nr = NUM_ACCOUNTS + nft                 # NUM_ACCOUNTS is a multiple of 16
    return Tx(i, [nr, sender], [nr, receiver], load_type="HOTSPOT")


def block_sequence(seed):
    rng = random.Random(seed)
    for b in range(N_BLOCKS):
        hot = 20 <= b < 40
        yield [make_hotspot(i, rng) if hot and rng.random() < HOT_FRACTION
               else make_mixed(i, rng) for i in range(N_BLOCK)]


def place(txs, homes, mapping, slack):
    if slack is None:
        load = [0] * W
        for h in homes:
            load[mapping[h]] += 1
        return load, 0
    assign, load = algorithm_D(txs, W, mapping, slack, homes=homes)
    return load, sum(a != mapping[h] for a, h in zip(assign, homes))


def run(policy, seed):
    mapping = {s: s % W for s in range(S_OBJ)}
    rows = []
    for b, txs in enumerate(block_sequence(seed)):
        homes = [home16(t) for t in txs]
        slack = None if policy == "Static affinity" else SLACK
        load, remote = place(txs, homes, mapping, slack)
        shard_loads = defaultdict(int)
        for h in homes:
            shard_loads[h] += 1
        moves = 0
        if policy == "Loop, E corrected":            # plan for next block
            mapping, ops, _ = algorithm_E(dict(shard_loads), mapping, W,
                                          DELTA, MAX_MOVES)
            moves = len(ops)
        elif policy == "Loop, original rule, corrected input":
            mapping, ops, _ = algorithm_E_original(
                mapping_loads(shard_loads, mapping, W), dict(shard_loads),
                mapping, DELTA)
            moves = len(ops)
        elif policy == "Loop, E original":
            mapping, ops, _ = algorithm_E_original(load, dict(shard_loads),
                                                   mapping, DELTA)
            moves = len(ops)
        rows.append({"block": b, "remote": remote / len(txs),
                     "imbalance": max(load) / (sum(load) / W),
                     "moves": moves,
                     "hot_share": shard_loads[HOT_SHARD] / len(txs)})
    return rows


def phase_mean(rows, key, lo, hi, skip=0):
    return st.mean(r[key] for r in rows if lo + skip <= r["block"] < hi)


def main():
    results = {p: [run(p, s) for s in SEEDS] for p in POLICIES}

    def curve(p, key):
        arr = np.array([[r[key] for r in rows] for rows in results[p]])
        return arr.mean(0), arr.std(0)

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("Feedback loop across 60 blocks (16 object shards on 4 workers, "
                 "n = 2,000 per block, 5 seeds; hotspot in blocks 20–39)",
                 fontsize=12.5, fontweight="bold")
    xs = np.arange(N_BLOCKS)
    for ax, key, title in [(axes[0], "remote", "(a) Remote placements (fraction)"),
                           (axes[1], "imbalance", "(b) Load imbalance (max / mean)")]:
        for p in POLICIES:
            if key == "remote" and p == "Static affinity":
                continue
            m, s = curve(p, key)
            ax.plot(xs, m, color=COLORS[p], lw=1.8, label=p)
            ax.fill_between(xs, m - s, m + s, color=COLORS[p], alpha=0.15)
        ax.axvspan(20, 40, color="#FAEEDA", alpha=0.5, zorder=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Block")
        ax.legend(fontsize=8)
        ax.set_facecolor("#F8F7F2")
        ax.grid(axis="y", ls="--", alpha=0.35)
    ax = axes[2]
    for p in ["Loop, E original", "Loop, original rule, corrected input",
              "Loop, E corrected"]:
        m, _ = curve(p, "moves")
        ax.step(xs, m, where="post", color=COLORS[p], lw=1.8, label=p)
    ax.axvspan(20, 40, color="#FAEEDA", alpha=0.5, zorder=0)
    ax.set_title("(c) Shard migrations planned per block", fontweight="bold")
    ax.set_xlabel("Block")
    ax.legend(fontsize=8)
    ax.set_facecolor("#F8F7F2")
    ax.grid(axis="y", ls="--", alpha=0.35)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("fig_feedback.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── table ─────────────────────────────────────────────────────────────
    def agg(p, key, lo, hi, skip=0):
        return st.mean(phase_mean(rows, key, lo, hi, skip) for rows in results[p])

    lines = [
        "\\begin{table}[ht]", "\\centering\\small",
        "\\caption{Feedback loop across blocks (mean over "
        f"{len(SEEDS)} seeds). Steady state excludes the first two blocks "
        "of each phase. Messages and rounds are identical for all policies "
        "because placement does not change the reads-from relation.}",
        "\\label{tab:feedback}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        "& \\multicolumn{2}{c}{MIXED (blocks 2--19)} & "
        "\\multicolumn{2}{c}{Hotspot (blocks 22--39)} & "
        "Migrations & First hotspot \\\\",
        "Policy & Remote & Imbalance & Remote & Imbalance & (total) & "
        "block remote \\\\", "\\midrule",
    ]
    numbers = {}
    for p in POLICIES:
        r1, i1 = agg(p, "remote", 0, 20, 2), agg(p, "imbalance", 0, 20, 2)
        r2, i2 = agg(p, "remote", 20, 40, 2), agg(p, "imbalance", 20, 40, 2)
        mig = st.mean(sum(r["moves"] for r in rows) for rows in results[p])
        first = st.mean(rows[20]["remote"] for rows in results[p])
        numbers[p] = (r1, i1, r2, i2, mig, first)
        lines.append(f"{p} & {100*r1:.1f}\\% & {i1:.2f} & {100*r2:.1f}\\% & "
                     f"{i2:.2f} & {mig:.1f} & {100*first:.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    open("tab_feedback.tex", "w").write("\n".join(lines) + "\n")

    bo, lc, lo_ = (numbers["Balance only"], numbers["Loop, E corrected"],
                   numbers["Loop, E original"])
    hot = st.mean(phase_mean(rows, "hot_share", 20, 40)
                  for rows in results["Balance only"])
    macros = [
        "% Auto-generated by feedback_experiment.py -- do not edit by hand.",
        f"\\newcommand{{\\FBstaticImb}}{{{numbers['Static affinity'][3]:.2f}}}",
        f"\\newcommand{{\\FBstaticImbMixed}}{{{numbers['Static affinity'][1]:.2f}}}",
        f"\\newcommand{{\\FBboRemoteMixed}}{{{100*bo[0]:.1f}}}",
        f"\\newcommand{{\\FBlcRemoteMixed}}{{{100*lc[0]:.1f}}}",
        f"\\newcommand{{\\FBboRemoteHot}}{{{100*bo[2]:.1f}}}",
        f"\\newcommand{{\\FBlcRemoteHot}}{{{100*lc[2]:.1f}}}",
        f"\\newcommand{{\\FBloRemoteHot}}{{{100*lo_[2]:.1f}}}",
        f"\\newcommand{{\\FBlcMigrations}}{{{lc[4]:.1f}}}",
        f"\\newcommand{{\\FBloMigrations}}{{{lo_[4]:.1f}}}",
        f"\\newcommand{{\\FBmixMigrations}}"
        f"{{{numbers['Loop, original rule, corrected input'][4]:.0f}}}",
        f"\\newcommand{{\\FBmixRemoteMixed}}"
        f"{{{100*numbers['Loop, original rule, corrected input'][0]:.1f}}}",
        f"\\newcommand{{\\FBmixRemoteHot}}"
        f"{{{100*numbers['Loop, original rule, corrected input'][2]:.1f}}}",
        f"\\newcommand{{\\FBlcFirstHot}}{{{100*lc[5]:.1f}}}",
        f"\\newcommand{{\\FBhotShare}}{{{100*hot:.0f}}}",
        f"\\newcommand{{\\FBredMixed}}{{{100*(1 - lc[0]/bo[0]):.0f}}}",
        f"\\newcommand{{\\FBredHot}}{{{100*(1 - lc[2]/bo[2]):.0f}}}",
    ]
    open("feedback_numbers.tex", "w").write("\n".join(macros) + "\n")

    for p in POLICIES:
        r1, i1, r2, i2, mig, first = numbers[p]
        print(f"{p:<18} MIXED remote {100*r1:5.1f}% imb {i1:.2f} | "
              f"hotspot remote {100*r2:5.1f}% imb {i2:.2f} | "
              f"migrations {mig:5.1f} | first hot block remote {100*first:.1f}%")
    print(f"hotspot share of shard {HOT_SHARD} during burst: {100*hot:.0f}%")


if __name__ == "__main__":
    main()
