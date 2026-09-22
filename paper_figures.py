"""
paper_figures.py
================
Regenerates the three illustrative figures whose content depends on the
corrected definitions (per-transaction hotness, Algorithm D placement,
clipped goodness components). Output filenames match main.tex.

Run: python paper_figures.py
"""

from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clef_core import (
    CASE_NAMES, WORKLOADS, algorithm_A, algorithm_B, algorithm_D, algorithm_F,
    algorithm_G, generate_block, metrics, toy_case,
)

PURPLE, TEAL, CORAL, AMBER, GRAY = "#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#888780"


def style(ax):
    ax.set_facecolor("#F8F7F2")
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def short(c):
    return f"Case {c}: " + CASE_NAMES[c].replace("$a\\to b\\to c\\to d$", "a→b→c→d")


def figure_B():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.suptitle("Algorithm B: hot-resource detection with per-transaction "
                 "frequency η(r) (window W = 1000)", fontsize=13, fontweight="bold")

    b = generate_block(10_000, "DEX_BURSTY", 1)
    _, hot, ext, _, scores = algorithm_B(b)
    top = sorted(scores, key=lambda r: -scores[r])[:20]
    ax = axes[0, 0]
    cols = [CORAL if r in ext else AMBER if r in hot else PURPLE for r in top]
    ax.bar(range(len(top)), [scores[r] for r in top], color=cols, edgecolor="white")
    ax.axhline(0.9, color=CORAL, ls="--", lw=1.2, label="τ_e = 0.9 (extreme)")
    ax.axhline(0.7, color=AMBER, ls="--", lw=1.2, label="τ_h = 0.7 (hot)")
    ax.set_title("(a) Top-20 η(r), DEX_BURSTY, n = 10,000", fontweight="bold")
    ax.set_xlabel("Resource rank")
    ax.set_ylabel("η(r)")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[0, 1]
    for w, c in zip(WORKLOADS, [PURPLE, TEAL, CORAL, AMBER, GRAY]):
        _, _, _, _, sc = algorithm_B(generate_block(10_000, w, 1))
        ax.hist(list(sc.values()), bins=np.linspace(0, 1, 41), histtype="step",
                lw=1.6, color=c, label=w)
    ax.axvline(0.7, color=AMBER, ls="--", lw=1)
    ax.axvline(0.9, color=CORAL, ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_title("(b) Distribution of η(r), all workloads", fontweight="bold")
    ax.set_xlabel("η(r)")
    ax.set_ylabel("Resources (log)")
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1, 0]
    stats = defaultdict(list)
    for w in WORKLOADS:
        plan = algorithm_G(generate_block(10_000, w, 1))
        for k in ("n_hot", "n_extreme", "n_avoidable", "n_relocated"):
            stats[k].append(plan[k])
    x = np.arange(len(WORKLOADS))
    for i, (k, lab, c) in enumerate([("n_hot", "hot resources", AMBER),
                                     ("n_extreme", "extreme resources", CORAL),
                                     ("n_avoidable", "avoidable readers (A)", TEAL),
                                     ("n_relocated", "relocated readers (B)", PURPLE)]):
        vals = stats[k]
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=c, label=lab, edgecolor="white")
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1.5) * 0.2, v + 0.03, str(v), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", "\n") for w in WORKLOADS], fontsize=9)
    ax.set_title("(c) Pre-batching activity at n = 10,000", fontweight="bold")
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=8, loc="upper left")
    style(ax)

    ax = axes[1, 1]
    naive, after = [], []
    for c in range(9):
        t = toy_case(c)
        order, *_ = algorithm_B(t)
        naive.append(metrics(t)["msgs"])
        after.append(metrics(order)["msgs"])
    x = np.arange(9)
    ax.bar(x - 0.18, naive, 0.36, color=CORAL, label="Naive", edgecolor="white")
    ax.bar(x + 0.18, after, 0.36, color=PURPLE, label="After Algorithm B",
           edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{c}" for c in range(9)])
    ax.set_title("(d) Messages on Cases 0–8 after Algorithm B alone",
                 fontweight="bold")
    ax.set_ylabel("Cross-shard messages")
    ax.legend(fontsize=8)
    style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("algo_B_hot_resource.png", dpi=145, bbox_inches="tight")
    plt.close(fig)


def figure_F():
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Algorithm F: goodness components of the CLEF block for Cases 0–8 "
                 "(red dashed = G(B) of CLEF, grey dotted = G(B) of naive order)",
                 fontsize=13, fontweight="bold")
    labels = ["P", "1−S/n", "1−C/n", "L", "H"]
    for c, ax in zip(range(9), axes.flat):
        t = toy_case(c)
        plan = algorithm_G(t)
        g = plan["goodness"]
        a0, _ = algorithm_D(t)
        g0 = algorithm_F(t, a0)["G"]
        vals = [g["P"], g["S"], g["C"], g["L"], g["H"]]
        ax.bar(labels, vals, color=[PURPLE, TEAL, CORAL, AMBER, GRAY],
               edgecolor="white", alpha=0.9)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(g["G"], color=CORAL, ls="--", lw=1.5)
        ax.axhline(g0, color=GRAY, ls=":", lw=1.5)
        ax.set_ylim(0, 1.18)
        ax.set_title(f"{short(c)}\nG(B): naive {g0:.2f} → CLEF {g['G']:.2f}",
                     fontsize=9, fontweight="bold")
        style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("algo_F_block_goodness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_G():
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Algorithm G (full pipeline): naive order vs CLEF on Cases 0–8",
                 fontsize=13, fontweight="bold")
    for c, ax in zip(range(9), axes.flat):
        t = toy_case(c)
        m0 = metrics(t)
        plan = algorithm_G(t)
        m1 = plan["metrics"]
        x = np.arange(3)
        n_vals = [m0["msgs"], m0["rounds"], m0["F0"]]
        c_vals = [m1["msgs"], m1["rounds"], m1["F0"]]
        ax.bar(x - 0.18, n_vals, 0.36, color=CORAL, label="Naive", edgecolor="white")
        ax.bar(x + 0.18, c_vals, 0.36, color=PURPLE, label="CLEF", edgecolor="white")
        for xi, (a, b) in enumerate(zip(n_vals, c_vals)):
            ax.text(xi - 0.18, a + 0.05, str(a), ha="center", fontsize=8)
            ax.text(xi + 0.18, b + 0.05, str(b), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Cross-shard\nmessages", "Comm.\nrounds", "|F₀|"])
        ax.set_ylim(0, max(n_vals + c_vals + [1]) * 1.3)
        ax.set_title(f"{short(c)}\nG(B) = {plan['goodness']['G']:.2f}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("algo_G_execution_plan.png", dpi=140, bbox_inches="tight")
    plt.close(fig)



def figure_E():
    """Corrected vs originally specified rebalancing on 16 shards / 4 workers."""
    import random
    from collections import defaultdict
    import feedback_experiment as fe
    from clef_core import (algorithm_E, algorithm_E_original, mapping_loads,
                           coefficient_of_variation, make_mixed)
    S, Wk = 16, 4
    rng = random.Random(7)
    mixed = defaultdict(int)
    for i in range(2000):
        mixed[fe.home16(make_mixed(i, rng))] += 1
    scen = {
        "Balanced": {s: 125 for s in range(S)},
        "MIXED block (measured)": dict(mixed),
        "One hot shard (52%)": {s: (1040 if s == 5 else 64) for s in range(S)},
        "Hot worker (shards 0,4,8,12)": {s: (300 if s % 4 == 0 else 45) for s in range(S)},
        "Zipf shard loads": {s: round(2000 / (s + 1)) for s in range(S)},
        "Two hot shards, same worker": {s: (500 if s in (1, 5) else 83) for s in range(S)},
    }
    base = {s: s % Wk for s in range(S)}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Algorithm E on 16 shards / 4 workers: mapping-induced worker "
                 "load before and after rebalancing", fontsize=13, fontweight="bold")
    for ax, (name, sl) in zip(axes.flat, scen.items()):
        before = mapping_loads(sl, base, Wk)
        m_orig, ops_o, _ = algorithm_E_original(before, dict(sl), base)
        after_o = mapping_loads(sl, m_orig, Wk)
        m_corr, ops_c, after_c = algorithm_E(dict(sl), base, Wk, max_moves=16)
        x = np.arange(Wk)
        for k, (vals, lab, col) in enumerate(
                [(before, "Before", CORAL),
                 (after_o, f"Original rule ({len(ops_o)} moves)", GRAY),
                 (after_c, f"Corrected ({len(ops_c)} moves)", PURPLE)]):
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=col, label=lab,
                   edgecolor="white")
        cv = lambda v: coefficient_of_variation(v)
        ax.set_title(f"{name}\nCV {cv(before):.2f} → original {cv(after_o):.2f}, "
                     f"corrected {cv(after_c):.2f}", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"W{w}" for w in range(Wk)])
        ax.set_ylabel("Transactions")
        ax.legend(fontsize=7)
        style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("algo_E_shard_rebalancing.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_A():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Algorithm A: naive order vs shard-aware construction on Cases 0–5",
                 fontsize=13, fontweight="bold")
    from clef_core import order_shard_grouped
    for c, ax in zip(range(6), axes.flat):
        t = toy_case(c)
        m0, m1 = metrics(t), metrics(algorithm_A(t)[0])
        x = np.arange(2)
        ax.bar(x - 0.18, [m0["msgs"], m0["rounds"]], 0.36, color=CORAL, label="Naive",
               edgecolor="white")
        ax.bar(x + 0.18, [m1["msgs"], m1["rounds"]], 0.36, color=PURPLE,
               label="Algorithm A", edgecolor="white")
        for xi, (a, b) in enumerate([(m0["msgs"], m1["msgs"]), (m0["rounds"], m1["rounds"])]):
            ax.text(xi - 0.18, a + 0.05, str(a), ha="center", fontsize=9)
            ax.text(xi + 0.18, b + 0.05, str(b), ha="center", fontsize=9)
        homes = Counter(tx.home for tx in t)
        dist = "  ".join(f"S{s}:{homes.get(s, 0)}" for s in range(4))
        ax.set_xticks(x)
        ax.set_xticklabels(["Cross-shard\nmessages", "Comm.\nrounds"])
        ax.set_ylim(0, max(m0["msgs"], m0["rounds"], 1) * 1.35)
        ax.set_title(f"{short(c)}\nhome shards: {dist}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("algo_A_block_construction.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_C():
    fig, axes = plt.subplots(3, 3, figsize=(16, 12.5))
    fig.suptitle("Algorithm C (Model A): execution frontier |F_k| of the naive order and "
                 "of Algorithm C; the curves coincide (Proposition 1)",
                 fontsize=13, fontweight="bold")
    from clef_core import algorithm_C
    for c, ax in zip(range(9), axes.flat):
        t = toy_case(c)
        f0, f1 = metrics(t)["frontier"], metrics(algorithm_C(t))["frontier"]
        L = max(len(f0), len(f1))
        a = f0 + [0] * (L - len(f0))
        b = f1 + [0] * (L - len(f1))
        xs = np.arange(L)
        ax.fill_between(xs, a, color=CORAL, alpha=0.25, step="mid")
        ax.step(xs, a, where="mid", color=CORAL, lw=3.5, label="Naive")
        ax.step(xs, b, where="mid", color=PURPLE, lw=1.6, ls="--", label="Algorithm C")
        ax.set_xticks(xs)
        ax.set_xlabel("Communication round k")
        ax.set_ylabel("|F_k|")
        ax.set_ylim(0, max(a + [1]) * 1.25)
        ax.set_title(f"{short(c)}\nmsgs {metrics(t)['msgs']} → {metrics(algorithm_C(t))['msgs']}, "
                     f"rounds {metrics(t)['rounds']} → {metrics(algorithm_C(t))['rounds']}",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("algo_C_comm_round_reduction.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def figure_D():
    from clef_core import coefficient_of_variation
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Algorithm D at n = 10,000: per-worker load under strict shard affinity "
                 "and under Algorithm D (4 shards on 4 workers)",
                 fontsize=13, fontweight="bold")
    summary = []
    for w, ax in zip(WORKLOADS, axes.flat):
        b = generate_block(10_000, w, 1)
        plan = algorithm_G(b)
        aff = [0] * 4
        for tx in plan["order"]:
            aff[tx.home] += tx.gas
        placed = plan["goodness"]["loads"]
        remote = sum(1 for tx, a in zip(plan["order"], plan["assign"]) if a != tx.home)
        cv0, cv1 = coefficient_of_variation(aff), coefficient_of_variation(placed)
        summary.append((w, cv0, cv1, remote / len(b)))
        x = np.arange(4)
        ax.bar(x - 0.18, aff, 0.36, color=CORAL, label="Strict affinity", edgecolor="white")
        ax.bar(x + 0.18, placed, 0.36, color=PURPLE, label="Algorithm D", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"W{i}" for i in range(4)])
        ax.set_ylabel("Transactions")
        ax.set_title(f"{w}\nCV {cv0:.2f} → {cv1:.2f}, remote placements {100*remote/len(b):.1f}%",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        style(ax)
    ax = axes.flat[5]
    x = np.arange(len(summary))
    ax.bar(x - 0.18, [s[1] for s in summary], 0.36, color=CORAL, label="CV, strict affinity")
    ax.bar(x + 0.18, [s[2] for s in summary], 0.36, color=PURPLE, label="CV, Algorithm D")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0].replace("_", "\n") for s in summary], fontsize=8)
    ax.set_title("Summary: load CV per workload", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("algo_D_worker_placement.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    with open("figure_numbers.tex", "w") as f:
        f.write("% Auto-generated by paper_figures.py -- do not edit by hand.\n")
        f.write(f"\\newcommand{{\\DaffCVmax}}{{{max(s[1] for s in summary):.2f}}}\n")
        f.write(f"\\newcommand{{\\DplaceCVmax}}{{{max(s[2] for s in summary):.2f}}}\n")
        f.write(f"\\newcommand{{\\DremoteMax}}{{{100*max(s[3] for s in summary):.1f}}}\n")


def figure_worst():
    from clef_core import algorithm_C, typed_edge_counts
    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle("Cases 6–8: conflict structure, frontier, and naive vs Algorithm C vs CLEF",
                 fontsize=13, fontweight="bold")
    kinds = [("RAW", True), ("RAW", False), ("WAW", True), ("WAW", False),
             ("WAR", True), ("WAR", False)]
    kcol = [CORAL, "#F0997B", PURPLE, "#AFA9EC", TEAL, "#9FE1CB"]
    for col, c in enumerate((6, 7, 8)):
        t = toy_case(c)
        cnt = typed_edge_counts(t)
        ax = axes[0, col]
        vals = [cnt.get(k, 0) for k in kinds]
        ax.bar([f"{k}\n{'cross' if x else 'same'}" for k, x in kinds], vals, color=kcol,
               edgecolor="white")
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v + 0.1, str(v), ha="center", fontsize=9)
        ax.set_title(f"{short(c)}\nconflict edges: {sum(v for (k, x), v in cnt.items() if x)} cross, "
                     f"{sum(v for (k, x), v in cnt.items() if not x)} same-shard",
                     fontsize=9, fontweight="bold")
        style(ax)
        m0, mc = metrics(t), metrics(algorithm_C(t))
        mg = algorithm_G(t)["metrics"]
        ax = axes[1, col]
        L = max(len(m0["frontier"]), len(mg["frontier"]))
        pad = lambda f: f + [0] * (L - len(f))
        xs = np.arange(L)
        ax.step(xs, pad(m0["frontier"]), where="mid", color=CORAL, lw=3.5, label="Naive")
        ax.step(xs, pad(mc["frontier"]), where="mid", color=GRAY, lw=1.6, ls="--",
                label="Algorithm C")
        ax.step(xs, pad(mg["frontier"]), where="mid", color=PURPLE, lw=1.8, label="CLEF")
        ax.set_xticks(xs)
        ax.set_xlabel("Communication round k")
        ax.set_ylabel("|F_k|")
        ax.set_title(f"{short(c)}\nexecution frontier", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        style(ax)
        ax = axes[2, col]
        x = np.arange(3)
        for k, (m, lab, colr) in enumerate([(m0, "Naive", CORAL), (mc, "Algorithm C", GRAY),
                                            (mg, "CLEF", PURPLE)]):
            v = [m["msgs"], m["rounds"], m["F0"]]
            ax.bar(x + (k - 1) * 0.26, v, 0.26, color=colr, label=lab, edgecolor="white")
            for xi, vi in enumerate(v):
                ax.text(xi + (k - 1) * 0.26, vi + 0.05, str(vi), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Cross-shard\nmessages", "Comm.\nrounds", "|F₀|"])
        ax.set_title(f"{short(c)}\nnaive vs Algorithm C vs CLEF", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        style(ax)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("algo_worst_cases.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_A(); figure_B(); figure_C(); figure_D(); figure_E()
    figure_F(); figure_G(); figure_worst()
    print("Saved all eight illustrative figures and figure_numbers.tex")
