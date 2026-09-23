from collections import defaultdict
from clef_core import *
import csv

def typed_edges(txs):
    """(kind, cross) counts over the conflict relation used in the paper."""
    c = defaultdict(int); lw = {}; lr = defaultdict(list)
    for j, tx in enumerate(txs):
        for r in tx.reads:
            if r in lw: c[("RAW", txs[lw[r]].home != tx.home)] += 1
        for r in tx.writes:
            if r in lw: c[("WAW", txs[lw[r]].home != tx.home)] += 1
            for p in lr[r]: c[("WAR", txs[p].home != tx.home)] += 1
            lw[r] = j; lr[r] = []
        for r in tx.reads: lr[r].append(j)
    return c

ok = []
def claim(text, cond): ok.append(cond); print(("OK   " if cond else "FAIL ") + text)

e6, e7, e8 = typed_edges(toy_case(6)), typed_edges(toy_case(7)), typed_edges(toy_case(8))
claim("Case 6: 9 cross RAW + 1 same RAW", e6[("RAW",True)] == 9 and e6[("RAW",False)] == 1)
cross7 = sum(v for (k, x), v in e7.items() if x); same7 = sum(v for (k, x), v in e7.items() if not x)
claim(f"Case 7: 10 edges, 3 cross (2 RAW, 1 WAR), 7 same  [got {cross7}/{same7}]",
      cross7 == 3 and same7 == 7 and e7[("RAW",True)] == 2 and e7[("WAR",True)] == 1)
claim(f"Case 8: 15 cross + 5 same WAR, naive msgs 0, |F0| 6",
      e8[("WAR",True)] == 15 and e8[("WAR",False)] == 5 and metrics(toy_case(8))["msgs"] == 0
      and metrics(toy_case(8))["F0"] == 6)
sg = lambda c: metrics(order_shard_grouped(toy_case(c)))
claim("Alg A: Case 0 3/3 -> 2/1, Case 4 5/5 -> 4/3, Cases 1,2,3,5 unchanged",
      (sg(0)["msgs"], sg(0)["rounds"]) == (2, 1) and (sg(4)["msgs"], sg(4)["rounds"]) == (4, 3)
      and all((sg(c)["msgs"], sg(c)["rounds"]) == (metrics(toy_case(c))["msgs"], metrics(toy_case(c))["rounds"]) for c in (1,2,3,5)))
b = lambda c: metrics(algorithm_B(toy_case(c))[0])["msgs"]
claim("Alg B alone: Case 6 9 -> 6, Case 7 2 -> 0", (b(6), b(7)) == (6, 0))
t = {int(r["case"]): r for r in csv.DictReader(open("results_toy.csv"))}
f = lambda c, k: round(float(t[c][k]), 2)
claim("Alg F: Case 6 0.30->0.47, Case 4 0.33->0.48, Case 8 0.93->0.68, Case 1 1.00",
      (f(6,"naive_G"), f(6,"clef_G"), f(4,"naive_G"), f(4,"clef_G"), f(8,"naive_G"), f(8,"clef_G"), f(1,"clef_G"))
      == (0.30, 0.47, 0.33, 0.48, 0.93, 0.68, 1.00))
claim("Case 6 toy CLEF: 4 msgs, 2 rounds; Case 8 toy CLEF: 2 msgs",
      (int(t[6]["clef_msgs"]), int(t[6]["clef_rounds"]), int(t[8]["clef_msgs"])) == (4, 2, 2))
s = {int(r["case"]): r for r in csv.DictReader(open("results_structural.csv"))}
claim("Scaled: Case 0 rounds 9,999 -> 3; Case 6 375,000/999 -> 108,975/3; Case 8 0 -> 17,250",
      (s[0]["naive_rounds"], s[0]["clef_rounds"], s[6]["naive_msgs"], s[6]["naive_rounds"],
       s[6]["clef_msgs"], s[6]["clef_rounds"], s[8]["naive_msgs"], s[8]["clef_msgs"])
      == ("9999", "3", "375000", "999", "108975", "3", "0", "17250"))
claim("Scaled: Cases 3, 5, 7 unchanged", all(s[c]["naive_msgs"] == s[c]["clef_msgs"] for c in (3, 5, 7)))
w = [r for r in csv.DictReader(open("results_workloads.csv")) if r["workload"] == "NFT"]
ch = [100 * (int(r["clef_msgs"]) - int(r["naive_msgs"])) / int(r["naive_msgs"]) for r in w]
claim(f"NFT per-block range -25%..+35%  [got {min(ch):.1f}..{max(ch):.1f}]", round(min(ch)) == -25 and round(max(ch)) == 35)
_, _, _, _, sc = algorithm_B(generate_block(10_000, "DEX_BURSTY", 1))
claim(f"DEX_BURSTY pool eta ~0.93  [got {max(sc.values()):.3f}]", abs(max(sc.values()) - 0.93) < 0.01)
print(f"\n{sum(ok)}/{len(ok)} hard-coded claims verified")
