from clef_core import *
import random

print("1) Reproduce the earlier n=200, seed=42 benchmark")
expected = {"P2P": (0, 0), "DEX_AVG": (112, 25), "DEX_BURSTY": (144, 8),
            "NFT": (1, 4), "MIXED": (70, 23)}
for w in WORKLOADS:
    b = generate_block(200, w, 42)
    got = (metrics(b)["msgs"], metrics(order_shard_grouped(b))["msgs"])
    print(f"   {w:<11} naive/shard-grouped = {got}  expected {expected[w]}  "
          f"{'OK' if got == expected[w] else 'MISMATCH'}")

print("\n2) Toy cases 0-8 (naive order)")
for c in range(9):
    m = metrics(toy_case(c))
    print(f"   case {c}: msgs={m['msgs']} rounds={m['rounds']} frontier={m['frontier']}")

print("\n3) Hypothesis P1: any conflict-preserving reorder leaves msgs/rounds/frontier unchanged")
bad = 0
for trial in range(300):
    w = random.Random(trial).choice(WORKLOADS)
    b = generate_block(random.Random(trial).randint(20, 400), w, trial)
    a = algorithm_C(b)
    assert is_linear_extension(b, a)
    m0, m1 = metrics(b), metrics(a)
    if (m0["msgs"], m0["rounds"], m0["frontier"]) != (m1["msgs"], m1["rounds"], m1["frontier"]):
        bad += 1
for c in range(9):
    b = toy_case(c); a = algorithm_C(b)
    if metrics(b)["msgs"] != metrics(a)["msgs"] or metrics(b)["rounds"] != metrics(a)["rounds"]:
        bad += 1
print(f"   counterexamples in 309 blocks: {bad}")

print("\n4) Hypothesis P2: msgs >= RMW lower bound for every order; shard-grouped attains it on DEX")
viol = tight = total_dex = 0
for trial in range(200):
    w = random.Random(trial).choice(WORKLOADS)
    b = generate_block(random.Random(trial + 7).randint(20, 400), w, trial)
    lb, lbr = rmw_lower_bound(b)
    for o in (b, order_shard_grouped(b), algorithm_C(b),
              random.Random(trial).sample(b, len(b))):
        m = metrics(o)
        if m["msgs"] < lb or m["rounds"] < lbr:
            viol += 1
    if w.startswith("DEX"):
        total_dex += 1
        tight += metrics(order_shard_grouped(b))["msgs"] == lb
print(f"   lower-bound violations: {viol};  shard-grouped == LB on {tight}/{total_dex} DEX blocks")

print("\n5) Hypothesis P3: shard-grouped rounds <= k-1 for every block")
worst = max(metrics(order_shard_grouped(generate_block(300, w, s)))["rounds"]
            for w in WORKLOADS for s in range(20))
worst_toy = max(metrics(order_shard_grouped(scaled_case(c, 300)))["rounds"] for c in range(9))
print(f"   max rounds observed: workloads={worst}, structural={worst_toy}  (k-1={NUM_SHARDS-1})")

print("\n6) Scaled generators reproduce toy cases; Case 6 closed form")
for c in (0, 1, 6, 8):
    t, s = toy_case(c), scaled_case(c, len(toy_case(c)))
    same = (metrics(t)["msgs"], metrics(t)["rounds"]) == (metrics(s)["msgs"], metrics(s)["rounds"])
    print(f"   case {c}: {'OK' if same else 'DIFF'}")
for n in (5, 50, 300):
    sim = metrics(scaled_case(6, n)); cf = case6_closed_form(n)
    print(f"   case 6 n={n}: sim=({sim['msgs']},{sim['rounds']}) closed={cf}")
