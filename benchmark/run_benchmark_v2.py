"""
Expanded benchmark: 3 usable free Zen models x 20 markets (4 countries x 5 industries).
Models are called sequentially per market (no concurrent hammering of the free tier).
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zen_client import extract_companies, USABLE_FREE_MODELS
import run_benchmark as rb

HERE = os.path.dirname(os.path.abspath(__file__))

snippets = json.load(open(os.path.join(HERE, "snippets_v2.json"), encoding="utf-8"))
gt = json.load(open(os.path.join(HERE, "ground_truth_v2.json"), encoding="utf-8"))
markets = [m for m in gt if not m.startswith("_")]
total_gold = sum(len(v) for m, v in gt.items() if not m.startswith("_"))

print(f"Markets: {len(markets)}, Gold entities total: {total_gold}\n")

all_results = []

for model in USABLE_FREE_MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print('='*60)
    per_market = {}
    for mk in markets:
        ctx = snippets.get(mk, "")
        res = extract_companies(mk, ctx, model=model, max_tokens=8000, timeout=300)
        sc = rb.score_market(res["companies"], gt[mk], ctx)
        per_market[mk] = {**res, **sc}
        status = "ok" if res["ok"] else "ERR"
        print(f"  [{status}] {mk[:40]:40} tp={sc['tp']:2} fn={sc['fn']:2} fp={sc['fp']:2} "
              f"n={len(res['companies']):2} lat={res['latency_s']}s", flush=True)
        time.sleep(1)  # polite gap between calls

    TP = sum(m["tp"] for m in per_market.values())
    FP = sum(m["fp"] for m in per_market.values())
    FN = sum(m["fn"] for m in per_market.values())
    halluc = sum(len(m["hallucinations"]) for m in per_market.values())
    errors = sum(1 for m in per_market.values() if not m["ok"])
    lat = sum(m["latency_s"] for m in per_market.values()) / len(per_market)
    tok = sum(m["usage"].get("total_tokens", 0) for m in per_market.values())
    P = round(TP / (TP + FP), 3) if TP + FP else 0
    R = round(TP / (TP + FN), 3) if TP + FN else 0
    F1 = round(2*P*R/(P+R), 3) if P+R else 0

    summary = dict(model=model, TP=TP, FP=FP, FN=FN, precision=P, recall=R, f1=F1,
                   hallucinations=halluc, errors=errors, avg_latency_s=round(lat, 2),
                   total_tokens=tok, per_market=per_market)
    all_results.append(summary)
    print(f"  -> P={P}  R={R}  F1={F1}  halluc={halluc}  errors={errors}/{len(per_market)}", flush=True)

all_results.sort(key=lambda s: (s["f1"], s["recall"]), reverse=True)
json.dump({"total_gold": total_gold, "n_markets": len(markets), "models": all_results},
          open(os.path.join(HERE, "results_v2.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

n = len(per_market)
print("\n\n## Final Results (ranked by F1)\n")
print("| Rank | Model | Precision | Recall | F1 | Halluc | Errors | Avg latency | Total tokens |")
print("|---|---|---|---|---|---|---|---|---|")
for i, s in enumerate(all_results, 1):
    print(f"| {i} | {s['model']} | {s['precision']} | {s['recall']} | {s['f1']} "
          f"| {s['hallucinations']} | {s['errors']}/{n} | {s['avg_latency_s']}s | {s['total_tokens']:,} |")

print("\n\n## Per-country breakdown\n")
countries = ["United States", "Germany", "India", "Japan"]
for s in all_results:
    print(f"\n### {s['model']}")
    print("| Country | TP | FP | FN | P | R | F1 |")
    print("|---|---|---|---|---|---|---|")
    for country in countries:
        cmarkets = [m for m in s["per_market"] if country in m]
        TP = sum(s["per_market"][m]["tp"] for m in cmarkets)
        FP = sum(s["per_market"][m]["fp"] for m in cmarkets)
        FN = sum(s["per_market"][m]["fn"] for m in cmarkets)
        P = round(TP/(TP+FP), 3) if TP+FP else 0
        R = round(TP/(TP+FN), 3) if TP+FN else 0
        F1 = round(2*P*R/(P+R), 3) if P+R else 0
        print(f"| {country} | {TP} | {FP} | {FN} | {P} | {R} | {F1} |")

print("\n\n## Per-industry breakdown\n")
industries = ["Pharmaceutical Market", "Automotive Tire Market", "Solar Panel Market", "Steel Market", "Semiconductor Market"]
for s in all_results:
    print(f"\n### {s['model']}")
    print("| Industry | TP | FP | FN | P | R | F1 |")
    print("|---|---|---|---|---|---|---|")
    for ind in industries:
        imarkets = [m for m in s["per_market"] if m.startswith(ind)]
        TP = sum(s["per_market"][m]["tp"] for m in imarkets)
        FP = sum(s["per_market"][m]["fp"] for m in imarkets)
        FN = sum(s["per_market"][m]["fn"] for m in imarkets)
        P = round(TP/(TP+FP), 3) if TP+FP else 0
        R = round(TP/(TP+FN), 3) if TP+FN else 0
        F1 = round(2*P*R/(P+R), 3) if P+R else 0
        print(f"| {ind} | {TP} | {FP} | {FN} | {P} | {R} | {F1} |")
