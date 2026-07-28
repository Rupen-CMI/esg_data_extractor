"""
Run mimo-v2.5-free and north-mini-code-free on the expanded 20-market set.
Deepseek already completed; we merge results at the end.
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zen_client import extract_companies
import run_benchmark as rb

HERE = os.path.dirname(os.path.abspath(__file__))

snippets = json.load(open(os.path.join(HERE, "snippets_v2.json"), encoding="utf-8"))
gt       = json.load(open(os.path.join(HERE, "ground_truth_v2.json"), encoding="utf-8"))
markets  = [m for m in gt if not m.startswith("_")]
total_gold = sum(len(v) for k, v in gt.items() if not k.startswith("_"))

print(f"Markets: {len(markets)}, Gold total: {total_gold}\n", flush=True)

MODELS = ["mimo-v2.5-free", "north-mini-code-free"]
all_results = []

for model in MODELS:
    print(f"\n{'='*60}", flush=True)
    print(f"Model: {model}", flush=True)
    print('='*60, flush=True)
    per_market = {}
    for mk in markets:
        ctx = snippets.get(mk, "")
        res = extract_companies(mk, ctx, model=model, max_tokens=8000, timeout=300)
        sc  = rb.score_market(res["companies"], gt[mk], ctx)
        per_market[mk] = {**res, **sc}
        status = "ok" if res["ok"] else "ERR"
        print(f"  [{status}] {mk[:40]:40} tp={sc['tp']:2} fn={sc['fn']:2} fp={sc['fp']:2} "
              f"n={len(res['companies']):2} lat={res['latency_s']}s", flush=True)
        time.sleep(1)

    TP = sum(m["tp"] for m in per_market.values())
    FP = sum(m["fp"] for m in per_market.values())
    FN = sum(m["fn"] for m in per_market.values())
    halluc = sum(len(m["hallucinations"]) for m in per_market.values())
    errors = sum(1 for m in per_market.values() if not m["ok"])
    lat = sum(m["latency_s"] for m in per_market.values()) / len(per_market)
    tok = sum(m["usage"].get("total_tokens", 0) for m in per_market.values())
    P  = round(TP/(TP+FP), 3) if TP+FP else 0
    R  = round(TP/(TP+FN), 3) if TP+FN else 0
    F1 = round(2*P*R/(P+R), 3) if P+R else 0
    all_results.append(dict(model=model, TP=TP, FP=FP, FN=FN, precision=P, recall=R, f1=F1,
                            hallucinations=halluc, errors=errors, avg_latency_s=round(lat,2),
                            total_tokens=tok, per_market=per_market))
    print(f"  -> P={P}  R={R}  F1={F1}  halluc={halluc}  errors={errors}/{len(per_market)}", flush=True)

json.dump(all_results, open(os.path.join(HERE, "results_v2b.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print(f"\nSaved results_v2b.json ({len(all_results)} models)", flush=True)
