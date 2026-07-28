"""
Benchmark every free Zen model on the company-identification task.

For each (model, market) the model extracts companies from the SAME cached
snippets. Predictions are scored against benchmark/ground_truth.json using
alias-aware fuzzy matching, yielding precision / recall / F1, plus latency,
token cost, JSON-validity, and hallucination rate.

Run:  venv/Scripts/python.exe benchmark/run_benchmark.py
"""

import os
import re
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zen_client import extract_companies, FREE_MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
SNIPPETS = os.path.join(HERE, "snippets.json")
GROUND_TRUTH = os.path.join(HERE, "ground_truth.json")
RESULTS = os.path.join(HERE, "results.json")

# Suffixes / noise tokens stripped during normalization so that
# "PPG Industries, Inc." and "PPG Industries" compare equal.
_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "co",
    "llc", "plc", "nv", "se", "gmbh", "kgaa", "ag", "holdings", "holding",
    "company", "companies", "group", "the", "sa", "spa", "as", "llp",
}


def norm(s: str) -> str:
    """Lowercase, drop punctuation and corporate suffixes -> compact token string."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def joined(s: str) -> str:
    return norm(s).replace(" ", "")


def matches_entity(pred: str, entity: dict) -> bool:
    pj = joined(pred)
    if not pj:
        return False
    for alias in entity["aliases"]:
        aj = joined(alias)
        if not aj:
            continue
        if pj == aj:
            return True
        # substring match, guarded by length to avoid spurious hits
        if len(aj) >= 4 and (aj in pj or pj in aj):
            return True
    return False


def score_market(preds: list[str], entities: list[dict], snippet_text: str) -> dict:
    matched_entities = set()
    fps = []

    # A prediction that matches ANY gold entity is a "hit" (it found that
    # company). Multiple predictions mapping to the same entity are duplicates,
    # NOT false positives. Only predictions matching no entity count as FP.
    for pred in preds:
        hit = None
        for ei, ent in enumerate(entities):
            if matches_entity(pred, ent):
                hit = ei
                break
        if hit is None:
            fps.append(pred)
        else:
            matched_entities.add(hit)

    tp = len(matched_entities)
    fn = len(entities) - tp
    fp = len(fps)

    # Hallucination = a predicted name whose core does not appear in the
    # snippet text at all (fabricated, not just mis-categorized).
    snip_join = joined(snippet_text)
    hallucinations = [p for p in fps if joined(p) and joined(p) not in snip_join]

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "false_positives": fps,
        "hallucinations": hallucinations,
    }


def run_model(model: str, snippets: dict, gt: dict) -> dict:
    per_market = {}

    def work(market):
        ctx = snippets.get(market, "")
        res = extract_companies(market, ctx, model=model, max_tokens=8000, timeout=300)
        return market, res

    markets = [m for m in gt if not m.startswith("_")]
    with ThreadPoolExecutor(max_workers=3) as ex:
        for market, res in ex.map(work, markets):
            sc = score_market(res["companies"], gt[market], snippets.get(market, ""))
            per_market[market] = {**res, **sc}

    # Aggregate (micro-averaged across markets).
    TP = sum(m["tp"] for m in per_market.values())
    FP = sum(m["fp"] for m in per_market.values())
    FN = sum(m["fn"] for m in per_market.values())
    halluc = sum(len(m["hallucinations"]) for m in per_market.values())
    n_calls = len(per_market)
    json_ok = sum(1 for m in per_market.values() if m["ok"] and m["companies"])
    errors = sum(1 for m in per_market.values() if not m["ok"])
    latency = sum(m["latency_s"] for m in per_market.values())
    total_tokens = sum(m["usage"].get("total_tokens", 0) for m in per_market.values())

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "model": model,
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "hallucinations": halluc,
        "json_ok_calls": json_ok,
        "errors": errors,
        "n_calls": n_calls,
        "avg_latency_s": round(latency / n_calls, 2) if n_calls else 0,
        "total_tokens": total_tokens,
        "per_market": per_market,
    }


def main():
    with open(SNIPPETS, encoding="utf-8") as f:
        snippets = json.load(f)
    with open(GROUND_TRUTH, encoding="utf-8") as f:
        gt = json.load(f)

    total_gold = sum(len(v) for k, v in gt.items() if not k.startswith("_"))
    print(f"Gold entities total: {total_gold}\n")

    summaries = []
    for model in FREE_MODELS:
        print(f"=== {model} ===")
        t0 = time.perf_counter()
        summary = run_model(model, snippets, gt)
        summaries.append(summary)
        print(f"  P={summary['precision']}  R={summary['recall']}  F1={summary['f1']}  "
              f"halluc={summary['hallucinations']}  errors={summary['errors']}  "
              f"({time.perf_counter()-t0:.0f}s)\n")

    summaries.sort(key=lambda s: (s["f1"], s["recall"]), reverse=True)

    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump({"total_gold": total_gold, "models": summaries}, f, indent=2,
                  ensure_ascii=False)

    # Print a markdown table to stdout.
    print("\n## Results (ranked by F1)\n")
    hdr = ("| Rank | Model | Precision | Recall | F1 | Halluc. | JSON ok | Errors "
           "| Avg latency | Tokens |")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(summaries, 1):
        print(f"| {i} | {s['model']} | {s['precision']} | {s['recall']} | {s['f1']} "
              f"| {s['hallucinations']} | {s['json_ok_calls']}/{s['n_calls']} "
              f"| {s['errors']} | {s['avg_latency_s']}s | {s['total_tokens']} |")
    print(f"\nFull per-market detail saved to {RESULTS}")


if __name__ == "__main__":
    main()
