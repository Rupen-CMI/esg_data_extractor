"""
tune_saturation.py — offline (A,k) grid search for the v5 saturation formula.

WHY OFFLINE: (A,k) only affect saturation_score.py's final aggregation, NOT
evidence gathering. So we run the expensive pipeline ONCE per company (dumping
each pillar's baseline + Contribution list + ground truth to disk), then sweep
dozens of (A,k) combos in seconds by re-running only the saturation arithmetic
over the cached contributions.

DISCIPLINE (anti-overfit): tune on one seed, validate on a DIFFERENT held-out
seed. A grid that picks the best number on the same sample it scores against
memorizes that sample; the held-out check is the only thing that proves the
params generalize.

Usage:
    # 1. dump contributions for a sample (expensive, ~20min for n=30):
    python -m calibration.tune_saturation dump --seed 42 --n 30 --out calibration/sat_contribs_seed42.json
    python -m calibration.tune_saturation dump --seed 101 --n 30 --out calibration/sat_contribs_seed101.json

    # 2. sweep the grid offline (instant), tune on seed42, validate on seed101:
    python -m calibration.tune_saturation sweep --tune calibration/sat_contribs_seed42.json --holdout calibration/sat_contribs_seed101.json
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


# ── Spearman (pure Python, no scipy) ──────────────────────────────────────────

def _spearman(x, y):
    n = len(x)
    if n < 2:
        return None
    def rank(vals):
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[idx[k]] = avg
            i = j + 1
        return ranks
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((r - mx) ** 2 for r in rx) * sum((r - my) ** 2 for r in ry)) ** 0.5
    return num / den if den else None


# ── DUMP: run the pipeline once, serialize contributions + baseline + truth ───

def _contrib_to_dict(c) -> dict:
    return {"factor": c.factor, "weight": c.weight, "confidence": c.confidence,
            "delta": c.delta, "method": c.method}


def _dump_one(truth, reg_wsum) -> Optional[dict]:
    """Gather + score ONE company for the dump: formula contributions PLUS the
    holistic LLM vote per pillar (so the offline sweep can reproduce the full
    reconciled ensemble score, not just formula-alone). Returns None on failure."""
    from agentic_estimation import calibration_harness as ch
    from agentic_estimation.layer_3.holistic_estimator import holistic_vote

    row = ch.BacktestRow(name=truth.name, country=truth.country, industry=truth.industry)
    try:
        signals, metadata, country, _claims, formula_scores = ch._gather_and_score_formula(truth, row)
        holistic = holistic_vote(truth.name, truth.industry or "", country, signals, metadata)
    except Exception as exc:
        print(f"  {truth.name}: FAILED ({type(exc).__name__}: {exc})")
        return None

    def _hv(p):
        return getattr(holistic, f"{p.lower()}_score", None) if holistic is not None else None

    rec = {
        "name": truth.name,
        "truth": {"E": truth.truth_e, "S": truth.truth_s, "G": truth.truth_g},
        "holistic": {p: _hv(p) for p in ("E", "S", "G")},
        "pillars": {},
    }
    for p in ("E", "S", "G"):
        pfs = formula_scores.get(p)
        if pfs is None:
            continue
        rec["pillars"][p] = {
            "baseline": pfs.baseline,
            "registry_weight_sum": reg_wsum[p],
            "contributions": [_contrib_to_dict(c) for c in pfs.contributions],
        }
    n_contribs = sum(len(rec["pillars"].get(p, {}).get("contributions", [])) for p in ("E", "S", "G"))
    print(f"  {truth.name}: dumped ({n_contribs} contribs, holistic={'yes' if holistic else 'none'})")
    return rec


def dump(seed: int, n: int, out_path: Path, workers: int = 6) -> None:
    """Run the full ensemble gather+score for a sample IN PARALLEL; dump each
    company's per-pillar baseline + contributions + holistic vote + ground truth.
    Reuses the harness's sampling + gathering so the sample matches a real
    ensemble backtest exactly, letting the offline sweep tune (A,k) against the
    real reconciled score. The process-wide rate limiters in zen_client keep
    `workers` concurrency safe (same as the harness's own worker pool)."""
    import concurrent.futures
    from agentic_estimation import calibration_harness as ch
    from agentic_estimation.layer_3.formula_estimator import _registry_weight_sum

    ch._USE_SATURATION = False   # raw contributions; sweep redoes the reduction offline
    truths = ch.load_bcorp_truth(n=n, seed=seed)
    reg_wsum = {p: _registry_weight_sum(p) for p in ("E", "S", "G")}
    print(f"dumping {len(truths)} companies (seed={seed}, workers={workers})...")

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_dump_one, t, reg_wsum) for t in truths]
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records -> {out_path}")


# ── SWEEP: re-run saturation + RECONCILE offline over dumped records ──────────
# CRITICAL: (A,k) must be tuned against the RECONCILED ensemble score (formula
# blended with the holistic vote), NOT the formula-alone score -- that is what
# production actually scores against ground truth. Tuning formula-alone rank
# optimizes the wrong target. So each candidate (A,k) -> saturation formula
# score -> reconcile with the stored holistic vote -> compare to truth.

def _formula_sat_pillar(pillar_rec: dict, pillar: str, a: float, k: float):
    """Return (score, breakdown) from saturate_pillar at (a,k), or None."""
    from agentic_estimation.layer_3.saturation_score import saturate_pillar, PillarSatParams
    contribs = [SimpleNamespace(**c, points=0.0, claim_reasoning="") for c in pillar_rec["contributions"]]
    bd = saturate_pillar(
        pillar=pillar, baseline=pillar_rec["baseline"], contributions=contribs,
        registry_weight_sum=pillar_rec["registry_weight_sum"],
        params=PillarSatParams(a_pos=a, a_neg=a, k=k),
    )
    return bd


def _reconciled_pillar_score(rec: dict, pillar: str, a: float, k: float) -> Optional[float]:
    """Full offline reproduction of what production scores: saturation formula
    score for this (a,k), reconciled with the dumped holistic vote via the SAME
    reconcile_pillar (so _PILLAR_WEIGHTS + the v8 M_trust confidence are exact)."""
    from agentic_estimation.layer_3.formula_estimator import PillarFormulaScore, Contribution
    from agentic_estimation.layer_3.reconcile import reconcile_pillar

    pr = rec["pillars"].get(pillar)
    if pr is None:
        return None
    bd = _formula_sat_pillar(pr, pillar, a, k)

    # Rebuild a real PillarFormulaScore so reconcile's _formula_confidence (v8
    # M_trust over .contributions) sees the true weight/confidence/method of each.
    contribs = [Contribution(factor=c["factor"], weight=c["weight"], confidence=c["confidence"],
                             delta=c["delta"], points=0.0, claim_reasoning="", method=c["method"])
                for c in pr["contributions"]]
    pfs = PillarFormulaScore(pillar=pillar, baseline=pr["baseline"], baseline_source="exact",
                             score=bd.score, contributions=contribs)
    holistic_val = (rec.get("holistic") or {}).get(pillar)
    return reconcile_pillar(pillar, pfs, holistic_val).score


def _pillar_spearman(records: list, pillar: str, a: float, k: float) -> Optional[float]:
    preds, truths = [], []
    for rec in records:
        tv = rec["truth"].get(pillar)
        if tv is None:
            continue
        s = _reconciled_pillar_score(rec, pillar, a, k)
        if s is None:
            continue
        preds.append(s)
        truths.append(float(tv))
    return _spearman(preds, truths)


def sweep(tune_path: Path, holdout_path: Path) -> None:
    tune = json.loads(tune_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    print(f"tune sample: {len(tune)} companies ({tune_path.name})")
    print(f"holdout sample: {len(holdout)} companies ({holdout_path.name})\n")

    A_GRID = [25, 30, 35, 40, 45, 50]
    K_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    DEFAULT = (40.0, 1.0)

    best = {}
    for pillar in ("E", "S", "G"):
        results = []
        for a in A_GRID:
            for k in K_GRID:
                rho = _pillar_spearman(tune, pillar, a, k)
                if rho is not None:
                    results.append((rho, a, k))
        results.sort(reverse=True)
        best_rho, best_a, best_k = results[0]
        default_rho = _pillar_spearman(tune, pillar, *DEFAULT)
        best[pillar] = (best_a, best_k)
        print(f"=== {pillar} (tune seed) ===")
        print(f"  default (A=40,k=1.0): rho={default_rho:+.3f}")
        print(f"  BEST    (A={best_a},k={best_k}): rho={best_rho:+.3f}")
        print(f"  top 5: " + ", ".join(f"A{a}/k{k}={r:+.3f}" for r, a, k in results[:5]))
        print()

    print("=" * 60)
    print("HELD-OUT VALIDATION (seed101) — tuned vs default:")
    print("=" * 60)
    for pillar in ("E", "S", "G"):
        a, k = best[pillar]
        tuned_ho = _pillar_spearman(holdout, pillar, a, k)
        default_ho = _pillar_spearman(holdout, pillar, *DEFAULT)
        verdict = "KEEP TUNED" if (tuned_ho or -9) > (default_ho or -9) else "KEEP DEFAULT (tuned regressed)"
        print(f"  {pillar}: tuned(A={a},k={k})={tuned_ho:+.3f}  vs  default(40,1.0)={default_ho:+.3f}  -> {verdict}")


def _cli() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    import argparse
    if cmd == "dump":
        ap = argparse.ArgumentParser()
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--n", type=int, default=30)
        ap.add_argument("--out", required=True)
        args = ap.parse_args(sys.argv[2:])
        dump(args.seed, args.n, Path(args.out))
    elif cmd == "sweep":
        ap = argparse.ArgumentParser()
        ap.add_argument("--tune", required=True)
        ap.add_argument("--holdout", required=True)
        args = ap.parse_args(sys.argv[2:])
        sweep(Path(args.tune), Path(args.holdout))
    else:
        print(f"unknown command {cmd!r}"); sys.exit(1)


if __name__ == "__main__":
    _cli()
