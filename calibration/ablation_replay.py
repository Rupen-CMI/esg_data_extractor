"""
ablation_replay.py — freeze evidence once, replay through pipeline variants
offline, to answer two questions plans/DEFECT_FIX_PLAN.md-adjacent work left
open:

  1. Which layer (Tier-0 validation, holistic LLM vote, peer anchor, evidence
     freshness) contributes most to final-score error?
  2. Is the current ensemble route (formula 0.7 / holistic 0.3, reconcile v8)
     the best of the candidate routes, or would a different blend do better?

WHY OFFLINE REPLAY: a fresh gather carries +/-0.10-0.17 Spearman/pillar noise
at n=30 (LLM call variance, live signal-fetch timing, live peer-table state)
-- far larger than any single layer's real effect. Freezing the evidence once
(dump) and replaying deterministic variants against the SAME frozen evidence
(run) removes that noise from the comparison; only the variant's own logic
differs between runs.

This mirrors tune_saturation.py's dump/sweep split (same rationale, same
pure-python-no-scipy house style) but captures MORE of the pipeline: raw
pre-Tier-0 claims, signals, metadata, and the peer-anchor vote, so knock-outs
that tune_saturation's dump can't replay (no_tier0, no_freshness, and any
variant touching peer anchor) become possible here.

Usage:
    # 1. dump (expensive, ~4 LLM calls/company, no DB writes):
    python -m calibration.ablation_replay dump --seed 777 --n 100 --out calibration/abl_seed777_n100.json

    # 1b. or a large stratified corpus (PHASE_5_PLAN.md §0) -- even thirds
    #     well_known/medium/obscure per truth source, checkpointed/resumable:
    python -m calibration.ablation_replay dump-stratified --seed 4001 --n-bcorp 350 --n-upright 150 \\
        --out calibration/abl_seed4001_tune500.json --skip-newsapi

    # 2. replay the variant matrix (free, offline, seconds):
    python -m calibration.ablation_replay run --dump calibration/abl_seed777_n100.json --variants all
"""

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


def _p(msg: str) -> None:
    """Print, degrading to '?'-replaced ASCII on Windows consoles whose
    default codepage (cp1252) can't encode a company name's Unicode
    characters -- found live: a name containing U+014D crashed a 40-company
    background dump on its LAST company (after ~35 min of real gathering),
    discarding the entire completed gather because out_path.write_text()
    never got to run. Never let a display encoding problem lose real work."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


# ── Spearman + percentile-MAE (pure Python, no scipy/pandas — matches
#    tune_saturation.py's house style) ─────────────────────────────────────

def _spearman(preds: list, truths: list) -> Optional[float]:
    n = len(preds)
    if n < 3 or len(set(preds)) < 2 or len(set(truths)) < 2:
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

    rx, ry = rank(preds), rank(truths)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((r - mx) ** 2 for r in rx) * sum((r - my) ** 2 for r in ry)) ** 0.5
    return num / den if den else None


def _percentile_ranks(vals: list) -> list:
    """Within-sample percentile rank (0-100), average rank on ties -- same
    convention as pandas' `.rank(pct=True) * 100` used elsewhere in the harness."""
    n = len(vals)
    idx = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1   # 1-based average rank
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank / n * 100.0
        i = j + 1
    return ranks


def _per_company_errors(preds: list, truths: list) -> list:
    """|percentile_rank(pred) - percentile_rank(truth)| per company, within
    this sample -- the same error definition as calibration_harness's
    confidence-label calibration block, so route/variant errors are
    comparable to numbers already reported elsewhere."""
    if len(preds) < 3:
        return [None] * len(preds)
    pp = _percentile_ranks(preds)
    pt = _percentile_ranks(truths)
    return [abs(a - b) for a, b in zip(pp, pt)]


def _percentile_mae(preds: list, truths: list) -> Optional[float]:
    errs = _per_company_errors(preds, truths)
    errs = [e for e in errs if e is not None]
    return round(sum(errs) / len(errs), 1) if errs else None


# ── Paired bootstrap (percentile bootstrap on the mean, fixed seed — no scipy) ─

def _bootstrap_ci(deltas: list, seed: int = 12345, n_resamples: int = 10_000) -> tuple:
    """95% percentile-bootstrap CI on the MEAN of paired per-company error
    deltas: resample deltas WITH REPLACEMENT n_resamples times, take the
    2.5/97.5 percentiles of the resample means. This is a CI on the true
    effect size (unlike a sign-flip null distribution, which is centered at
    zero by construction and would make "does the CI exclude zero" always
    trivially about noise magnitude, not about the observed effect). A
    single company (n=1) has no resampling variance -- CI collapses to a
    point at that one delta. Returns (lo, hi, mean)."""
    if not deltas:
        return None, None, None
    rng = random.Random(seed)
    n = len(deltas)
    mean = sum(deltas) / n
    if n == 1:
        return deltas[0], deltas[0], mean
    resample_means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        resample_means.append(s / n)
    resample_means.sort()
    lo = resample_means[int(0.025 * n_resamples)]
    hi = resample_means[int(0.975 * n_resamples) - 1]
    return lo, hi, mean


def _verdict(lo: Optional[float], hi: Optional[float]) -> str:
    """Delta = variant_error - base_error, per company. Negative delta =
    variant did BETTER (lower error) than base."""
    if lo is None or hi is None:
        return "UNDECIDABLE"
    if hi < 0:
        return "BETTER"
    if lo > 0:
        return "WORSE"
    return "UNDECIDABLE"


# ── DUMP v2 ─────────────────────────────────────────────────────────────

def _claim_to_dict(c) -> dict:
    return {"factor": c.factor, "pillar": c.pillar, "polarity": c.polarity,
            "strength": c.strength, "confidence": c.confidence, "value": c.value,
            "source_tag": c.source_tag, "reasoning": c.reasoning, "method": c.method}


def _dict_to_claim(d: dict):
    from agentic_estimation.shared.claim_types import ExtractedClaim
    return ExtractedClaim(factor=d["factor"], pillar=d["pillar"], polarity=d["polarity"],
                           strength=d["strength"], confidence=d["confidence"], value=d["value"],
                           source_tag=d["source_tag"], reasoning=d["reasoning"], method=d["method"])


def _flag_to_dict(f) -> dict:
    return {"factor": f.factor, "pillar": f.pillar, "rule": f.rule, "action": f.action, "detail": f.detail}


def _anchor_to_dict(a) -> Optional[dict]:
    if a is None:
        return None
    return {"pillar": a.pillar, "percentile": a.percentile, "confidence": a.confidence,
            "n_peers": a.n_peers, "tier": a.tier, "basis": a.basis}


def _dict_to_anchor(d: Optional[dict]):
    if d is None:
        return None
    from agentic_estimation.layer_3.peer_anchor import PeerAnchorVote
    return PeerAnchorVote(pillar=d["pillar"], percentile=d["percentile"], confidence=d["confidence"],
                           n_peers=d["n_peers"], tier=d["tier"], basis=d["basis"])


def _dump_one(truth, reg_wsum: dict) -> Optional[dict]:
    """Gather + score ONE company, capturing everything a variant might need
    to replay: signals, metadata, raw pre-Tier-0 claims, Tier-0's flags and
    kept claims, the frozen peer-anchor vote per pillar, and the holistic
    LLM vote. Returns None on failure (mirrors tune_saturation._dump_one)."""
    from agentic_estimation import calibration_harness as ch
    from agentic_estimation.layer_3.holistic_estimator import holistic_vote

    row = ch.BacktestRow(name=truth.name, country=truth.country, industry=truth.industry)
    capture: dict = {}
    try:
        signals, metadata, country, _claims, formula_scores = ch._gather_and_score_formula(
            truth, row, capture=capture)
        holistic = holistic_vote(truth.name, truth.industry or "", country, signals, metadata)
    except Exception as exc:
        _p(f"  {truth.name}: FAILED ({type(exc).__name__}: {exc})")
        return None

    def _hv(p):
        return getattr(holistic, f"{p.lower()}_score", None) if holistic is not None else None

    rec = {
        "name": truth.name,
        "truth": {"E": truth.truth_e, "S": truth.truth_s, "G": truth.truth_g},
        "country": country,
        "metadata": metadata or {},
        "signals": dict(signals or {}),
        "raw_claims": [_claim_to_dict(c) for c in capture.get("raw_claims", [])],
        "kept_claims": [_claim_to_dict(c) for c in capture.get("kept_claims", [])],
        "flags": [_flag_to_dict(f) for f in capture.get("flags", [])],
        "holistic": {p: _hv(p) for p in ("E", "S", "G")},
        "pillars": {},
    }
    for p in ("E", "S", "G"):
        pfs = formula_scores.get(p)
        if pfs is None:
            continue
        rec["pillars"][p] = {
            "baseline": pfs.baseline,
            "baseline_source": pfs.baseline_source,
            "registry_weight_sum": reg_wsum[p],
            "contributions": [
                {"factor": c.factor, "weight": c.weight, "confidence": c.confidence,
                 "delta": c.delta, "method": c.method}
                for c in pfs.contributions
            ],
            "peer_anchor": _anchor_to_dict(pfs.peer_anchor),
        }
    n_contribs = sum(len(rec["pillars"].get(p, {}).get("contributions", [])) for p in ("E", "S", "G"))
    _p(f"  {truth.name}: dumped ({len(rec['raw_claims'])} raw claims, "
          f"{n_contribs} contribs, holistic={'yes' if holistic else 'none'})")
    return rec


def dump(seed: int, n: int, out_path: Path, workers: int = 6) -> None:
    import concurrent.futures
    from agentic_estimation import calibration_harness as ch
    from agentic_estimation.layer_3.formula_estimator import _registry_weight_sum

    ch._USE_SATURATION = False   # raw contributions; replay redoes reduction offline
    truths = ch.load_bcorp_truth(n=n, seed=seed)
    reg_wsum = {p: _registry_weight_sum(p) for p in ("E", "S", "G")}
    _p(f"dumping {len(truths)} companies (seed={seed}, workers={workers})...")

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_dump_one, t, reg_wsum) for t in truths]
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _p(f"wrote {len(records)} records -> {out_path}")


# ── Stratified corpus sampling (PHASE_5_PLAN.md §0) ───────────────────────
#
# Visibility tiers exist because the pipeline's error profile differs
# sharply between famous, web-visible companies (rich evidence, formula and
# holistic both have something to work with) and obscure SMEs (thin/no
# evidence, most of the score comes from country baseline + peer anchor).
# Averaging across an unstratified random sample hides that split; even
# thirds per tier gives every stratum equal statistical power, matching the
# per-tier "evidence-conditional routing" analysis the Phase 5 plan calls for.
#
# bcorp uses its own `size` employee-bucket column as the visibility proxy;
# upright uses `revenue_usd`. The two truth sources are tiered SEPARATELY
# (never blended) so bcorp-truth and upright-truth conclusions stay reportable
# on their own, per the standing house rule (EVALUATION_STRATEGIES.md).

_BCORP_WELLKNOWN_SIZES = {"1000+", "250-999"}     # oversampled toward here for G coverage
_BCORP_OBSCURE_SIZES = {"0", "1-9"}
# everything else ("10-49", "50-249") is "medium"


def _load_bcorp_pool(seed: int) -> list:
    """All bcorp rows with usable ground truth (E/S/G/total + size), shuffled
    once by `seed` -- the same fresh-seed discipline as load_bcorp_truth, just
    keeping `size` alongside so the caller can stratify instead of taking a
    flat head-of-list sample."""
    from agentic_estimation.calibration_harness import _db_conn, TruthRecord

    conn = _db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT company_name, country, industry, size,
               overall_score, impact_area_environment, impact_area_governance,
               impact_area_workers, impact_area_community, impact_area_customers
        FROM bcorp_lookup
        WHERE overall_score IS NOT NULL
          AND company_name IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()

    rng = random.Random(seed)
    rng.shuffle(rows)

    out = []
    for (name, country, industry, size, overall, env, gov, workers, community, customers) in rows:
        social_parts = [v for v in (workers, community, customers) if v is not None]
        social = sum(social_parts) / len(social_parts) if social_parts else None
        truth = TruthRecord(name=name, country=country, industry=industry,
                             truth_e=env, truth_s=social, truth_g=gov, truth_total=overall)
        out.append((truth, size or ""))
    return out


def _bcorp_tier(size: str) -> str:
    if size in _BCORP_WELLKNOWN_SIZES:
        return "well_known"
    if size in _BCORP_OBSCURE_SIZES:
        return "obscure"
    return "medium"


def _load_upright_pool(seed: int) -> list:
    """All upright rows with usable ground truth, shuffled once by `seed`,
    keeping revenue_usd alongside for tiering (mirrors load_upright_truth's
    query but returns the full pool instead of head-biased n rows)."""
    from agentic_estimation.calibration_harness import _db_conn, TruthRecord

    conn = _db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name, country, industry, revenue_usd, net_impact_ratio_percentile,
               e1_ghg_positive, e2_non_ghg_positive, e3_scarce_resources_positive,
               e4_biodiversity_positive, e5_waste_positive,
               e1_ghg_negative, e2_non_ghg_negative, e3_scarce_resources_negative,
               e4_biodiversity_negative, e5_waste_negative,
               h1_physical_diseases_positive, h2_mental_diseases_positive,
               h3_nutrition_positive, h4_relationships_positive, h5_meaning_joy_positive,
               s1_jobs_positive, s3_societal_infra_positive, s4_societal_stability_positive,
               s5_equality_positive,
               h1_physical_diseases_negative, h2_mental_diseases_negative,
               h4_relationships_negative, h5_meaning_joy_negative,
               s4_societal_stability_negative, s5_equality_negative
        FROM upright_lookup
        WHERE net_impact_ratio_percentile IS NOT NULL
          AND name IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()

    rng = random.Random(seed)
    rng.shuffle(rows)

    def _net(pos_vals, neg_vals):
        if not any(v is not None for v in pos_vals + neg_vals):
            return None
        return sum(v for v in pos_vals if v is not None) - sum(v for v in neg_vals if v is not None)

    revenues = [r[3] for r in rows if r[3] is not None]
    revenues.sort()

    def _pctile_rank(v):
        if v is None or not revenues:
            return 0.0
        import bisect
        return bisect.bisect_left(revenues, v) / len(revenues)

    out = []
    for r in rows:
        (name, country, industry, rev, pctile,
         e1p, e2p, e3p, e4p, e5p, e1n, e2n, e3n, e4n, e5n,
         h1p, h2p, h3p, h4p, h5p, s1p, s3p, s4p, s5p,
         h1n, h2n, h4n, h5n, s4n, s5n) = r
        e_net = _net([e1p, e2p, e3p, e4p, e5p], [e1n, e2n, e3n, e4n, e5n])
        s_net = _net([h1p, h2p, h3p, h4p, h5p, s1p, s3p, s4p, s5p],
                     [h1n, h2n, h4n, h5n, s4n, s5n])
        from agentic_estimation.calibration_harness import TruthRecord
        truth = TruthRecord(name=name, country=country, industry=industry,
                             truth_e=e_net, truth_s=s_net, truth_g=None, truth_total=pctile)
        out.append((truth, _pctile_rank(rev)))
    return out


def _upright_tier(rev_pctile: float) -> str:
    if rev_pctile >= 2.0 / 3.0:
        return "well_known"
    if rev_pctile < 1.0 / 3.0:
        return "obscure"
    return "medium"


def _stratified_sample(pool: list, tier_fn, n: int) -> list:
    """Split `pool` (list of (truth, tier_key)) into 3 tiers via `tier_fn`,
    draw as close to n/3 from each as available (pool is pre-shuffled by the
    loader, so within-tier order is already randomized -- this just slices).
    A tier short of its quota donates the shortfall to the other two tiers
    (round-robin) rather than silently returning < n."""
    tiers = {"well_known": [], "medium": [], "obscure": []}
    for truth, key in pool:
        tiers[tier_fn(key)].append(truth)

    per_tier = n // 3
    quotas = {"well_known": per_tier, "medium": per_tier, "obscure": n - 2 * per_tier}

    picked: list = []
    shortfall = 0
    for tier_name, quota in quotas.items():
        available = tiers[tier_name]
        take = min(quota, len(available))
        picked.extend(available[:take])
        shortfall += quota - take
        tiers[tier_name] = available[take:]  # remainder, for shortfall backfill

    if shortfall > 0:
        remainder = tiers["well_known"] + tiers["medium"] + tiers["obscure"]
        picked.extend(remainder[:shortfall])
        if len(remainder) < shortfall:
            _p(f"WARNING: stratified sample short by {shortfall - len(remainder)} "
               f"(pool exhausted across all tiers)")

    return picked


def build_stratified_corpus(seed: int, n_bcorp: int, n_upright: int) -> list:
    """The Phase 5 corpus sampler: even thirds (well_known/medium/obscure) per
    truth source, bcorp's well_known tier drawn from its largest size buckets
    first (1000+ before 250-999) so G-pillar decisions -- G only exists in
    bcorp truth -- aren't made purely on tiny SMEs (user decision: oversample
    large B Corps into the well-known tier)."""
    bcorp_pool = _load_bcorp_pool(seed)
    # within well_known, prefer 1000+ over 250-999 (already grouped by
    # _bcorp_tier; re-sort well_known rows so 1000+ rows sort first)
    bcorp_pool.sort(key=lambda t: 0 if t[1] == "1000+" else 1)
    bcorp_sample = _stratified_sample(bcorp_pool, _bcorp_tier, n_bcorp)

    upright_pool = _load_upright_pool(seed)
    upright_sample = _stratified_sample(upright_pool, _upright_tier, n_upright)

    _p(f"stratified corpus: {len(bcorp_sample)} bcorp + {len(upright_sample)} upright "
       f"= {len(bcorp_sample) + len(upright_sample)} companies (seed={seed})")
    return bcorp_sample + upright_sample


# ── Checkpointed dump (survives a crash/kill mid-gather) ──────────────────
#
# A single-shot dump (see `dump()` above) holds every record in memory and
# writes once at the end -- exactly the failure mode that lost a completed
# 40-company gather to a Windows console encoding crash on the LAST company
# (see _p's docstring). At 650 companies and ~5h wall-clock, a checkpointed
# JSONL append -- one line per completed company, flushed immediately -- means
# a crash/kill/network blip loses at most the one in-flight company, and a
# re-invocation with the same --out path skips everything already done.

def _checkpoint_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".checkpoint.jsonl")


def _load_checkpoint(ckpt_path: Path) -> dict:
    """Returns {company_name: record}. Corrupt/partial trailing lines (from a
    kill mid-write) are skipped, not fatal -- the company just gets re-dumped."""
    if not ckpt_path.exists():
        return {}
    done: dict = {}
    for line in ckpt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[rec["name"]] = rec
    return done


def dump_stratified(seed: int, n_bcorp: int, n_upright: int, out_path: Path,
                     workers: int = 6, skip_newsapi: bool = False) -> None:
    """Checkpointed stratified corpus dump (PHASE_5_PLAN.md §0). Resumable:
    re-running with the same --out skips companies already recorded in the
    .checkpoint.jsonl sidecar. Writes the final consolidated JSON array only
    once ALL companies are done (matching dump()'s output format so `run`
    doesn't need to know which dump command produced a file); the checkpoint
    file is the crash-safe intermediate, kept after a successful run so a
    user can inspect it or add more companies later without re-gathering.

    skip_newsapi: unsets NEWS_API_KEY for this process only (restored after),
    so this one gather skips NewsAPI without touching the production default
    (per user: NewsAPI stays on for the real pipeline, "the more the merrier" --
    it's only this corpus that's skipping it as a deliberate one-off)."""
    import concurrent.futures
    import os
    from agentic_estimation import calibration_harness as ch
    from agentic_estimation.layer_3.formula_estimator import _registry_weight_sum

    ch._USE_SATURATION = False
    reg_wsum = {p: _registry_weight_sum(p) for p in ("E", "S", "G")}

    truths = build_stratified_corpus(seed, n_bcorp, n_upright)

    ckpt_path = _checkpoint_path(out_path)
    done = _load_checkpoint(ckpt_path)
    remaining = [t for t in truths if t.name not in done]
    _p(f"{len(done)} already dumped (resuming from {ckpt_path.name}), "
       f"{len(remaining)} remaining (workers={workers})")

    old_key = os.environ.pop("NEWS_API_KEY", None) if skip_newsapi else None
    try:
        if skip_newsapi:
            _p("NEWS_API_KEY unset for this gather (--skip-newsapi)")
        ckpt_file = ckpt_path.open("a", encoding="utf-8")
        try:
            completed = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_dump_one, t, reg_wsum): t for t in remaining}
                for fut in concurrent.futures.as_completed(futures):
                    rec = fut.result()
                    if rec is not None:
                        ckpt_file.write(json.dumps(rec) + "\n")
                        ckpt_file.flush()
                        os.fsync(ckpt_file.fileno())
                        done[rec["name"]] = rec
                    completed += 1
                    _p(f"  progress: {completed}/{len(remaining)} this run "
                       f"({len(done)}/{len(truths)} total)")
        finally:
            ckpt_file.close()
    finally:
        if skip_newsapi and old_key is not None:
            os.environ["NEWS_API_KEY"] = old_key

    records = [done[t.name] for t in truths if t.name in done]
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    _p(f"wrote {len(records)} records -> {out_path} "
       f"({len(truths) - len(records)} failed/missing)")


# ── REPLAY: contributions-level rescore (v1-dump compatible) ─────────────

def _contribs_from_dicts(dicts: list):
    from agentic_estimation.layer_3.formula_estimator import Contribution
    return [Contribution(factor=c["factor"], weight=c["weight"], confidence=c["confidence"],
                          delta=c["delta"], points=0.0, claim_reasoning="", method=c["method"])
            for c in dicts]


def _saturate(pillar_rec: dict, pillar: str, contrib_dicts: list):
    from agentic_estimation.layer_3.saturation_score import saturate_pillar
    contribs = [SimpleNamespace(**c, points=0.0, claim_reasoning="") for c in contrib_dicts]
    return saturate_pillar(pillar=pillar, baseline=pillar_rec["baseline"], contributions=contribs,
                            registry_weight_sum=pillar_rec["registry_weight_sum"])


def _reconcile_score(pillar: str, pillar_rec: dict, contrib_dicts: list, holistic_val: Optional[float],
                      pillar_weights_override: Optional[dict] = None) -> Optional[float]:
    from agentic_estimation.layer_3.formula_estimator import PillarFormulaScore
    from agentic_estimation.layer_3 import reconcile as reconcile_mod

    bd = _saturate(pillar_rec, pillar, contrib_dicts)
    contribs = _contribs_from_dicts(contrib_dicts)
    pfs = PillarFormulaScore(pillar=pillar, baseline=pillar_rec["baseline"],
                              baseline_source=pillar_rec.get("baseline_source", "exact"),
                              score=bd.score, contributions=contribs)

    if pillar_weights_override is None:
        return reconcile_mod.reconcile_pillar(pillar, pfs, holistic_val).score

    original = reconcile_mod._PILLAR_WEIGHTS
    try:
        reconcile_mod._PILLAR_WEIGHTS = {**original, pillar: pillar_weights_override}
        return reconcile_mod.reconcile_pillar(pillar, pfs, holistic_val).score
    finally:
        reconcile_mod._PILLAR_WEIGHTS = original


# ── REPLAY: claims-level rescore (needs v2 dump: raw_claims/signals/metadata) ─

def _rescore_claims_level(rec: dict, pillar: str, *, skip_tier0: bool, freeze_freshness: bool,
                           peer_anchor_from_dump: bool) -> Optional[dict]:
    """Re-run validate_claims (optionally skipped) -> compute_formula_scores
    (peer anchor frozen from the dump, DB-free) for ONE pillar. Returns a
    pillar-shaped dict compatible with `_reconcile_score`'s contrib_dicts arg,
    or None if the dump has no raw_claims (v1-dump)."""
    if not rec.get("raw_claims"):
        return None   # v1 dump (or a company with zero claims) -- not replayable at claims level
    from agentic_estimation.layer_2.claim_validators import validate_claims
    from agentic_estimation.layer_3.formula_estimator import compute_formula_scores

    claims = [_dict_to_claim(d) for d in rec["raw_claims"]]
    signals = rec.get("signals") or {}
    country = rec.get("country")
    metadata = rec.get("metadata") or {}

    if not skip_tier0:
        claims, _flags = validate_claims(claims, signals=signals, country=country)
    else:
        # "no_tier0" means only the 5 lexical/numeric/polarity/corroboration/
        # known-failure-shape rules are skipped -- NOT cluster severity, which
        # is a separate, later-added step that happens to live inside the same
        # function. Run it directly so this variant's name still matches what
        # it measures (see validate_claims' apply_cluster_severity docstring).
        from agentic_estimation.layer_2.evidence_clusters import apply_cluster_severity
        apply_cluster_severity(claims)  # adjusts claim objects in place

    ctx = _freshness_frozen() if freeze_freshness else _null_ctx()
    with ctx:
        peer_override = None
        if peer_anchor_from_dump:
            peer_override = {
                p: _dict_to_anchor(rec["pillars"].get(p, {}).get("peer_anchor"))
                for p in ("E", "S", "G")
            }
        formula_scores = compute_formula_scores(
            claims, country, metadata, company_name=rec["name"], signals=signals,
            use_saturation=False, peer_anchor_override=peer_override,
        )
    pfs = formula_scores.get(pillar)
    if pfs is None:
        return None
    return {
        "baseline": pfs.baseline,
        "baseline_source": pfs.baseline_source,
        "registry_weight_sum": rec["pillars"][pillar]["registry_weight_sum"],
        "contributions": [{"factor": c.factor, "weight": c.weight, "confidence": c.confidence,
                            "delta": c.delta, "method": c.method} for c in pfs.contributions],
    }


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _freshness_frozen:
    """Monkeypatch evidence_freshness.freshness_multiplier_for_signal -> 1.0
    for the duration of one rescore. Offline analysis tool only -- no
    production flag needed (see formula_estimator docstring on `signals`)."""
    def __enter__(self):
        from agentic_estimation.layer_2 import evidence_freshness
        self._mod = evidence_freshness
        self._orig = evidence_freshness.freshness_multiplier_for_signal
        evidence_freshness.freshness_multiplier_for_signal = lambda *a, **k: 1.0
        return self

    def __exit__(self, *a):
        self._mod.freshness_multiplier_for_signal = self._orig
        return False


# ── Variant matrix ────────────────────────────────────────────────────────

_CONTRIB_LEVEL_VARIANTS = {
    "base": lambda rec, p: rec["pillars"][p]["contributions"],
    "no_holistic": lambda rec, p: rec["pillars"][p]["contributions"],
    "holistic_only": lambda rec, p: rec["pillars"][p]["contributions"],
    "formula_only": lambda rec, p: rec["pillars"][p]["contributions"],
    "blend_60_40": lambda rec, p: rec["pillars"][p]["contributions"],
    "blend_50_50": lambda rec, p: rec["pillars"][p]["contributions"],
    "no_peer_anchor": lambda rec, p: [c for c in rec["pillars"][p]["contributions"] if c["method"] != "peer_anchor"],
    "baseline_only": lambda rec, p: [],
}

_CLAIMS_LEVEL_VARIANTS = {"no_tier0", "no_freshness"}

# Peer-first baseline routes: invert the current hierarchy (country baseline
# as anchor, peer percentile as an additive vote) -- instead START from where
# the company's real peer group sits, and let the country baseline either
# vanish (peer_baseline) or shift the peer anchor proportionally to how much
# we trust the peer vote (peer_baseline_blend). Both drop the _peer_anchor
# pseudo-contribution to avoid double-counting the same signal. Companies
# whose peer vote abstained fall back to the country baseline unchanged --
# exactly what a production peer-first route would have to do. Needs the v2
# dump's stored peer_anchor field (v1 dumps: not replayable, excluded).
_PEER_BASELINE_VARIANTS = {"peer_baseline", "peer_baseline_blend"}

_ALL_VARIANTS = (list(_CONTRIB_LEVEL_VARIANTS) + sorted(_PEER_BASELINE_VARIANTS)
                 + list(_CLAIMS_LEVEL_VARIANTS))


def _score_variant(rec: dict, pillar: str, variant: str) -> Optional[float]:
    pr = rec["pillars"].get(pillar)
    if pr is None:
        return None
    holistic_val = (rec.get("holistic") or {}).get(pillar)

    if variant in _CLAIMS_LEVEL_VARIANTS:
        skip_tier0 = variant == "no_tier0"
        freeze_freshness = variant == "no_freshness"
        pr2 = _rescore_claims_level(rec, pillar, skip_tier0=skip_tier0,
                                     freeze_freshness=freeze_freshness, peer_anchor_from_dump=True)
        if pr2 is None:
            return None   # v1 dump has no raw_claims -- variant not replayable
        return _reconcile_score(pillar, pr2, pr2["contributions"], holistic_val)

    if variant in _PEER_BASELINE_VARIANTS:
        if "peer_anchor" not in pr:
            return None   # v1 dump never stored the vote -- not replayable
        anchor = pr["peer_anchor"] or {}
        contribs = [c for c in pr["contributions"] if c["method"] != "peer_anchor"]
        if anchor.get("percentile") is None:
            # peer vote abstained -- fall back to the country baseline, same
            # as a production peer-first route would.
            return _reconcile_score(pillar, pr, contribs, holistic_val)
        # The peer percentile is rank-based (0-100 within the peer table);
        # our accuracy metrics are rank-based too (Spearman + percentile-MAE),
        # so using it directly on the baseline's 0-100 scale is valid for
        # ranking purposes even though it is not calibrated in absolute terms.
        if variant == "peer_baseline":
            new_baseline = float(anchor["percentile"])
        else:   # peer_baseline_blend: trust-weighted, country baseline shifts it
            conf = float(anchor.get("confidence") or 0.0)
            new_baseline = conf * float(anchor["percentile"]) + (1.0 - conf) * pr["baseline"]
        pr2 = {**pr, "baseline": new_baseline}
        return _reconcile_score(pillar, pr2, contribs, holistic_val)

    contribs = _CONTRIB_LEVEL_VARIANTS[variant](rec, pillar)

    if variant == "no_holistic":
        return _reconcile_score(pillar, pr, contribs, None)
    if variant == "holistic_only":
        return holistic_val
    if variant == "formula_only":
        bd = _saturate(pr, pillar, contribs)
        return bd.score
    if variant == "blend_60_40":
        return _reconcile_score(pillar, pr, contribs, holistic_val,
                                 pillar_weights_override={"formula": 0.6, "holistic": 0.4})
    if variant == "blend_50_50":
        return _reconcile_score(pillar, pr, contribs, holistic_val,
                                 pillar_weights_override={"formula": 0.5, "holistic": 0.5})
    # base, no_peer_anchor, baseline_only
    return _reconcile_score(pillar, pr, contribs, holistic_val)


# ── RUN: replay the matrix, compute paired stats vs base ──────────────────

def run(dump_path: Path, variants: list, out_path: Optional[Path] = None) -> dict:
    records = json.loads(dump_path.read_text(encoding="utf-8"))
    _p(f"loaded {len(records)} companies ({dump_path.name})\n")

    is_v2 = any(r.get("raw_claims") for r in records)
    if not is_v2:
        skipped = [v for v in variants if v in _CLAIMS_LEVEL_VARIANTS]
        if skipped:
            _p(f"NOTE: v1 dump (no raw_claims) -- skipping claims-level variants: {skipped}\n")
        variants = [v for v in variants if v not in _CLAIMS_LEVEL_VARIANTS]

    report: dict = {"dump": str(dump_path), "n_companies": len(records), "pillars": {}}

    for pillar in ("E", "S", "G"):
        truths = [r["truth"].get(pillar) for r in records]
        pillar_report = {}
        base_preds = None
        base_errs = None

        for variant in variants:
            preds, truths_kept = [], []
            for r, t in zip(records, truths):
                if t is None:
                    continue
                s = _score_variant(r, pillar, variant)
                if s is None:
                    continue
                preds.append(s)
                truths_kept.append(float(t))

            rho = _spearman(preds, truths_kept)
            mae = _percentile_mae(preds, truths_kept)
            errs = _per_company_errors(preds, truths_kept)
            entry = {"n": len(preds), "spearman": rho, "pct_mae": mae}

            if variant == "base":
                base_preds, base_truths, base_errs = preds, truths_kept, errs
                entry["delta_vs_base"] = 0.0
                entry["ci95"] = [0.0, 0.0]
                entry["verdict"] = "BASE"
            elif base_errs is not None and len(errs) == len(base_errs) and errs:
                deltas = [e - b for e, b in zip(errs, base_errs) if e is not None and b is not None]
                lo, hi, mean = _bootstrap_ci(deltas)
                entry["delta_vs_base"] = round(mean, 2) if mean is not None else None
                entry["ci95"] = [round(lo, 2), round(hi, 2)] if lo is not None else None
                entry["verdict"] = _verdict(lo, hi)
            else:
                entry["delta_vs_base"] = None
                entry["ci95"] = None
                entry["verdict"] = "UNDECIDABLE"

            pillar_report[variant] = entry

        report["pillars"][pillar] = pillar_report

    _print_report(report)
    if out_path is not None:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _p(f"\nwrote report -> {out_path}")
    return report


def _print_report(report: dict) -> None:
    for pillar, variants in report["pillars"].items():
        _p(f"=== {pillar} ===")
        _p(f"{'variant':<16}{'n':>5}{'spearman':>11}{'pct_MAE':>10}{'d_err':>9}{'CI95':>18}  verdict")
        for name, e in variants.items():
            rho = f"{e['spearman']:+.3f}" if e["spearman"] is not None else "n/a"
            mae = f"{e['pct_mae']:.1f}" if e["pct_mae"] is not None else "n/a"
            delta = f"{e['delta_vs_base']:+.2f}" if e.get("delta_vs_base") is not None else "n/a"
            ci = f"[{e['ci95'][0]:+.2f},{e['ci95'][1]:+.2f}]" if e.get("ci95") else "n/a"
            _p(f"{name:<16}{e['n']:>5}{rho:>11}{mae:>10}{delta:>9}{ci:>18}  {e['verdict']}")
        _p("")


# ── CLI ─────────────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 2:
        _p(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    import argparse
    if cmd == "dump":
        ap = argparse.ArgumentParser()
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--n", type=int, default=30)
        ap.add_argument("--out", required=True)
        ap.add_argument("--workers", type=int, default=6)
        args = ap.parse_args(sys.argv[2:])
        dump(args.seed, args.n, Path(args.out), workers=args.workers)
    elif cmd == "dump-stratified":
        ap = argparse.ArgumentParser()
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--n-bcorp", type=int, required=True)
        ap.add_argument("--n-upright", type=int, required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--workers", type=int, default=6)
        ap.add_argument("--skip-newsapi", action="store_true")
        args = ap.parse_args(sys.argv[2:])
        dump_stratified(args.seed, args.n_bcorp, args.n_upright, Path(args.out),
                         workers=args.workers, skip_newsapi=args.skip_newsapi)
    elif cmd == "run":
        ap = argparse.ArgumentParser()
        ap.add_argument("--dump", required=True)
        ap.add_argument("--variants", default="all",
                         help="comma-separated variant names, or 'all'")
        ap.add_argument("--out", default=None)
        args = ap.parse_args(sys.argv[2:])
        variants = _ALL_VARIANTS if args.variants == "all" else args.variants.split(",")
        out_path = Path(args.out) if args.out else None
        run(Path(args.dump), variants, out_path)
    else:
        _p(f"unknown command {cmd!r}"); sys.exit(1)


if __name__ == "__main__":
    _cli()
