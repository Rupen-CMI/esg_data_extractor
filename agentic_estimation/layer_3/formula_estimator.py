"""
formula_estimator.py — deterministic E/S/G score computation (Phase 2, Step 4;
see PHASE_2_PLAN.md). No LLM call anywhere in this module.

DEFAULT aggregation (v5 saturation, EQUATION_CHANGES.md CHANGES 5.0 -- the
default since 2026-07-21, validated on fixed evidence across two samples):

    Delta = Sum_C w_i*c_i'*delta_i / Sum_C w_i          (normalized swing)
    pillar_score = clamp( B + A_p * tanh(k_p * Delta) * CovMult , 0, 100 )

  (full math -- method-trust confidence c', evidence gate, coverage
  multiplier -- lives in saturation_score.py; this module builds the
  per-factor contributions either way)

LEGACY linear aggregation (use_saturation=False, kept for A/B):

    pillar_score = clamp( B_country_pillar + Sum_i  w_i * c_i * delta_i ,  0, 100 )

  B_country_pillar : country baseline (World Bank, via country_baseline_agent)
  w_i              : hand-set factor weight (factor_registry.py)
  c_i              : claim confidence (0 when no evidence -> factor contributes nothing)
  delta_i          : signed delta in [-1, +1], computed per factor shape

No evidence for a factor -> it simply doesn't appear in the claims list -> the
pillar score sits exactly at the country baseline for that contribution. This
is the direct fix for the old scorer's "if evidence is thin, score near 45-55"
failure mode (baseline is a REAL statistic about the country, not a fake-
confident midpoint asserted by an LLM with no evidence to back it).

delta computation:
  benchmark_band, has a numeric value:
      intensity-normalise by revenue ($M) if the metric specifies one, then
      map onto the (value_for_100, value_for_0) band:
          s01 = clamp((v - v0) / (v100 - v0), 0, 1)
          delta = 2*s01 - 1
      Band ordering handles both directions automatically (direction="lower"
      metrics have v0 > v100; "higher" metrics have v0 < v100).
      No revenue available for an intensity-normalised metric -> cannot
      compute a ratio -> degrade to the event fallback with confidence halved
      (still real evidence, just can't be placed on the band without knowing
      company size).
  benchmark_band, no value (LLM only saw a qualitative mention):
      falls back to the event shape below.
  event:
      delta = polarity * strength

Multiple claims on the same factor: pick ONE, never blend. Rank by
(confidence, method trust, recency) and take the best -- averaging would
dilute a real measured value with a weaker guess, and summing would let
repeated news mentions of one controversy stack unboundedly past the
factor's weight. One factor = one bounded contribution, always auditable.

CLI:
    python -m agentic_estimation.layer_3.formula_estimator dry "Nvidia" --country USA
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.layer_2.factor_registry import get_factor, FACTORS

log = get_logger("formula_estimator")

_METHOD_RANK = {"dataset_lookup": 3, "extracted": 2, "peer_ratio_fallback": 1, "coarse_bucket": 0}
_DEFAULT_BASELINE = 50.0


@dataclass
class Contribution:
    factor: str
    weight: float
    confidence: float
    delta: float
    points: float           # weight * confidence * delta -- the actual score swing
    claim_reasoning: str
    method: str


@dataclass
class PillarFormulaScore:
    pillar: str
    baseline: float
    baseline_source: str    # "exact" | "regional" | "global_average" | "no_country" -- see country_baseline_agent.get_country_baseline_with_fallback
    score: float
    contributions: list[Contribution] = field(default_factory=list)
    peer_anchor: Optional[object] = None   # PeerAnchorVote, when a real peer group was used
    breakdown: Optional[object] = None     # SaturationBreakdown, when use_saturation=True (see saturation_score.py) -- carries evidence_mass/coverage/gate_fired for the QC gate
    ladder: Optional[dict] = None          # set by graph.py's _apply_evidence_ladder when the low-evidence ladder fired -- audit trail (rung, prior_pct, w_claims)
    controversy_overlay: Optional[dict] = None   # set by graph.py's _apply_evidence_ladder when a confirmed negative event pulled the score down


def _claim_sort_key(c: ExtractedClaim):
    """Higher is better: confidence first, then method trust, then arbitrary
    stable order (no created_at on this dataclass -- claims are in-memory
    only until persisted, so list order is the closest thing to recency)."""
    return (c.confidence, _METHOD_RANK.get(c.method, 0))


def _pick_best_claim(claims: list[ExtractedClaim]) -> ExtractedClaim:
    return max(claims, key=_claim_sort_key)


def _benchmark_delta(claim: ExtractedClaim, revenue_musd: Optional[float]) -> tuple[float, float]:
    """Returns (delta, confidence) -- confidence may be halved if intensity
    normalisation was required but no revenue was available."""
    factor = get_factor(claim.factor)
    metric = factor.metric
    v100, v0 = metric["benchmark"]

    value = claim.value
    if value is not None and metric.get("intensity") == "annual_revenue":
        if not revenue_musd or revenue_musd <= 0:
            # Can't normalise a per-revenue intensity metric without revenue --
            # real evidence, just not placeable on the band. Degrade to event
            # shape rather than discard it entirely.
            log.info("[%s] no revenue available to normalise -- degrading to event shape", claim.factor)
            return claim.polarity * claim.strength, claim.confidence * 0.5
        value = value / revenue_musd

    if value is None:
        # Qualitative mention only, no measured value -- event fallback.
        return claim.polarity * claim.strength, claim.confidence

    s01 = max(0.0, min(1.0, (value - v0) / (v100 - v0)))
    delta = 2 * s01 - 1
    return delta, claim.confidence


def _event_delta(claim: ExtractedClaim) -> tuple[float, float]:
    return claim.polarity * claim.strength, claim.confidence


def _contribution_for_factor(
    factor_key: str, claims: list[ExtractedClaim], revenue_musd: Optional[float],
    signals: Optional[dict] = None,
) -> Contribution:
    factor = get_factor(factor_key)
    best = _pick_best_claim(claims)

    if factor.delta_shape == "benchmark_band":
        delta, confidence = _benchmark_delta(best, revenue_musd)
    else:
        delta, confidence = _event_delta(best)

    # Recency decay -- EVENT-shaped claims only. A 3-year-old controversy or
    # fine genuinely matters less than last month's. But disclosed DATA
    # (benchmark_band factors: emissions figures, board percentages) is
    # deliberately NOT decayed: ESG disclosure is annual by nature, so a
    # year-old report number is usually the newest data that exists --
    # decaying it punishes the disclosure calendar, not stale evidence (and
    # the bcorp/upright ground truth we calibrate against was itself assessed
    # in the past; over-favoring last-week news decorrelates from it).
    # dataset_lookup claims (Climate TRACE anchors) also stay at 1.0 -- their
    # recency is handled at the source (latest harvested year only).
    freshness = 1.0
    if (signals is not None and best.method == "extracted"
            and factor.delta_shape == "event"
            and not (best.source_tag or "").startswith("report_pdf")):
        # report_pdf_* is EXCLUDED from date-based freshness on purpose.
        # freshness_multiplier_for_signal takes max() over every date it can
        # parse in the cited text. That is sound for a news snippet, which
        # carries one publication date, and wrong for a 200-page report, which
        # carries hundreds -- historical comparison tables, GRI index
        # references, and forward targets like "net zero by 2030". The max is
        # therefore whatever the document's furthest-future mention happens to
        # be, and _recency_decay clamps negative ages to 0, so a target year
        # scores a perfect 1.0 for a report that may be years old.
        # Rather than reward that, report claims keep freshness 1.0, matching
        # how dataset_lookup claims are already handled above.
        from agentic_estimation.layer_2.evidence_freshness import freshness_multiplier_for_signal
        freshness = freshness_multiplier_for_signal(signals.get(best.source_tag))
        confidence *= freshness

    points = factor.weight * confidence * delta
    return Contribution(
        factor=factor_key, weight=factor.weight, confidence=confidence, delta=delta,
        points=points, claim_reasoning=best.reasoning, method=best.method,
    )


def compute_formula_scores(
    claims: list[ExtractedClaim],
    country: Optional[str],
    metadata: Optional[dict] = None,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    signals: Optional[dict] = None,
    use_saturation: bool = True,
    sat_params: Optional[dict] = None,
    peer_anchor_override: Optional[dict] = None,
    truth_source: Optional[str] = None,
) -> dict[str, PillarFormulaScore]:
    """Pure, deterministic. No LLM. DB access is read-only, cached peer/country
    statistics (see peer_anchor.py, country_baseline_agent.py) -- no writes.

    use_saturation: the v5 aggregate-saturation final step (saturation_score.py:
    normalize + coverage + tanh + evidence gate) is the DEFAULT as of 2026-07-21
    -- validated on fixed evidence across two independent samples, beating the
    linear reduction on all six pillar/sample combinations (see
    EQUATION_CHANGES.md v5 and the calibration notes). Pass False to get the
    legacy linear `baseline + sum(points)` reduction for A/B comparison. Every
    upstream step (best-claim-per-factor, deltas, freshness, peer anchor) is
    identical either way -- only the last reduction differs. sat_params:
    optional {pillar: PillarSatParams} override for tuning; None uses the
    module defaults.

    company_name is used ONLY to exclude the company's own row from its peer
    group (ground-truth leakage guard) -- pass the real name when scoring a
    company that might itself be a bcorp_lookup row (e.g. calibration backtests).

    sector: explicit industry/sector hint for peer matching, e.g. the ground-
    truth record's own industry field, or a caller-supplied hint. Falls back to
    metadata['industry'] if not given. Found live that relying on metadata
    alone silently drops the peer anchor for small/private companies -- Wikidata
    has no industry field for most of them (confirmed: Thomson & Scott Ltd's
    metadata came back completely empty), even when the CALLER already knows
    the sector (e.g. from bcorp_lookup's own industry column during a backtest,
    or a user-supplied hint) -- that known sector was being silently discarded.

    signals: the {source_tag: text} dict claims were extracted from (see
    pillar_extractors.py) -- optional, but when given, each "extracted" claim's
    confidence is scaled by a recency-decay factor computed from the most
    recent date found in its cited signal's text (see evidence_freshness.py).
    None (the default) preserves today's exact behavior -- no decay applied.

    peer_anchor_override: optional {pillar: PeerAnchorVote | None} -- when
    given, SKIPS the live peer_anchor_vote() DB call and uses the frozen vote
    instead (None for a pillar = no vote, i.e. abstain). Exists for the
    ablation/route-comparison harness (calibration/ablation_replay.py): peer
    tables mutate over time, so a live re-query at replay time would make
    "same dump, same variant" non-reproducible. None (the default) preserves
    today's exact behavior -- live DB query.

    truth_source: 'bcorp' | 'upright' | None (default) -- passed straight
    through to peer_anchor_vote() to restrict which peer pool it draws from
    (see that function's docstring for the contamination bug this fixes).
    None preserves today's exact behavior: no restriction, both pools
    eligible. Pass the real truth source during backtests/calibration; live
    production scoring (a company with no ground truth row at all) has no
    correct value to pass and should leave this None."""
    from agentic_estimation.layer_1.country_baseline_agent import get_country_baseline_with_fallback
    from agentic_estimation.layer_3.peer_anchor import peer_anchor_vote, PeerAnchorVote

    if country:
        baseline, baseline_source = get_country_baseline_with_fallback(country)
    else:
        baseline, baseline_source = None, "no_country"
        log.warning("no country provided -- using global default %.0f", _DEFAULT_BASELINE)

    revenue_musd = None
    if metadata and metadata.get("revenue"):
        # company_metadata.py's "revenue" field is raw USD; CORE_METRICS
        # benchmark bands are calibrated per $M of revenue.
        revenue_musd = float(metadata["revenue"]) / 1_000_000.0

    sector = sector or (metadata.get("industry") if metadata else None)

    by_factor: dict[str, list[ExtractedClaim]] = {}
    for c in claims:
        if get_factor(c.factor) is None:
            log.warning("dropping claim for unknown factor %r", c.factor)
            continue
        by_factor.setdefault(c.factor, []).append(c)

    results: dict[str, PillarFormulaScore] = {}
    for pillar in ("E", "S", "G"):
        pillar_baseline = getattr(baseline, {"E": "e_score", "S": "s_score", "G": "g_score"}[pillar]) \
            if baseline else _DEFAULT_BASELINE

        contributions = [
            _contribution_for_factor(factor_key, factor_claims, revenue_musd, signals)
            for factor_key, factor_claims in by_factor.items()
            if FACTORS[factor_key].pillar == pillar
        ]

        # Real peer-company statistic as an additional, confidence-weighted
        # input -- this is what lets two thin-evidence companies in the same
        # country end up with DIFFERENT scores instead of an identical tie
        # (measured live: without this, up to ~47% of a bcorp sample tied at
        # one value). Blended in ALONGSIDE the country baseline + claims, not
        # instead of them -- more evidence-weight still dominates when real
        # claims exist (peer confidence caps at 0.5, well below a strong claim).
        if peer_anchor_override is not None:
            anchor = peer_anchor_override.get(pillar)
            if anchor is None:
                anchor = PeerAnchorVote(pillar=pillar, percentile=None, confidence=0.0,
                                         n_peers=0, tier="abstain", basis="peer_anchor_override: no vote")
        else:
            anchor = peer_anchor_vote(pillar, company_name or "", sector, country,
                                       truth_source=truth_source)
        if anchor.percentile is not None:
            anchor_delta = 2 * (anchor.percentile / 100.0) - 1   # -1..+1, same scale as claim deltas
            anchor_points = 10.0 * anchor.confidence * anchor_delta   # weight=10: comparable to a mid-strength factor
            contributions.append(Contribution(
                factor="_peer_anchor", weight=10.0, confidence=anchor.confidence, delta=anchor_delta,
                points=anchor_points, claim_reasoning=anchor.basis, method="peer_anchor",
            ))

        # EXIOBASE structural industry-median vote, added 2026-08-27. This is
        # a DIFFERENT signal from _peer_anchor above: peer_anchor is a real
        # bcorp/upright PEER median (other companies' disclosed/rated
        # performance); this is the sector's own physical/structural profile
        # (emissions/water/material intensity for E, workforce composition
        # for S, an E/S-derived estimate for G -- see exio_lookup.py's
        # module docstring for why G has no real EXIOBASE column). Was
        # previously wired ONLY into evidence_ladder.py's low-evidence path
        # (exio_structural rung) -- added here too so companies with plenty
        # of evidence still get an industry-median contribution at all,
        # which today's high-evidence formula had none of. E's exio_e_vote
        # has a measured +0.615 Spearman on real holdout (see exio_lookup.py);
        # S and G do not yet (exio_s_vote/exio_g_vote are unvalidated -- set
        # to a lower weight/confidence ceiling than peer_anchor for that
        # reason, same "smaller swing for a coarser/less-proven signal" rule
        # evidence_ladder.py's a_prior already follows).
        from agentic_estimation.layer_3.exio_lookup import exio_e_vote, exio_s_vote, exio_g_vote
        _exio_vote_fn = {"E": exio_e_vote, "S": exio_s_vote, "G": exio_g_vote}[pillar]
        exio_vote = _exio_vote_fn(sector)
        if exio_vote is not None:
            exio_delta = 2 * (exio_vote.percentile / 100.0) - 1   # -1..+1, same scale as claim deltas
            exio_weight = 10.0 if pillar == "E" else 6.0   # E is measured; S/G are unvalidated -- smaller max swing
            exio_points = exio_weight * exio_vote.confidence * exio_delta
            contributions.append(Contribution(
                factor="_industry_median", weight=exio_weight, confidence=exio_vote.confidence,
                delta=exio_delta, points=exio_points, claim_reasoning=exio_vote.basis,
                method="exio_structural",
            ))

        breakdown = None
        if use_saturation:
            from agentic_estimation.layer_3.saturation_score import saturate_pillar
            params = (sat_params or {}).get(pillar)
            breakdown = saturate_pillar(
                pillar=pillar, baseline=pillar_baseline, contributions=contributions,
                registry_weight_sum=_registry_weight_sum(pillar), params=params,
            )
            score = breakdown.score
        else:
            score = pillar_baseline + sum(c.points for c in contributions)
            score = max(0.0, min(100.0, score))

        results[pillar] = PillarFormulaScore(pillar=pillar, baseline=pillar_baseline,
                                              baseline_source=baseline_source,
                                              score=score, contributions=contributions,
                                              peer_anchor=anchor, breakdown=breakdown)

    return results


_REGISTRY_WEIGHT_SUM_CACHE: dict[str, float] = {}


def _registry_weight_sum(pillar: str) -> float:
    """Σ w_i over EVERY registered factor in this pillar -- the coverage
    denominator for saturation scoring. Cached (registry is static)."""
    if pillar not in _REGISTRY_WEIGHT_SUM_CACHE:
        from agentic_estimation.layer_2.factor_registry import factors_for_pillar
        _REGISTRY_WEIGHT_SUM_CACHE[pillar] = sum(f.weight for f in factors_for_pillar(pillar))
    return _REGISTRY_WEIGHT_SUM_CACHE[pillar]


async def load_claims(company_id) -> list[ExtractedClaim]:
    """DB path: load a company's persisted claims back out for graph runs
    (where extraction already happened and persisted in an earlier stage)."""
    import os
    import asyncpg

    db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set")

    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT pillar, factor, polarity, strength, confidence, value, reasoning, method "
            "FROM company_evidence_claims WHERE company_id = $1 AND confidence > 0",
            str(company_id),
        )
    finally:
        await conn.close()

    return [
        ExtractedClaim(
            factor=r["factor"], pillar=r["pillar"], polarity=r["polarity"],
            strength=r["strength"], confidence=r["confidence"], value=r["value"],
            source_tag="", reasoning=r["reasoning"] or "", method=r["method"],
        )
        for r in rows
    ]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Formula Estimator — deterministic E/S/G scoring (dry run)")
    ap.add_argument("mode", choices=["dry"])
    ap.add_argument("company")
    ap.add_argument("--industry", default="")
    ap.add_argument("--country", default=None)
    args = ap.parse_args()

    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.governance_collector import fetch_governance_signals
    from agentic_estimation.layer_1.facility_extractor import fetch_facility_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    from agentic_estimation.layer_2.pillar_extractors import extract_all_claims
    from agentic_estimation.layer_2.climate_trace_anchor import ct_anchor_claims

    log_header(log, "Formula Estimator — dry run", company=args.company, country=args.country or "auto-detect")

    signals = {}
    signals.update(fetch_company_signals(args.company, args.industry))
    signals.update(fetch_governance_signals(args.company))
    signals.update(fetch_facility_signals(args.company, args.industry))
    metadata = get_company_metadata(args.company)
    country = args.country or metadata.get("country")

    claims = extract_all_claims(args.company, signals, metadata)
    claims += ct_anchor_claims(args.company, country=country)
    print(f"\n{len(claims)} claims gathered ({len(signals)} signals)\n")

    scores = compute_formula_scores(claims, country, metadata, company_name=args.company, sector=args.industry or None, signals=signals)
    for pillar, pfs in scores.items():
        print(f"=== {pillar}: {pfs.score:.1f}/100 (baseline {pfs.baseline:.1f}, source={pfs.baseline_source}) ===")
        for c in sorted(pfs.contributions, key=lambda x: -abs(x.points)):
            print(f"  {c.factor:28s} points={c.points:+6.2f}  "
                  f"(w={c.weight:.1f} c={c.confidence:.2f} d={c.delta:+.2f})  [{c.method}]")
            print(f"    {c.claim_reasoning}")
        print()


if __name__ == "__main__":
    _cli()
