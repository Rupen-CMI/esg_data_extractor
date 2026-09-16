"""
ratio_estimator.py — deterministic Tier-3 back-fill (NOT an LLM call).

Sits between Layer 1 (Collectors) and Layer 2 (Extractors) in the rebuild
pipeline. When Tier 1/2 collection misses a factor — the common case for
small/private companies, per the Company Profiler reliability finding
(10/10 hit rate for large/known companies, 0/5 for small/private, tested
live during Phase 1) — this module fills it via the STRICT, ORDERED
fallback chain specified in the plan:

    1. sector+country peer median   (confidence ~0.3)
    2. sector-only peer median      (confidence ~0.2)
    3. absent                       (confidence 0, no claim written)

UPRIGHT-ONLY (2026-09-12): peers come from upright_lookup exclusively --
bcorp_lookup is never queried here, per this pipeline's own truth-source
decision (bcorp disagrees with Upright at Spearman -0.538 on overall score,
already rejected as a ground-truth/calibration source elsewhere; using it to
estimate a missing value here would quietly reintroduce a source the
pipeline doesn't otherwise trust). This removed the old step 3 (coarse
size-bucket fallback) entirely -- it was a bcorp_lookup.size-only mechanism
(upright_lookup has no size-bucket column, only continuous revenue_usd), so
there was no Upright equivalent to keep it alive.

Every filled value in steps 1-2 is a REAL STATISTIC over real comparable
companies already in the DB (via peer_anchor_collector), NEVER a value
invented by an LLM. This is the load-bearing distinction that keeps the
"no fabricated numbers" property true even when Layer 1 collection fails.

factory_workforce_share and gender_ratio are OPTIONAL ENRICHERS per user
direction: the fallback chain is still attempted for them, but a miss at
every step leaves them absent (step 4) rather than forcing a weaker
estimate through — they never block or degrade the rest of the estimate.

Every filled value writes a company_evidence_claims row (method=
'peer_ratio_fallback', source_note describing the peer group) — satisfying
the DB-level "no source, no claim" constraint via source_note rather than
source_signal_id, exactly as that constraint was designed to allow (see
db_migrations/002_evidence_claims.sql).

CLI:
    python -m agentic_estimation.ratio_estimator "Kitchen Bath Ventures SL" --sector manufacturing --country Spain
"""

from dataclasses import dataclass
from statistics import median

from agentic_estimation.layer_1.peer_anchor_collector import find_peers
from agentic_estimation.shared.pipeline_logger import get_logger, log_header

log = get_logger("ratio_estimator")

# Minimum peer-group size before a median is trusted. Upright peer groups
# are commonly 50-200+ companies (see Phase 1 verification: 205
# manufacturing-sector peers found for one test case, across bcorp+upright
# combined at the time -- Upright alone is smaller but still comfortably
# above this floor for common sectors), so 3 is workable there. The
# wikirate-backed real-metric path (used for employee_count, annual_revenue)
# is known to be extremely sparse (5 total rows in the entire DB as of
# Phase 1 build) AND noisy — verified directly: a 5-point sample of
# [0, 0.84, 1, 1000, 402614] produces a median of 1.0, a nonsensical
# real-world estimate despite the sample nominally clearing a naive n>=3
# bar. A larger floor for this specific path is a deliberate, data-driven
# guard, not an arbitrary number.
_MIN_PEERS_UPRIGHT = 3
_MIN_PEERS_REAL_METRIC = 15

# Factors that are REQUIRED to attempt the full fallback chain (the formula
# generally benefits from something to anchor to, even at low confidence).
REQUIRED_FACTORS = ("employee_count", "annual_revenue", "factory_count")

# Factors that are explicitly OPTIONAL — per user direction, absence must
# never block or degrade the estimate. Fallback chain is still tried, but a
# miss at every step is fine and expected, not a failure.
OPTIONAL_FACTORS = ("factory_workforce_share", "female_employees_pct")

# Field name each factor maps to on upright_lookup peer records, where
# applicable. Factors with no direct peer field (e.g. factory_count --
# upright doesn't track this) stay absent -- there is no lower fallback
# step anymore (see UPRIGHT-ONLY note above).
#
# HONEST CURRENT COVERAGE (verified during Phase 1 build, not aspirational):
#   employee_count  -- ONLY resolvable via the wikirate real-metric path
#                       (metric_key), which has just 5 rows in the whole DB
#                       and requires _MIN_PEERS_REAL_METRIC=15 to trust a
#                       median -- so this factor will realistically come
#                       back ABSENT for most companies until more real
#                       employee-count disclosures are ingested.
#   annual_revenue  -- resolvable via upright_lookup.revenue_usd, but
#                       requires an upright-vocabulary sector string
#                       ("Industrial Manufacturing and Services", not
#                       "manufacturing") -- see the KNOWN LIMITATION note in
#                       peer_anchor_collector.py. Fixing this needs a
#                       sector-vocabulary mapping table, intentionally
#                       deferred rather than guessed at here.
#   factory_count, factory_workforce_share, female_employees_pct -- no
#                       upright field exists at all; these stay absent,
#                       which is the correct behavior for data that
#                       genuinely isn't tracked by this source.
_PEER_FIELD_MAP = {
    "employee_count": None,       # via metric_key on the real-metric path only
    "annual_revenue": "revenue_usd",   # blocked by sector-vocab mismatch today
    "factory_count": None,        # no peer source tracks this at all
    "factory_workforce_share": None,
    "female_employees_pct": None,  # no direct upright field; left for Phase 2 wikirate expansion
}


@dataclass
class RatioEstimate:
    factor: str
    value: float | None
    confidence: float
    method: str            # 'peer_ratio_fallback' | 'absent'
    source_note: str | None
    n_peers: int = 0


def _combined_sample(peers: list, upright_field: str | None,
                      metric_field: str | None) -> list[float]:
    """Values for one logical factor across a MIXED peer list, where upright
    and the wikirate real-metric path store it under DIFFERENT field names
    (e.g. upright's revenue_usd vs. the metric_key 'annual_revenue' used on
    company_metric_values rows).

    REAL BUG FOUND AND FIXED 2026-09-12: estimate_factor() used to check only
    ONE field name (metric_key when given, else peer_field), so passing
    metric_key="annual_revenue" for annual_revenue silently searched for a
    field literally named "annual_revenue" on EVERY peer, including upright
    ones -- but upright peers only ever carry the value under "revenue_usd"
    (see _PEER_FIELD_MAP). Confirmed live: find_peers(sector="Retail",
    metric_key="annual_revenue") returned 200 real upright peers, 0 of which
    had a truthy .fields.get("annual_revenue"), while all 200 had a real
    .fields.get("revenue_usd") -- the exact data this step exists to use was
    silently discarded every time metric_key was set for a field upright
    tracks under a different name. This is WHY the module's own comment
    said annual_revenue "currently NEVER matches" -- it blamed the sector-
    vocabulary mismatch, which is real too, but this field-name mismatch was
    ALSO independently zeroing the sample even on a sector match.

    Collects both field names (whichever exist) and returns the union of
    values, so upright's plentiful data and the sparser real-metric rows
    both count."""
    fields = {f for f in (upright_field, metric_field) if f}
    values: list[float] = []
    for p in peers:
        for f in fields:
            v = p.fields.get(f)
            if v is not None:
                values.append(v)
                break   # one peer contributes at most one value per factor
    return values


def estimate_factor(
    factor: str,
    sector: str | None,
    country: str | None,
    exclude_name: str | None = None,
    metric_key: str | None = None,
) -> RatioEstimate:
    """
    Run the ordered fallback chain for ONE factor. metric_key, if given,
    also searches real (non-agentic) company_metric_values via
    peer_anchor_collector's wikirate-backed path — needed for physical-unit
    factors like employee_count that upright doesn't carry directly.
    """
    peer_field = _PEER_FIELD_MAP.get(factor)

    # If NEITHER an upright field nor a wikirate metric_key exists for this
    # factor, there is nothing to search for at any step -- go straight to
    # absent (avoids a same-shape-as-before false "0 peers" log for factors
    # that were never resolvable in the first place, e.g. factory_count).
    if not peer_field and not metric_key:
        log.info("[%s] no peer field or metric_key mapped -- absent", factor)
        return RatioEstimate(factor=factor, value=None, confidence=0.0, method="absent", source_note=None)

    # Upright contributes real, plentiful data whenever peer_field exists --
    # use the normal, lower floor even if the sparser wikirate path also
    # contributes to the same combined sample (a few noisy wikirate rows
    # mixed into 50+ real upright ones don't meaningfully skew a median).
    # ONLY when peer_field is None (no upright equivalent at all, e.g.
    # employee_count) is the WHOLE sample wikirate-sourced -- keep the
    # higher, data-driven floor for that case, per this module's own
    # measured noise example ([0, 0.84, 1, 1000, 402614] at n=5).
    min_peers = _MIN_PEERS_UPRIGHT if peer_field else _MIN_PEERS_REAL_METRIC

    # UPRIGHT-ONLY (2026-09-12): bcorp is excluded from every step here, per
    # the pipeline's own truth-source decision -- bcorp disagrees with Upright
    # at Spearman -0.538 on overall score, so it was already rejected as a
    # calibration/ground-truth source elsewhere; using it to peer-estimate a
    # missing value would let a source the pipeline itself doesn't trust
    # quietly set real numbers. This costs the old step 3 (coarse size-bucket
    # fallback) entirely -- confirmed live: upright_lookup has NO size-bucket
    # column at all (revenue_usd is its only continuous size signal), so
    # size-bucket peer-matching is a bcorp-only mechanism with no Upright
    # equivalent to fall back to, not just a weaker one. Accepted rather than
    # worked around -- a 3-step chain (sector+country, sector-only, absent)
    # that only ever estimates from a trusted source beats a 4-step chain
    # whose last rung quietly reintroduces the source everything else here
    # was built to avoid.

    # Step 1: sector + country peer median
    if sector and country:
        peers = find_peers(sector=sector, country=country, exclude_name=exclude_name,
                            metric_key=metric_key, include_bcorp=False)
        values = _combined_sample(peers, peer_field, metric_key)
        if len(values) >= min_peers:
            val = float(median(values))
            log.info("[%s] step1 sector+country median = %.2f (n=%d)", factor, val, len(values))
            return RatioEstimate(
                factor=factor, value=val, confidence=0.3, method="peer_ratio_fallback",
                source_note=f"sector+country peer median (sector={sector}, country={country}, n={len(values)})",
                n_peers=len(values),
            )

    # Step 2: sector-only peer median (country dropped)
    if sector:
        peers = find_peers(sector=sector, country=None, exclude_name=exclude_name,
                            metric_key=metric_key, include_bcorp=False)
        values = _combined_sample(peers, peer_field, metric_key)
        if len(values) >= min_peers:
            val = float(median(values))
            log.info("[%s] step2 sector-only median = %.2f (n=%d)", factor, val, len(values))
            return RatioEstimate(
                factor=factor, value=val, confidence=0.2, method="peer_ratio_fallback",
                source_note=f"sector-only peer median (sector={sector}, n={len(values)})",
                n_peers=len(values),
            )

    # Step 3 (coarse bcorp size-bucket fallback) REMOVED -- see UPRIGHT-ONLY
    # note above. Falls straight through to absent if steps 1-2 miss.

    # Step 4: absent — no fabricated value, honest terminal state.
    log.info("[%s] step4 absent -- no peer data available at any fallback level", factor)
    return RatioEstimate(factor=factor, value=None, confidence=0.0, method="absent", source_note=None)


def estimate_missing_factors(
    known_factors: dict[str, float | None],
    sector: str | None,
    country: str | None,
    exclude_name: str | None = None,
) -> dict[str, RatioEstimate]:
    """
    Given a dict of {factor: value_or_None} already collected in Layer 1,
    run the fallback chain for every REQUIRED factor that's missing, and
    every OPTIONAL factor that's missing (best-effort, absence is fine).
    """
    results: dict[str, RatioEstimate] = {}

    for factor in REQUIRED_FACTORS + OPTIONAL_FACTORS:
        if known_factors.get(factor) is not None:
            continue  # already have a real value, no back-fill needed
        metric_key = factor if factor in ("employee_count", "annual_revenue") else None
        results[factor] = estimate_factor(
            factor, sector, country,
            exclude_name=exclude_name, metric_key=metric_key,
        )

    n_filled = sum(1 for r in results.values() if r.value is not None)
    n_optional_absent = sum(
        1 for f, r in results.items() if f in OPTIONAL_FACTORS and r.value is None
    )
    log.info("estimate_missing_factors: %d/%d factors back-filled (%d optional factors left absent, as designed)",
              n_filled, len(results), n_optional_absent)
    return results


# ── company_evidence_claims persistence ───────────────────────────────────────

async def save_ratio_estimates(company_id, pillar_map: dict[str, str], estimates: dict[str, RatioEstimate]) -> int:
    """
    Persist RatioEstimate results as company_evidence_claims rows. pillar_map
    maps factor -> pillar ('E'/'S'/'G') since RatioEstimate itself is
    pillar-agnostic (a peer statistic, not a pillar judgment). Only writes
    rows for estimates with confidence > 0 (absent/step-4 results produce no
    row at all — consistent with "no source, no claim").

    Returns the number of rows written.
    """
    import os

    import asyncpg

    db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set")

    written = 0
    conn = await asyncpg.connect(db_url)
    try:
        async with conn.transaction():
            # No unique constraint on this table to ON CONFLICT against, so a
            # rerun for the same company would otherwise just append on top of
            # every prior run's rows forever. Clear this producer's own rows
            # for this company first -- scoped to produced_by so it never
            # touches rows written by a different producer (e.g.
            # pillar_extractors.persist_claims()).
            await conn.execute(
                "DELETE FROM company_evidence_claims WHERE company_id = $1 AND produced_by = 'ratio_estimator'",
                str(company_id),
            )
            for factor, est in estimates.items():
                if est.confidence <= 0 or est.value is None:
                    continue  # absent — no claim, per the "no source, no claim" rule
                pillar = pillar_map.get(factor, "S")  # most Tier-3 factors default to S if unmapped
                await conn.execute(
                    """
                    INSERT INTO company_evidence_claims
                        (company_id, pillar, factor, polarity, strength, confidence,
                         value, reasoning, source_note, produced_by, method)
                    VALUES ($1, $2, $3, 0, $4, $5, $6, $7, $8, 'ratio_estimator', $9)
                    """,
                    str(company_id), pillar, factor,
                    min(est.confidence, 1.0),  # strength: reuse confidence magnitude, polarity neutral (0) — a peer statistic isn't inherently positive/negative
                    est.confidence, est.value,
                    f"Peer-ratio fallback ({est.method}), n_peers={est.n_peers}",
                    est.source_note, est.method,
                )
                written += 1
    finally:
        await conn.close()

    log.info("saved %d ratio-estimate claims to company_evidence_claims", written)
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Ratio Estimator — deterministic Tier-3 back-fill")
    ap.add_argument("company")
    ap.add_argument("--sector", default=None)
    ap.add_argument("--country", default=None)
    args = ap.parse_args()

    log_header(log, "Ratio Estimator", company=args.company, sector=args.sector or "N/A", country=args.country or "N/A")

    # Assume nothing known yet — demo of the "full miss" path
    known = {"employee_count": None, "annual_revenue": None, "factory_count": None,
              "factory_workforce_share": None, "female_employees_pct": None}
    results = estimate_missing_factors(known, sector=args.sector, country=args.country, exclude_name=args.company)

    print(f"\nRatio estimates for '{args.company}' (sector={args.sector}, country={args.country}):\n")
    for factor, est in results.items():
        if est.value is not None:
            print(f"  {factor:28s} = {est.value:>12.2f}  conf={est.confidence:.2f}  method={est.method}")
            print(f"    {'':<28s}   {est.source_note}")
        else:
            required = "REQUIRED" if factor in REQUIRED_FACTORS else "optional"
            print(f"  {factor:28s} = ABSENT ({required}, no fabricated value)")


if __name__ == "__main__":
    _cli()
