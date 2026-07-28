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
    3. coarse size bucket           (confidence ~0.15)
    4. absent                       (confidence 0, no claim written)

Every filled value in steps 1-3 is a REAL STATISTIC over real comparable
companies already in the DB (via peer_anchor_collector), NEVER a value
invented by an LLM. This is the load-bearing distinction that keeps the
"no fabricated numbers" property true even when Layer 1 collection fails.

factory_workforce_share and gender_ratio are OPTIONAL ENRICHERS per user
direction: the fallback chain is still attempted for them, but a miss at
every step leaves them absent (step 4) rather than forcing a weaker
estimate through — they never block or degrade the rest of the estimate.

Every filled value writes a company_evidence_claims row (method=
'peer_ratio_fallback' or 'coarse_bucket', source_note describing the peer
group) — satisfying the DB-level "no source, no claim" constraint via
source_note rather than source_signal_id, exactly as that constraint was
designed to allow (see db_migrations/002_evidence_claims.sql).

CLI:
    python -m agentic_estimation.ratio_estimator "Kitchen Bath Ventures SL" --sector manufacturing --country Spain
"""

import sys
from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.peer_anchor_collector import find_peers, peer_median, peer_sample_size

log = get_logger("ratio_estimator")

# Minimum peer-group size before a median is trusted. bcorp/upright peer
# groups are commonly 50-200+ companies (see Phase 1 verification: 205
# manufacturing-sector peers found for one test case), so 3 is workable
# there. The wikirate-backed real-metric path (used for employee_count,
# annual_revenue) is known to be extremely sparse (5 total rows in the
# entire DB as of Phase 1 build) AND noisy — verified directly: a 5-point
# sample of [0, 0.84, 1, 1000, 402614] produces a median of 1.0, a
# nonsensical real-world estimate despite the sample nominally clearing a
# naive n>=3 bar. A larger floor for this specific path is a deliberate,
# data-driven guard, not an arbitrary number.
_MIN_PEERS_BCORP_UPRIGHT = 3
_MIN_PEERS_REAL_METRIC = 15

# Factors that are REQUIRED to attempt the full fallback chain (the formula
# generally benefits from something to anchor to, even at low confidence).
REQUIRED_FACTORS = ("employee_count", "annual_revenue", "factory_count")

# Factors that are explicitly OPTIONAL — per user direction, absence must
# never block or degrade the estimate. Fallback chain is still tried, but a
# miss at every step is fine and expected, not a failure.
OPTIONAL_FACTORS = ("factory_workforce_share", "female_employees_pct")

# Field name each factor maps to on bcorp/upright peer records, where applicable.
# Factors with no direct peer field (e.g. factory_count — neither bcorp nor
# upright track this) fall through to the coarse-bucket step only.
#
# HONEST CURRENT COVERAGE (verified during Phase 1 build, not aspirational):
#   employee_count  -- ONLY resolvable via the wikirate real-metric path
#                       (metric_key), which has just 5 rows in the whole DB
#                       and requires _MIN_PEERS_REAL_METRIC=15 to trust a
#                       median -- so this factor will realistically come
#                       back ABSENT for most companies until more real
#                       employee-count disclosures are ingested.
#   annual_revenue  -- ONLY resolvable via upright_lookup.revenue_usd, which
#                       requires an upright-vocabulary sector string. Since
#                       Ratio Estimator callers pass bcorp-style sector
#                       strings (e.g. "manufacturing"), and upright uses a
#                       different, finer-grained vocabulary (e.g. "Industrial
#                       Manufacturing and Services"), this path currently
#                       NEVER matches through find_peers(sector=...) -- see
#                       the KNOWN LIMITATION note in peer_anchor_collector.py.
#                       Fixing this needs a sector-vocabulary mapping table,
#                       intentionally deferred rather than guessed at here.
#   factory_count, factory_workforce_share, female_employees_pct -- no bcorp/
#                       upright field exists at all; these only ever resolve
#                       via step 3 (coarse bucket) if peer_field is added
#                       later, or stay absent, which is the correct behavior
#                       for data that genuinely isn't tracked by either source.
_PEER_FIELD_MAP = {
    "employee_count": None,       # via metric_key on the real-metric path only
    "annual_revenue": "revenue_usd",   # upright only, blocked by sector-vocab mismatch today
    "factory_count": None,        # no peer source tracks this at all
    "factory_workforce_share": None,
    "female_employees_pct": None,  # no direct bcorp/upright field; left for Phase 2 wikirate expansion
}


@dataclass
class RatioEstimate:
    factor: str
    value: Optional[float]
    confidence: float
    method: str            # 'peer_ratio_fallback' | 'coarse_bucket' | 'absent'
    source_note: Optional[str]
    n_peers: int = 0


def _size_bucket_from_employees(employees: Optional[float]) -> Optional[str]:
    """Map a raw employee count to bcorp_lookup's size-bucket vocabulary."""
    if employees is None:
        return None
    if employees < 1:
        return "0"
    if employees < 10:
        return "1-9"
    if employees < 50:
        return "10-49"
    if employees < 250:
        return "50-249"
    if employees < 1000:
        return "250-999"
    return "1000+"


def estimate_factor(
    factor: str,
    sector: Optional[str],
    country: Optional[str],
    size_bucket: Optional[str] = None,
    exclude_name: Optional[str] = None,
    metric_key: Optional[str] = None,
) -> RatioEstimate:
    """
    Run the ordered fallback chain for ONE factor. metric_key, if given,
    also searches real (non-agentic) company_metric_values via
    peer_anchor_collector's wikirate-backed path — needed for physical-unit
    factors like employee_count that bcorp/upright don't carry directly.
    """
    peer_field = _PEER_FIELD_MAP.get(factor)

    # Real-metric peers (wikirate-backed, e.g. employee_count/annual_revenue)
    # are known sparse+noisy — require a much larger sample before trusting
    # the median. bcorp/upright-backed fields use the normal, lower floor.
    min_peers = _MIN_PEERS_REAL_METRIC if metric_key else _MIN_PEERS_BCORP_UPRIGHT

    # Step 1: sector + country peer median
    if sector and country:
        peers = find_peers(sector=sector, country=country, exclude_name=exclude_name, metric_key=metric_key)
        field = metric_key if metric_key else peer_field
        if field:
            n = peer_sample_size(peers, field)  # NOT len(peers) -- see peer_sample_size docstring
            if n >= min_peers:
                val = peer_median(peers, field)
                log.info("[%s] step1 sector+country median = %.2f (n=%d)", factor, val, n)
                return RatioEstimate(
                    factor=factor, value=val, confidence=0.3, method="peer_ratio_fallback",
                    source_note=f"sector+country peer median (sector={sector}, country={country}, n={n})",
                    n_peers=n,
                )

    # Step 2: sector-only peer median (country dropped)
    if sector:
        peers = find_peers(sector=sector, country=None, exclude_name=exclude_name, metric_key=metric_key)
        field = metric_key if metric_key else peer_field
        if field:
            n = peer_sample_size(peers, field)
            if n >= min_peers:
                val = peer_median(peers, field)
                log.info("[%s] step2 sector-only median = %.2f (n=%d)", factor, val, n)
                return RatioEstimate(
                    factor=factor, value=val, confidence=0.2, method="peer_ratio_fallback",
                    source_note=f"sector-only peer median (sector={sector}, n={n})",
                    n_peers=n,
                )

    # Step 3: coarse size bucket (bcorp_lookup.size, matched on sector+bucket only)
    if sector and size_bucket:
        peers = find_peers(sector=sector, size_bucket=size_bucket, exclude_name=exclude_name)
        if peer_field:
            n = peer_sample_size(peers, peer_field)
            if n >= 1:
                val = peer_median(peers, peer_field)
                log.info("[%s] step3 coarse bucket median = %.2f (n=%d)", factor, val, n)
                return RatioEstimate(
                    factor=factor, value=val, confidence=0.15, method="coarse_bucket",
                    source_note=f"coarse size-bucket median (sector={sector}, size={size_bucket}, n={n})",
                    n_peers=n,
                )

    # Step 4: absent — no fabricated value, honest terminal state.
    log.info("[%s] step4 absent -- no peer data available at any fallback level", factor)
    return RatioEstimate(factor=factor, value=None, confidence=0.0, method="absent", source_note=None)


def estimate_missing_factors(
    known_factors: dict[str, Optional[float]],
    sector: Optional[str],
    country: Optional[str],
    exclude_name: Optional[str] = None,
) -> dict[str, RatioEstimate]:
    """
    Given a dict of {factor: value_or_None} already collected in Layer 1,
    run the fallback chain for every REQUIRED factor that's missing, and
    every OPTIONAL factor that's missing (best-effort, absence is fine).

    known_factors should include 'employee_count' if available (used to
    derive the coarse size bucket for step 3 of other factors).
    """
    size_bucket = _size_bucket_from_employees(known_factors.get("employee_count"))
    results: dict[str, RatioEstimate] = {}

    for factor in REQUIRED_FACTORS + OPTIONAL_FACTORS:
        if known_factors.get(factor) is not None:
            continue  # already have a real value, no back-fill needed
        metric_key = factor if factor in ("employee_count", "annual_revenue") else None
        results[factor] = estimate_factor(
            factor, sector, country, size_bucket=size_bucket,
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
