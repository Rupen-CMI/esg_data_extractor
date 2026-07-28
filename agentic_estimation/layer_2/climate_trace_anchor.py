"""
climate_trace_anchor.py — deterministic Climate-TRACE-grounded claims (NOT an
LLM call). Phase 2, Step 2 of the rebuild (see PHASE_2_PLAN.md).

Two independent, additive claim sources:

  1. Owner match — company name -> climate_trace_owners.name. On a strict
     match, sum the owner's harvested facility emissions into a REAL
     scope_1_emissions claim at high confidence (0.85). This is genuine
     disclosed/measured data, not a peer statistic or an LLM guess.

  2. Sector anchor — company -> market_company_link -> markets ->
     market_climate_trace_mapping (LLM-mapped market -> CT sector/subsector,
     built once in market_climate_trace_mapper.py) -> that sector's emissions
     percentile among the company's country's sectors, from
     climate_trace_country_emissions. A much weaker signal (0.25 confidence)
     since it's a sector-level proxy, not the company's own data.

Both queries fail closed to an empty list -- no company_id, no market link,
unmapped market, un-harvested owner, or missing country data all produce zero
claims, never an exception. This is a design requirement, not a shortcut: the
owner-emissions harvest is still running and the market mapper has only
processed 25/9,580 markets as of this writing -- this module gets more useful
automatically as both fill in, with no code change.

Owner-name matching is DELIBERATELY conservative: exact match after
normalisation (lowercase, strip punctuation, strip common legal-entity
suffixes), or a prefix match where the unmatched residue on the longer side is
purely suffix tokens. No edit-distance/fuzzy matching -- against 14,513 owner
names, a fuzzy threshold would eventually assert the WRONG company's real
emissions at 0.85 confidence, which is worse than the miss it would prevent.

CLI:
    python -m agentic_estimation.climate_trace_anchor owner-match "Nvidia"
    python -m agentic_estimation.climate_trace_anchor claims "Nvidia" --company-id <uuid>
"""

import re
import sys
from typing import Optional

import pandas as pd
from pathlib import Path

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.climate_trace_harvester import _db_conn
from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.shared.company_name_utils import LEGAL_SUFFIXES

log = get_logger("climate_trace_anchor")

_YEAR = 2024  # only year currently harvested; revisit if multi-year data lands


def _normalise(name: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    return [t for t in cleaned.split() if t]


def _strip_suffixes(tokens: list[str]) -> list[str]:
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return tokens


def _names_match(a: str, b: str) -> bool:
    """Exact match after normalisation + suffix-stripping, or a prefix match
    where the longer side's unmatched residue is purely legal-suffix tokens."""
    ta, tb = _strip_suffixes(_normalise(a)), _strip_suffixes(_normalise(b))
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if longer[: len(shorter)] == shorter:
        residue = longer[len(shorter):]
        return all(tok in LEGAL_SUFFIXES for tok in residue)
    return False


# ── owner cache (static reference data -- harvested offline, ~14.5k rows) ────
_owners_cache: Optional[list[tuple[str, str]]] = None


def _all_owners() -> list[tuple[str, str]]:
    """(owner_id, name) for every harvested Climate TRACE owner, cached at
    module scope after the first call -- this is static reference data, not
    worth re-fetching on every ct_owner_match() call across a whole backtest."""
    global _owners_cache
    if _owners_cache is None:
        conn = _db_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT owner_id, name FROM climate_trace_owners")
            _owners_cache = cur.fetchall()
        finally:
            conn.close()
    return _owners_cache


# ── 1. Owner match -> real facility emissions claim ──────────────────────────

def ct_owner_match(company: str) -> Optional[tuple[str, str]]:
    """Return (owner_id, matched_name) on a confident match, else None."""
    for owner_id, name in _all_owners():
        if _names_match(company, name):
            return owner_id, name
    return None


def _owner_emissions_claim(company: str) -> Optional[ExtractedClaim]:
    match = ct_owner_match(company)
    if not match:
        return None
    owner_id, matched_name = match

    conn = _db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT SUM(emissions_quantity), COUNT(*) FROM climate_trace_owner_emissions "
            "WHERE owner_id = %s AND gas = 'co2e_100yr' AND year = %s",
            (owner_id, _YEAR),
        )
        total, n_facilities = cur.fetchone()
    finally:
        conn.close()

    if total is None:
        log.info("[%s] owner match '%s' (id=%s) but no harvested emissions rows yet -- no claim",
                  company, matched_name, owner_id)
        return None

    log.info("[%s] owner match '%s' (id=%s): %.0f tCO2e across %d facilities",
              company, matched_name, owner_id, total, n_facilities)
    return ExtractedClaim(
        factor="scope_1_emissions",
        pillar="E",
        polarity=0,   # magnitude carries the signal via `value`, not a directional guess
        strength=0.0,
        confidence=0.85,
        value=float(total),
        source_tag="climate_trace_owner_match",
        reasoning=f"Climate TRACE owner match '{matched_name}' (owner_id={owner_id}, "
                  f"{n_facilities} facilities, year={_YEAR})",
        method="dataset_lookup",
    )


# ── 2. Sector anchor -> country x sector emissions percentile ────────────────

_METADATA_PATH = Path(__file__).parent.parent.parent / "raw_esg_data" / "esgdata_download-2026-05-01.xlsx"
_country_to_iso3_cache: Optional[dict[str, str]] = None


def _country_to_iso3(country: str) -> Optional[str]:
    """Country name -> ISO3, via the World Bank Metadata sheet (already used
    by country_baseline_agent for its own Excel fallback path -- reused here
    directly rather than depending on that module's DB-path cache, which does
    NOT populate its iso3 lookup when baselines load from the DB, the normal
    case). No new dependency (pycountry is not installed)."""
    global _country_to_iso3_cache
    if _country_to_iso3_cache is None:
        meta = pd.read_excel(_METADATA_PATH, sheet_name="Metadata")
        _country_to_iso3_cache = dict(zip(meta["Economy"], meta["ISO3 code"]))
    iso3 = _country_to_iso3_cache.get(country)
    if iso3:
        return iso3
    lower = country.lower()
    for name, code in _country_to_iso3_cache.items():
        if name.lower() == lower:
            return code
    return None


def get_country_total_emissions(country: Optional[str]) -> Optional[float]:
    """Total harvested emissions (tCO2e, latest year) for a country -- summed
    over the per-sector grand-total rows (NULL subsector, same rows
    _sector_anchor_claim ranks against). Used by Tier-0 numeric-bounds
    validation (claim_validators.py) as a sanity ceiling: no single company's
    claimed scope_1_emissions should exceed its whole country's total.
    Returns None (never raises) on no country, no ISO3 mapping, DB
    unavailable, or zero harvested rows -- callers must treat None as
    "cannot check", not as "zero emissions"."""
    if not country:
        return None
    iso3 = _country_to_iso3(country)
    if not iso3:
        return None
    try:
        conn = _db_conn()
    except Exception as e:
        log.warning("get_country_total_emissions: DB unavailable (%s) -- skipping check", e)
        return None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT SUM(emissions_quantity) FROM climate_trace_country_emissions "
            "WHERE country_iso3 = %s AND sector IS NOT NULL AND subsector IS NULL AND year = %s",
            (iso3, _YEAR),
        )
        row = cur.fetchone()
    except Exception as e:
        log.warning("get_country_total_emissions: query failed (%s) -- skipping check", e)
        return None
    finally:
        conn.close()
    total = row[0] if row else None
    return float(total) if total is not None else None


def _sector_anchor_claim(company_id, country: Optional[str]) -> Optional[ExtractedClaim]:
    if not company_id or not country:
        # Distinct from the "queried and found nothing" cases below -- if a
        # caller expected a sector anchor and gets none, this log line (vs.
        # the others in this function) tells them it's a wiring gap (missing
        # company_id/country), not a real data gap.
        log.info("no company_id/country provided -- cannot attempt sector anchor")
        return None
    iso3 = _country_to_iso3(country)
    if not iso3:
        log.info("no ISO3 mapping for country '%s' -- skipping sector anchor", country)
        return None

    conn = _db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT mm.ct_sector, mm.ct_subsector, mm.confidence
            FROM market_company_link mcl
            JOIN market_climate_trace_mapping mm ON mm.market_id = mcl.market_id
            WHERE mcl.company_id = %s AND mm.confidence >= 0.6 AND mm.ct_sector IS NOT NULL
            LIMIT 1
            """,
            (str(company_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        ct_sector, ct_subsector, mapping_confidence = row

        # This country's sector totals (NULL subsector = per-sector grand total row)
        cur.execute(
            "SELECT sector, emissions_quantity FROM climate_trace_country_emissions "
            "WHERE country_iso3 = %s AND sector IS NOT NULL AND subsector IS NULL AND year = %s",
            (iso3, _YEAR),
        )
        sector_totals = cur.fetchall()
    finally:
        conn.close()

    # A percentile needs at least a few sectors to rank against -- with 1-2
    # sectors present, rank=0 would silently compute as percentile=0.0 ("cleanest
    # sector"), a false-confident claim from an insufficient sample. Not reachable
    # today (every harvested country has the full 10-sector breakdown), but this
    # guards against a future partial-harvest or filter change reducing coverage.
    if len(sector_totals) < 3:
        log.info("only %d sector(s) harvested for iso3=%s -- too few to rank, skipping sector anchor",
                  len(sector_totals), iso3)
        return None

    ranked = sorted(sector_totals, key=lambda r: r[1])
    sectors_only = [r[0] for r in ranked]
    if ct_sector not in sectors_only:
        return None
    rank = sectors_only.index(ct_sector)
    percentile = rank / (len(sectors_only) - 1)  # 0 = cleanest sector, 1 = dirtiest

    log.info("[%s country=%s] sector anchor: %s -> percentile=%.2f (mapping conf=%.2f)",
              company_id, country, ct_sector, percentile, mapping_confidence)
    return ExtractedClaim(
        factor="sector_emissions_intensity",
        pillar="E",
        polarity=-1,
        strength=percentile,
        confidence=0.25,
        value=None,
        source_tag="climate_trace_sector_anchor",
        reasoning=f"Sector '{ct_sector}'/{ct_subsector} emissions percentile {percentile:.2f} "
                  f"among {iso3} sectors (market mapping confidence={mapping_confidence:.2f})",
        method="dataset_lookup",
    )


# ── Public entry point ────────────────────────────────────────────────────────

def ct_anchor_claims(company: str, company_id=None, country: Optional[str] = None) -> list[ExtractedClaim]:
    """Both CT-grounded claims, best-effort. Never raises; missing data at any
    stage just means fewer claims, not an error."""
    claims = []
    try:
        c = _owner_emissions_claim(company)
        if c:
            claims.append(c)
    except Exception as exc:
        log.warning("[%s] owner-match claim failed (%s) -- skipped", company, exc)

    try:
        c = _sector_anchor_claim(company_id, country)
        if c:
            claims.append(c)
    except Exception as exc:
        log.warning("[%s] sector-anchor claim failed (%s) -- skipped", company, exc)

    return claims


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Climate TRACE anchor claims (deterministic)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    om = sub.add_parser("owner-match")
    om.add_argument("company")

    cl = sub.add_parser("claims")
    cl.add_argument("company")
    cl.add_argument("--company-id", default=None)
    cl.add_argument("--country", default=None)

    args = ap.parse_args()
    log_header(log, "Climate TRACE Anchor", cmd=args.cmd)

    if args.cmd == "owner-match":
        match = ct_owner_match(args.company)
        if match:
            print(f"MATCH: owner_id={match[0]} name={match[1]!r}")
        else:
            print("no match")

    elif args.cmd == "claims":
        claims = ct_anchor_claims(args.company, company_id=args.company_id, country=args.country)
        if not claims:
            print("no claims produced")
        for c in claims:
            print(f"\nfactor={c.factor} pillar={c.pillar} confidence={c.confidence} "
                  f"value={c.value} polarity={c.polarity} strength={c.strength:.3f}")
            print(f"  {c.reasoning}")


if __name__ == "__main__":
    _cli()
