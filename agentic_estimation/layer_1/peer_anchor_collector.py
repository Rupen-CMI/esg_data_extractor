"""
peer_anchor_collector.py — Layer 1 Peer/Anchor Collector.

Pure SQL, NO LLM call. Queries bcorp_lookup (10,337 companies), upright_lookup
(10,086 companies), and real (non-agentic-source) rows in company_metric_values
for comparable companies — same sector +/- country +/- size band — as the
Ratio Estimator's grounding data.

This is the mechanism that keeps Tier-3 back-fill honest: every value it
returns is a real statistic over real comparable companies already in the DB,
never a value invented by an LLM. See the rebuild plan
(all-data-metadata-linked-dijkstra.md) Ratio Estimator section for the
fallback chain this collector feeds:
    1. sector+country peer median
    2. sector-only peer median
    3. coarse size bucket
    4. absent (no fabricated value)

Usage:
    from agentic_estimation.layer_1.peer_anchor_collector import find_peers, peer_median

    peers = find_peers(sector="manufacturing", country="Germany", exclude_name="Bosch")
    median_score = peer_median(peers, field="overall_score")

CLI:
    python -m agentic_estimation.peer_anchor_collector peers manufacturing Germany
"""

import os
import sys
import threading
from dataclasses import dataclass
from statistics import median
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import psycopg2
from dotenv import load_dotenv

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.shared.company_name_utils import normalize_company_name

log = get_logger("peer_anchor_collector")
load_dotenv()

_MIN_PEERS_FOR_SECTOR_COUNTRY = 5   # below this, fall back to sector-only
_MIN_PEERS_FOR_SECTOR_ONLY = 5      # below this, fall back to coarse bucket


# ── DB connection (same URL handling used across the pipeline) ────────────────
#
# CACHED, not opened fresh per call. Measured live: each fresh psycopg2.connect()
# to the remote Neon instance costs ~1.5s for the TCP/TLS handshake alone, before
# any query runs. find_peers() is called once per pillar (up to 3x/company) with
# multiple fallback tiers each potentially calling it again -- a 150-company
# formula-only backtest (calibration/run_formula_only_150.py) measured ~60s/company
# with a fresh connection every time, almost entirely connection overhead, not
# query time (a trivial `SELECT 1` on an already-open connection took ~0.5s vs
# ~1.5s to open a new one). One process-lifetime connection, reused across every
# call, cuts that back to query time only.
#
# LIVENESS CHECKED, not just cached blindly: a network change mid-run (observed
# live -- switching networks left a stale resolved connection whose DNS name no
# longer resolved) must not wedge every subsequent call on a dead connection.
# `conn.closed` catches an explicitly-closed connection; a stale-but-still-open
# one is caught by the SELECT 1 probe, which raises and triggers a reconnect.
#
# ONE CONNECTION PER THREAD, not one shared module-level connection. Fixed
# 2026-08-18 after a real, repeated bug under concurrent load: with a single
# shared psycopg2 connection, thread A's liveness probe (SELECT 1) could fail
# and call conn.close() while thread B was mid-query on the SAME connection
# object -- B's cursor then raised "cursor already closed". Measured live in
# the niche-10 production demo (3 concurrent workers): 3/10 companies lost
# their formula score to this exact error (Slatto Value Add, Mali Lithium,
# Strandberg Guitars). psycopg2 connections are not thread-safe for concurrent
# use from multiple threads even with a lock around each call, because the
# close-on-dead-probe path races with an in-flight query on another thread.
# threading.local() keeps the original amortization goal (no ~1.5s handshake
# per call) while giving each worker thread its own connection -- 3 workers
# means 3 live connections instead of 1, a small, acceptable cost against a
# real correctness bug.
_thread_local = threading.local()


def _db_conn():
    conn = getattr(_thread_local, "conn", None)
    if conn is not None and conn.closed == 0:
        try:
            with conn.cursor() as probe:
                probe.execute("SELECT 1")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _thread_local.conn = None

    db_url = os.getenv("ASYNC_DB_URL", os.getenv("DB_URL", ""))
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    sslmode = qs.pop("ssl", [None])[0]
    clean_qs = urlencode({k: v[0] for k, v in qs.items()})
    if sslmode:
        clean_qs = (clean_qs + "&" if clean_qs else "") + f"sslmode={sslmode}"
    conn = psycopg2.connect(urlunparse(parsed._replace(query=clean_qs)))
    _thread_local.conn = conn
    return conn


# ── Peer record ────────────────────────────────────────────────────────────────

@dataclass
class PeerRecord:
    source: str              # 'bcorp' | 'upright' | 'wikirate'
    name: str
    sector: Optional[str]
    country: Optional[str]
    size_bucket: Optional[str]
    fields: dict              # source-specific numeric fields (e.g. overall_score, employees)


def _drop_self(peers: list[PeerRecord], exclude_name: Optional[str]) -> list[PeerRecord]:
    """Ground-truth-leakage guard, layer 2: the SQL `!= %s` filters in each
    _find_*_peers function are cheap exact-string pre-filters, but bcorp_lookup/
    upright_lookup company names carry legal suffixes and punctuation that vary
    from how a caller names the same company ("Fish Tales" vs "Fish Tales
    Holding BV") -- an exact-string compare lets the target company survive as
    its own "peer". This normalizes both sides (normalize_company_name) before
    comparing, so suffix/casing/punctuation variants are caught too. Falls back
    to a plain `.strip().lower()` comparison when the normal form is empty on
    either side (an all-suffix name like "Ltd" alone normalizes to "" and can
    never positively identify anything -- comparing two empty strings would
    incorrectly drop every peer with an unparseable name)."""
    if not exclude_name:
        return peers
    target_norm = normalize_company_name(exclude_name)
    target_fallback = exclude_name.strip().lower()

    def _is_self(peer_name: str) -> bool:
        peer_norm = normalize_company_name(peer_name)
        if target_norm and peer_norm:
            return peer_norm == target_norm
        return peer_name.strip().lower() == target_fallback

    return [p for p in peers if not _is_self(p.name)]


# ── bcorp_lookup peers ────────────────────────────────────────────────────────

def _find_bcorp_peers(sector: Optional[str], country: Optional[str], size_bucket: Optional[str],
                       exclude_name: Optional[str], limit: int, sector_column: str = "sasb_sector") -> list[PeerRecord]:
    """sector_column: which bcorp_lookup column to match `sector` against.
    Default 'sasb_sector' (4 coarse values: apparel_retail/general/
    manufacturing/services) is the original tier. 'industry_category' (22
    real, finer-grained values, e.g. "Manufactured Goods", "Energy",
    "Agriculture, forestry & fishing" -- confirmed via direct query) is a
    second, better-resolution bcorp tier for sector_matcher.py's fuzzy match
    to target before falling all the way through to upright's 30 labels."""
    conn = _db_conn()
    cur = conn.cursor()
    conditions, params = [], []
    if sector:
        conditions.append(f"{sector_column} = %s")
        params.append(sector)
    if country:
        conditions.append("country = %s")
        params.append(country)
    if size_bucket:
        conditions.append("size = %s")
        params.append(size_bucket)
    if exclude_name:
        conditions.append("company_name != %s")
        params.append(exclude_name)
    conditions.append("overall_score IS NOT NULL")

    where = " AND ".join(conditions) if conditions else "TRUE"
    params.append(limit)
    cur.execute(
        f"""
        SELECT company_name, sasb_sector, country, size, overall_score,
               impact_area_environment, impact_area_governance, impact_area_workers,
               impact_area_community, impact_area_customers
        FROM bcorp_lookup
        WHERE {where}
        LIMIT %s
        """,
        params,
    )
    rows = cur.fetchall()
    # connection is process-cached (see _db_conn) -- do not close it here

    out = []
    for name, sasb, ctry, size, overall, env, gov, workers, community, customers in rows:
        social_parts = [v for v in (workers, community, customers) if v is not None]
        social = sum(social_parts) / len(social_parts) if social_parts else None
        out.append(PeerRecord(
            source="bcorp", name=name, sector=sasb, country=ctry, size_bucket=size,
            fields={"overall_score": overall, "e_score": env, "g_score": gov, "s_score": social},
        ))
    return _drop_self(out, exclude_name)


def bcorp_industry_category_labels() -> list[str]:
    """The 22 distinct real values of bcorp_lookup.industry_category (confirmed
    live), for sector_matcher.py's fuzzy match -- finer-grained than
    sasb_sector's 4 coarse buckets, and unlike upright's peer pool, still
    yields real per-pillar E/S/G scores (not just a single total-impact
    percentile)."""
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT industry_category FROM bcorp_lookup WHERE industry_category IS NOT NULL")
    return [r[0] for r in cur.fetchall()]
    # connection is process-cached (see _db_conn) -- do not close it here


# ── upright_lookup peers ──────────────────────────────────────────────────────

def _find_upright_peers(sector: Optional[str], country: Optional[str],
                         exclude_name: Optional[str], limit: int) -> list[PeerRecord]:
    """
    Upright has no size-bucket column — revenue_usd is used as a continuous
    size signal instead, but this collector only filters by sector/country;
    callers wanting size-banding should filter the returned fields.revenue_usd.
    """
    conn = _db_conn()
    cur = conn.cursor()
    conditions, params = [], []
    if sector:
        conditions.append("industry = %s")
        params.append(sector)
    if country:
        conditions.append("country = %s")
        params.append(country)
    if exclude_name:
        conditions.append("name != %s")
        params.append(exclude_name)
    conditions.append("net_impact_ratio_percentile IS NOT NULL")

    where = " AND ".join(conditions) if conditions else "TRUE"
    params.append(limit)
    cur.execute(
        f"""
        SELECT name, industry, country, revenue_usd, net_impact_ratio_percentile
        FROM upright_lookup
        WHERE {where}
        LIMIT %s
        """,
        params,
    )
    rows = cur.fetchall()
    # connection is process-cached (see _db_conn) -- do not close it here

    out = [
        PeerRecord(
            source="upright", name=name, sector=ind, country=ctry, size_bucket=None,
            fields={"net_impact_ratio_percentile": pctile, "revenue_usd": rev},
        )
        for name, ind, ctry, rev, pctile in rows
    ]
    return _drop_self(out, exclude_name)


# ── Real (non-agentic) company_metric_values peers — for physical-unit factors ─

def _find_real_metric_peers(metric_key: str, sector_hint: Optional[str],
                             country: Optional[str], limit: int,
                             exclude_name: Optional[str] = None) -> list[PeerRecord]:
    """
    Peers with a REAL (non-agentic-source) value for a specific core metric key
    (e.g. 'employee_count', 'scope_1_emissions') — used when the Ratio Estimator
    needs a physical-unit anchor rather than a pillar score. sector_hint filters
    on companies.country only today (no sector column on `companies`); callers
    needing sector precision should cross-reference bcorp/upright peers instead.

    Deduplicates to ONE row per company (most recent reporting_year) before
    returning — raw Wikirate data has multiple year-rows per company, and
    without dedup a company with 5 years on file would get 5x the weight of
    a company with 1 in any downstream median/statistic. Real-data quality
    issues (e.g. a company reporting 0 or <1 employees) are NOT filtered here
    — that would be an undocumented judgment call the Ratio Estimator should
    make explicitly, not something silently baked into the collector.

    GROUND-TRUTH LEAKAGE GUARD: exclude_name works the same way as the other
    two peer-finders (SQL `!=` pre-filter + _drop_self post-filter) -- this
    path queries `companies`/`company_metric_values` directly, which real
    disclosed/reported data for the SAME company being scored could otherwise
    leak through as its own peer.
    """
    conn = _db_conn()
    cur = conn.cursor()
    conditions = ["d.key = %s", "v.numeric_value IS NOT NULL",
                  "(v.source IS NULL OR v.source NOT LIKE 'agentic_%%')"]
    params = [metric_key]
    if country:
        conditions.append("c.country = %s")
        params.append(country)
    if exclude_name:
        conditions.append("c.name != %s")
        params.append(exclude_name)
    where = " AND ".join(conditions)
    params.append(limit)
    cur.execute(
        f"""
        SELECT DISTINCT ON (c.id) c.name, c.country, v.numeric_value, v.reporting_year
        FROM company_metric_values v
        JOIN esg_metric_definitions d ON d.id = v.metric_id
        JOIN companies c ON c.id = v.company_id
        WHERE {where}
        ORDER BY c.id, v.reporting_year DESC NULLS LAST
        LIMIT %s
        """,
        params,
    )
    rows = cur.fetchall()
    # connection is process-cached (see _db_conn) -- do not close it here

    out = [
        PeerRecord(source="wikirate", name=name, sector=sector_hint, country=ctry, size_bucket=None,
                   fields={metric_key: val})
        for name, ctry, val, _year in rows
    ]
    return _drop_self(out, exclude_name)


# ── Public API ─────────────────────────────────────────────────────────────────

def find_peers(
    sector: Optional[str] = None,
    country: Optional[str] = None,
    size_bucket: Optional[str] = None,
    exclude_name: Optional[str] = None,
    metric_key: Optional[str] = None,
    limit: int = 200,
    bcorp_sector_column: str = "sasb_sector",
    include_upright: bool = True,
    include_bcorp: bool = False,
) -> list[PeerRecord]:
    """
    Find comparable companies for a given sector/country/size. If metric_key is
    given, also searches real (non-agentic) company_metric_values rows for that
    specific physical-unit metric. Returns the union across sources — callers
    (Ratio Estimator) decide which fields to use and compute medians themselves
    via peer_median().

    BCORP DISABLED BY DEFAULT (2026-09-16): see the "BCORP REMOVAL" note at the
    top of agentic_estimation/layer_3/peer_anchor.py for the full picture of
    every place bcorp was wired in and why it's off. Pass include_bcorp=True
    explicitly to opt back in for a specific call (e.g. a bcorp-truth backtest
    that genuinely wants bcorp peers) — this default only governs callers that
    don't ask for a source either way.

    KNOWN LIMITATION: `sector` is matched literally against bcorp_lookup.sasb_sector
    (coarse: apparel_retail/general/manufacturing/services) AND upright_lookup.industry
    (fine-grained GICS-like: "Automotive", "Industrial Manufacturing and Services", etc.)
    — these are DIFFERENT vocabularies with no shared taxonomy. Passing a bcorp-style
    sector string will always return 0 upright peers, and vice versa. Callers wanting
    cross-source peers must call find_peers() once per source's own vocabulary, or this
    should be normalized via a sector-mapping table in a later phase. Not fixed here to
    avoid guessing a mapping without real validation.

    COUNTRY CROSSWALK (DEFECT_FIX_PLAN.md 2.1): `country` is expected as a
    World Bank Economy name (resolve_country_name()'s canonical output, e.g.
    from graph.py's _resolve_state_country) -- this function converts it to
    each source's own vocabulary before querying (bcorp_lookup.country: full
    names with some source-specific spellings; upright_lookup.country: ISO3
    codes) via country_crosswalk.py. Previously the SAME unconverted string
    was passed to both, silently making every upright country+sector-tier
    lookup unreachable (bcorp names never match ISO3 codes) regardless of
    sector-match quality.
    """
    from agentic_estimation.layer_1.country_crosswalk import country_for_bcorp, country_for_upright

    peers: list[PeerRecord] = []
    if include_bcorp:
        bcorp_country = country_for_bcorp(country)
        peers += _find_bcorp_peers(sector, bcorp_country, size_bucket, exclude_name, limit, sector_column=bcorp_sector_column)
    if include_upright:
        upright_country = country_for_upright(country)
        peers += _find_upright_peers(sector, upright_country, exclude_name, limit)
    if metric_key:
        # company_metric_values/companies.country is populated from the same
        # sources as the caller's own resolved country (metadata/DB), not
        # bcorp/upright's vocabularies -- no crosswalk needed here.
        peers += _find_real_metric_peers(metric_key, sector, country, limit, exclude_name=exclude_name)
    log.info("find_peers(sector=%s, country=%s, size=%s, metric=%s) -> %d peers",
              sector, country, size_bucket, metric_key, len(peers))
    return peers


def peer_median(peers: list[PeerRecord], field: str) -> Optional[float]:
    """Median of `field` across peers that have a non-null value for it, or None."""
    values = [p.fields[field] for p in peers if p.fields.get(field) is not None]
    if not values:
        return None
    return float(median(values))


def peer_sample_size(peers: list[PeerRecord], field: str) -> int:
    """
    Count of peers that ACTUALLY contribute a non-null value for `field` —
    distinct from len(peers), which includes every source's records
    regardless of whether they carry this specific field. Callers computing
    a confidence/reliability threshold on a median MUST use this, not
    len(peers) — using the raw union size overstates sample support (e.g.
    205 bcorp/upright peers + 5 real wikirate employee_count rows returns
    len(peers)==210, but only 5 of those actually have employee_count).
    """
    return sum(1 for p in peers if p.fields.get(field) is not None)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "peers":
        print("Usage: python -m agentic_estimation.peer_anchor_collector peers <sasb_sector> [country]")
        sys.exit(1)
    sector = sys.argv[2] if len(sys.argv) > 2 else None
    country = sys.argv[3] if len(sys.argv) > 3 else None

    log_header(log, "Peer/Anchor Collector", sector=sector or "any", country=country or "any")
    peers = find_peers(sector=sector, country=country)
    print(f"\nFound {len(peers)} peers (sector={sector}, country={country})\n")
    for src in ("bcorp", "upright", "wikirate"):
        sub = [p for p in peers if p.source == src]
        print(f"  {src}: {len(sub)}")
        for p in sub[:5]:
            print(f"    {p.name:<40s} {p.fields}")

    if peers:
        overall_median = peer_median(peers, "overall_score")
        print(f"\n  peer_median(overall_score) = {overall_median}")


if __name__ == "__main__":
    _cli()
