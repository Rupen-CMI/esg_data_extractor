"""
climate_trace_harvester.py — batch harvester for Climate TRACE v7 API data.

Simple ETL: fetch data from the public, no-key-needed Climate TRACE v7 API
(api.climatetrace.org) fully into memory, THEN open a DB connection and
write it. The DB connection only exists for the brief write window — never
held open across the (slow) API calls — because Neon (serverless Postgres)
closes connections left idle, which is what caused "SSL connection has been
closed unexpectedly" in earlier drafts that opened the connection first.

Two tables:
  1. climate_trace_owners            -- name->id lookup, ~14,500 rows.
        Enumerated via GET /v7/owners?name=<a-z single letter>, deduped by id
        (no "list all owners" endpoint exists; a-z substring search reaches
        the same ~14,500 scale as Climate TRACE's own published count).
  2. climate_trace_owner_emissions   -- per-facility emissions for a known
        owner id (GET /v7/sources?ownerIds=).
  3. climate_trace_country_emissions -- country x sector x subsector totals
        (GET /v7/sources/emissions?gadmId=&year=). sector/subsector NULL =
        that country's grand total. Worldwide sector totals are a GROUP BY
        over this table (climate_trace_global_sector_emissions view), not
        separately harvested.

Idempotent: every write is an upsert on the natural unique key, safe to re-run.

CLI:
    python -m agentic_estimation.climate_trace_harvester owners
    python -m agentic_estimation.climate_trace_harvester owner-emissions --year 2024
    python -m agentic_estimation.climate_trace_harvester country-emissions --year 2024
    python -m agentic_estimation.climate_trace_harvester all --year 2024
"""

import argparse
import os
import string
import time
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import psycopg2
from psycopg2.extras import execute_batch
import requests
from dotenv import load_dotenv

from agentic_estimation.shared.pipeline_logger import get_logger, log_header

log = get_logger("climate_trace_harvester")
load_dotenv()

_API_BASE = "https://api.climatetrace.org"
_HEADERS = {"User-Agent": "ESG-Data-Extractor/1.0 (research-pipeline)"}
_TIMEOUT = 30
_MIN_GAP = 1.0  # polite pacing between API calls (public free API, batch job)


def _db_conn():
    """A fresh, short-lived Postgres connection. Callers should open it right
    before writing and close it right after — never hold it open across API
    calls (Neon closes idle connections)."""
    db_url = os.getenv("ASYNC_DB_URL", os.getenv("DB_URL", ""))
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    sslmode = qs.pop("ssl", [None])[0]
    clean_qs = urlencode({k: v[0] for k, v in qs.items()})
    if sslmode:
        clean_qs = (clean_qs + "&" if clean_qs else "") + f"sslmode={sslmode}"
    return psycopg2.connect(urlunparse(parsed._replace(query=clean_qs)))


def _write_rows(sql: str, rows: list[tuple]) -> int:
    """Open a fresh connection, batch-write rows, commit, close. All DB work
    for a harvest phase funnels through here so no connection is ever held
    open during API calls."""
    if not rows:
        return 0
    conn = _db_conn()
    try:
        cur = conn.cursor()
        execute_batch(cur, sql, rows, page_size=500)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _get(path: str, params: dict) -> Optional[dict]:
    try:
        r = requests.get(f"{_API_BASE}{path}", params=params, headers=_HEADERS, timeout=_TIMEOUT)
        time.sleep(_MIN_GAP)
        if r.status_code == 404:
            return None  # e.g. "no owners matched" -- expected, not an error
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        log.warning("GET %s %s -> %s", path, params, exc)
        return None


# ── 1. Owners ─────────────────────────────────────────────────────────────────

_OWNERS_SQL = """
    INSERT INTO climate_trace_owners (owner_id, name, fetched_at)
    VALUES (%s, %s, now())
    ON CONFLICT (owner_id) DO UPDATE SET name = EXCLUDED.name, fetched_at = now()
"""


def harvest_owners() -> int:
    """Enumerate all owners via a-z substring search, then write once."""
    log_header(log, "Climate TRACE Harvester", phase="owners")

    all_owners: dict[str, str] = {}  # id -> name
    for letter in string.ascii_lowercase:
        data = _get("/v7/owners", {"name": letter, "limit": 100000})
        if data:
            for row in data:
                all_owners[row["id"]] = row["name"]
        log.info("letter '%s': cumulative %d owners", letter, len(all_owners))

    n = _write_rows(_OWNERS_SQL, list(all_owners.items()))
    log.info("harvest_owners: wrote %d owners to DB", n)
    return n


# ── 2. Owner emissions ────────────────────────────────────────────────────────

_OWNER_EMISSIONS_SQL = """
    INSERT INTO climate_trace_owner_emissions
        (owner_id, source_id, source_name, country_iso3, sector, subsector,
         asset_type, gas, emissions_quantity, activity, activity_units,
         capacity, capacity_units, year, fetched_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (owner_id, source_id, gas, year) DO UPDATE SET
        source_name = EXCLUDED.source_name,
        emissions_quantity = EXCLUDED.emissions_quantity,
        activity = EXCLUDED.activity,
        capacity = EXCLUDED.capacity,
        fetched_at = now()
"""


def _load_owner_ids(limit_owners: Optional[int], year: int, skip_done: bool) -> list[str]:
    conn = _db_conn()
    try:
        cur = conn.cursor()
        if skip_done:
            # Resume support: a prior run may have died partway (e.g. network
            # drop) -- every completed owner already has a row for this year
            # (even owners with zero facilities are worth marking, but today
            # we only insert rows when data exists, so this is "at least
            # attempted with results", not "attempted at all"; good enough to
            # avoid redoing the ~2600 owners already confirmed done without
            # extra bookkeeping).
            cur.execute(
                "SELECT owner_id FROM climate_trace_owners "
                "WHERE owner_id NOT IN (SELECT DISTINCT owner_id FROM climate_trace_owner_emissions WHERE year = %s) "
                "ORDER BY owner_id",
                (year,),
            )
        else:
            cur.execute("SELECT owner_id FROM climate_trace_owners ORDER BY owner_id")
        owner_ids = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()
    return owner_ids[:limit_owners] if limit_owners else owner_ids


def harvest_owner_emissions(year: int, limit_owners: Optional[int] = None,
                             flush_every: int = 200, skip_done: bool = True) -> int:
    """
    For each cached owner, fetch its facilities' emissions. Accumulates rows
    in memory and flushes to the DB every `flush_every` owners via a fresh
    short-lived connection, so no connection is held open across the long
    API-call loop (~14,500 owners at full scale).

    skip_done=True (default): skip owners that already have at least one
    emissions row for this year -- lets a run resume after an interruption
    (e.g. a network drop) without re-fetching owners already completed.
    Pass skip_done=False to force a full refresh of every owner.
    """
    log_header(log, "Climate TRACE Harvester", phase="owner-emissions", year=year)
    owner_ids = _load_owner_ids(limit_owners, year, skip_done)
    log.info("fetching owner emissions for %d owners (year=%d, skip_done=%s)", len(owner_ids), year, skip_done)

    rows_written = 0
    buffer: list[tuple] = []
    for i, owner_id in enumerate(owner_ids):
        data = _get("/v7/sources", {"ownerIds": owner_id, "year": year, "limit": 500})
        if data:
            for src in data:
                buffer.append((
                    owner_id, src.get("id"), src.get("name"), src.get("country"),
                    src.get("sector"), src.get("subsector"), src.get("assetType"),
                    src.get("gas", "co2e_100yr"), src.get("emissionsQuantity"),
                    src.get("activity"), src.get("activityUnits"),
                    src.get("capacity"), src.get("capacityUnits"), year,
                ))
        if (i + 1) % flush_every == 0 or i == len(owner_ids) - 1:
            if buffer:
                rows_written += _write_rows(_OWNER_EMISSIONS_SQL, buffer)
                buffer = []
            log.info("progress %d/%d owners, %d facility rows written", i + 1, len(owner_ids), rows_written)

    log.info("harvest_owner_emissions: wrote %d facility rows to DB", rows_written)
    return rows_written


# ── 3. Country emissions ──────────────────────────────────────────────────────

_COUNTRY_EMISSIONS_SQL = """
    INSERT INTO climate_trace_country_emissions
        (country_iso3, sector, subsector, gas, emissions_quantity, percentage_of_total, year, fetched_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, now())
    ON CONFLICT (country_iso3, sector, subsector, gas, year) DO UPDATE SET
        emissions_quantity = EXCLUDED.emissions_quantity,
        percentage_of_total = EXCLUDED.percentage_of_total,
        fetched_at = now()
"""


def _country_rows(country_iso3: str, year: int, response: dict) -> list[tuple]:
    """Flatten one /v7/sources/emissions response into rows: grand total
    (sector/subsector NULL), per-sector rows, and per-subsector rows."""
    rows = []
    g = "co2e_100yr"
    for s in response.get("totals", {}).get("summaries", []):
        rows.append((country_iso3, None, None, s.get("gas", g),
                     s.get("emissionsQuantity"), s.get("percentage"), year))
    for s in response.get("sectors", {}).get("summaries", []):
        rows.append((country_iso3, s.get("sector"), None, s.get("gas", g),
                     s.get("emissionsQuantity"), s.get("percentage"), year))
    for s in response.get("subsectors", {}).get("summaries", []):
        rows.append((country_iso3, s.get("sector"), s.get("subsector"), s.get("gas", g),
                     s.get("emissionsQuantity"), s.get("percentage"), year))
    return rows


def harvest_country_emissions(year: int, flush_every: int = 25) -> int:
    """
    Fetch every country's sector/subsector emissions (~252 calls), buffering
    rows and flushing to the DB every `flush_every` countries via a fresh
    short-lived connection.
    """
    log_header(log, "Climate TRACE Harvester", phase="country-emissions", year=year)

    countries = _get("/v7/definitions/countries", {}) or []
    log.info("fetching sector emissions for %d countries (year=%d)", len(countries), year)

    rows_written = 0
    buffer: list[tuple] = []
    for i, c in enumerate(countries):
        iso3 = c.get("id")
        if not iso3:
            continue
        resp = _get("/v7/sources/emissions", {"year": year, "gadmId": iso3})
        if resp:
            buffer.extend(_country_rows(iso3, year, resp))
        if (i + 1) % flush_every == 0 or i == len(countries) - 1:
            if buffer:
                rows_written += _write_rows(_COUNTRY_EMISSIONS_SQL, buffer)
                buffer = []
            log.info("progress %d/%d countries, %d rows written", i + 1, len(countries), rows_written)

    log.info("harvest_country_emissions: wrote %d rows to DB", rows_written)
    return rows_written


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Harvest Climate TRACE v7 data into local DB cache.")
    ap.add_argument("phase", choices=["owners", "owner-emissions", "country-emissions", "all"])
    ap.add_argument("--year", type=int, default=2024, help="emissions year (default 2024)")
    ap.add_argument("--limit-owners", type=int, default=None, help="cap owner-emissions harvest for testing")
    args = ap.parse_args()

    if args.phase in ("owners", "all"):
        print(f"Owners harvested: {harvest_owners()}")
    if args.phase in ("owner-emissions", "all"):
        print(f"Owner-emissions rows: {harvest_owner_emissions(args.year, limit_owners=args.limit_owners)}")
    if args.phase in ("country-emissions", "all"):
        print(f"Country-emissions rows: {harvest_country_emissions(args.year)}")


if __name__ == "__main__":
    _cli()
