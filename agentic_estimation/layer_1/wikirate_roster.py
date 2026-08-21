"""
wikirate_roster.py — harvest the Wikirate COMPANY ROSTER (identity only) into
`wikirate_lookup`, so runtime answer-fetching has a local name/identifier index
to resolve against instead of guessing.

WHY IDENTITY-ONLY, AND WHY THIS TABLE MATTERS MORE THAN THE ANSWERS

Our measured blocker is not missing ESG answers, it is missing IDENTIFIERS:
across 685 frozen companies only 15.5% carry a website, 35% an LEI, 16.9% a
Wikidata QID -- and 50% carry NOTHING that could verify a candidate domain.
company_site_resolver.py therefore has to abstain for half the corpus, which is
why a company like Felton Road (which really does publish a report) gets
nothing from us.

A Wikirate company card carries exactly the identifiers we lack:

    wikidata_id, legal_entity_identifier (LEI), sec_central_index_key (CIK),
    open_corporates_id, uk_company_number, ISIN, headquarters, website, alias

So harvesting the roster is an IDENTIFIER-RESOLUTION play, not an ESG-data
play. Answers are fetched lazily per company at runtime (one card fetch), and
the answer columns here stay NULL until then.

API BEHAVIOUR, MEASURED LIVE 2026-08-05. Each point cost a wrong assumption,
so they are recorded with the evidence that corrected it:

  1. A browser User-Agent is REQUIRED -- Cloudflare 403s everything without
     one. This is genuinely Cloudflare, unlike points 2-3.

  2. WITHOUT an API key, limit/offset are capped at 1000 -- and the 403 body
     says so explicitly:
         {"errors": {"items view":
          "limit parameter exceeds maximum for anonymous users (1000)"}}
     Decko checks `limit` and `offset` SEPARATELY (mod/collection/set/abstract/
     paging.rb), which is why limit=700&offset=320 passed while
     limit=50&offset=1050 failed -- page size never mattered. It is a
     PermissionDenied raise, so it is instant and never clears with backoff.
     This was originally misdiagnosed here as a cumulative throttle; the
     status code alone looks identical, the RESPONSE BODY is what
     distinguishes them. Always read the body.

  3. WITH the key (WIKIRATE_API_KEY), that cap disappears, but a second,
     honest server limit appears at offset 5000:
         {"error":"pagination_limit_exceeded",
          "message":"Offset values above 5000 are not supported for this
           endpoint because they create excessive database load"}
     offset=5000 works, offset=5001 does not. So ONE query reaches at most
     ~6000 companies (offset 5000 + limit 1000), against a ~150k roster.

  4. ?page= is silently IGNORED, authenticated or not -- page=1,2,3,5,10 all
     return byte-identical items. So do p/start/skip/from, and so do
     filter[id][gt] and every sort= variant. The API accepts unknown params
     without error rather than rejecting them, which makes parameter guessing
     actively misleading. Verified against wikirate4py's own Cursor, which is
     plain offset increment and would hit the same wall.

  5. THE PARTITION THAT WORKS: filter[country]. Verified -- every returned
     company matched the requested country (20/20 for Germany, India, Japan,
     Brazil) and each country yields a distinct set. It is one of the five
     filters wikirate4py documents (name, company_category, company_group,
     country, company_identifier); of the others, company_group and
     company_category returned 0 items for every value tried.

     Most countries fit under the 6000 budget (Germany exhausts before
     offset 3000). Large ones (India still returning at offset 5000) need a
     name sub-shard, which is why _harvest_slice falls back to filter[name]
     within a country rather than silently truncating.

  6. `headquarters` is often SUB-NATIONAL ("California (United States)",
     "Texas (United States)") -- 103 distinct values in one sample. That is
     why the queried country is stored in its own `country` column: the raw
     headquarters string is not a usable partition key.

USAGE
    python -m agentic_estimation.layer_1.wikirate_roster init      # create table
    python -m agentic_estimation.layer_1.wikirate_roster harvest   # resumable
    python -m agentic_estimation.layer_1.wikirate_roster status
"""

import json
import os
import string
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from agentic_estimation.shared.pipeline_logger import get_logger, log_header

log = get_logger("wikirate_roster")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

_BASE = "https://wikirate.org/Company.json"
_API_KEY = os.getenv("WIKIRATE_API_KEY", "")

# Browser UA is REQUIRED (Cloudflare); X-API-Key lifts the anonymous 1000 cap.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json",
}
if _API_KEY:
    _HEADERS["X-API-Key"] = _API_KEY

_PAGE_LIMIT = 500        # 1000 works but is slow; >=2000 times out server-side
_MAX_OFFSET = 5000       # hard server limit: offset 5001 -> pagination_limit_exceeded
# The endpoint is genuinely slow -- MEASURED: limit=500 takes ~24s with a
# country filter, ~45s without; limit=100 ~6-12s. Roughly 20 rows/second, so a
# full ~150k harvest is a 2-4 HOUR job no matter how it is paced. An artificial
# client-side gap on top of a 24s server round-trip buys nothing, so there is
# none: the request latency IS the rate limiting.
_MIN_GAP = 0.0
_TIMEOUT = 180           # a 500-row page can legitimately take ~45s
_MAX_RETRIES = 3

# Name sub-shards, used ONLY inside a country slice too large for the offset
# budget. Not the primary strategy -- see docstring point 5.
_NAME_SHARDS = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
_NAME_SHARDS += list(string.digits)

# Partition keys. Wikirate has no listable Country card (/Country.json returns
# only the cardtype itself), so this is a standard country list -- unknown
# names simply return 0 items, which costs one request and no correctness.
_COUNTRIES = [
    "United States", "United Kingdom", "China", "India", "Japan", "Germany",
    "France", "Italy", "Spain", "Netherlands", "Belgium", "Switzerland",
    "Austria", "Sweden", "Norway", "Denmark", "Finland", "Ireland", "Portugal",
    "Greece", "Poland", "Czech Republic", "Hungary", "Romania", "Bulgaria",
    "Croatia", "Slovenia", "Slovakia", "Estonia", "Latvia", "Lithuania",
    "Luxembourg", "Iceland", "Malta", "Cyprus",
    "Canada", "Mexico", "Brazil", "Argentina", "Chile", "Colombia", "Peru",
    "Venezuela", "Ecuador", "Uruguay", "Paraguay", "Bolivia", "Costa Rica",
    "Panama", "Guatemala", "Honduras", "Nicaragua", "El Salvador",
    "Dominican Republic", "Jamaica", "Trinidad and Tobago", "Puerto Rico",
    "Australia", "New Zealand",
    "South Korea", "Taiwan", "Hong Kong", "Singapore", "Malaysia", "Thailand",
    "Indonesia", "Philippines", "Vietnam", "Bangladesh", "Pakistan",
    "Sri Lanka", "Nepal", "Myanmar", "Cambodia", "Laos", "Mongolia", "Macau",
    "Russia", "Ukraine", "Turkey", "Israel", "Saudi Arabia",
    "United Arab Emirates", "Qatar", "Kuwait", "Bahrain", "Oman", "Jordan",
    "Lebanon", "Iraq", "Iran", "Egypt", "Morocco", "Tunisia", "Algeria",
    "Libya", "Kazakhstan", "Uzbekistan", "Azerbaijan", "Georgia", "Armenia",
    "Belarus", "Serbia", "Bosnia and Herzegovina", "Albania", "North Macedonia",
    "Moldova",
    "South Africa", "Nigeria", "Kenya", "Ghana", "Ethiopia", "Tanzania",
    "Uganda", "Zambia", "Zimbabwe", "Botswana", "Namibia", "Mozambique",
    "Angola", "Senegal", "Ivory Coast", "Cameroon", "Rwanda", "Malawi",
    "Mauritius", "Madagascar", "Democratic Republic of the Congo", "Sudan",
]

_last_call = [0.0]


def _pace() -> None:
    gap = time.time() - _last_call[0]
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)
    _last_call[0] = time.time()


def _fetch_page(offset: int, country: Optional[str] = None,
                name: Optional[str] = None) -> Optional[list]:
    """One page. Returns [] for a legitimate end-of-slice, None for a failure
    the caller should treat as "this slice is incomplete"."""
    params = {"limit": _PAGE_LIMIT, "offset": offset}
    if country:
        params["filter[country]"] = country
    if name:
        params["filter[name]"] = name
    desc = f"country={country!r} name={name!r} offset={offset}"

    for attempt in range(_MAX_RETRIES):
        _pace()
        try:
            r = requests.get(_BASE, headers=_HEADERS, timeout=_TIMEOUT, params=params)
        except Exception as exc:
            # Connection resets/timeouts are transient -- retry rather than
            # abandoning the slice (an earlier version treated these as
            # terminal and silently truncated shard 'aa' at 200 records).
            log.warning("%s: %s (attempt %d/%d)", desc, type(exc).__name__,
                        attempt + 1, _MAX_RETRIES)
            time.sleep(5 * (attempt + 1))
            continue

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                log.warning("%s: unparseable JSON", desc)
                return None
            return data.get("items", []) if isinstance(data, dict) else []

        if r.status_code == 422:
            # pagination_limit_exceeded -- expected when a slice is bigger than
            # the offset budget. Caller decides whether to sub-shard.
            return None

        if r.status_code == 403:
            body = r.text[:160].replace("\n", " ")
            log.error("%s: 403 -- %s", desc, body)
            return None

        log.warning("%s: HTTP %d (attempt %d/%d)", desc, r.status_code,
                    attempt + 1, _MAX_RETRIES)
        time.sleep(3 * (attempt + 1))

    log.error("%s: failed after %d attempts", desc, _MAX_RETRIES)
    return None


def _first(value):
    """Wikirate returns some identifier fields as a list (a company can carry
    several ISINs/LEIs). Keep the first and stash the rest in `extra`, rather
    than silently dropping them."""
    if isinstance(value, list):
        return (value[0] if value else None), (value[1:] if len(value) > 1 else None)
    return value, None


def _row_from_item(item: dict) -> dict:
    lei, lei_extra = _first(item.get("legal_entity_identifier"))
    isin, isin_extra = _first(item.get("international_securities_identification_number"))
    cik, _ = _first(item.get("sec_central_index_key"))
    oc, _ = _first(item.get("open_corporates_id"))
    ukcn, _ = _first(item.get("uk_company_number"))
    wd, _ = _first(item.get("wikidata_id"))
    aliases = item.get("alias") or []
    if isinstance(aliases, str):
        aliases = [aliases]

    extra = {k: v for k, v in (("lei_additional", lei_extra),
                               ("isin_additional", isin_extra)) if v}

    return {
        "wikirate_id": item.get("id"),
        "name": item.get("name"),
        "url": item.get("url"),
        "website": item.get("website"),
        "headquarters": item.get("headquarters"),
        # Set by the caller to the country we QUERIED by. Deliberately separate
        # from `headquarters`, which is frequently sub-national
        # ("California (United States)") and so cannot serve as a partition key.
        "country": None,
        "aliases": aliases or None,
        "lei": lei,
        "isin": isin,
        "sec_cik": cik,
        "open_corporates_id": oc,
        "uk_company_number": ukcn,
        "wikidata_id": wd,
        "extra": json.dumps(extra) if extra else None,
    }


# ── DB ───────────────────────────────────────────────────────────────────────

def _conn():
    from agentic_estimation.calibration_harness import _db_conn
    return _db_conn()


_DDL = """
CREATE TABLE IF NOT EXISTS wikirate_lookup (
    wikirate_id         BIGINT PRIMARY KEY,
    name                TEXT NOT NULL,
    url                 TEXT,
    website             TEXT,
    headquarters        TEXT,   -- raw, often sub-national ("Texas (United States)")
    country             TEXT,   -- the country we queried by: canonical partition key
    aliases             TEXT[],
    lei                 TEXT,
    isin                TEXT,
    sec_cik             TEXT,
    open_corporates_id  TEXT,
    uk_company_number   TEXT,
    wikidata_id         TEXT,
    extra               JSONB,
    -- Answer columns stay NULL until a runtime fetch fills them in; the roster
    -- harvest deliberately never populates these.
    answers_fetched_at  TIMESTAMPTZ,
    answer_count        INTEGER,
    first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_wikirate_name       ON wikirate_lookup (LOWER(name));
CREATE INDEX IF NOT EXISTS idx_wikirate_country    ON wikirate_lookup (country)     WHERE country IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wikirate_lei        ON wikirate_lookup (lei)        WHERE lei IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wikirate_wikidata   ON wikirate_lookup (wikidata_id) WHERE wikidata_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_wikirate_cik        ON wikirate_lookup (sec_cik)    WHERE sec_cik IS NOT NULL;
"""

_UPSERT = """
INSERT INTO wikirate_lookup
    (wikirate_id, name, url, website, headquarters, country, aliases, lei, isin,
     sec_cik, open_corporates_id, uk_company_number, wikidata_id, extra)
VALUES (%(wikirate_id)s, %(name)s, %(url)s, %(website)s, %(headquarters)s,
        %(country)s, %(aliases)s, %(lei)s, %(isin)s, %(sec_cik)s,
        %(open_corporates_id)s, %(uk_company_number)s, %(wikidata_id)s, %(extra)s)
ON CONFLICT (wikirate_id) DO UPDATE SET
    name               = EXCLUDED.name,
    url                = COALESCE(EXCLUDED.url, wikirate_lookup.url),
    website            = COALESCE(EXCLUDED.website, wikirate_lookup.website),
    headquarters       = COALESCE(EXCLUDED.headquarters, wikirate_lookup.headquarters),
    country            = COALESCE(EXCLUDED.country, wikirate_lookup.country),
    aliases            = COALESCE(EXCLUDED.aliases, wikirate_lookup.aliases),
    lei                = COALESCE(EXCLUDED.lei, wikirate_lookup.lei),
    isin               = COALESCE(EXCLUDED.isin, wikirate_lookup.isin),
    sec_cik            = COALESCE(EXCLUDED.sec_cik, wikirate_lookup.sec_cik),
    open_corporates_id = COALESCE(EXCLUDED.open_corporates_id, wikirate_lookup.open_corporates_id),
    uk_company_number  = COALESCE(EXCLUDED.uk_company_number, wikirate_lookup.uk_company_number),
    wikidata_id        = COALESCE(EXCLUDED.wikidata_id, wikirate_lookup.wikidata_id),
    extra              = COALESCE(EXCLUDED.extra, wikirate_lookup.extra),
    updated_at         = NOW()
"""


def init_table() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        log.info("wikirate_lookup ready")
        print("wikirate_lookup created/verified")
    finally:
        conn.close()


def _save(rows: list) -> int:
    if not rows:
        return 0
    conn = _conn()
    try:
        with conn.cursor() as cur:
            for row in rows:
                if row["wikirate_id"] is None or not row["name"]:
                    continue
                cur.execute(_UPSERT, row)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def _existing_ids() -> set:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT wikirate_id FROM wikirate_lookup")
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def _harvest_slice(seen: set, country: Optional[str] = None,
                   name: Optional[str] = None) -> tuple:
    """Page through one slice. Returns (new_rows, exhausted).

    exhausted=False means the slice hit the offset ceiling and is INCOMPLETE --
    the caller must sub-shard it. Reporting that honestly is the whole point:
    a truncated slice is indistinguishable from a complete one in the output
    unless it is tracked explicitly.
    """
    offset, new_count = 0, 0
    label = f"{country or 'ALL'}{'/' + name if name else ''}"
    while offset <= _MAX_OFFSET:
        t0 = time.time()
        items = _fetch_page(offset, country=country, name=name)
        if items is None:
            return new_count, False          # ceiling or hard failure
        if not items:
            return new_count, True           # genuinely exhausted

        rows = []
        for it in items:
            wid = it.get("id")
            if wid is None:
                continue
            if wid not in seen:
                new_count += 1
                seen.add(wid)
            row = _row_from_item(it)
            row["country"] = country         # canonical partition key
            rows.append(row)
        _save(rows)

        # Per-PAGE progress. Printing only per country meant a slow country
        # looked identical to a hung process for minutes at a time -- each page
        # is a ~24s round-trip, so silence between countries is not acceptable
        # feedback.
        print(f"      {label} offset={offset}: {len(items)} fetched, "
              f"{new_count} new so far ({time.time() - t0:.0f}s)", flush=True)

        if len(items) < _PAGE_LIMIT:
            return new_count, True           # short page = last page
        offset += _PAGE_LIMIT

    return new_count, False                  # ran out of offset budget


def harvest(countries: Optional[list] = None) -> None:
    """Enumerate the roster country by country, sub-sharding oversized slices.

    Resumable by construction: the upsert is idempotent and already-seen ids
    are skipped in memory, so re-running after an interruption costs only HTTP
    time, never duplicate rows.
    """
    if not _API_KEY:
        # Hard stop, not a warning. Measured: anonymous requests past the cap
        # return HTTP 403, so the harvest does not fail -- it just silently
        # stops short, and the truncated roster is indistinguishable from a
        # complete one on inspection.
        raise RuntimeError(
            "WIKIRATE_API_KEY not set -- anonymous requests are capped at "
            "offset 1000 (403 beyond it) and the harvest would be silently "
            "truncated. Set the key in .env before harvesting."
        )

    countries = countries or _COUNTRIES
    log_header(log, "Wikirate roster harvest",
               countries=len(countries), page=_PAGE_LIMIT, authed=bool(_API_KEY))

    seen = _existing_ids()
    print(f"{len(seen)} companies already in wikirate_lookup")
    print(f"harvesting {len(countries)} country slices "
          f"({'authenticated' if _API_KEY else 'ANONYMOUS'})\n")

    total_new, truncated = 0, []

    for ci, country in enumerate(countries, 1):
        got, exhausted = _harvest_slice(seen, country=country)

        if not exhausted:
            # Too big for the offset budget -- split by name rather than
            # silently dropping the remainder.
            log.info("country %r exceeded the offset budget -- sub-sharding by name",
                     country)
            sub_new = 0
            for shard in _NAME_SHARDS:
                n, ok = _harvest_slice(seen, country=country, name=shard)
                sub_new += n
                if not ok:
                    truncated.append(f"{country}/{shard}")
            got += sub_new
            print(f"  [{ci}/{len(countries)}] {country}: +{got} new "
                  f"(sub-sharded, total {len(seen)})", flush=True)
        else:
            print(f"  [{ci}/{len(countries)}] {country}: +{got} new "
                  f"(total {len(seen)})", flush=True)

        total_new += got

    print(f"\nharvest done: {total_new} new companies, {len(seen)} total")
    if truncated:
        print(f"WARNING: {len(truncated)} slices still truncated: {truncated[:10]}")


def status() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM wikirate_lookup")
            total = cur.fetchone()[0]
            print(f"wikirate_lookup: {total} companies")
            if not total:
                return
            for col in ("website", "lei", "wikidata_id", "sec_cik",
                        "open_corporates_id", "uk_company_number", "isin",
                        "headquarters", "country", "aliases"):
                cur.execute(f"SELECT COUNT(*) FROM wikirate_lookup WHERE {col} IS NOT NULL")
                c = cur.fetchone()[0]
                print(f"  {col:20s} {c:7d}  ({100.0 * c / total:5.1f}%)")
            cur.execute("SELECT COUNT(*) FROM wikirate_lookup WHERE answers_fetched_at IS NOT NULL")
            print(f"  {'answers fetched':20s} {cur.fetchone()[0]:7d}")
            cur.execute("SELECT country, COUNT(*) FROM wikirate_lookup "
                        "WHERE country IS NOT NULL GROUP BY country "
                        "ORDER BY COUNT(*) DESC LIMIT 12")
            rows = cur.fetchall()
            if rows:
                print("\n  top countries:")
                for c, n in rows:
                    print(f"    {c:28s} {n:7d}")
    finally:
        conn.close()


def _cli() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "init":
        init_table()
    elif cmd == "harvest":
        countries = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        harvest(countries)
    elif cmd == "status":
        status()
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli()
