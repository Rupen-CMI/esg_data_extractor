"""SRN (srnav.com) report lookup: company name -> sustainability report PDF.

ON-DEMAND, not a bulk harvest. Nothing is downloaded until a company we are
actually scoring turns out to have a report. The only bulk step is the INDEX --
one page fetch that yields every company/report pair at once -- and it is cached
to disk so repeated lookups cost nothing.

WHY THE HTML PAGE AND NOT THE API: srnav.com/reports is server-rendered with the
full record set embedded as a JS array, including fields the REST API does not
return -- lei, isin, sector, country, and pdfpage_sust_start/end (the page range
where the sustainability statement sits inside a combined annual report). The
LEI/ISIN in particular are what let us match a company without trusting names
alone.

TWO SRN DATASETS, DO NOT CONFLATE:
  * this page    -- 1,898 reports / 1,101 companies, FY2024-25, CSRD-era, with
                    identifiers. Direct `original_link` to the publisher's PDF.
  * the REST API -- 12,335 documents / 1,994 companies, 2010-2023 historical,
                    served through SRN's own /download endpoint.
This module uses the page. The API remains available if historical depth is
ever wanted.

MEASURED (2026-08-06):
  index fetch          1.0 MB, ~3-15s, no auth
  parsed               1,898 records, 1,101 companies, 1,691 with a .pdf link
  identifiers          1,786 ISIN, 923 LEI
  direct download      6/8 sampled links returned real PDFs, ~3s each
                       (2 failed with ConnectionError -- publisher hosts, not SRN)
  coverage             France 273, Germany 263, Italy 167, Finland 145, NL 138

KNOWN LIMIT -- read before expecting this to move a score. Overlap with our
scored corpora is near zero:
    heldout250 0/193 | abl tune 3/373 | abl holdout 3/94 | bcorp 1/421
SRN indexes EU-listed large caps; our ground truth is B Corp SMEs. This source
is correct and cheap, and it will do nothing for the current corpora. It becomes
valuable when a listed population is scored.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("srn_report_source")

_PAGE_URL = "https://www.srnav.com/reports"
_INDEX_PATH = Path("raw_esg_data/srn_index.json")
# Any CSV exported from srnav.com's own export button, dropped into raw_esg_data.
_CSV_PATHS = sorted(Path("raw_esg_data").glob("sustainability-reports-*.csv"))
_INDEX_TTL_S = 14 * 24 * 3600          # CSRD filings move slowly; refetch fortnightly

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA}
_TIMEOUT = 180

# The page embeds records as JS object literals with UNQUOTED keys, so this is
# not JSON and json.loads cannot be used. The `(?:(?!\{id:").)*?` guard is what
# makes it safe: it forbids the wildcard from crossing into the NEXT record,
# which a plain `.*?` does whenever a field is missing -- silently pairing one
# company's link with another company's name.
_REC_RE = re.compile(
    r'\{id:"[0-9a-f-]{36}",year:"(\d{4})",type:"([^"]*)"'
    r'(?:(?!\{id:").)*?original_link:"([^"]*)"'
    r'(?:(?!\{id:").)*?company:\{id:"[0-9a-f-]{36}",'
    r'lei:(null|"[^"]*"),isin:(null|"[^"]*"),'
    r'name:"([^"]*)",sector:"([^"]*)",country:"([^"]*)"',
    re.S)

_PAGES_RE = re.compile(r'pdfpage_sust_start:"(\d+)",pdfpage_sust_end:"(\d+)"')


def _unq(v: str) -> Optional[str]:
    return None if v == "null" else v.strip('"') or None


def _parse(html: str) -> list[dict]:
    out = []
    for m in _REC_RE.finditer(html):
        year, typ, link, lei, isin, name, sector, country = m.groups()
        seg = html[m.start():m.end()]
        pg = _PAGES_RE.search(seg)
        out.append({
            "name": name, "country": country, "sector": sector,
            "lei": _unq(lei), "isin": _unq(isin),
            "year": year, "type": typ, "url": link,
            "sust_pages": [int(pg.group(1)), int(pg.group(2))] if pg else None,
        })
    return out


def _load_csv_export() -> list[dict]:
    """SRN's own CSV export, when present -- preferred over scraping.

    srnav.com has an export button; the resulting CSV is the same 1,900
    rows / 1,101 companies this module scrapes, but parsed by SRN rather than
    by a regex over their markup, so it cannot break when their page changes.

    It carries Company/Country/Sector/Industry/ISIN/Year/Published/
    CSRD Compliant/Report Year/Pages/Report Link.

    ONE THING IT LACKS: `Pages` is the LENGTH of the sustainability section,
    not its location -- verified, Vestas Pages=77 == scraped span 132-56+1.
    Without the start offset we cannot jump to the section, and jumping to it
    is the single biggest quality win available (Vestas G density 2.5 -> 7.4).
    So the scraped index is still consulted for page ranges; see
    report_collector._srn_page_ranges.
    """
    import csv
    import io

    if not _CSV_PATHS:
        return []
    path = max(_CSV_PATHS, key=lambda p: p.stat().st_mtime)
    try:
        txt = path.read_bytes().decode("utf-8-sig", errors="replace")
        rows = list(csv.DictReader(io.StringIO(txt)))
    except Exception as exc:
        log.warning("SRN csv export unreadable (%s): %s", path.name, exc)
        return []

    out = []
    for r in rows:
        link = (r.get("Report Link") or "").strip()
        if not link:
            continue
        out.append({
            "name": (r.get("Company") or "").strip(),
            "country": (r.get("Country") or "").strip(),
            "sector": (r.get("Sector") or "").strip(),
            "industry": (r.get("Industry") or "").strip(),
            "lei": None,
            "isin": (r.get("ISIN") or "").strip() or None,
            "year": (r.get("Year") or "").strip(),
            "type": "sustainability report",
            "url": link,
            "csrd": (r.get("CSRD Compliant") or "").strip(),
            "sust_page_count": int(r["Pages"]) if (r.get("Pages") or "").strip().isdigit() else None,
            "sust_pages": None,          # start offset is not in the export
            "source": "csv_export",
        })
    log.info("SRN csv export: %d reports, %d companies (%s)",
             len(out), len({x["name"] for x in out}), path.name)
    return out


def fetch_index(force: bool = False) -> list[dict]:
    """The full company->report index.

    Prefers SRN's own CSV export when one is on disk (no network, no regex
    against their markup), and falls back to scraping the page.
    """
    csv_rows = _load_csv_export()
    if csv_rows and not force:
        return csv_rows

    if not force and _INDEX_PATH.exists():
        age = time.time() - _INDEX_PATH.stat().st_mtime
        if age < _INDEX_TTL_S:
            try:
                return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass                    # corrupt cache -> refetch

    last: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(_PAGE_URL, headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            recs = _parse(r.text)
            if not recs:
                raise RuntimeError("index page parsed to 0 records "
                                   "-- page structure probably changed")
            _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            _INDEX_PATH.write_text(json.dumps(recs, indent=1), encoding="utf-8")
            log.info("SRN index: %d reports, %d companies",
                     len(recs), len({x["name"] for x in recs}))
            return recs
        except Exception as exc:        # 1 MB page; transient read timeouts seen
            last = exc
            log.warning("SRN index fetch attempt %d: %s", attempt + 1, type(exc).__name__)
            time.sleep(4 * (attempt + 1))

    if _INDEX_PATH.exists():            # stale beats nothing
        log.warning("using STALE SRN index (%s)", last)
        try:
            return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


_by_name: Optional[dict] = None
_by_id: Optional[dict] = None


def _norm(name: str) -> str:
    from agentic_estimation.shared.company_name_utils import normalize_company_name
    try:
        return normalize_company_name(name).lower()
    except Exception:
        return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _build_lookups() -> None:
    global _by_name, _by_id
    if _by_name is not None:
        return
    _by_name, _by_id = {}, {}
    for r in fetch_index():
        _by_name.setdefault(_norm(r["name"]), []).append(r)
        for k in ("lei", "isin"):
            if r.get(k):
                _by_id.setdefault(r[k].upper(), []).append(r)


def lookup(company: str, lei: str = "", isin: str = "",
           prefer_type: str = "sustainability report") -> list[dict]:
    """Reports for one company, newest first. [] when not indexed.

    Identifier match is tried FIRST and is authoritative -- an ISIN/LEI cannot
    collide the way a name can. Name matching is exact-on-normalised only: no
    fuzzy fallback, because attributing another company's report is a worse
    outcome than returning nothing (the Dialog/Lion/BGF lesson).
    """
    _build_lookups()
    hits: list[dict] = []
    for ident in (lei, isin):
        if ident and ident.upper() in _by_id:
            hits = list(_by_id[ident.upper()])
            break
    if not hits:
        hits = list(_by_name.get(_norm(company), []))
    if not hits:
        return []
    hits.sort(key=lambda r: (r.get("type", "").lower() != prefer_type.lower(),
                             -int(r.get("year") or 0)))
    return hits


_scraped_by_name: Optional[dict] = None


def scraped_records(company: str) -> list[dict]:
    """Records from the SCRAPED index only -- the sole carrier of `sust_pages`.

    Separate from lookup() on purpose: lookup() prefers the CSV export (more
    robust, SRN-parsed), but the export omits the sustainability section's
    START page, and that offset is what lets the parser jump straight to the
    ESG content instead of reading financial front-matter. Returns [] when no
    scraped index has been built.
    """
    global _scraped_by_name
    if _scraped_by_name is None:
        _scraped_by_name = {}
        if _INDEX_PATH.exists():
            try:
                for r in json.loads(_INDEX_PATH.read_text(encoding="utf-8")):
                    _scraped_by_name.setdefault(_norm(r["name"]), []).append(r)
            except Exception as exc:
                log.debug("scraped index unreadable: %s", exc)
    return _scraped_by_name.get(_norm(company), [])


_DEST_DIR = Path(os.getenv("ESG_REPORT_DIR", "raw_esg_data/company_esg_reports"))


def download(rec: dict, dest_dir: Path = None) -> dict:
    """Fetch one indexed report to disk, reusing the vetted downloader.

    report_coverage.download_pdf already handles magic-byte verification, the
    size cap, per-host limiting and atomic replace -- all of which this needs
    and none of which is worth reimplementing.
    """
    from calibration.report_coverage import download_pdf
    return download_pdf(rec["name"], rec["url"], dest_dir=dest_dir or _DEST_DIR)


def fetch_srn_report(company: str, lei: str = "", isin: str = "") -> Optional[str]:
    """End-to-end: look up, download the best report, return its path or None."""
    for rec in lookup(company, lei=lei, isin=isin)[:2]:
        res = download(rec)
        if res.get("path") and not res.get("error"):
            log.info("SRN report for %s: %s (%s %s)", company,
                     Path(res["path"]).name, rec["year"], rec["type"])
            return res["path"]
        log.info("SRN download failed for %s: %s", company, res.get("error"))
    return None
