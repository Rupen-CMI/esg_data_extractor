"""
sec_filings.py — shared SEC EDGAR access for Layer 1 collectors.

Extracted from facility_extractor.py (Phase 1) into a shared module so
governance_collector.py can reuse the same CIK-resolution and
document-fetch logic instead of duplicating it — both Facility and
Governance evidence live in the same class of legally-mandated, free,
no-API-key-needed SEC filings.

Filing types used across collectors:
  - 10-K  "Item 2. Properties"        -> facility_extractor.py
  - 10-K  "Item 3. Legal Proceedings" -> governance_collector.py
  - DEF 14A (proxy statement)         -> governance_collector.py
    (board composition, committee structure, exec comp — the single
    richest structured governance source for a US-listed company)

Section-extraction detail, verified against a real filing (Nvidia FY2026
10-K) during Phase 1: a section heading like "Item 2. Properties" appears
TWICE in a 10-K — once in the table of contents (page-number reference
only, no real content), once as the actual section header with real prose.
The TOC occurrence is always first. extract_section() always returns text
from the LAST occurrence, never the first, to avoid returning TOC noise as
if it were real evidence. This same risk applies to DEF 14A sub-section
headings, so the same last-occurrence logic is used there too.
"""

import re
import threading
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.layer_1.signal_agent import _RateLimiter, _get

log = get_logger("sec_filings")

_TIMEOUT = 20

# SEC publishes a 10 req/s ceiling. The previous 1/6s+0.15 gap (~4.1 req/s)
# looked like comfortable headroom and was NOT: a 150-company run took 4 HTTP
# 429s from www.sec.gov and aborted at 45.
#
# The reason the published ceiling is misleading here is nesting. This module is
# called from inside the pipeline's worker pool (5 threads), and
# fetch_sec_fulltext_signals opens its OWN pool of 6 threads per company, each
# issuing a search plus up to 2 document fetches across 16 phrases. So up to 30
# threads contend for this one limiter and sustain the maximum rate for minutes
# at a stretch -- a burst profile the "10 req/s" figure does not describe.
#
# 0.5s+0.4s (~1.4 req/s) trades wall-clock for not being throttled. SEC full-text
# is our only entity-CERTAIN source (CIK-scoped, no name matching to get wrong),
# so losing it to a block costs more than the extra minutes.
_SEC_LIMITER = _RateLimiter(min_gap=0.5, jitter=0.4)


def _sec_get(url: str, params: Optional[dict] = None, timeout: int = _TIMEOUT):
    """Rate-limited SEC fetch. Use for every sec.gov / efts.sec.gov request."""
    _SEC_LIMITER.wait()
    return _get(url, params=params, timeout=timeout)

# EDGAR full-text search. Covers 2001-present across ALL form types and returns
# the filings whose text contains a phrase -- unlike the submissions index,
# which only tells you a filing exists, not what is in it.
_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"

# Official name -> CIK map (~10.4k US registrants, ~800KB). Fetched once per
# process. Preferred over resolve_cik()'s browse-edgar name search, which is
# fuzzy and returns whichever registrant EDGAR ranks first for a query string.
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Words that may trail a registrant's name without changing WHICH company it is --
# legal forms, geographies, and corporate-unit words. Used to decide whether
# "<registrant> <extra words>" is the same company with qualifiers
# ("Bonduelle Americas US" -> Bonduelle) or a different company that merely shares
# a first word ("Wise Investments Ltd" is not the registrant "Wise").
_NAME_QUALIFIERS = frozenset({
    # legal forms
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lp",
    "llp", "ltd", "limited", "plc", "gmbh", "ag", "sa", "nv", "bv", "spa", "srl",
    "pty", "pvt", "private", "kk", "as", "ab", "oyj", "oy",
    # geographies / units
    "usa", "us", "u.s.", "america", "americas", "north", "south", "east", "west",
    "europe", "european", "asia", "asia-pacific", "apac", "international", "global",
    "worldwide", "uk", "canada", "india", "japan", "china", "australia", "brasil",
    "brazil", "mexico", "deutschland", "france", "italia",
    # corporate structure
    "holdings", "holding", "group", "groupe", "grp", "sub", "subsidiary",
    "the", "and", "of",
})
_cik_map: Optional[dict[str, str]] = None
_cik_map_lock = threading.Lock()


def resolve_cik(company: str) -> Optional[str]:
    """
    Resolve a company name to its SEC CIK via EDGAR's company search (atom feed).
    Returns None if no match — the common case for non-US-listed companies,
    NOT an error.
    """
    r = _sec_get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={"action": "getcompany", "company": company, "type": "10-K",
                "dateb": "", "owner": "include", "count": "5", "output": "atom"},
    )
    if not r:
        return None
    m = re.search(r"<cik>(\d+)</cik>", r.text)
    return m.group(1) if m else None


def _today_iso() -> str:
    """Today as YYYY-MM-DD, for EFTS's required enddt bound."""
    from datetime import date
    return date.today().isoformat()


def _normalize_name(name: str) -> str:
    """Lowercase, strip punctuation and legal-form suffixes, for CIK matching.
    "NIKE, Inc." and "Nike Inc" must collapse to the same key."""
    n = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    n = re.sub(
        r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|"
        r"lp|holdings?|group|the)\b", " ", n)
    return " ".join(n.split())


def _load_cik_map() -> dict[str, str]:
    """Fetch and index SEC's official ticker->CIK file. Keyed by normalized
    company name AND by ticker, so either resolves. Cached for the process;
    returns {} on failure so callers degrade to resolve_cik()."""
    global _cik_map
    with _cik_map_lock:
        if _cik_map is not None:
            return _cik_map
        _cik_map = {}
        r = _sec_get(_TICKERS_URL, timeout=_TIMEOUT)
        if not r:
            log.warning("company_tickers.json unavailable — CIK lookup falls back to browse-edgar")
            return _cik_map
        try:
            for row in (r.json() or {}).values():
                cik = str(row.get("cik_str") or "").zfill(10)
                title = row.get("title") or ""
                ticker = (row.get("ticker") or "").lower()
                if not cik:
                    continue
                key = _normalize_name(title)
                if key:
                    _cik_map.setdefault(key, cik)
                # Tickers are indexed under a "ticker:" prefix, NOT bare. A bare
                # ticker key collides with ordinary company names: our corpus
                # contains B Corps called "Mine" and "FEED", which matched the
                # tickers MINE (Mayfair Gold) and FEED (EnVue Medical) and inherited
                # their CIKs. Callers who genuinely hold a ticker can look up
                # "ticker:<sym>"; a company NAME must never resolve this way.
                if ticker:
                    _cik_map.setdefault(f"ticker:{ticker}", cik)
            log.info("loaded CIK map: %d keys", len(_cik_map))
        except (ValueError, AttributeError) as exc:
            log.warning("company_tickers.json parse failed: %s", exc)
        return _cik_map


def cik_for_company(company: str) -> Optional[str]:
    """Zero-padded 10-digit CIK for a company name, or None.

    Tries the official name map first (exact, and prefix-matched for names
    that carry extra qualifiers like "Bonduelle Americas US"), then falls
    back to EDGAR's fuzzy name search. Returning None is the normal outcome
    for the many non-US-listed companies in our corpus, not an error.

    The 10-digit zero-padding is REQUIRED by EFTS: verified live that
    ciks=320187 returns 0 hits while ciks=0000320187 returns 10 for the same
    query. An unpadded CIK fails silently rather than erroring.
    """
    key = _normalize_name(company)
    if not key:
        return None
    cmap = _load_cik_map()
    if key in cmap:
        return cmap[key]
    # "Bonduelle Americas US" -> match registrant "Bonduelle": the QUERY may carry
    # extra qualifiers beyond a registrant's name.
    #
    # The reverse direction is NOT safe and was removed 2026-08-05. Allowing a
    # registrant to be longer than the query lets any short company name claim an
    # unrelated filer: "Metropolitan Group" (a small agency in our corpus) matched
    # registrant "Metropolitan Bank" and inherited CIK 0001476034. A CIK is used to
    # scope EDGAR full-text search, so a wrong one silently attributes another
    # company's filings as this company's evidence -- worse than returning None,
    # which is the normal and correct outcome for the many non-US firms we score.
    # The trailing words must look like QUALIFIERS (a region, a legal form, a unit),
    # not a different company name. "Bonduelle Americas US" is Bonduelle; "Wise
    # Investments Ltd" is NOT the registrant "Wise" -- 'investments' is part of the
    # name, and accepting it handed a UK micro-firm CIK 0002099039.
    for cand_key, cik in cmap.items():
        if len(cand_key) > 3 and key.startswith(cand_key + " "):
            extra = key[len(cand_key):].split()
            if extra and all(w in _NAME_QUALIFIERS for w in extra):
                return cik

    # NO browse-edgar fallback. Removed 2026-08-05 for two independent reasons:
    #
    # 1. UNSAFE. The search is fuzzy with no similarity floor and returns whichever
    #    registrant EDGAR ranks first, so "Wise Investments Ltd" (a UK micro-firm)
    #    came back as CIK 0002099039. A wrong CIK silently scopes full-text search
    #    to ANOTHER company's filings and attributes them as this company's
    #    evidence -- strictly worse than returning None.
    # 2. NEARLY PURE WASTE. Measured on the 193-company held-out corpus: 2 companies
    #    resolve from the cached ticker map; the other 191 would each fire a live
    #    EDGAR request that returns nothing usable. That burst is what tripped the
    #    503 guard during testing on 2026-08-05.
    #
    # The ticker map covers ~18k registrants and is fetched once per process, so
    # every genuine US filer resolves at zero request cost. Returning None for the
    # rest is the correct outcome, not a gap -- most companies we score are not SEC
    # registrants at all.
    #
    # resolve_cik() is retained for explicit, one-off callers who accept the fuzzy
    # semantics; it is deliberately no longer on the automatic path.
    return None



def _find_latest_filing(cik: str, form_type: str) -> Optional[tuple[str, str]]:
    """Returns (accession_no_dashes, primary_document) for the most recent
    filing of `form_type`, or None if no such filing exists for this CIK."""
    cik_padded = cik.zfill(10)
    r = _sec_get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
    if not r:
        return None
    try:
        data = r.json()
        recent = data["filings"]["recent"]
        forms = recent["form"]
        idx = next((i for i, f in enumerate(forms) if f == form_type), None)
        if idx is None:
            log.info("CIK %s: no %s in recent filings", cik, form_type)
            return None
        accession = recent["accessionNumber"][idx].replace("-", "")
        primary_doc = recent["primaryDocument"][idx]
        return accession, primary_doc
    except (KeyError, IndexError, ValueError) as exc:
        log.warning("CIK %s: submissions parse failed for %s: %s", cik, form_type, exc)
        return None


def filing_url(company_or_cik: str, form_type: str, is_cik: bool = False) -> Optional[str]:
    """Public EDGAR URL of the most recent `form_type` filing for a company.

    Exists so callers that surface filing text as evidence can cite the exact
    document it came from. The section-extracting helpers below return plain
    strings (their callers embed them straight into signal text), so the URL
    built inside _fetch_filing_text was previously unreachable from outside.
    Resolves through the same cik lookup + submissions index those helpers use,
    so the URL refers to the same filing they read.
    """
    cik = company_or_cik if is_cik else resolve_cik(company_or_cik)
    if not cik:
        return None
    found = _find_latest_filing(cik, form_type)
    if not found:
        return None
    accession, primary_doc = found
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"


def _fetch_filing_text(cik: str, accession: str, primary_doc: str) -> Optional[str]:
    """Fetch a filing document and return its plain (tag-stripped) text."""
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
    r = _sec_get(doc_url, timeout=_TIMEOUT)
    if not r:
        return None
    text = re.sub(r"<[^>]+>", " ", r.text)
    text = re.sub(r"&#160;|&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_section(text: str, heading_pattern: str, window: int = 2000) -> Optional[str]:
    """
    Extract the body text following the LAST occurrence of heading_pattern
    (a regex) in `text`. Returns None if the heading doesn't appear at all.
    Trims at the next "Item N." heading within the window if one appears
    well into the window (avoids bleeding into the following section), but
    keeps at least the first 100 chars regardless so a heading appearing
    immediately after isn't treated as "no content."
    """
    matches = list(re.finditer(heading_pattern, text, re.IGNORECASE))
    if not matches:
        return None
    body_start = matches[-1].end()
    body = text[body_start:body_start + window].strip()
    next_item = re.search(r"Item\s*\d+[A-Z]?\.", body)
    if next_item and next_item.start() > 100:
        body = body[:next_item.start()]
    return body.strip() or None


def fetch_10k_section(company_or_cik: str, heading_pattern: str, is_cik: bool = False,
                       window: int = 2000) -> Optional[str]:
    """
    Convenience: resolve CIK (if a name was given) -> fetch latest 10-K ->
    extract the section matching heading_pattern. Returns None at any stage
    that fails to resolve (no CIK, no 10-K on file, section not found) — all
    expected outcomes for non-US-listed companies, not errors.
    """
    cik = company_or_cik if is_cik else resolve_cik(company_or_cik)
    if not cik:
        return None
    found = _find_latest_filing(cik, "10-K")
    if not found:
        return None
    accession, primary_doc = found
    text = _fetch_filing_text(cik, accession, primary_doc)
    if not text:
        return None
    return extract_section(text, heading_pattern, window=window)


def fetch_def14a_section(company_or_cik: str, heading_pattern: Optional[str] = None,
                          is_cik: bool = False, window: int = 2000) -> Optional[str]:
    """
    Convenience: resolve CIK -> fetch latest DEF 14A (proxy statement) ->
    extract a section, or return a leading window of the whole document if
    heading_pattern is None (DEF 14A sub-sections are less consistently
    named "Item N." than 10-Ks, so callers may prefer to search the raw text
    for topic keywords instead of a fixed heading).
    """
    cik = company_or_cik if is_cik else resolve_cik(company_or_cik)
    if not cik:
        return None
    found = _find_latest_filing(cik, "DEF 14A")
    if not found:
        return None
    accession, primary_doc = found
    text = _fetch_filing_text(cik, accession, primary_doc)
    if not text:
        return None
    if heading_pattern is None:
        return text[:window].strip() or None
    return extract_section(text, heading_pattern, window=window)


def full_text_search(phrase: str, cik: Optional[str] = None, forms: str = "",
                      date_from: str = "", date_to: str = "", limit: int = 10) -> list[dict]:
    """EDGAR full-text search: which filings actually CONTAIN this phrase.

    Returns [{accession, filename, form, file_date, display_name, url}], most
    recent first. Empty list when nothing matches — the normal outcome, not an
    error.

    `cik` MUST be zero-padded to 10 digits (use cik_for_company); an unpadded
    CIK silently returns zero hits rather than erroring. Scoping by CIK is the
    reason this source sidesteps the entity-matching problem that plagues our
    web-search sources: results are filings BY that registrant, not pages that
    merely mention a similar name.

    `forms` is left empty by default on purpose. Filtering to 10-K drops the
    richest ESG material -- PX14A6G (shareholder-proposal exhibits, which are
    adversarial and specific) and Form SD (conflict minerals) both carry
    heavier ESG content than the 10-K itself, and neither is a 10-K.

    Date filtering needs BOTH `date_from` and `date_to`. Verified live that
    startdt alone is silently ignored -- an identical query returned the same
    26 hits (oldest 2002) with and without it, and only began filtering once
    enddt was supplied. Passing one alone therefore looks like it works while
    doing nothing, so this helper supplies today's date when only date_from
    is given rather than letting the filter quietly fail.
    """
    params: dict[str, str] = {"q": f'"{phrase}"'}
    if cik:
        params["ciks"] = cik
    if forms:
        params["forms"] = forms
    if date_from or date_to:
        params["startdt"] = date_from or "2001-01-01"
        params["enddt"] = date_to or _today_iso()

    r = _sec_get(_EFTS_URL, params=params, timeout=_TIMEOUT)
    if not r:
        return []
    try:
        hits = (r.json().get("hits") or {}).get("hits") or []
    except (ValueError, AttributeError) as exc:
        log.warning("EFTS parse failed for %r: %s", phrase, exc)
        return []

    out: list[dict] = []
    for h in hits[:limit]:
        src = h.get("_source") or {}
        # _id is "accession-with-dashes:primary_document"
        hid = h.get("_id") or ""
        if ":" not in hid:
            continue
        accession, filename = hid.split(":", 1)
        display = (src.get("display_names") or [""])[0]
        # CIK for the Archives path: prefer the one embedded in display_names
        # ("NIKE, Inc. (NKE) (CIK 0000320187)"), else the requested cik.
        m = re.search(r"CIK\s*(\d{10})", display)
        doc_cik = m.group(1) if m else (cik or "")
        if not doc_cik:
            continue
        out.append({
            "accession": accession,
            "filename": filename,
            "form": src.get("root_form") or src.get("file_type") or "",
            "file_date": src.get("file_date") or "",
            "display_name": display,
            "url": (f"https://www.sec.gov/Archives/edgar/data/{int(doc_cik)}/"
                    f"{accession.replace('-', '')}/{filename}"),
        })
    return out


def fetch_document_text(url: str, max_chars: int = 40000) -> Optional[str]:
    """Fetch an EDGAR document and return tag-stripped plain text.

    Skips PDFs: EFTS indexes them (they appear as .pdf hits) but they need a
    binary parser, and every PDF hit we have seen is a courtesy copy of an
    HTML filing that is also in the results.
    """
    if url.lower().endswith(".pdf"):
        return None
    r = _sec_get(url, timeout=_TIMEOUT)
    if not r:
        return None
    text = re.sub(r"<[^>]+>", " ", r.text)
    text = re.sub(r"&#160;|&nbsp;", " ", text)
    text = re.sub(r"&#8217;|&#8216;", "'", text)
    text = re.sub(r"&#8220;|&#8221;", '"', text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] or None


def extract_phrase_context(text: str, phrase: str, context_chars: int = 700,
                            max_windows: int = 3) -> list[str]:
    """Windows of text around each occurrence of `phrase`.

    A 30k-character filing has to be reduced to the passages that actually
    discuss the topic before it reaches the extractor -- sending the whole
    document wastes context on unrelated sections and buries the evidence.
    Overlapping windows are merged so one dense passage yields one window
    rather than three near-duplicates.
    """
    if not text or not phrase:
        return []
    windows: list[tuple[int, int]] = []
    for m in re.finditer(re.escape(phrase), text, re.IGNORECASE):
        start = max(0, m.start() - context_chars // 3)
        end = min(len(text), m.end() + context_chars)
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
        if len(windows) >= max_windows:
            break
    return [text[s:e].strip() for s, e in windows]


def search_def14a_keyword(company_or_cik: str, keyword_pattern: str, is_cik: bool = False,
                           context_chars: int = 800) -> Optional[str]:
    """
    Fetch the latest DEF 14A and return a window of text around the FIRST
    match of keyword_pattern (case-insensitive). Used for governance topics
    (e.g. "independent director", "audit committee") that appear as prose
    within the proxy statement rather than under a single fixed heading.
    """
    cik = company_or_cik if is_cik else resolve_cik(company_or_cik)
    if not cik:
        return None
    found = _find_latest_filing(cik, "DEF 14A")
    if not found:
        return None
    accession, primary_doc = found
    text = _fetch_filing_text(cik, accession, primary_doc)
    if not text:
        return None
    m = re.search(keyword_pattern, text, re.IGNORECASE)
    if not m:
        return None
    start = max(0, m.start() - 100)
    return text[start:start + context_chars].strip() or None
