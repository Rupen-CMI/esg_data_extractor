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
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.layer_1.signal_agent import _get

log = get_logger("sec_filings")

_TIMEOUT = 20


def resolve_cik(company: str) -> Optional[str]:
    """
    Resolve a company name to its SEC CIK via EDGAR's company search (atom feed).
    Returns None if no match — the common case for non-US-listed companies,
    NOT an error.
    """
    r = _get(
        "https://www.sec.gov/cgi-bin/browse-edgar",
        params={"action": "getcompany", "company": company, "type": "10-K",
                "dateb": "", "owner": "include", "count": "5", "output": "atom"},
    )
    if not r:
        return None
    m = re.search(r"<cik>(\d+)</cik>", r.text)
    return m.group(1) if m else None


def _find_latest_filing(cik: str, form_type: str) -> Optional[tuple[str, str]]:
    """Returns (accession_no_dashes, primary_document) for the most recent
    filing of `form_type`, or None if no such filing exists for this CIK."""
    cik_padded = cik.zfill(10)
    r = _get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
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
    r = _get(doc_url, timeout=_TIMEOUT)
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
