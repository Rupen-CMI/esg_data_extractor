"""
facility_extractor.py — Layer 1 Facility Extractor (NEW).

Gathers factory/facility count, scale, and locations — the source category
identified during architecture research as having no reliable structured
API. Source priority, per that research (see project memory /
all-data-metadata-linked-dijkstra.md):

  1. SEC 10-K "Item 2. Properties" — US-listed public companies only, but
     legally mandated, structured, highest reliability. Fetched here via
     SEC EDGAR's free submissions + document APIs (no API key needed).
  2. Web search fallback (sustainability report / "manufacturing plants
     locations" query via signal_agent's existing DDG helper) — broader
     coverage, lower reliability, used when no 10-K match exists.

Both paths return raw evidence TEXT, not parsed facility counts — turning
this prose into a typed {factory_count, facility_scale, confidence} claim is
Layer 2's E Extractor's job (per the plan: Extractors read evidence and tag
it; this collector's job stops at gathering real, attributable text).

CLI:
    python -m agentic_estimation.facility_extractor "Nvidia"
    python -m agentic_estimation.facility_extractor "Kitchen Bath Ventures SL"
"""

import sys

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.signal_agent import _ddg_fallback
from agentic_estimation.layer_1.sec_filings import resolve_cik, fetch_10k_section, filing_url

log = get_logger("facility_extractor")


# ── SEC 10-K "Item 2. Properties" extraction ──────────────────────────────────
# CIK resolution + document fetch logic lives in sec_filings.py (shared with
# governance_collector.py, which also reads SEC 10-K/DEF 14A filings).

def _sec_10k_properties_signal(company: str) -> str:
    """Full pipeline: name -> CIK -> latest 10-K -> Properties section text."""
    log.info("[%s] sec_10k -> resolving CIK", company)
    cik = resolve_cik(company)
    if not cik:
        log.info("[%s] sec_10k -> no CIK match (likely not US-listed)", company)
        return ""
    log.info("[%s] sec_10k -> CIK %s, fetching Properties section", company, cik)
    body = fetch_10k_section(cik, r"Item\s*2\.?\s*Properties", is_cik=True)
    if not body:
        log.info("[%s] sec_10k -> no Properties section extracted", company)
        return ""
    log.info("[%s] sec_10k -> hit (%d chars)", company, len(body))
    src = filing_url(cik, "10-K", is_cik=True)
    return f"SEC 10-K Item 2 Properties: {body[:1500]}" + (f" <{src}>" if src else "")


# ── Web search fallback (non-US-listed / private companies) ──────────────────

def _facility_web_signal(company: str, industry: str = "") -> str:
    """DDG fallback — manufacturing footprint mentions in sustainability
    reports / company sites. Lower reliability than the 10-K path; the
    resulting claim's confidence should reflect that downstream (Layer 2)."""
    sector_hint = f" {industry}" if industry else ""
    log.info("[%s] facility_web -> searching manufacturing footprint", company)
    result = _ddg_fallback(
        f'"{company}"{sector_hint} manufacturing plants factories facilities locations number of',
        prefix="Facility Footprint", min_len=60, reject_wikipedia=True,
    )
    log.info("[%s] facility_web -> %s", company, "hit" if result else "no result")
    return result


# ── Main entry point ─────────────────────────────────────────────────────────

def fetch_facility_signals(company: str, industry: str = "") -> dict[str, str]:
    """
    Gather facility evidence for one company. Tries the SEC 10-K path first
    (highest reliability); falls through to web search regardless of whether
    the 10-K hit, since a company can have BOTH a 10-K properties summary and
    additional detail in its sustainability report — more evidence for
    Layer 2 to weigh, not either/or.

    Returns dict of source_name -> evidence text (empty sources omitted),
    same shape as signal_agent.fetch_company_signals.
    """
    log_header(log, "Facility Extractor", company=company, industry=industry or "N/A")
    signals: dict[str, str] = {}

    sec_result = _sec_10k_properties_signal(company)
    if sec_result:
        signals["sec_10k_properties"] = sec_result

    web_result = _facility_web_signal(company, industry)
    if web_result:
        signals["facility_web"] = web_result

    log.info("[%s] facility extractor done -- %d/2 sources returned data", company, len(signals))
    return signals


# ── DB persistence (reuses company_esg_signals) ───────────────────────────────

def save_facility_signals(company_id, signals: dict[str, str]) -> None:
    from agentic_estimation.layer_1.signal_agent import _save_signals_to_db
    _save_signals_to_db(company_id, signals)


def get_or_fetch_facility_signals(company_id, company: str, industry: str = "") -> dict[str, str]:
    from agentic_estimation.layer_1.signal_agent import _load_signals_from_db

    cached = _load_signals_from_db(company_id)
    cached_fac = {k: v for k, v in cached.items() if k in ("sec_10k_properties", "facility_web")}
    if cached_fac:
        log.info("[%s] facility signal cache hit -- %d sources", company, len(cached_fac))
        return cached_fac

    log.info("[%s] facility signal cache miss -- gathering", company)
    signals = fetch_facility_signals(company, industry)
    save_facility_signals(company_id, signals)
    return signals


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agentic_estimation.facility_extractor \"<company>\" [industry]")
        sys.exit(1)
    company = sys.argv[1]
    industry = sys.argv[2] if len(sys.argv) > 2 else ""
    signals = fetch_facility_signals(company, industry)
    print(f"\nFacility signals for '{company}': {len(signals)} sources\n")
    for source, text in signals.items():
        print(f"[{source}]")
        print(f"  {text[:400]}{'...' if len(text) > 400 else ''}\n")


if __name__ == "__main__":
    _cli()
