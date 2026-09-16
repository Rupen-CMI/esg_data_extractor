"""
governance_collector.py — Layer 1 Governance Collector (NEW).

Fills the source category that was entirely missing before this rebuild —
diagnosed as the direct cause of G's flat ~45-55 predictions in the
calibration baseline (no governance-specific evidence meant the scoring LLM
had nothing to differentiate on). Gathers:
  - Board composition / independence
  - Regulatory fines & violations / litigation
  - Anti-corruption / compliance certifications

SOURCE TIERING (added after Phase 1 verification showed uneven DDG hit
rates and a real false-positive case — see KNOWN PRECISION ISSUE below):
  Tier 1 — SEC filings (sec_filings.py), US-listed public companies only,
           but legally mandated + structured + free, no API key:
             gov_board_sec       <- DEF 14A proxy statement (board
                                     composition, committee structure)
             gov_litigation_sec  <- 10-K Item 3 "Legal Proceedings"
             gov_board_count     <- Wikidata board_member_count (already
                                     fetched by company_metadata.py for
                                     every company, previously unused
                                     downstream — a free structured signal)
  Tier 2 — DDG web search (existing, kept as fallback for non-US-listed
           companies and topics SEC filings don't cover, e.g. ISO 37001
           certification status):
             gov_board, gov_fines, gov_compliance, gov_litigation

Both tiers are attempted for every company — Tier 1 doesn't replace Tier 2,
it adds a higher-reliability layer on top, so coverage for non-US-listed
companies is unaffected.

Reuses signal_agent's existing DDG rate-limiter and fallback helper
(_ddg_fallback, _DDG_LIMITER) rather than reimplementing — this collector's
DDG queries are serialized through the SAME shared limiter as every other
DDG-backed source in the pipeline, so it does not add extra rate-limit
exposure beyond what signal_agent already manages. SEC/Wikidata sources are
separate free APIs with their own existing rate-limit handling
(sec_filings.py reuses signal_agent._get; company_metadata.py has its own
Wikidata throttle).

Results are persisted to company_esg_signals (source tags prefixed 'gov_')
so they land in the same evidence pool signal_agent already populates — no
new storage path, consistent with the plan's "all data/metadata extraction
happens in Layer 1" direction.

KNOWN PRECISION ISSUE (found during Phase 1 verification, not yet fixed):
for smaller/less-documented companies, these DDG queries sometimes return
real, non-empty text that passes min_len/reject_wikipedia but is NOT actually
about governance — e.g. querying "Blackmores" (a vitamins brand) for board
composition returned genuine web content about yoga poses, because DDG fell
back to generic brand-website content when no governance-specific page
existed. This passed every current filter (it's real text, isn't Wikipedia,
is long enough) while being semantically irrelevant. Layer 2's Governance
Extractor MUST NOT treat "has a source_id" as sufficient for confidence > 0
— the source text itself must be checked for actual topical relevance before
a claim is extracted from it. This is exactly the gap the "no source, no
claim" schema rule (company_evidence_claims) doesn't fully close by itself:
a source_id can point to a real-but-irrelevant document. Extractor prompts
in Phase 2 need an explicit relevance check, not just an extraction task.

Usage:
    from agentic_estimation.layer_1.governance_collector import fetch_governance_signals
    signals = fetch_governance_signals("Bosch")
    # -> {"gov_board": "...", "gov_fines": "...", "gov_compliance": "...", "gov_litigation": "..."}

CLI:
    python -m agentic_estimation.governance_collector "Bosch"
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.signal_agent import RateLimitTripped, _ddg_fallback
from agentic_estimation.layer_1.sec_filings import (
    resolve_cik, search_def14a_keyword, fetch_10k_section, filing_url)

log = get_logger("governance_collector")


# ── Tier 1: SEC filings + Wikidata (structured, high-reliability) ────────────

def _board_composition_sec_signal(company: str) -> str:
    """
    DEF 14A proxy statement — real board/committee structure text. Far more
    reliable than the DDG board query (Tier 2 gov_board), which is prone to
    Wikipedia false-rejects and, for smaller companies, irrelevant-content
    false-positives (see KNOWN PRECISION ISSUE below). US-listed only.
    """
    log.info("[%s] gov_board_sec -> resolving CIK for DEF 14A", company)
    cik = resolve_cik(company)
    if not cik:
        log.info("[%s] gov_board_sec -> no CIK match (likely not US-listed)", company)
        return ""
    body = search_def14a_keyword(cik, r"independent director", is_cik=True)
    if not body:
        log.info("[%s] gov_board_sec -> no independent-director text found in DEF 14A", company)
        return ""
    log.info("[%s] gov_board_sec -> hit (%d chars)", company, len(body))
    src = filing_url(cik, "DEF 14A", is_cik=True)
    return (f"SEC DEF 14A Proxy Statement (board/independence): {body[:1200]}"
            + (f" <{src}>" if src else ""))


def _litigation_sec_signal(company: str) -> str:
    """
    10-K Item 3 'Legal Proceedings' — mandated litigation disclosure. US-listed only.

    KNOWN LIMITATION (found during verification): companies with MATERIAL
    litigation commonly cross-reference Item 3 to a financial-statement note
    instead of repeating the detail (e.g. Nvidia's Item 3 reads only "Please
    see Note 12 of the Notes to the Consolidated Financial Statements...").
    This is REAL, non-fabricated text, but it is a pointer, not litigation
    content — verified this is standard 10-K practice, not an extraction bug.
    Following the cross-reference into the financial statement notes would
    need a much deeper document crawl; deliberately out of scope here. The
    DDG-backed gov_litigation/gov_fines Tier-2 sources partially compensate
    (e.g. they independently surfaced Nvidia's real China SAMR antitrust
    investigation, which this SEC path does not capture at all). Layer 2's
    G Extractor should treat a boilerplate cross-reference as weak/no
    evidence, not as "no litigation exists."
    """
    log.info("[%s] gov_litigation_sec -> resolving CIK for 10-K Item 3", company)
    cik = resolve_cik(company)
    if not cik:
        log.info("[%s] gov_litigation_sec -> no CIK match (likely not US-listed)", company)
        return ""
    body = fetch_10k_section(cik, r"Item\s*3\.?\s*Legal Proceedings", is_cik=True)
    if not body:
        log.info("[%s] gov_litigation_sec -> no Legal Proceedings section extracted", company)
        return ""
    log.info("[%s] gov_litigation_sec -> hit (%d chars)", company, len(body))
    src = filing_url(cik, "10-K", is_cik=True)
    return (f"SEC 10-K Item 3 Legal Proceedings: {body[:1200]}"
            + (f" <{src}>" if src else ""))


def _board_count_wikidata_signal(company: str) -> str:
    """
    Wikidata board_member_count — already fetched by company_metadata.py for
    every company via its existing Wikidata SPARQL query, but was previously
    unused downstream. Free, structured, global (not US-listed-only), sparse
    (only populated when Wikidata itself has the fact).
    """
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    log.info("[%s] gov_board_count -> checking Wikidata metadata", company)
    meta = get_company_metadata(company)
    count = meta.get("board_member_count")
    if not count:
        log.info("[%s] gov_board_count -> no board_member_count in Wikidata", company)
        return ""
    log.info("[%s] gov_board_count -> hit (%s members)", company, count)
    return f"Wikidata: Board of Directors has {count} members."


# ── Tier 2: DDG web search (broader coverage, lower reliability) ─────────────

# One topic-term tuple per DDG function below, passed as topic_terms= to
# _ddg_fallback. Fixes the KNOWN PRECISION ISSUE documented in this module's
# own docstring (Blackmores/yoga-poses) plus its three sibling functions,
# which share the identical gap: entity filtering (company=) only proves a
# result is ABOUT the right company, never that it is about the SUBJECT this
# specific query exists to find. Same root cause and same fix pattern as
# facility_extractor.py's _FACILITY_TOPIC_TERMS (confirmed live 2026-09-11:
# a YouTube video description and an Outlook sign-in page both genuinely
# mentioned "Microsoft" and passed entity filtering while being pure noise).

_BOARD_TOPIC_TERMS = (
    "board of directors", "board composition", "independent director",
    "independent directors", "audit committee", "proxy statement",
    "board member", "board members", "director nominee", "chairman",
    "chairperson", "non-executive director", "supervisory board",
)

_FINES_TOPIC_TERMS = (
    "fine", "fined", "penalty", "penalties", "violation", "violations",
    "settlement", "settled", "enforcement action", "regulatory action",
    "consent decree", "cease and desist", "sanction", "sanctioned",
)

_COMPLIANCE_TOPIC_TERMS = (
    "anti-corruption", "anti-bribery", "iso 37001", "compliance program",
    "compliance certification", "whistleblower", "code of conduct",
    "ethics policy", "ethics hotline", "ungc", "un global compact",
)

_LITIGATION_TOPIC_TERMS = (
    "lawsuit", "lawsuits", "litigation", "legal dispute", "legal disputes",
    "securities fraud", "shareholder suit", "shareholder lawsuit",
    "class action", "court filing", "plaintiff", "defendant",
)


def _board_composition_signal(company: str) -> str:
    """
    Board composition / independence — via DDG. Queries proxy-statement /
    investor-relations phrasing specifically ("independent directors",
    "proxy statement") rather than generic "board of directors" wording,
    which is Wikipedia-infobox-attracting and gets discarded by
    reject_wikipedia (Wikipedia almost always leads for generic board
    queries on well-known companies — this is the filter working as
    intended, not a bug, so the fix is a more specific query, not disabling
    the guard).
    """
    log.info("[%s] gov_board -> searching board composition/independence", company)
    result = _ddg_fallback(
        f'"{company}" proxy statement independent directors board composition audit committee 2023 2024 2025',
        prefix="Board Composition", min_len=60, reject_wikipedia=True,
        company=company, topic_terms=_BOARD_TOPIC_TERMS,
    )
    log.info("[%s] gov_board -> %s", company, "hit" if result else "no result")
    return result


def _regulatory_fines_signal(company: str) -> str:
    """Regulatory fines / violations — via DDG, focused on enforcement actions."""
    log.info("[%s] gov_fines -> searching regulatory fines/violations", company)
    result = _ddg_fallback(
        f'"{company}" regulatory fine penalty violation settlement enforcement 2023 2024 2025 -site:wikipedia.org',
        prefix="Regulatory Fines/Violations", min_len=60, reject_wikipedia=True,
        company=company, topic_terms=_FINES_TOPIC_TERMS,
    )
    log.info("[%s] gov_fines -> %s", company, "hit" if result else "no result")
    return result


def _compliance_certs_signal(company: str) -> str:
    """Anti-corruption / compliance certifications (ISO 37001, UNGC, etc.) — via DDG."""
    log.info("[%s] gov_compliance -> searching compliance certifications", company)
    result = _ddg_fallback(
        f'"{company}" anti-corruption policy ISO 37001 compliance certification whistleblower ethics',
        prefix="Compliance Certifications", min_len=60, reject_wikipedia=True,
        company=company, topic_terms=_COMPLIANCE_TOPIC_TERMS,
    )
    log.info("[%s] gov_compliance -> %s", company, "hit" if result else "no result")
    return result


def _litigation_signal(company: str) -> str:
    """Litigation records — via DDG, focused on lawsuits/legal disputes with governance relevance."""
    log.info("[%s] gov_litigation -> searching litigation records", company)
    result = _ddg_fallback(
        f'"{company}" lawsuit litigation legal dispute securities fraud shareholder 2023 2024 2025 -site:wikipedia.org',
        prefix="Litigation Records", min_len=60, reject_wikipedia=True,
        company=company, topic_terms=_LITIGATION_TOPIC_TERMS,
    )
    log.info("[%s] gov_litigation -> %s", company, "hit" if result else "no result")
    return result


# ── Main entry point (mirrors signal_agent.fetch_company_signals shape) ───────

def fetch_governance_signals(company: str) -> dict[str, str]:
    """
    Fetch all governance-specific signals for one company in parallel.
    Returns dict of source_name -> evidence text. Empty sources are omitted.
    Same shape/contract as signal_agent.fetch_company_signals so callers can
    merge the two dicts directly.
    """
    tasks: dict[str, callable] = {
        # Tier 1 — SEC filings + Wikidata (structured, high-reliability)
        "gov_board_sec":       lambda: _board_composition_sec_signal(company),
        "gov_litigation_sec":  lambda: _litigation_sec_signal(company),
        "gov_board_count":     lambda: _board_count_wikidata_signal(company),
        # Tier 2 — DDG web search (broader coverage, lower reliability, kept
        # as fallback for non-US-listed companies and topics SEC filings
        # don't cover)
        "gov_board":       lambda: _board_composition_signal(company),
        "gov_fines":       lambda: _regulatory_fines_signal(company),
        "gov_compliance":  lambda: _compliance_certs_signal(company),
        "gov_litigation":  lambda: _litigation_signal(company),
    }

    signals: dict[str, str] = {}
    log_header(log, "Governance Collector", company=company, sources=len(tasks))
    log.info("[%s] starting governance signal fetch (%d sources)", company, len(tasks))

    # DDG calls queue through the shared _DDG_LIMITER inside _ddg_fallback, so
    # running these concurrently here doesn't bypass the rate-limit protection
    # — it just lets non-DDG work (none in this module) overlap; DDG requests
    # themselves still serialize at the source, same as signal_agent's sources.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = (fut.result() or "").strip()
                if result:
                    signals[name] = result
                    log.info("[%s] %s -> OK (%d chars)", company, name, len(result))
                else:
                    log.info("[%s] %s -> empty", company, name)
            except RateLimitTripped:
                raise   # never swallow the abort signal
            except Exception as e:
                log.warning("[%s] %s -> exception: %s", company, name, e)

    log.info("[%s] governance collector done -- %d/%d sources returned data", company, len(signals), len(tasks))
    return signals


# ── DB persistence (reuses company_esg_signals, same table signal_agent uses) ──

def save_governance_signals(company_id, signals: dict[str, str]) -> None:
    """Upsert governance signals into company_esg_signals — same table/upsert
    pattern as signal_agent._save_signals_to_db, reused directly."""
    from agentic_estimation.layer_1.signal_agent import _save_signals_to_db
    _save_signals_to_db(company_id, signals)


def get_or_fetch_governance_signals(company_id, company: str) -> dict[str, str]:
    """DB-cached variant, mirroring signal_agent.get_or_fetch_signals. Only
    checks for the gov_* keys specifically so it doesn't collide with a
    signal_agent cache hit that lacks governance sources."""
    from agentic_estimation.layer_1.signal_agent import _load_signals_from_db

    cached = _load_signals_from_db(company_id)
    cached_gov = {k: v for k, v in cached.items() if k.startswith("gov_")}
    if cached_gov:
        log.info("[%s] governance signal cache hit -- %d sources", company, len(cached_gov))
        return cached_gov

    log.info("[%s] governance signal cache miss -- gathering from web", company)
    signals = fetch_governance_signals(company)
    save_governance_signals(company_id, signals)
    return signals


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agentic_estimation.governance_collector \"<company>\"")
        sys.exit(1)
    company = sys.argv[1]
    signals = fetch_governance_signals(company)
    print(f"\nGovernance signals for '{company}': {len(signals)} sources\n")
    for source, text in signals.items():
        snippet = f"{text[:300]}{'...' if len(text) > 300 else ''}"
        # Windows console (cp1252) can't encode some Unicode punctuation that
        # shows up in scraped web text (curly quotes, em-dashes) -- degrade
        # gracefully instead of crashing the CLI on a display-only issue.
        safe = snippet.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"[{source}]")
        print(f"  {safe}\n")


if __name__ == "__main__":
    _cli()
