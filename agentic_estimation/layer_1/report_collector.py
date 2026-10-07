"""Company sustainability-report PDFs -> signals dict.

The module named (but never written) at calibration/pdf_parser_test.py:9. This
is the seam that finally connects the document-acquisition work
(report_coverage.py, download_all.py, the Wayback recovery) to the SCORING
path. Before this, 42 downloaded PDFs sat on disk and no scoring script ever
opened one.

Contract matches its sibling collectors (signal_agent.fetch_company_signals,
governance_collector.fetch_governance_signals): company name in, flat
{source_tag: text} out, {} on a miss.

────────────────────────────────────────────────────────────────────────────
THE TRUNCATION PROBLEM -- the reason this file is not three lines long
────────────────────────────────────────────────────────────────────────────
pillar_extractors._signals_block truncates EACH source to
_MAX_CHARS_PER_SIGNAL = 4000 chars. A 200-page report is ~400k chars, so a
naive {"sustainability_report_pdf": full_text} delivers the first 4000
chars -- cover page, CEO letter, table of contents. That is the least
extractable content in the document; the KPI tables that justify the whole
exercise are at pages 40-120 and would never be seen.

Worse, that loop `break`s (not `continue`s) at _MAX_TOTAL_CHARS = 60000. A
huge blob early in the dict silently drops EVERY LATER SOURCE. Since the
harness merges collectors with dict.update() in a fixed order, a careless PDF
source could delete country_governance/facility/gov_* evidence from the prompt
without a single log line.

So this module does the selection itself rather than letting a blind [:4000]
do it:

  1. score each page-ish block for ESG-metric density (numbers + units +
     pillar vocabulary), and
  2. emit SEVERAL small tags -- report_pdf_e / _s / _g -- each already under
     the 4000 cap, pillar-targeted, and appended LAST.

Emitting per-pillar rather than one blob also means the E extractor is not
paying prompt budget to read S content and vice versa.

WHY NOT JUST RAISE _MAX_CHARS_PER_SIGNAL: it is global. Raising it to fit a
report multiplies every other source's budget too and would blow past
_MAX_TOTAL_CHARS, re-triggering the same silent-drop bug on the sources that
currently work.

KNOWN TENSION (documented, not solved here): claim_validators corroboration
counts DISTINCT source_tags, so splitting one report into three tags lets a
report corroborate itself. Mitigated by _SHARED_TAG_PREFIX -- see
report_tags_are_one_source().
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.layer_1.report_parser import PAGE_MARK, parse_pdf

log = get_logger("report_collector")

_PDF_DIRS = [
    Path("raw_esg_data/company_esg_reports"),
    Path("raw_esg_data/wayback_recovered"),
]
# A run that downloads into its own folder (ESG_REPORT_DIR, as srn_report_source
# honours) would otherwise be invisible here: these paths are fixed, so the
# documents land on disk and the collector reports "no report" for every company
# -- the with/without arms then see identical evidence and the experiment
# silently measures nothing. Prepend the run's folder when one is set.
_RUN_DIR = os.getenv("ESG_REPORT_DIR")
if _RUN_DIR and Path(_RUN_DIR) not in _PDF_DIRS:
    _PDF_DIRS.insert(0, Path(_RUN_DIR))

# The index must be per-directory too: a cached index built from the default
# folders lists none of the run's files, and find_reports reads the index.
_INDEX_PATH = Path(os.getenv(
    "ESG_REPORT_INDEX",
    f"raw_esg_data/{Path(_RUN_DIR).name}_collector_index.json" if _RUN_DIR
    else "raw_esg_data/report_index.json"))

# Comfortably under pillar_extractors._MAX_CHARS_PER_SIGNAL (4000) so nothing
# is silently clipped mid-sentence after we did the work of selecting it.
_PER_TAG_CHARS = int(os.getenv("ESG_REPORT_TAG_CHARS", "3500"))

# All emitted tags start with this. Anything counting "independent sources"
# must treat them as ONE -- see report_tags_are_one_source().
_SHARED_TAG_PREFIX = "report_pdf"

# Minimum pillar-term hits per 1000 chars for a selection to be shipped.
#
# MEASURED, not assumed, across all 30 indexed companies (80 non-zero
# pillar selections):
#     deciles  p0=0.4  p10=1.7  p25=2.5  p50=5.9  p75=9.2  p90=14.7  max=21.5
#
# There is NO clean gap between "sustainability report" and "financial filing"
# -- the distribution is smooth. An earlier guess of 2.5 sat on the 25th
# percentile and would have silently dropped a quarter of all selections,
# including legitimate ones (Ebro Foods S=2.4, Soitec E/S/G=2.4/2.5/2.0).
#
# So this is set LOW and does one narrow job: reject selections with almost no
# pillar vocabulary at all (Arrow Financial E=0.9, Job&Talent E=0.4, Marshalls
# E=0.9 -- all 10-K text). Everything above that is left for the extractor to
# judge, which is what the extractor is for. Raising this trades recall for
# precision and should be re-measured, not eyeballed.
_MIN_PILLAR_DENSITY = float(os.getenv("ESG_REPORT_MIN_DENSITY", "1.2"))

# Download an indexed report on demand when the company has nothing on disk.
# ESG_REPORT_FETCH=0 forces a strictly offline run (replay of a frozen corpus,
# or any run where new network calls would make results non-reproducible).
_FETCH_MISSING = os.getenv("ESG_REPORT_FETCH", "1") != "0"

# Fall back to a real web search (Bing primary, Startpage/DDG secondary) when
# the SRN index has nothing. SRN only covers EU-listed large caps (confirmed
# 2026-08-18: 5/150 hit rate on a 35-country corpus) -- a genuinely niche or
# non-EU company can still have a real, publicly hosted report (e.g. AVZ
# Minerals' report on Squarespace) that SRN was never going to index. This
# reuses calibration/discover_reports.py's `from_search`, which already does
# search -> download -> STRICT entity verification (rejects a document that
# doesn't actually belong to this company -- see _entity_ok's docstring for
# why that check exists and how it was hardened against false positives).
# Same on-demand-single-company discipline as the SRN path above: one
# company, at most one search, only when actually being scored -- not a bulk
# crawl. ESG_REPORT_SEARCH=0 disables it (e.g. for a strictly offline replay,
# same reasoning as ESG_REPORT_FETCH=0 above).
_SEARCH_FALLBACK = os.getenv("ESG_REPORT_SEARCH", "1") != "0"


def _fetch_via_search(company: str, country: Optional[str] = None) -> Optional[str]:
    """One company, one search-discovered PDF, downloaded to the same
    directory the SRN path uses so find_reports() picks it up on the next
    call. Returns the local path, or None on no hit / any failure -- never
    raises (matches _fetch_from_srn's fail-open discipline: an unreachable
    search engine or a rejected document is a missing signal, not a crash).

    country: appended to the search query when known (2026-08-26). Confirmed
    live: "Humana" (US health insurer) matched "Humana AB" -- a real,
    unrelated, separately-listed Swedish care-services company with the
    exact same bare name. No content-quality check can tell two genuine
    self-authored ESG reports for two different real companies apart;
    steering the search query itself toward the right country resolves the
    ambiguity before a wrong candidate is even downloaded."""
    try:
        from calibration.discover_reports import from_search
        from calibration.report_coverage import _PDF_DIR
    except Exception as exc:
        log.info("search-fallback import failed for %s: %s", company, exc)
        return None
    try:
        path = from_search(company, _PDF_DIR, country=country)
    except Exception as exc:
        log.info("search-fallback failed for %s: %s: %s", company, type(exc).__name__, exc)
        return None
    return path

def _fetch_pillar_via_search(company: str, pillar: str,
                              country: Optional[str] = None) -> Optional[str]:
    """Like _fetch_via_search, but for a pillar-specific follow-up document
    (see fetch_report_signals()'s MISSING-PILLAR FALLBACK). Downloaded to the
    SAME directory _fetch_via_search uses -- find_reports()/build_index()
    would otherwise never see it -- but this is a one-off lookup, not
    persisted into the index under the company's normal key, since it is
    consumed immediately by the caller and re-searching next run is cheap
    and always fresh (a stale second-pillar document silently going stale
    forever is worse than paying for a fresh search each time it's needed).
    """
    try:
        from calibration.discover_reports import from_search
        from calibration.report_coverage import _PDF_DIR
    except Exception as exc:
        log.info("pillar-search-fallback import failed for %s/%s: %s", company, pillar, exc)
        return None
    try:
        return from_search(company, _PDF_DIR, country=country, pillar=pillar)
    except Exception as exc:
        log.info("pillar-search-fallback failed for %s/%s: %s: %s",
                  company, pillar, type(exc).__name__, exc)
        return None


_PILLAR_TERMS = {
    "E": ("emission", "scope 1", "scope 2", "scope 3", "ghg", "carbon",
          "energy", "renewable", "water", "waste", "recycl", "biodiversity",
          "climate", "net zero", "net-zero", "sbti", "cdp", "environment",
          "pollution", "effluent", "co2", "tco2", "kwh", "mwh", "gj"),
    "S": ("employee", "workforce", "diversity", "inclusion", "gender",
          "pay gap", "safety", "injury", "trir", "fatalit", "human rights",
          "modern slavery", "supply chain", "training", "community",
          "labour", "labor", "union", "turnover", "wellbeing", "csr"),
    "G": ("board", "governance", "independent director", "audit committee",
          "remuneration", "executive compensation", "shareholder", "ethics",
          "anti-corruption", "anti-bribery", "whistleblow", "compliance",
          "risk management", "tax", "data privacy", "code of conduct"),
}

# Phrases that mark a block as FINANCIAL-STATEMENT text rather than a real
# ESG disclosure, even when it scores densely on a pillar's own keywords.
# Confirmed live 2026-09-10: a 10-K's tax/compensation notes clear G's
# density bar and its own cross-pillar dominance check cleanly (Apple:
# density 10.26, E/S near zero) purely because "tax", "compliance" and
# "shareholder" are legitimate G-adjacent words that ALSO appear constantly
# in ordinary financial statements -- there is no density threshold that
# separates the two, because the underlying vocabulary genuinely overlaps.
# This is a SHAPE check, not a vocabulary check: real financial statements
# are dense with these specific phrases in a way no genuine ESG narrative
# (even a governance-heavy one) is.
_FINANCIAL_STATEMENT_MARKERS = (
    "income tax", "effective tax rate", "provision for income",
    "stock option", "class b common stock", "repurchase of",
    "share repurchase program", "repurchased", "non-gaap", "restructuring",
    "income before income tax", "federal income tax rate",
    "statutory tax rate", "deferred tax", "tax holiday", "transition tax",
    "tax cuts and jobs act", "net income per share",
    "diluted earnings per share", "weighted average shares",
    "goodwill impairment", "% change fiscal", "fiscal 20",
    "dollars in millions",
)


def _financial_hits(text: str) -> int:
    """Count of financial-statement marker phrases in `text`. Used to demote
    a pillar selection that is dense with G/S-adjacent words but is actually
    an accounting note, not a real disclosure -- see _FINANCIAL_STATEMENT_MARKERS."""
    low = text.lower()
    return sum(low.count(m) for m in _FINANCIAL_STATEMENT_MARKERS)


# Coverage verdict thresholds. Deliberately HIGHER than _MIN_PILLAR_DENSITY
# (1.2, which only rejects near-zero content) -- this asks a stricter
# question: is the document GENUINELY GOOD on this pillar, not just
# non-empty? Set from the measured density of confirmed-real selections
# (JPMorgan/Siemens/Samsung, all three pillars: 8.0-18.5) vs. confirmed-thin
# ones (Nike's own E on its 10-K: 3.08) -- see verification notes 2026-09-10.
_COVERAGE_STRONG_DENSITY = 5.0
# 2+ financial-statement markers inside an already-selected, budget-limited
# (3500-char) FINAL selection is a real signal, not noise -- a genuine ESG
# narrative essentially never uses phrases like "effective tax rate" or
# "diluted earnings per share" even once, let alone twice.
_COVERAGE_FINANCIAL_VETO = 2
# Lower bar for a single ~1200-char BLOCK (see _is_financial_statement):
# confirmed live 2026-09-10 (Nike 10-K) that individual blocks like "In June
# 2022, the Board of Directors approved a four-year, $18 billion share
# repurchase program..." carry exactly ONE marker phrase each, never two,
# simply because the block is too short to repeat itself -- requiring 2 at
# block level let every one of these slip through scoring untouched. One
# clear marker phrase in a single page-sized block is already conclusive.
_BLOCK_FINANCIAL_VETO = 1


def rate_pdf_coverage(text: str) -> dict[str, dict]:
    """Score one already-downloaded, already-parsed document against all
    three pillars' keyword sets and rate how well it covers each.

    Returns {"E": {...}, "S": {...}, "G": {...}}, each with:
      density        -- pillar-term hits per 1000 chars of that pillar's
                         best-scoring selection from THIS document (same
                         _select/_pillar_density used everywhere else in
                         this module, so this rating means the same thing
                         as every other density number already logged).
      financial_hits -- count of financial-statement marker phrases found
                         in that same selection.
      verdict        -- "strong"  : dense AND not financial-statement-shaped
                                     -- this document alone can supply real
                                     evidence for this pillar.
                        "weak"    : some relevant content, but either too
                                    thin (below _MIN_PILLAR_DENSITY) or
                                    dense only because it is financial-
                                    statement text wearing this pillar's
                                    vocabulary -- worth trying to replace.
                        "none"    : no selectable content for this pillar
                                    at all.

    This is the "does one PDF cover all three pillars, or only one?" check:
    call this once per freshly downloaded candidate, before committing to
    it as the company's sole source. A document that rates "strong" on all
    three needs no further searching. One that rates "weak"/"none" on one
    or two pillars should trigger a pillar-specific re-search for exactly
    those pillars (see fetch_report_signals's COVERAGE-DRIVEN RE-SEARCH) --
    if that re-search also fails to find anything better, the pillar is
    left as-is (or empty): genuinely irredeemable for this company.
    """
    out: dict[str, dict] = {}
    for pillar in ("E", "S", "G"):
        sel = _select(text, pillar)
        if not sel or len(sel) <= 200:
            out[pillar] = {"density": 0.0, "financial_hits": 0, "verdict": "none"}
            continue
        density = _pillar_density(sel, pillar)
        fin_hits = _financial_hits(sel)
        if density < _MIN_PILLAR_DENSITY:
            verdict = "none"
        elif density >= _COVERAGE_STRONG_DENSITY and fin_hits < _COVERAGE_FINANCIAL_VETO:
            verdict = "strong"
        else:
            verdict = "weak"
        out[pillar] = {"density": round(density, 2), "financial_hits": fin_hits,
                        "verdict": verdict}
    return out

# A number that carries a UNIT, a decimal point, or a percent sign -- i.e. a
# measurement rather than a page number. Bare integers are deliberately NOT
# matched; scoring them is what let contents pages outrank KPI tables.
_UNIT_NUM_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s?"
    r"(?:%|tco2e?|co2e?|kg|tonnes?|tons?|mwh|kwh|gwh|gj|tj|m3|litres?|liters?|"
    r"million|billion|mn|bn|employees|hours|days|years)\b"
    r"|\b\d+\.\d+\b|\b\d[\d,]*\s?%",
    re.I)


def _blocks(text: str, target: int = 1200) -> list[str]:
    """Split into scoreable blocks, one per page where possible.

    ORIGINALLY split on blank lines. That silently failed: pdfplumber emits no
    blank line between pages, so a 277k-char report became ONE block and 99% of
    the document was discarded before scoring ever ran. The symptom was subtle
    -- every pillar returned the cover/contents page, which superficially looks
    like a plausible extract.

    So: split on the page marker first (authoritative), and only fall back to
    blank lines when the marker is absent (e.g. the ODL backend, which emits
    markdown with real paragraph breaks).
    """
    if PAGE_MARK.strip("\n ") in text:
        pages = [p.strip() for p in re.split(re.escape(PAGE_MARK.strip("\n ")), text)]
        units = [p for p in pages if p.strip()]
    else:
        units = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    out: list[str] = []
    for u in units:
        # A dense page can exceed target; split it rather than truncating, so
        # a table at the bottom of a long page is not thrown away.
        if len(u) <= target * 2:
            out.append(u)
        else:
            for i in range(0, len(u), target):
                piece = u[i:i + target]
                if piece.strip():
                    out.append(piece)
    return out


def _is_navigation(block: str) -> bool:
    """Table-of-contents / index page, which must never be selected.

    These score deceptively well on a naive metric: a contents page mentions
    every pillar term in the document AND is full of numbers (page numbers).
    Measured on the ANTA report, the contents page outranked every real KPI
    table for all three pillars.

    Signature: many short lines, a high share of which END in a bare number.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 6:
        return False
    endnum = sum(1 for ln in lines if re.search(r"\s\d{1,3}$", ln))
    short = sum(1 for ln in lines if len(ln) < 60)
    return endnum >= max(4, len(lines) * 0.3) and short >= len(lines) * 0.7


def _is_financial_statement(block: str) -> bool:
    """Accounting text (10-K notes, tax/compensation tables), which must
    never be selected regardless of how densely it happens to use a
    pillar's own vocabulary.

    REJECTED HERE, BEFORE SCORING -- not after, as an afterthought veto on
    the already-built selection. Confirmed live 2026-09-10 (Apple, Nike,
    Pfizer): scoring first and vetoing the FINAL 3500-char selection still
    wastes the entire _select() pass building a selection out of blocks
    that were always going to be thrown away, and (worse) if only some of a
    document's blocks are financial, a selection built from a MIX of one
    real block and several financial ones can dilute below the marker
    threshold and slip through. Checking per-block, at the same point
    _is_navigation already vetoes TOC pages, means a financial block never
    enters the candidate pool at all -- the real blocks in the same
    document (if any) are unaffected and still compete normally.

    Signature: 1+ financial-statement marker phrase in a SINGLE ~1200-char
    block -- see _BLOCK_FINANCIAL_VETO for why the bar is lower here than
    on a full multi-block selection.
    """
    return _financial_hits(block) >= _BLOCK_FINANCIAL_VETO


def _score_block(block: str, pillar: str) -> float:
    """Higher = more likely to carry an extractable ESG metric for `pillar`.

    Term hits alone rank the CEO letter top -- it name-drops every pillar and
    quantifies nothing. What makes a claim extractable is a number WITH A UNIT
    next to a pillar term, so units are scored separately from bare digits;
    page numbers and years carry no unit and no longer inflate the score.
    """
    if _is_navigation(block) or _is_financial_statement(block):
        return 0.0
    low = block.lower()
    terms = sum(low.count(t) for t in _PILLAR_TERMS[pillar])
    if not terms:
        return 0.0
    # Bare integers are mostly page numbers and list indices. A measurement
    # almost always carries a unit or a decimal/percentage.
    united = len(_UNIT_NUM_RE.findall(block))
    density = united / max(1.0, len(block) / 400.0)
    return terms * (1.0 + 2.0 * density)


def _pillar_density(text: str, pillar: str) -> float:
    """Pillar-term hits per 1000 chars of the FINAL selection.

    Applied to the selection rather than the whole document: a 10-K mentions
    "board" and "compliance" often enough to pass a document-level check, while
    the blocks actually selected are financial tables.
    """
    low = text.lower()
    hits = sum(low.count(t) for t in _PILLAR_TERMS[pillar])
    return 1000.0 * hits / max(1, len(text))


def _select(text: str, pillar: str, budget: int = _PER_TAG_CHARS) -> str:
    """Best-scoring blocks for one pillar, in document order, under budget."""
    blocks = _blocks(text)
    scored = [(i, b, _score_block(b, pillar)) for i, b in enumerate(blocks)]
    scored = [s for s in scored if s[2] > 0]
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[2])
    picked, total = [], 0
    for i, b, sc in scored:
        if total + len(b) > budget:
            continue
        picked.append((i, b))
        total += len(b)
        if total >= budget * 0.9:
            break
    if not picked:
        return ""
    picked.sort(key=lambda x: x[0])          # restore document order
    return "\n\n".join(b for _, b in picked)[:budget]


# ── company -> PDF resolution ────────────────────────────────────────────────

def _norm(name: str) -> str:
    from agentic_estimation.shared.company_name_utils import normalize_company_name
    try:
        return normalize_company_name(name).lower()
    except Exception:
        return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def _slug_tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2}


def build_index(dirs: Iterable[Path] = None) -> dict:
    """Map company -> [pdf paths] from the download filenames.

    download_pdf names files "<company>__<basename>__<hash>.pdf", so the company
    is recoverable from the filename. The Wayback recovery used the same
    convention. Written to disk so the scoring path does no directory scanning.
    """
    dirs = list(dirs or _PDF_DIRS)
    index: dict[str, list[str]] = {}
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.pdf")):
            comp = f.name.split("__")[0].replace("_", " ").strip()
            if not comp:
                continue
            index.setdefault(_norm(comp), []).append(str(f))
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_PATH.write_text(json.dumps(index, indent=1), encoding="utf-8")
    log.info("report index: %d companies, %d files",
             len(index), sum(len(v) for v in index.values()))
    return index


_index_cache: Optional[dict] = None


def _index() -> dict:
    global _index_cache
    if _index_cache is None:
        if _INDEX_PATH.exists():
            try:
                _index_cache = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:
                _index_cache = build_index()
        else:
            _index_cache = build_index()
    return _index_cache


def find_reports(company: str) -> list[str]:
    """PDFs belonging to `company`. Exact normalized match, then token subset.

    Token matching is deliberately strict (every index token must appear in the
    query or vice versa) because a loose match here attributes ANOTHER
    company's disclosures to this one -- the same failure mode that made
    company_site_resolver abstain on Dialog/Lion/BGF.

    SUBSET GUARD (added 2026-08-26): the subset rule above still let "Bank of
    China" match "Agricultural Bank of China" and "China Railway Construction"
    match "China Railway" -- both real, distinct, separately-listed companies
    whose full name happens to be a strict superset of another real
    company's name. A generic corporate-suffix word ("bank", "group",
    "holdings", "construction"...) being the only extra token is fine to
    subset-match through (that's the case this rule exists FOR -- e.g. a
    filing under "X Holdings" for query "X"); a DISTINGUISHING qualifier
    word being the only extra token means the two names are almost
    certainly different real companies that happen to share a common noun.
    """
    idx = _index()
    key = _norm(company)
    if key in idx:
        return idx[key]
    qt = _slug_tokens(key)
    if not qt:
        return []
    best: list[str] = []
    for k, paths in idx.items():
        kt = _slug_tokens(k)
        if not kt:
            continue
        if kt <= qt or qt <= kt:
            extra = (kt | qt) - (kt & qt)
            if extra and not extra <= _GENERIC_CORP_WORDS:
                continue
            if len(best) < len(paths):
                best = paths
    return best


# Corporate-suffix words ONLY -- legal-entity-form words that never, by
# themselves, distinguish one real company from another (a filing under
# "X Holdings Ltd" for a query of "X" is the case the subset rule exists
# for). Deliberately NOT industry/sector/business-line words: confirmed
# live 2026-08-26 that "Bank of China" vs "Agricultural Bank of China" and
# "China Railway" vs "China Railway Construction" are each two entirely
# separate, independently-listed companies whose full name differs by
# exactly one such word -- "construction"/"bank" must NOT be treated as
# safe-to-ignore, or the subset rule silently misattributes one company's
# disclosures to the other. When in doubt, leave a word OUT of this set --
# that only costs a fresh search instead of a wrong document.
_GENERIC_CORP_WORDS = {
    "group", "holdings", "holding", "corporation", "corp", "company", "co",
    "limited", "ltd", "inc", "plc",
}


# ── public collector ─────────────────────────────────────────────────────────

def _srn_page_ranges(company: str) -> dict:
    """{downloaded_filename: (start, end)} from the SRN index, when known.

    Best-effort: SRN covers EU-listed companies only, so most lookups return
    {} and parsing falls back to reading from page 1. Never raises -- a
    missing index must not cost us the report entirely.
    """
    try:
        from agentic_estimation.layer_1 import srn_report_source as srn
        from calibration.report_coverage import _safe_filename
    except Exception:
        return {}
    out: dict = {}
    try:
        # Deliberately queries the SCRAPED index, not whichever index lookup()
        # currently prefers. SRN's CSV export gives `Pages` -- the LENGTH of the
        # sustainability section -- but not its START offset (verified: Vestas
        # Pages=77 == scraped span 132-56+1). Only the scraped records carry
        # sust_pages, and the start offset is the whole point here.
        for rec in srn.scraped_records(company):
            if rec.get("sust_pages"):
                out[_safe_filename(rec["name"], rec["url"])] = tuple(rec["sust_pages"])
    except Exception as exc:
        log.debug("SRN page-range lookup failed for %s: %s", company, exc)
    return out


def _fetch_from_srn(company: str) -> bool:
    """Download this company's SRN report if it is indexed. True if we got one.

    Deliberately narrow: one company, one PDF, only when that company is being
    scored. Never raises -- an unreachable publisher host is a missing signal,
    not a failed run (measured: 2 of 8 sampled SRN links were unreachable, all
    publisher-side).
    """
    try:
        from agentic_estimation.layer_1.srn_report_source import fetch_srn_report
    except Exception:
        return False
    try:
        path = fetch_srn_report(company)
    except Exception as exc:
        log.info("SRN fetch failed for %s: %s: %s", company, type(exc).__name__, exc)
        return False
    if not path:
        return False
    # find_reports reads a prebuilt index; a file that landed after it was
    # built is invisible until the cache is dropped.
    global _index_cache
    _index_cache = None
    build_index()
    return True


def report_tags_are_one_source() -> str:
    """Prefix shared by every tag this collector emits.

    Corroboration logic counts distinct source_tags; without collapsing on this
    prefix, one report would corroborate itself three times over.
    """
    return _SHARED_TAG_PREFIX


def fetch_report_signals(company: str, max_reports: int = 2,
                         fetch_missing: bool = _FETCH_MISSING,
                         country: Optional[str] = None) -> dict[str, str]:
    """{tag: text} from this company's reports, or {}.

    Emits up to three small pillar-targeted tags rather than one oversized blob
    -- see the module docstring for why that is not an optimisation but a
    correctness requirement.

    fetch_missing=True (default) will download the report on demand when the
    company is in the SRN index but has nothing on disk yet. Set False -- or
    ESG_REPORT_FETCH=0 -- for a strictly offline run.

    country: optional, forwarded to the search fallback ONLY (SRN is keyed
    by LEI/ISIN, not name+country, so it needs no disambiguation help).
    Disambiguates a bare company name that collides with a real, unrelated
    company elsewhere (e.g. "Humana" the US insurer vs. "Humana AB", a
    Swedish care-services company) -- see _fetch_via_search's docstring.
    None (default) preserves prior behavior exactly.

    ORDER FLIPPED 2026-10-06 (user instruction): SEARCH FIRST, disk cache
    is now a FALLBACK, not the primary source. Confirmed live on a real
    company (Toyota): the old disk-first order returned a 2009
    sustainability report every single run -- 17 years stale -- because
    find_reports() found SOMETHING on disk and the code never looked
    further. There is no staleness check anywhere in this pipeline, and
    adding one (e.g. parsing a year out of the filename) is a weaker fix
    than just not trusting old cached files as the default path. The
    search call itself now uses year-qualified queries (see
    discover_reports._year_qualified_templates) that try the current year
    first, so a correct, CURRENT document is what gets found and cached
    going forward -- the disk cache still exists (fetch results are still
    written to the same directory find_reports() indexes), it is just no
    longer checked BEFORE a fresh search is attempted.

    Real cost accepted knowingly: every company now pays a live web-search
    round-trip on every call, not just companies with nothing cached
    before. fetch_missing=False (or ESG_REPORT_FETCH=0) skips search
    entirely and falls back to whatever is on disk, same as before, for a
    strictly offline run.
    """
    paths: list[str] = []
    if fetch_missing and _SEARCH_FALLBACK:
        # SEARCH FIRST, not last resort -- see docstring above for why.
        found_path = _fetch_via_search(company, country=country)
        if found_path:
            global _index_cache
            _index_cache = None
            build_index()
            paths = find_reports(company)
            if not paths:
                # find_reports's strict name matching didn't recognize the
                # downloaded file's location against the index rebuild --
                # fall back to using the search hit directly rather than
                # losing a real, entity-verified document to an index miss.
                paths = [found_path]

    if not paths and fetch_missing:
        # Search found nothing live -- try SRN's own index (exact LEI/ISIN
        # match, zero network cost beyond the initial download). Kept as
        # the SECOND source now, not the first, since a live search result
        # is more likely to be current than SRN's periodically-refreshed
        # index.
        if _fetch_from_srn(company):
            paths = find_reports(company)

    if not paths:
        # LAST RESORT: whatever is already on disk from a prior run,
        # however old. Only reached when search is disabled
        # (fetch_missing=False / ESG_REPORT_FETCH=0) or both search and
        # SRN genuinely found nothing live -- a real report, even a stale
        # one, is still better evidence than none for a company search
        # engines can't find anything current for.
        paths = find_reports(company)

    if not paths:
        return {}

    ranges = _srn_page_ranges(company)
    texts: list[str] = []
    for p in paths[:max_reports]:
        # A combined annual report hides its sustainability statement in the
        # middle; parsing from page 1 returns financial front-matter instead.
        # SRN publishes the exact range, and using it measurably improves the
        # selection (Vestas: G density 2.5 -> 7.4, S from a section header to
        # actual injury counts, E 20.2 -> 22.0 with Scope 3 figures).
        res = parse_pdf(p, page_range=ranges.get(Path(p).name))
        if res.get("ok"):
            texts.append(res["text"])
        else:
            log.info("report parse miss for %s (%s): %s",
                     company, Path(p).name, res.get("error"))
    if not texts:
        return {}

    # PER-DOCUMENT SCORING, NEVER A JOINED BLOB. Confirmed live 2026-09-10:
    # joining every cached document's text before selecting (the old
    # `full = "\n\n".join(texts)`) meant a STALE candidate left on disk from
    # an earlier, since-fixed search bug (e.g. Apple's real Environmental
    # Progress Report sitting alongside a mis-fetched 10-K from a prior run)
    # permanently contaminated every future fetch -- the 10-K's tax-rate and
    # RSU-vesting text won the S/G pillar selections outright once merged
    # into the same pool, even though the real report alone scores cleanly.
    # Nothing on disk is ever cleaned up, so this bug compounds silently
    # over time rather than self-correcting.
    #
    # Fix: when only one document exists, it is used for all three pillars
    # (unchanged behaviour -- there is nothing to compare it against). When
    # multiple documents exist, each pillar is scored SEPARATELY against
    # EACH document, and only the single best-scoring document is used as
    # that pillar's source -- different pillars may legitimately draw from
    # different documents (this is exactly the shape the missing-pillar
    # fallback below already produces; scoring per-document here makes the
    # PRIMARY fetch behave the same way when disk already holds >1 candidate).
    out: dict[str, str] = {}
    for pillar in ("E", "S", "G"):
        if len(texts) == 1:
            candidates = [texts[0]]
        else:
            candidates = texts

        best_sel, best_density = None, 0.0
        for text in candidates:
            sel = _select(text, pillar)
            if not sel or len(sel) <= 200:
                continue
            # Not every downloaded PDF is a sustainability report. Several are
            # 10-Ks and annual financial reports, where the best-scoring blocks
            # for a pillar can still be tax tables or non-GAAP reconciliations
            # (measured on Cardinal Health's 10-K). Emitting those invites the
            # extractor to invent ESG claims from accounting text, so require
            # the selection to be genuinely about the pillar before shipping it.
            own_density = _pillar_density(sel, pillar)
            if own_density < _MIN_PILLAR_DENSITY:
                continue
            # CROSS-PILLAR DOMINANCE CHECK. Confirmed live 2026-09-09
            # (Microsoft): the absolute floor above is not enough on its own
            # -- Microsoft's S selection cleared it (density 2.84) purely
            # from repeated hits of the single word "employee", every one of
            # them actually part of "Employee Commuting", a Scope 3
            # EMISSIONS category name, not a real S-pillar claim.
            # Block-level bidding confirmed the same text is overwhelmingly
            # E-dominant (E density on this selection: ~20, vs S's 2.84 on
            # itself). An absolute floor asks "does this clear a low bar for
            # MY pillar?"; this asks the question that actually matters:
            # "is this text more about a DIFFERENT pillar than the one
            # claiming it?"
            other_densities = {p: _pillar_density(sel, p) for p in ("E", "S", "G") if p != pillar}
            dominant_other = max(other_densities, key=other_densities.get)
            if other_densities[dominant_other] > 2.0 * own_density:
                continue
            # FINANCIAL-STATEMENT VETO. Confirmed live 2026-09-10 (Apple,
            # Nike, Pfizer): the two checks above are not enough for G
            # specifically -- a 10-K's tax/compensation notes are genuinely
            # G-dominant relative to their OWN E/S content (there is no
            # E/S text to compete with inside a tax note), so they clear
            # both the density floor and the cross-pillar dominance check
            # while still being pure accounting text, not a real governance
            # disclosure. See _financial_hits/_FINANCIAL_STATEMENT_MARKERS.
            if _financial_hits(sel) >= _COVERAGE_FINANCIAL_VETO:
                log.info("%s: %s selection looks like financial-statement "
                         "text (%d marker hits), not a real disclosure -- "
                         "rejecting", company, pillar, _financial_hits(sel))
                continue
            if own_density > best_density:
                best_sel, best_density = sel, own_density

        if best_sel is None:
            log.info("%s: no document scored well enough for %s "
                     "(checked %d candidate document(s))", company, pillar, len(candidates))
            continue
        out[f"{_SHARED_TAG_PREFIX}_{pillar.lower()}"] = best_sel

    # MISSING-PILLAR FALLBACK. Confirmed live 2026-09-09 (Microsoft): some
    # companies do not publish one combined E+S+G report at all -- Microsoft's
    # "Environmental Sustainability Report" is genuinely E-only (bidding-
    # tested against the full 423-block document: 261 blocks won by E, 0 by
    # G, 2 by S and both of those were content-free "Employee Commuting"
    # table-header fragments, not real S evidence). No amount of better
    # selection recovers a pillar that document structurally never covers.
    # Microsoft's actual social-pillar content lives in a SEPARATE, standalone
    # "Global Diversity & Inclusion Report" PDF; a generic "sustainability
    # report" search just keeps finding the same E-only document, so the
    # follow-up here uses a PILLAR-SPECIFIC query (see discover_reports.py's
    # _PILLAR_QUERY_TEMPLATES) instead of retrying the same search.
    #
    # PROPORTIONAL BY DESIGN: fires at most once per missing pillar per
    # company, and only when this run already found the company has SOME
    # report but it's silent on a pillar -- a company with one good combined
    # report never triggers this at all, so the common case pays zero extra
    # search cost. Tagged report_pdf2_* (still under the shared report_pdf
    # prefix so report_tags_are_one_source() keeps treating every report-
    # derived tag as one corroboration source) to avoid colliding with the
    # primary document's tags.
    missing = [p for p in ("E", "S", "G") if f"{_SHARED_TAG_PREFIX}_{p.lower()}" not in out]
    if missing and fetch_missing and _SEARCH_FALLBACK:
        for pillar in missing:
            extra_path = _fetch_pillar_via_search(company, pillar, country=country)
            if not extra_path:
                continue
            res = parse_pdf(extra_path)
            if not res.get("ok"):
                continue
            extra_sel = _select(res["text"], pillar)
            if not extra_sel or len(extra_sel) <= 200:
                continue
            if _pillar_density(extra_sel, pillar) < _MIN_PILLAR_DENSITY:
                log.info("%s: %s fallback search found a document but it's "
                         "also too thin on %s -- dropping", company, pillar, pillar)
                continue
            if _financial_hits(extra_sel) >= _COVERAGE_FINANCIAL_VETO:
                log.info("%s: %s fallback search found another financial-"
                         "statement-shaped document -- dropping, pillar left "
                         "unfilled (genuinely irredeemable for now)", company, pillar)
                continue
            out[f"{_SHARED_TAG_PREFIX}2_{pillar.lower()}"] = extra_sel
            log.info("%s: recovered %s via pillar-specific fallback search", company, pillar)

    if out:
        log.info("%s: report signals %s (from %d pdf(s), %d chars total parsed)",
                 company, {k: len(v) for k, v in out.items()}, len(texts),
                 sum(len(t) for t in texts))
    return out
