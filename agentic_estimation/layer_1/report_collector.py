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


def _fetch_via_search(company: str) -> Optional[str]:
    """One company, one search-discovered PDF, downloaded to the same
    directory the SRN path uses so find_reports() picks it up on the next
    call. Returns the local path, or None on no hit / any failure -- never
    raises (matches _fetch_from_srn's fail-open discipline: an unreachable
    search engine or a rejected document is a missing signal, not a crash)."""
    try:
        from calibration.discover_reports import from_search
        from calibration.report_coverage import _PDF_DIR
    except Exception as exc:
        log.info("search-fallback import failed for %s: %s", company, exc)
        return None
    try:
        path = from_search(company, _PDF_DIR)
    except Exception as exc:
        log.info("search-fallback failed for %s: %s: %s", company, type(exc).__name__, exc)
        return None
    return path

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


def _score_block(block: str, pillar: str) -> float:
    """Higher = more likely to carry an extractable ESG metric for `pillar`.

    Term hits alone rank the CEO letter top -- it name-drops every pillar and
    quantifies nothing. What makes a claim extractable is a number WITH A UNIT
    next to a pillar term, so units are scored separately from bare digits;
    page numbers and years carry no unit and no longer inflate the score.
    """
    if _is_navigation(block):
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
            if len(best) < len(paths):
                best = paths
    return best


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
                         fetch_missing: bool = _FETCH_MISSING) -> dict[str, str]:
    """{tag: text} from this company's reports, or {}.

    Emits up to three small pillar-targeted tags rather than one oversized blob
    -- see the module docstring for why that is not an optimisation but a
    correctness requirement.

    fetch_missing=True (default) will download the report on demand when the
    company is in the SRN index but has nothing on disk yet. Set False -- or
    ESG_REPORT_FETCH=0 -- for a strictly offline run.
    """
    paths = find_reports(company)
    if not paths and fetch_missing:
        # Nothing on disk -- but this company may be in the SRN index, in which
        # case its report is one download away. Without this the collector was
        # limited to whatever a previous bulk run happened to fetch: Kering,
        # Covestro and Balder all sat in the SRN index with 2 reports each and
        # still returned {} because nobody ever pulled them.
        #
        # On-demand by design: one company, at most one PDF, only when it is
        # actually being scored.
        if _fetch_from_srn(company):
            paths = find_reports(company)

    if not paths and fetch_missing and _SEARCH_FALLBACK:
        # SRN has nothing -- fall back to real web search (see
        # _fetch_via_search docstring). This is what catches the companies
        # SRN structurally cannot: non-EU, private, or otherwise unlisted
        # ones that still publish a real, findable report (confirmed
        # concretely 2026-08-18: AVZ Minerals, an Australian mining company,
        # has a real ESG/financial report hosted on Squarespace -- SRN will
        # never index it, search finds it directly).
        found_path = _fetch_via_search(company)
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

    full = "\n\n".join(texts)
    out: dict[str, str] = {}
    for pillar in ("E", "S", "G"):
        sel = _select(full, pillar)
        if not sel or len(sel) <= 200:
            continue
        # Not every downloaded PDF is a sustainability report. Several are
        # 10-Ks and annual financial reports, where the best-scoring blocks for
        # a pillar can still be tax tables or non-GAAP reconciliations
        # (measured on Cardinal Health's 10-K). Emitting those invites the
        # extractor to invent ESG claims from accounting text, so require the
        # selection to be genuinely about the pillar before shipping it.
        if _pillar_density(sel, pillar) < _MIN_PILLAR_DENSITY:
            log.info("%s: dropping %s selection -- too few pillar terms "
                     "(likely a financial filing, not an ESG report)",
                     company, pillar)
            continue
        out[f"{_SHARED_TAG_PREFIX}_{pillar.lower()}"] = sel
    if out:
        log.info("%s: report signals %s (from %d pdf, %d chars parsed)",
                 company, {k: len(v) for k, v in out.items()}, len(texts), len(full))
    return out
