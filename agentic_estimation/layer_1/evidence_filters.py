"""
evidence_filters.py — cheap, deterministic pre-LLM evidence hygiene:
fingerprint dedup and keyword relevance gating.

WHY THIS EXISTS: analyzed via the code-review-graph MCP tool against a
separate, already-shipped ESG product's news-collection pipeline
(rupa/code/trends/collector.py). Two mechanisms there directly fix gaps in
our own signal gathering, which today has neither dedup nor a pre-LLM
relevance filter -- every headline line, however noisy or repeated, goes
straight into the pillar-extractor prompt.

1. FINGERPRINT DEDUP: sha256(normalized_title) catches the same story
   appearing across multiple sources (Google News RSS, Reuters feed,
   localized ESG feed can all surface the same headline) before it reaches
   the LLM -- fewer duplicate mentions inflating one controversy's apparent
   weight, fewer tokens spent on repeats.

2. KEYWORD PRE-GATE: a compiled, word-boundary, case-insensitive match
   against the same ESG keyword vocabulary already used in the NewsAPI/
   Google News queries. This does NOT replace the LLM's Step-1 relevance
   check in pillar_extractors.py (that catches semantic irrelevance a
   keyword match can't, e.g. the "yoga page" bleed-through) -- it's a
   cheaper, earlier filter that drops headline lines with literally zero
   ESG-adjacent vocabulary before they're even sent to the model, so the
   LLM's limited context budget (_MAX_TOTAL_CHARS) isn't spent on lines
   that were never going anywhere.

Both operate on our signal shape (one multi-line text blob per source, each
line typically "[date] headline") rather than rupa's per-article DB rows --
this module works at that finer grain: per HEADLINE LINE within a signal's
text, not per whole signal.
"""

import hashlib
import re
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("evidence_filters")

# ── Entity gate ──────────────────────────────────────────────────────────────
# Legal-form suffixes and filler that carry no identifying power. "Ltd" appearing
# in a snippet says nothing about whether that snippet is about THIS company, so
# matching on it would pass essentially every corporate page.
_ENTITY_STOPWORDS = frozenset((
    "the", "and", "for", "of", "a", "an",
    "inc", "inc.", "ltd", "ltd.", "llc", "l.l.c", "plc", "co", "co.", "corp",
    "corp.", "corporation", "company", "group", "holdings", "holding",
    "limited", "international", "global", "worldwide", "enterprises",
    "sa", "s.a", "ag", "gmbh", "bv", "b.v", "nv", "n.v", "as", "ab", "oy",
    "spa", "s.p.a", "srl", "s.r.l", "sas", "sarl", "pty", "pte", "kk",
    "services", "solutions", "systems", "technologies", "technology",
))

# Ground-truth leakage. Our accuracy answer-key (bcorp / upright) is published
# on the open web, so a plain company search can return the very score the
# pipeline is trying to predict. Any snippet carrying these phrases would feed
# the target back in as an input feature -- inflating tune-set accuracy while
# teaching the model nothing that generalizes. Dropped before extraction.
_LEAKAGE_MARKERS = (
    "b impact assessment",
    "median score for ordinary businesses",
    "overall score of",
    "certified b corporation",
    "certified since",
    "b corp score",
    "upright net impact",
    "net impact ratio",
)


def _company_tokens(company: str) -> list[str]:
    """Identifying tokens from a company name: alphanumeric, >2 chars, not a
    legal-form suffix. "Sales: Untangled(TM)" -> ["sales", "untangled"]."""
    words = re.sub(r"[^a-z0-9 ]+", " ", (company or "").lower()).split()
    return [w for w in words if len(w) > 2 and w not in _ENTITY_STOPWORDS]


def mentions_company(text: str, company: str, url: str = "") -> bool:
    """True if `text` or `url` plausibly refers to `company`.

    A hit on ANY identifying token is deliberate: search snippets truncate
    names ("Isigny Sainte-Mere" -> "Isigny"), and requiring the full name
    would drop legitimate evidence. The URL counts because a company's own
    domain often omits the name from the visible snippet.

    Companies whose name is entirely legal-form/filler yield no tokens; those
    return True (fail open) rather than dropping every result for them.
    """
    toks = _company_tokens(company)
    if not toks:
        return True
    haystack = f"{text} {url}".lower()
    return any(t in haystack for t in toks)


def has_ground_truth_leakage(text: str) -> bool:
    """True if the text contains a benchmark score we are trying to predict."""
    low = (text or "").lower()
    return any(m in low for m in _LEAKAGE_MARKERS)

# Same breadth as signal_agent._GOOGLE_RSS_KEYWORDS / news_api query, flattened
# to bare words/phrases for compiled word-boundary matching (no OR/quote syntax).
_ESG_RELEVANCE_TERMS = (
    "esg", "sustainability", "sustainable", "climate", "emissions", "emission",
    "carbon", "renewable", "pollution", "waste", "recycling", "environmental",
    "net zero", "carbon neutral", "workers", "labor", "labour", "employees",
    "lawsuit", "human rights", "discrimination", "safety", "supply chain",
    "community", "controversy", "fine", "penalty", "governance", "board",
    "executive", "fraud", "compliance", "transparency", "scandal", "corruption",
    "data breach", "diversity", "biodiversity", "deforestation", "pledge",
    "certification", "audit", "regulator", "regulation", "violation", "strike",
    "union", "wage", "harassment", "injury", "spill", "contamination",
    # E — extend: energy/water/climate-target specifics an English feed uses
    "greenhouse gas", "scope 1", "scope 2", "scope 3", "co2", "ghg",
    "renewable energy", "solar", "wind power", "energy efficiency",
    "water", "wastewater", "effluent", "hazardous", "landfill", "circular economy",
    "sbti", "science based targets", "cdp", "tcfd", "decarbonization",
    "decarbonisation", "climate risk", "greenwashing", "epa",
    # S — extend: pay/safety/rights specifics
    "gender pay", "pay gap", "pay equity", "inclusion", "gender balance",
    "workforce", "occupational", "fatality", "trir", "osha", "accident",
    "modern slavery", "forced labor", "forced labour", "child labor",
    "child labour", "living wage", "layoffs", "whistleblower", "grievance",
    "collective bargaining", "product recall", "worker",
    # G — extend: compliance/ethics/board specifics
    "independent director", "board independence", "proxy statement",
    "executive pay", "shareholder", "bribery", "antitrust", "money laundering",
    "sanctions", "tax", "settlement", "sec ", "litigation", "misconduct",
    "conflict of interest", "code of conduct", "ethics", "privacy",
)

_compiled_cache: dict[tuple, list[re.Pattern]] = {}


def _compiled_patterns(terms: tuple) -> list[re.Pattern]:
    """Compile once per distinct term set (word-boundary, case-insensitive).
    Cached by the term tuple itself (not a single module-global), so callers
    passing a different vocabulary (e.g. a country's localized ESG keywords)
    don't collide with each other's compiled patterns -- a bug documented as
    present in the source pipeline this pattern is adapted from."""
    key = terms
    if key not in _compiled_cache:
        _compiled_cache[key] = [re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE) for t in terms]
    return _compiled_cache[key]


def matches_esg_keywords(text: str, extra_terms: tuple = ()) -> bool:
    """True if `text` contains at least one ESG-relevance term (default
    vocabulary plus any caller-supplied extra terms, e.g. a country's
    localized keyword list). Cheap pre-filter -- NOT a substitute for the
    LLM's semantic relevance check."""
    terms = _ESG_RELEVANCE_TERMS + tuple(extra_terms)
    patterns = _compiled_patterns(terms)
    return any(p.search(text) for p in patterns)


def filter_search_results(
    results: list[dict],
    company: str,
    prefix: str = "",
    require_entity: bool = True,
) -> list[dict]:
    """Gate a list of raw search results (each {"body", "href"}) individually,
    BEFORE they are joined into one signal blob.

    This is the fix for the dominant evidence-quality failure: joining first
    meant one irrelevant result (a different company's proxy statement, or
    generic "what is an anti-bribery policy" filler returned when a site:
    query found nothing) became indistinguishable from real evidence inside
    a single string. Scoring each result on its own lets the bad ones be
    dropped while the good ones survive.

    Returns the kept results, in order. An empty list means the search found
    nothing about this company -- callers MUST treat that as "no signal"
    rather than falling back to the unfiltered text, or the filter is moot.
    """
    kept: list[dict] = []
    dropped_entity = 0
    dropped_leak = 0

    for r in results:
        body = (r.get("body") or "").strip()
        if not body:
            continue
        url = r.get("href") or ""

        if has_ground_truth_leakage(body):
            dropped_leak += 1
            continue
        if require_entity and not mentions_company(body, company, url):
            dropped_entity += 1
            continue
        kept.append(r)

    if dropped_entity or dropped_leak:
        log.info(
            "filter_search_results [%s/%s]: kept %d/%d (dropped %d off-entity, %d leakage)",
            prefix or "?", company, len(kept), len(results), dropped_entity, dropped_leak,
        )
    return kept


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _fingerprint(title: str) -> str:
    return hashlib.sha256(_normalize_title(title).encode("utf-8")).hexdigest()


def dedup_and_filter_lines(text: str, extra_terms: tuple = (), require_keyword: bool = True) -> str:
    """Given a multi-line signal blob (one headline per line, typically
    "[date] headline text" or "prefix: headline text"), drop exact-duplicate
    lines (by normalized-title fingerprint) and, optionally, lines with no
    ESG-relevance keyword hit at all. Non-headline signals (a single prose
    paragraph from Wikipedia/DDG, not one-line-per-article) pass through
    unchanged -- this only meaningfully applies to the multi-headline feeds
    (Google News RSS, Reuters, localized ESG, NewsAPI).

    require_keyword=False for feeds that are already query-targeted (a
    country's localized ESG feed, a site:-restricted feed) -- mirrors the
    source pipeline's own rule that query-targeted feeds skip the keyword
    gate since the query string itself is the relevance filter.
    """
    lines = text.splitlines()
    if len(lines) < 2:
        return text  # single-blob prose signal, nothing to dedup/filter

    seen_fp: set = set()
    kept: list[str] = []
    dropped_dupes = 0
    dropped_irrelevant = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        fp = _fingerprint(stripped)
        if fp in seen_fp:
            dropped_dupes += 1
            continue

        if require_keyword and not matches_esg_keywords(stripped, extra_terms):
            dropped_irrelevant += 1
            continue

        seen_fp.add(fp)
        kept.append(line)

    if dropped_dupes or dropped_irrelevant:
        log.debug("dedup_and_filter_lines: kept %d, dropped %d dupes, %d irrelevant",
                   len(kept), dropped_dupes, dropped_irrelevant)

    return "\n".join(kept) if kept else text  # never return empty -- fail open to original
