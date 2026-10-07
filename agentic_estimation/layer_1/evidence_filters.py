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

import numpy as np

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
    # S — extend again: found live against calibration/trainset_evidence.jsonl
    # (194/669 real-evidence rows had zero keyword hit at all -- see
    # matches_esg_keywords's own miss analysis, 2026-09-16). "attrition" and
    # "gender" are safe bare -- checked against the full RSS frozen corpus
    # (4,709 real headlines), no false-positive collisions found (every
    # "gender" hit was a real Bloomberg Gender-Equality Index story). "ltifr"
    # (the actual industry acronym for lost-time injury frequency rate) has no
    # competing everyday meaning. "employee turnover"/"staff turnover" kept as
    # PHRASES, not bare "turnover" -- bare "turnover" hit exactly once in the
    # same corpus and it was revenue turnover ("Dufry sees hike in 2022
    # turnover"), not an HR signal; the phrase form avoids that collision
    # entirely since revenue turnover is essentially never phrased that way.
    # "female employees"/"female board" kept as phrases rather than bare
    # "female" for the same reason -- bare "female" collided with lifestyle/
    # marketing coverage ("Bumble... Isn't Female, It's Female Marketing") in
    # the same corpus check.
    "attrition", "gender", "ltifr", "employee turnover", "staff turnover",
    "female employees", "female board",
    # G — extend: compliance/ethics/board specifics
    "independent director", "board independence", "proxy statement",
    "executive pay", "shareholder", "bribery", "antitrust", "money laundering",
    "sanctions", "tax", "settlement", "sec ", "litigation", "misconduct",
    "conflict of interest", "code of conduct", "ethics", "privacy",
    # G — extend again: same miss analysis as the S block above. "certified"
    # and "gri" (the GRI reporting standard) are safe bare -- checked against
    # the RSS corpus, no false-positive collisions found. Kept as PHRASES
    # rather than bare words: "b corp" (a specific credential, not just any
    # "corp"), "impact report"/"disclosure report"/"non-financial statement"
    # (report-naming conventions, not generic "report"), "global compact" (the
    # UN initiative, not generic "compact"), "class action"/"legal
    # proceedings"/"material claim"/"related-party" (formal litigation
    # phrasing, distinct from casual "lawsuit" already in the list above),
    # "proxy advisers" (kept alongside the existing "proxy statement" --
    # bare "proxy" rejected: real corpus hits were a mix of governance signal
    # and generic finance/tech-proxy usage, e.g. "proxy fight"/"proxy server"
    # risk, too ambiguous to add unqualified).
    "certified", "gri", "remuneration", "b corp", "impact report",
    "disclosure report", "non-financial statement", "global compact",
    "class action", "legal proceedings", "material claim", "related-party",
    "proxy advisers",
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


# ── Embedding-based relevance (MiniLM, multi-part/max design) ───────────────
#
# Reuses evidence_classifier._get_embedder()'s module-level SentenceTransformer
# singleton (MiniLM, sentence-transformers/all-MiniLM-L6-v2) -- pre-warmed once
# at app startup (see app.py's _prewarm_embedder), not loaded here.
#
# DESIGN HISTORY (measured 2026-09-16, three candidate designs compared on a
# TEMPLATE-level train/test split of calibration/trainset_evidence.jsonl --
# a row-level split was tried first and rejected: 68% of "test" rows were the
# same synthetic sentence template as a train row with only the company name
# swapped, which silently inflated every score):
#   1. One averaged positive vector (mean of all seed terms) minus one
#      averaged negative vector: F1 0.794 on the held-out template split.
#      Blending ~30 seed terms into a single vector dilutes sharp single-word
#      matches -- a real "whistleblowing hotline" claim scored only +0.02 net
#      (barely above zero) because averaging in 10 unrelated S-pillar seeds
#      pulled the vector away from the one seed ("whistleblower") that
#      actually matched at 0.38 on its own.
#   2. Max-of-individual-terms on the positive side only (no negative
#      anchors): rejected earlier, even noisier -- a single incidental word
#      match (e.g. "employees" in "office snack menu for employees") spiked
#      the score with nothing to counterbalance it.
#   3. MULTI-PART (this one): max similarity across each INDIVIDUAL positive
#      seed term, minus max similarity across each INDIVIDUAL negative
#      (confounder) seed term. F1 0.874 on the same held-out split -- best of
#      the three, and the only one that gets both the "whistleblowing" case
#      (best positive term dominates, isn't diluted by averaging) AND the
#      "Labor Day" case (the negative anchor's own best match, "Labor Day
#      holiday celebration", outscores the positive "labor" match) right.
#
# NOT a replacement for matches_esg_keywords -- same "cheap pre-filter, not
# the LLM's real relevance check" role, just a second, differently-shaped net
# to catch what static word-boundary matching structurally cannot (headlines
# that are on-topic without using any of the listed words, e.g. "green AI
# projects" or "comply with EU gatekeeper rules" -- neither contains
# "environmental"/"compliance" verbatim). Callers should OR the two, not
# choose one -- each catches real cases the other misses.
_EMBED_POSITIVE_TERMS = (
    # E
    "carbon", "climate", "emissions", "environmental", "pollution",
    "renewable", "sustainability", "waste", "energy",
    # S -- "whistleblower" included here (not just G) since whistleblowing
    # mechanisms are as much a labor/workplace-conduct signal as a board one.
    "labor", "workers", "employees", "safety", "discrimination", "harassment",
    "wages", "diversity", "workforce", "human rights", "whistleblower",
    # G -- "board"/"director"/"oversight" given equal footing with the
    # misconduct terms (fraud/corruption/bribery/antitrust/ethics) rather than
    # averaged together, so a governance-STRUCTURE headline (e.g. "board
    # approves executive compensation plan") isn't drowned out by the
    # misconduct cluster's mutual similarity the way it was in an earlier,
    # unbalanced version of this list (measured: 0.165 vs 0.270 net after
    # rebalancing).
    "governance", "board", "executive compensation", "oversight", "director",
    "fraud", "corruption", "bribery", "antitrust", "ethics",
)

# Confounders: real headline patterns that share surface vocabulary with the
# positive terms above but are NOT ESG evidence -- calendar/holiday language
# ("Labor Day"), HR-marketing puff (snacks/perks/hiring announcements), and
# product-spec/routine-business text that happens to contain an E/G-flavored
# word (carbon fiber, energy efficiency spec, board game/leaderboard,
# quarterly earnings). Each subtracts from whichever positive term it
# happens to share vocabulary with, which is what correctly rejects "Labor
# Day" (negative anchor "Labor Day holiday celebration" scores 0.605 against
# it, beating the positive "labor" match at 0.406) without needing a
# hardcoded blocklist of exact phrases.
_EMBED_NEGATIVE_TERMS = (
    "Labor Day holiday celebration", "company holiday and office celebration",
    "new employee hire announcement", "office perks and free snacks",
    "product energy efficiency spec sheet", "carbon fiber material product design",
    "skateboard surfboard sports equipment", "video game leaderboard ranking",
    "quarterly earnings call routine update", "new product launch announcement",
)

_embed_pos_matrix = None
_embed_neg_matrix = None


def _embed_term_matrices():
    """Lazily encode the fixed positive/negative term lists once per process
    and cache the resulting matrices -- these never change at runtime, so
    there is no reason to re-encode them on every call. Uses the SAME
    embedder singleton evidence_classifier.py already loads (pre-warmed at
    app startup), not a second model instance."""
    global _embed_pos_matrix, _embed_neg_matrix
    if _embed_pos_matrix is None:
        from agentic_estimation.layer_2.evidence_classifier import _get_embedder
        embedder = _get_embedder()
        _embed_pos_matrix = embedder.encode(list(_EMBED_POSITIVE_TERMS), normalize_embeddings=True)
        _embed_neg_matrix = embedder.encode(list(_EMBED_NEGATIVE_TERMS), normalize_embeddings=True)
    return _embed_pos_matrix, _embed_neg_matrix


def matches_esg_embedding(text: str, threshold: float = 0.0) -> bool:
    """True if `text`'s best single-term match among _EMBED_POSITIVE_TERMS
    beats its best single-term match among _EMBED_NEGATIVE_TERMS by more
    than `threshold`. See the module comment above for why max-of-terms
    (not one averaged vector) is the design that was actually measured to
    work. threshold=0.0 (both maxes compete on equal footing) is the value
    validated in the template-split measurement above -- not yet re-tuned
    beyond that one measurement."""
    from agentic_estimation.layer_2.evidence_classifier import _get_embedder
    pos_matrix, neg_matrix = _embed_term_matrices()
    embedder = _get_embedder()
    emb = embedder.encode([text], normalize_embeddings=True)[0]
    pos_score = float(np.max(pos_matrix @ emb))
    neg_score = float(np.max(neg_matrix @ emb))
    return (pos_score - neg_score) > threshold


def matches_esg_relevance(text: str, extra_terms: tuple = ()) -> bool:
    """OR of the two independent relevance checks -- keyword match (cheap,
    exact, zero-cost) and embedding match (catches on-topic phrasing that
    uses none of the listed words). Each catches real cases the other
    misses (measured on a held-out template split, 2026-09-16); neither
    alone is a safe replacement for the other. Prefer this over calling
    matches_esg_keywords alone for any NEW call site -- existing call sites
    are migrated separately so each can be verified independently."""
    return matches_esg_keywords(text, extra_terms) or matches_esg_embedding(text)


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

        # OR of keyword + embedding relevance (see matches_esg_relevance's own
        # comment) -- catches on-topic headlines that use none of the listed
        # words (e.g. "green AI projects", "comply with EU gatekeeper rules"),
        # measured to matter on real RSS headlines this exact gate sees.
        if require_keyword and not matches_esg_relevance(stripped, extra_terms):
            dropped_irrelevant += 1
            continue

        seen_fp.add(fp)
        kept.append(line)

    if dropped_dupes or dropped_irrelevant:
        log.debug("dedup_and_filter_lines: kept %d, dropped %d dupes, %d irrelevant",
                   len(kept), dropped_dupes, dropped_irrelevant)

    return "\n".join(kept) if kept else text  # never return empty -- fail open to original
