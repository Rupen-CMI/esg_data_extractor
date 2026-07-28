"""
sector_matcher.py — fuzzy sector-label matching for peer_anchor.py, using a
pure-numpy TF-IDF cosine similarity (no sklearn/sentence-transformers/torch
dependency -- none of those are installed in this environment).

WHY THIS EXISTS: analyzed via the code-review-graph MCP tool against a
separate, already-shipped ESG product that matches companies to sectors via
OpenAI embeddings + cosine similarity, with an explicit confidence floor and
a "no match above threshold" flag rather than silently taking the best
candidate. This module adopts that STRUCTURE (score + floor + explicit
no-match signal) without the embedding dependency.

CONCRETE GAP THIS FIXES: peer_anchor_collector.find_peers() matches `sector`
with SQL `=` against BOTH bcorp_lookup.sasb_sector (only 4 coarse values:
apparel_retail/general/manufacturing/services) AND upright_lookup.industry
(30 real fine-grained values: "Automotive", "Food and Beverage", "Electronics",
etc. -- confirmed via direct query) using the SAME literal string. A real
sector like "Beverages" matches neither exactly, so upright's fine-grained
peer pool (10,086 companies across genuinely useful categories) is
effectively unreachable from peer_anchor.py today -- only the bcorp coarse
bucket fallback (classify_manufacturing_vs_services) was ever wired in.

This is deliberately NOT semantic (no embeddings) -- it is word-overlap
cosine similarity (TF-IDF-weighted) between the caller's sector string and
each of upright's 30 known industry labels. This still catches the exact
cases that mattered in practice: "Beverages" vs "Food and Beverage" share
the token "beverage"/"beverages" after stemming-free normalization; "Auto
Manufacturing" vs "Automotive" share "auto". It will NOT catch pure synonyms
with zero shared tokens (e.g. "Car Manufacturing" vs "Automotive" -- no
shared root) -- a real limitation, documented rather than hidden, and the
reason a confidence floor + explicit no-match path exists rather than always
returning the best-scoring label regardless of quality.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("sector_matcher")

# Below this cosine similarity, the match is not trusted -- caller falls back
# to its existing coarse-bucket path rather than accept a weak/wrong label.
_MIN_SIMILARITY = 0.30

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A few domain stopwords that appear in almost every upright label and would
# otherwise dominate every cosine score regardless of actual topical overlap.
# "manufacturing"/"manufacturer" is included deliberately: it appears across
# unrelated upright labels ("Consumer Product Manufacturing", "Industrial
# Manufacturing and Services") and, without stemming to catch the REAL
# semantic overlap (auto/automotive, steel/steel), was pulling every
# "X Manufacturing" query onto whichever manufacturing label happened to
# have the most other token matches -- worse than no match at all.
_STOPWORDS = {
    "and", "services", "service", "products", "product", "companies", "company",
    "manufacturing", "manufacturer", "manufacturers", "industry", "industrial",
}

# Irregular pairs no suffix rule catches -- small, curated, extend as misses
# are found (same discipline as company_metadata.py's _BRAND_ALIASES).
_STEM_ALIASES = {"automotive": "auto", "auto": "auto", "cars": "auto", "car": "auto"}

# Minimal suffix stripping (NOT a real stemmer) so "beverages"/"beverage",
# "electronics"/"electronic" share a root token. Order matters -- longer,
# more specific suffixes checked first: "electronics" must hit "-ics"->"-ic"
# before the generic "-s" rule would otherwise strip it to "electronic" too
# (harmless here) but "beverages" must NOT hit a blanket "-es"->"" rule
# (that gives "beverag", which no longer matches "beverage"'s stem) -- plain
# "-s" removal is the only regular-plural rule that keeps both forms equal.
_SUFFIX_RULES = [("ics", "ic"), ("ies", "y"), ("s", "")]


def _stem(token: str) -> str:
    if token in _STEM_ALIASES:
        return _STEM_ALIASES[token]
    for suffix, replacement in _SUFFIX_RULES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)] + replacement
    return token


@dataclass
class SectorMatch:
    label: str                 # the matched candidate label (e.g. upright's industry string)
    similarity: float           # cosine similarity, 0-1
    matched: bool               # False if similarity < _MIN_SIMILARITY -- caller should fall back
    all_scores: list[tuple]    # (label, similarity) for every candidate, sorted desc -- for audit


def _tokenize(text: str) -> list[str]:
    raw = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]
    return [_stem(t) for t in raw]


def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _idf(all_docs_tokens: list[list[str]]) -> dict[str, float]:
    """Standard smoothed IDF over the candidate label set itself (small, fixed
    vocabulary -- e.g. upright's 30 industry labels) plus the query, so a
    token appearing in every candidate (uninformative) scores near zero."""
    n_docs = len(all_docs_tokens)
    df: Counter = Counter()
    for tokens in all_docs_tokens:
        df.update(set(tokens))
    return {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = _tf(tokens)
    return {term: count * idf.get(term, 1.0) for term, count in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def best_sector_match(query_sector: Optional[str], candidate_labels: list[str]) -> Optional[SectorMatch]:
    """Fuzzy-match `query_sector` (a company's own free-text sector/industry
    string, e.g. from Wikidata or a ground-truth record) against a fixed list
    of candidate labels (e.g. the 30 distinct upright_lookup.industry values).

    Returns None only if query_sector is empty or no candidate has any token
    overlap at all (cosine of everything is exactly 0). Otherwise returns a
    SectorMatch with `matched=False` when the best score is below
    _MIN_SIMILARITY -- caller decides what to do (fall back, or use it anyway
    at reduced confidence), mirroring the source pipeline's explicit
    no-match-above-threshold pattern rather than silently taking an argmax
    that might be a bad match.
    """
    if not query_sector or not candidate_labels:
        return None

    query_tokens = _tokenize(query_sector)
    if not query_tokens:
        return None

    label_tokens = [_tokenize(label) for label in candidate_labels]
    idf = _idf(label_tokens + [query_tokens])
    query_vec = _tfidf_vector(query_tokens, idf)

    scores = []
    for label, tokens in zip(candidate_labels, label_tokens):
        label_vec = _tfidf_vector(tokens, idf)
        sim = _cosine(query_vec, label_vec)
        scores.append((label, sim))

    scores.sort(key=lambda x: -x[1])
    best_label, best_sim = scores[0]

    if best_sim <= 0.0:
        log.debug("best_sector_match(%r) -> no token overlap with any of %d candidates",
                   query_sector, len(candidate_labels))
        return None

    matched = best_sim >= _MIN_SIMILARITY
    log.debug("best_sector_match(%r) -> %r (sim=%.3f, matched=%s)",
              query_sector, best_label, best_sim, matched)
    return SectorMatch(label=best_label, similarity=best_sim, matched=matched, all_scores=scores)
