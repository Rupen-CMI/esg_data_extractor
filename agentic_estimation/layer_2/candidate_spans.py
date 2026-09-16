"""
candidate_spans.py -- keyword-gated candidate-window extraction, run BEFORE
evidence_classifier.predict() on long real signal text.

WHY THIS EXISTS: evidence_clf_v3 fixed the training/production length
mismatch (median 96 -> 1,081 chars) but only by making the classifier tolerate
long text, not by giving it a shorter, denser span to actually decide on. A
1,600-char Microsoft signal blob still buries the one sentence that carries a
factor's real evidence inside boilerplate, navigation text, and unrelated
paragraphs -- exactly the shape that produced the false wrong_entity/
no_evidence rejections seen in v3's Microsoft eval.

Naive per-SENTENCE classification was tried and rejected earlier this
session: it recovered more factors but a generic climate-science sentence
("global emissions must fall...") won net_zero_pledge on its own, out of
context from the company-specific sentence next to it. A pure sentence
boundary throws away that neighbouring context.

This module instead extracts a WINDOW (default +/-300 chars) around each
keyword hit -- wide enough to keep the sentence's neighbours (so "Microsoft
has committed to..." stays attached to "...net-zero by 2030" even if they're
split across sentences), narrow enough to cut the boilerplate that swamps the
char n-gram features (see evidence_classifier.py's clean_snippet notes on
long text diluting the same features).

Reuses claim_validators._FACTOR_TOPIC_TERMS / _PILLAR_FALLBACK_TERMS --
the SAME per-factor keyword vocabulary already trusted in production to drop
off-topic claims post-extraction (Tier-0 rule (a), the yoga-page catcher).
Not duplicated here: importing keeps both call sites -- post-extraction
validation and pre-classification gating -- honest to the same vocabulary
if either is ever tuned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agentic_estimation.layer_1.evidence_filters import _compiled_patterns
from agentic_estimation.layer_2.claim_validators import _FACTOR_TOPIC_TERMS, _PILLAR_FALLBACK_TERMS
from agentic_estimation.layer_2.factor_registry import FACTORS

WINDOW_RADIUS = 300   # chars kept on each side of a keyword hit
MIN_TEXT_LEN_FOR_GATING = 400   # short text is already dense enough; gate only long text

# SEC filing table-of-contents detector. Confirmed live: Microsoft's
# sec_10k_properties signal is a bare TOC dump -- "Item 3. Legal Proceedings
# 31 Item 4. Mine Safety Disclosures 31 ..." -- that scored litigation at 0.99
# confidence from evidence_clf_v3. The text is real, on-topic-looking
# (matches "Safety"/"Governance"/"Director" keywords), and long enough to pass
# every existing filter, but asserts nothing: it is a page index, not a claim.
#
# The signature that catches this without touching real prose: an "Item N."
# or "Item NA." heading followed within ~80 chars by a bare 1-3 digit page
# number is a structural artifact unique to TOC/index formatting -- genuine
# litigation text never reads "Item 3. Legal Proceedings 31", it reads
# "Item 3 of our Annual Report discusses ongoing litigation regarding...".
# Verified against all 18 cached real Microsoft signals (report_pdf_*,
# gov_litigation_sec, sec_ft_restatement, etc.): 0 false hits anywhere except
# the actual TOC (14 matches there). Threshold of >=3 leaves wide margin.
_TOC_ENTRY_PATTERN = re.compile(r"Item\s+\d+[A-Z]?\.\s*[^.]{0,80}?\s\d{1,3}\b")
_TOC_MIN_ENTRIES = 3


def is_toc_like(text: str) -> bool:
    """True if `text` looks like a filing's table of contents / item index
    rather than actual prose -- see _TOC_ENTRY_PATTERN above for why this
    signature is safe against real evidence."""
    if not text:
        return False
    return len(_TOC_ENTRY_PATTERN.findall(text)) >= _TOC_MIN_ENTRIES


@dataclass
class Candidate:
    factor_hint: str    # factor key, or pillar letter when only the fallback list matched
    pillar: str
    window: str
    match_term: str


def _windows_for_terms(text: str, terms: tuple[str, ...]) -> list[tuple[str, str]]:
    """Return (window_text, matched_term) pairs, one per keyword hit, with
    overlapping windows merged so a dense paragraph collapses into one span
    instead of N duplicates of nearly the same text."""
    patterns = _compiled_patterns(terms)
    hits: list[tuple[int, int, str]] = []
    for pat in patterns:
        for m in pat.finditer(text):
            hits.append((m.start(), m.end(), m.group(0)))
    if not hits:
        return []
    hits.sort(key=lambda h: h[0])

    spans: list[tuple[int, int, str]] = []
    cur_start, cur_end, cur_term = None, None, None
    for start, end, term in hits:
        w_start, w_end = max(0, start - WINDOW_RADIUS), min(len(text), end + WINDOW_RADIUS)
        if cur_start is None:
            cur_start, cur_end, cur_term = w_start, w_end, term
        elif w_start <= cur_end:
            cur_end = max(cur_end, w_end)   # merge overlapping windows
        else:
            spans.append((cur_start, cur_end, cur_term))
            cur_start, cur_end, cur_term = w_start, w_end, term
    spans.append((cur_start, cur_end, cur_term))

    return [(text[s:e].strip(), term) for s, e, term in spans]


def extract_candidates(text: str, pillar_hint: Optional[str] = None) -> list[Candidate]:
    """Find keyword-anchored candidate windows in `text`.

    pillar_hint ('E'/'S'/'G'): when the caller already knows which pillar a
    signal belongs to (most collectors tag signals this way), only that
    pillar's factors + fallback terms are searched -- narrower and faster,
    and avoids an E-heavy company's long report matching S/G windows that
    were never going to be scored against this signal anyway. None searches
    all three pillars' terms.

    Returns [] if text is short enough that gating isn't worth it (caller
    should classify the whole text directly) OR if no keyword anywhere
    matched (caller should still fall back to whole-text classification --
    absence of a keyword hit is not proof of absence of evidence, just a
    cheap first pass).
    """
    if not text or len(text) < MIN_TEXT_LEN_FOR_GATING:
        return []

    pillars = [pillar_hint] if pillar_hint in ("E", "S", "G") else ["E", "S", "G"]
    candidates: list[Candidate] = []

    for pillar in pillars:
        factor_keys = [k for k, f in FACTORS.items() if f.pillar == pillar and f.weight > 0]
        for factor_key in factor_keys:
            terms = _FACTOR_TOPIC_TERMS.get(factor_key)
            if not terms:
                continue
            for window, term in _windows_for_terms(text, terms):
                candidates.append(Candidate(factor_hint=factor_key, pillar=pillar,
                                             window=window, match_term=term))

        # Fallback pass: catches text that matches the pillar's generic terms
        # but none of the narrower per-factor lists (e.g. a factor with no
        # hand-tuned entry yet, or a claim that uses vocabulary the per-factor
        # list didn't anticipate).
        fallback_terms = _PILLAR_FALLBACK_TERMS.get(pillar, ())
        if fallback_terms:
            for window, term in _windows_for_terms(text, fallback_terms):
                candidates.append(Candidate(factor_hint=pillar, pillar=pillar,
                                             window=window, match_term=term))

    # Dedupe windows that are identical or near-identical across factors/
    # fallback (a single sentence mentioning "board" and "governance" would
    # otherwise appear twice, once per matching term set).
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for c in candidates:
        key = c.window[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _toc_rejection():
    """A Prediction shaped exactly like a normal no_evidence abstain, so
    callers never need to special-case the TOC path. See is_toc_like()."""
    from agentic_estimation.layer_2.evidence_classifier import Prediction
    return Prediction(factor=None, label="no_evidence", probability=1.0, margin=1.0,
                       polarity=None, strength=None, abstained=True)


def classify_with_gating(clf, text: str, company: Optional[str] = None,
                          pillar_hint: Optional[str] = None):
    """Drop-in replacement for clf.predict(text, company) that gates long
    text through keyword-anchored candidate windows first.

    Runs predict() on every candidate window, keeps the best NON-ABSTAINED
    result (highest probability). Falls back to whole-text predict() when
    gating found no candidates (short text, or no keyword hit anywhere) --
    this method can only add recall on long text, never remove it, since
    whole-text classification is always the floor.

    TOC/index-page text (is_toc_like()) is rejected outright before it ever
    reaches the classifier -- confirmed live that this shape (SEC filing
    "Item N. <title> <page#>" dumps) fools the factor head into a confident
    but false positive (sec_10k_properties -> litigation at 0.99) because the
    keywords are real even though the text asserts nothing.
    """
    if is_toc_like(text):
        return _toc_rejection()

    candidates = extract_candidates(text, pillar_hint)
    if not candidates:
        return clf.predict(text, company)

    whole = clf.predict(text, company)
    best = whole
    for c in candidates:
        p = clf.predict(c.window, company)
        if p.abstained:
            continue
        if best.abstained or p.probability > best.probability:
            best = p
    return best
