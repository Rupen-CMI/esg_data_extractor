"""
sec_fulltext_collector.py — ESG evidence from EDGAR full-text search.

WHY THIS EXISTS: measured against the frozen calibration dumps, our web-search
sources produce ~0.2 kept claims per fetch and roughly half their text is not
even about the right company (a site: query that finds nothing falls back to
another registrant's page or to topic-generic filler). The S pillar is worst:
54 claims across 336 companies, which is too little evidence to rank anything.

EDGAR full-text search fixes both problems at once for US registrants:

  * ENTITY CERTAINTY. Scoping by `ciks=` returns filings BY that registrant.
    There is no name-similarity step to get wrong -- the failure mode of every
    web-search source we have.
  * REAL TEXT, NOT SNIPPETS. A hit resolves to the filing document itself
    (verified: 30,574 characters of Nike-specific prose on the Uyghur Forced
    Labor Prevention Act investigation), versus ~200-character search blurbs.

The phrase list is deliberately weighted toward S and G. E is already our
best-covered pillar; S and G are where evidence volume is missing.

WHAT THIS SOURCE CANNOT DO: it covers SEC registrants only -- roughly 10.4k
US-listed companies. Most of our corpus (private, non-US) resolves to no CIK
and gets nothing from here. That is an expected miss, not a failure, and it
is why this supplements the web sources rather than replacing them.

Two behaviours verified live and easy to get wrong:
  * CIK must be zero-padded to 10 digits. `ciks=320187` returns 0 hits;
    `ciks=0000320187` returns 10 for the same query. It fails SILENTLY.
  * SEC requires a "Name email" User-Agent. Browser-style UAs get 403 --
    the opposite of the other agencies we fetch from.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.signal_agent import RateLimitTripped
from agentic_estimation.layer_1.sec_filings import (
    cik_for_company,
    extract_phrase_context,
    fetch_document_text,
    full_text_search,
)

log = get_logger("sec_fulltext")

# (signal_name, phrase, pillar). Phrases are exact-match in EFTS, so they are
# chosen to be the terms filings actually use -- "human rights due diligence"
# rather than "human rights", which matches boilerplate in every filing.
_ESG_PHRASES: tuple[tuple[str, str, str], ...] = (
    # S -- the starved pillar. Verified filing counts since 2025-01-01:
    # collective bargaining 10k+, forced labor 3091, child labor 2836,
    # modern slavery 1408, human rights due diligence 1358.
    ("sec_ft_forced_labor",      "forced labor",                "S"),
    ("sec_ft_child_labor",       "child labor",                 "S"),
    ("sec_ft_modern_slavery",    "modern slavery",              "S"),
    ("sec_ft_human_rights",      "human rights due diligence",  "S"),
    ("sec_ft_collective_barg",   "collective bargaining",       "S"),
    ("sec_ft_workplace_safety",  "workplace safety",            "S"),
    ("sec_ft_discrimination",    "discrimination lawsuit",      "S"),
    # G -- negative-polarity governance, which our corpus badly lacks
    # (our highest-volume sources run 97% positive self-disclosure).
    ("sec_ft_bribery",           "bribery",                     "G"),
    ("sec_ft_fcpa",              "Foreign Corrupt Practices Act", "G"),
    ("sec_ft_whistleblower",     "whistleblower complaint",     "G"),
    ("sec_ft_material_weakness", "material weakness",           "G"),
    ("sec_ft_restatement",       "restatement of previously issued", "G"),
    ("sec_ft_class_action",      "securities class action",     "G"),
    # E
    ("sec_ft_environmental",     "environmental violation",     "E"),
    ("sec_ft_remediation",       "environmental remediation",   "E"),
    ("sec_ft_ghg",               "greenhouse gas emissions",    "E"),
)

# Cap per phrase. EFTS ranks by relevance and most companies have only a
# handful of hits per phrase anyway; the cap bounds fetch cost for the few
# large filers that have many.
_MAX_HITS_PER_PHRASE = 2
_MAX_CHARS_PER_SIGNAL = 2500

# Only filings from this date forward. Verified necessary: an unbounded search
# surfaced Nike's 2002 securities class action settlement as if it were current
# evidence. ESG standing is a present-state question, and a 20-year-old
# resolved matter is not evidence about it.
_MIN_FILE_DATE = "2021-01-01"

# Exhibit forms are contract templates, not disclosure. Verified failure mode:
# searching "bribery" returned Nike's EMPLOYMENT AGREEMENT termination clause
# ("conviction ... involving fraud ... bribery, forgery"), and "material
# weakness" returned an underwriting agreement REPRESENTING THAT NONE EXISTS.
# Both are boilerplate that appears in essentially every filer's exhibits, so
# they carry no company-specific signal while reading as serious findings.
_EXCLUDED_FORM_PREFIXES = ("EX-",)

# Negation/hypothetical guards. Filings routinely state the ABSENCE of a
# problem ("we are not aware of any material weakness") or describe it as a
# generic risk ("any such violation could result in"). Counting those as
# evidence of the problem inverts their meaning -- the single most damaging
# error this collector could make, since polarity is decided downstream from
# text we hand over.
_NEGATION_MARKERS = (
    "not aware of any", "no material weakness", "did not identify any",
    "were no ", "was no ", "has not been", "have not been", "none of the",
    "no such", "did not have any", "are not subject to",
)


def _is_boilerplate_form(form: str) -> bool:
    """True for exhibit forms, which are contract templates rather than
    company-specific disclosure."""
    f = (form or "").upper()
    return any(f.startswith(p) for p in _EXCLUDED_FORM_PREFIXES)


def _is_negated(window: str, phrase: str) -> bool:
    """True if the phrase appears inside a denial or a generic risk statement.

    Checks only the text immediately BEFORE the phrase: "we are not aware of
    any material weakness" negates, whereas "a material weakness ... was not
    remediated until Q3" does not, and looking at the whole window would
    wrongly reject the second.
    """
    low = window.lower()
    idx = low.find(phrase.lower())
    if idx < 0:
        return False
    lead = low[max(0, idx - 90):idx]
    return any(m in lead for m in _NEGATION_MARKERS)


def _collect_phrase(company: str, cik: str, name: str, phrase: str) -> Optional[tuple[str, str]]:
    """Search one phrase for one company; return (signal_name, text) or None."""
    hits = full_text_search(phrase, cik=cik, date_from=_MIN_FILE_DATE,
                            limit=_MAX_HITS_PER_PHRASE * 3)
    if not hits:
        return None

    chunks: list[str] = []
    used = 0
    for hit in hits:
        if used >= _MAX_HITS_PER_PHRASE:
            break
        if _is_boilerplate_form(hit.get("form", "")):
            log.debug("[%s] %s → skipping exhibit form %s", company, name, hit.get("form"))
            continue
        text = fetch_document_text(hit["url"])
        if not text:
            continue
        kept_any = False
        for window in extract_phrase_context(text, phrase, max_windows=2):
            if _is_negated(window, phrase):
                log.debug("[%s] %s → dropping negated mention", company, name)
                continue
            chunks.append(
                f"[{hit['form'] or 'filing'} {hit['file_date']}] {window} <{hit['url']}>"
            )
            kept_any = True
        if kept_any:
            used += 1
        if sum(len(c) for c in chunks) > _MAX_CHARS_PER_SIGNAL:
            break

    if not chunks:
        return None
    body = " ".join(chunks)[:_MAX_CHARS_PER_SIGNAL]
    log.info("[%s] %s → %d chars from %d filing(s)", company, name, len(body), used)
    return name, f"SEC filing text ({phrase}): {body}"


def fetch_sec_fulltext_signals(company: str, max_workers: int = 2) -> dict[str, str]:
    """ESG evidence for one company from EDGAR full-text search.

    Returns {signal_name: evidence_text}; {} for any company without a CIK
    (most of our corpus) or with no matching filings. Mirrors the shape of
    signal_agent.fetch_company_signals so callers can merge the two dicts.
    """
    cik = cik_for_company(company)
    if not cik:
        log.info("[%s] no CIK — not an SEC registrant, skipping full-text search", company)
        return {}

    log_header(log, "SEC Full-Text", company=company, cik=cik, phrases=len(_ESG_PHRASES))
    signals: dict[str, str] = {}

    # 2, not 6. This function runs INSIDE the pipeline's 5-thread worker pool,
    # so the real concurrency against sec.gov is workers x max_workers: at 6 that
    # is 30 threads on one host, which drew 4 HTTP 429s and aborted a run.
    # The limiter caps the RATE either way; what this bounds is how long the
    # queue behind it grows, and therefore how bursty the traffic looks.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_collect_phrase, company, cik, name, phrase): name
            for name, phrase, _pillar in _ESG_PHRASES
        }
        for fut in as_completed(futures):
            try:
                got = fut.result()
            except RateLimitTripped:
                raise   # never swallow the abort signal
            except Exception as exc:
                log.warning("[%s] %s → exception: %s", company, futures[fut], exc)
                continue
            if got:
                signals[got[0]] = got[1]

    log.info("[%s] SEC full-text done — %d/%d phrases hit", company, len(signals), len(_ESG_PHRASES))
    return signals


if __name__ == "__main__":  # manual probe: python -m ...sec_fulltext_collector NIKE
    import sys
    target = " ".join(sys.argv[1:]) or "NIKE, Inc."
    for k, v in fetch_sec_fulltext_signals(target).items():
        print(f"\n=== {k} ===\n{v[:600]}")
