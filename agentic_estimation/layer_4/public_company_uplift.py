"""
public_company_uplift.py — Layer 4's 4th critic, distinct from
critic_panel.py's adversarial 3-lens panel (evidence_support/
peer_plausibility/internal_consistency). Those three try to catch the
pipeline being WRONG in either direction. This one exists for one
specific, known, one-directional failure mode: the pipeline's accuracy
ceiling is evidence coverage (large public companies routinely have more
disclosure than the pipeline actually finds/parses), so a real public
company can land a pillar score well below what its actual scale and
market standing would justify, purely because our evidence gathering
missed most of what exists about it -- not because the company is
actually a laggard.

This is NOT adversarial and NOT symmetric: it only ever pushes a score UP,
never down, and only for a company confirmed to be publicly traded (a
private/small company scoring low has no such prior to correct against).
Runs once per pillar per company, after estimate_verifier.py's critic
panel + retry loop has fully settled (see graph.py's node ordering) --
it corrects the FINAL number, not an intermediate one, and does not
interact with critic_panel.py's retry/round-2 logic at all.

Detection: yfinance's live ticker search (see api/v1/esg_data/fetchers/
yfinance_fetcher.py's _find_ticker, reused here) rather than the
pipeline's own (sparse, Wikidata-sourced, potentially stale) metadata --
a company that yfinance resolves to a real EQUITY ticker right now is
unambiguously publicly traded today, which the alternative
metadata['stock_exchanges'] field cannot promise (Wikidata can lag a
delisting/going-private event).
"""

from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.shared.llm_json import extract_json_object

log = get_logger("public_company_uplift")

_UPLIFT_THRESHOLD = 60.0
_UPLIFT_MAX_TOKENS = 600
_UPLIFT_TIMEOUT = 120

_UPLIFT_SYSTEM = (
    "You are an ESG analyst correcting for a known data gap: automated evidence gathering "
    "systematically under-counts real disclosure for large, well-known public companies. "
    "You only ever raise a score, and only when the company's real-world scale/reputation "
    "genuinely justifies it -- you do not inflate scores without grounds. After reasoning, "
    "you MUST end with a single valid JSON object of the exact shape requested."
)

_UPLIFT_PROMPT = """The {pillar_label} pillar score below for {company} ({industry}, {country}) came out \
below 60, but {company} is a publicly traded company (ticker: {ticker}).

Our pipeline's own evidence coverage is known to be incomplete -- especially for large public \
companies, where real disclosure (sustainability reports, regulatory filings, index inclusion, \
independent ratings) usually exists but our automated gathering did not find or use enough of \
it. A large, established public company scoring this low on {pillar_label} is more often a sign \
of missing evidence than a sign of genuinely poor {pillar_label} performance.

CURRENT {pillar_label} SCORE: {score:.1f}/100
CURRENT REASONING: {reasoning}

Using your own knowledge of {company}'s real-world scale, market standing, and general {pillar_label} \
reputation/history, decide a corrected score that is a FAIR, PROPORTIONATE reflection of what the \
evidence gap likely hid -- not a flat bump to a fixed number. A company you know little about beyond \
its listing should move only modestly above 60; a company with an extensive, well-known track record \
should move further. Do NOT default to the same number for every company.

Respond with ONLY this JSON object after your reasoning:
{{"corrected_score": <number, MUST be > {threshold:.0f} and <= 100>, \
"reasoning": "<1-2 sentences grounded in the company's actual history/scale, explaining the correction>"}}"""


@dataclass
class UpliftResult:
    pillar: str
    applied: bool
    original_score: float
    corrected_score: Optional[float] = None
    reasoning: Optional[str] = None


def find_ticker(company_name: str) -> Optional[str]:
    """Thin wrapper around yfinance_fetcher's own ticker search -- reused,
    not re-implemented, so both call sites stay in sync on what counts as
    a real match (EQUITY quoteType + name-similarity > 0.5)."""
    from api.v1.esg_data.fetchers.yfinance_fetcher import _find_ticker
    try:
        ticker, _matched_name = _find_ticker(company_name)
        return ticker
    except Exception as exc:
        log.warning("[%s] yfinance ticker lookup raised: %s -- treating as not public", company_name, exc)
        return None


def _parse_uplift_response(raw_text: str, original_score: float) -> Optional[tuple]:
    """Fail-closed: unparseable, missing, or non-improving corrected_score
    -> None (caller keeps the original score, never applies a bad
    correction). Returns (corrected_score, reasoning) on success."""
    parsed = extract_json_object(raw_text) if raw_text else None
    if not parsed:
        return None

    raw_score = parsed.get("corrected_score")
    try:
        corrected = float(raw_score)
    except (TypeError, ValueError):
        return None

    if not (_UPLIFT_THRESHOLD < corrected <= 100.0):
        log.warning("uplift LLM returned corrected_score=%r outside (%.0f, 100] -- discarding",
                    raw_score, _UPLIFT_THRESHOLD)
        return None
    if corrected <= original_score:
        log.warning("uplift LLM returned corrected_score=%.1f <= original %.1f -- discarding",
                    corrected, original_score)
        return None

    reasoning = str(parsed.get("reasoning", "")).strip()[:500] or "(no reasoning given)"
    return corrected, reasoning


def apply_pillar_uplift(
    pillar: str, company: str, industry: str, country: Optional[str], ticker: str,
    score: float, reasoning: str, model: Optional[str] = None,
) -> UpliftResult:
    """Runs the uplift LLM call for ONE pillar already confirmed to be
    below threshold, for a company already confirmed public (ticker
    resolved). Never raises -- an LLM failure or a rejected/invalid
    correction both fall back to `applied=False`, original score kept."""
    from zen_client import call_with_prompt

    pillar_label = {"E": "Environment", "S": "Social", "G": "Governance"}.get(pillar, pillar)
    prompt = _UPLIFT_PROMPT.format(
        pillar_label=pillar_label, company=company, industry=industry or "Not specified",
        country=country or "Unknown", ticker=ticker, score=score, reasoning=reasoning or "(none)",
        threshold=_UPLIFT_THRESHOLD,
    )

    try:
        resp = call_with_prompt(prompt, model=model, max_tokens=_UPLIFT_MAX_TOKENS,
                                 timeout=_UPLIFT_TIMEOUT, system=_UPLIFT_SYSTEM)
    except Exception as exc:
        log.warning("[%s/%s] uplift call raised: %s -- keeping original score", company, pillar, exc)
        return UpliftResult(pillar=pillar, applied=False, original_score=score)

    if not resp.get("ok"):
        log.warning("[%s/%s] uplift call failed: %s -- keeping original score",
                    company, pillar, resp.get("error"))
        return UpliftResult(pillar=pillar, applied=False, original_score=score)

    raw_text = resp.get("raw", "") or resp.get("reasoning", "")
    result = _parse_uplift_response(raw_text, score)
    if result is None:
        return UpliftResult(pillar=pillar, applied=False, original_score=score)

    corrected, new_reasoning = result
    log.info("[%s/%s] public-company uplift: %.1f -> %.1f (ticker=%s)",
              company, pillar, score, corrected, ticker)
    return UpliftResult(pillar=pillar, applied=True, original_score=score,
                         corrected_score=corrected, reasoning=new_reasoning)


def apply_public_company_uplift(
    company: str, industry: str, country: Optional[str],
    final_scores: dict, final_reasonings: dict, model: Optional[str] = None,
) -> dict:
    """Entry point. final_scores: {"E"|"S"|"G": float} -- the SETTLED
    post-verification score per pillar (caller resolves verified-vs-
    reconciled precedence before calling this; this module has no opinion
    on that, it only sees the number that already won). final_reasonings:
    matching {"E"|"S"|"G": str} reasoning text, for prompt context only.

    Returns {pillar: UpliftResult} for every pillar checked (i.e. every
    pillar in final_scores) -- callers can tell "checked, not applied"
    (already >=60, or not public, or LLM declined) apart from pillars
    never even in scope, by simply checking dict membership.

    Ticker lookup is done ONCE and only if at least one pillar is below
    threshold -- most companies need no uplift at all, so this avoids the
    yfinance call/cost for every other run."""
    low_pillars = {p: s for p, s in final_scores.items() if s < _UPLIFT_THRESHOLD}
    if not low_pillars:
        return {}

    ticker = find_ticker(company)
    if not ticker:
        log.info("[%s] %d pillar(s) below %.0f but not confirmed publicly traded -- no uplift",
                  company, len(low_pillars), _UPLIFT_THRESHOLD)
        return {}

    log.info("[%s] confirmed public (ticker=%s) -- checking uplift for pillars: %s",
              company, ticker, sorted(low_pillars))
    results = {}
    for pillar, score in low_pillars.items():
        results[pillar] = apply_pillar_uplift(
            pillar, company, industry, country, ticker, score,
            final_reasonings.get(pillar, ""), model=model,
        )
    return results
