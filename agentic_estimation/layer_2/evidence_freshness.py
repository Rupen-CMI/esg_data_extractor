"""
evidence_freshness.py — exponential recency decay applied to claim confidence.

WHY THIS EXISTS: analyzed via the code-review-graph MCP tool against a
separate, already-shipped ESG product (rupa/code/enrichment/industry_match.py
and rupa/code/scoring/company_score.py). Both apply a recency decay so a
5-year-old mention doesn't carry the same weight as a headline from last
week. We had no equivalent -- a claim's confidence today is set purely from
the LLM's own self-reported certainty, with zero adjustment for how old the
underlying evidence actually is.

Two decay shapes exist in the source pipeline: a piecewise-linear one
(company_score.py, tuned for a 180-day sales-freshness window) and a simple
exponential one (industry_match.py: exp(-days/half_life)). We use the
exponential form -- it needs one constant instead of five knots, and ESG
evidence (annual reports, certifications, multi-year pledges) plausibly
wants a longer, single-parameter half-life than sales signals rather than a
shape tuned to a 180-day sales cycle.

ONE DELIBERATE DIVERGENCE from the source pipeline: there, an undated item
defaults to age=0 (maximum freshness) -- documented in that codebase itself
as a "fail open" bias. For ESG evidence this is the wrong default: an
undated controversy claim should NOT be treated as equally fresh as a dated
one from yesterday. Undated claims here get a fixed penalty confidence
multiplier instead (_UNDATED_FRESHNESS), applied once, not decayed further.

This module does NOT change ExtractedClaim's schema (no date field is
threaded through the LLM extraction prompt -- that would require re-running
Phase 2's already-verified extraction contract). Instead, it works from the
signal TEXT itself: each headline-style signal line already carries a date
prefix ("[Mon, 13 Jul 2026] ..." from Google News RSS/Reuters/localized_esg,
or "[YYYY-MM-DD] ..." from NewsAPI). freshness_multiplier_for_signal() scans
a signal's text for the MOST RECENT parseable date and returns a decay
factor; formula_estimator.py multiplies a claim's confidence by its cited
signal's freshness before computing the contribution.

SCOPE (decided after review, see formula_estimator.py's call site): decay
applies to EVENT-shaped claims only (controversies, fines, pledges --
news-cycle evidence that genuinely fades). Disclosed DATA (benchmark_band
factors: emissions figures, board percentages) is never decayed -- ESG
disclosure is annual, so a year-old report figure is usually the newest
data that exists. And the decay is floored at _FRESHNESS_FLOOR: old events
count less, never nothing.
"""

import math
import re
from datetime import datetime, timezone
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("evidence_freshness")

# e-folding constant in days (not a literal half-life -- see module docstring
# and the source pipeline's own note that exp(-days/h) gives ~37% at h days,
# not 50%). ESG evidence should decay slower than sales-signal freshness:
# annual reports and multi-year pledges stay relevant far longer than a
# single news cycle, so this is set well above the source pipeline's 30-day
# sales-tuned constant.
_HALF_LIFE_DAYS = 365

# Old evidence never decays below this multiplier. Two reasons this floor
# exists (added after review): (1) ESG disclosure is ANNUAL by nature -- a
# year-old report figure is often the newest data that exists, and letting
# it decay toward zero punishes the disclosure calendar, not stale evidence;
# (2) the ground truth this pipeline calibrates against (bcorp/upright
# assessments) was itself assessed in the past -- over-favoring last-week
# news can decorrelate predictions from the very truth being predicted. Old
# evidence counts LESS, never NOTHING. With half_life=365 the floor is hit
# at ~8.4 months; everything older sits at exactly 0.5.
_FRESHNESS_FLOOR = 0.5

# Confidence multiplier for signals where no date could be parsed at all.
# The source pipeline defaults undated items to age=0 (max freshness) --
# deliberately NOT copied here (see module docstring): for ESG evidence,
# "we don't know how old this is" should be penalized, not treated as fresh.
# Sits between the floor (0.5, "definitely old") and 1.0 ("definitely fresh").
_UNDATED_FRESHNESS = 0.6

_RFC822_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dates(text: str) -> list[datetime]:
    """Every parseable date found anywhere in `text` (order not assumed)."""
    found: list[datetime] = []

    for m in _RFC822_DATE_RE.finditer(text):
        day, mon_str, year = m.groups()
        mon = _MONTHS.get(mon_str[:3].lower())
        if not mon:
            continue
        try:
            found.append(datetime(int(year), mon, int(day), tzinfo=timezone.utc))
        except ValueError:
            continue

    for m in _ISO_DATE_RE.finditer(text):
        year, mon, day = m.groups()
        try:
            found.append(datetime(int(year), int(mon), int(day), tzinfo=timezone.utc))
        except ValueError:
            continue

    return found


def _recency_decay(days: float, half_life: float = _HALF_LIFE_DAYS) -> float:
    """max(floor, exp(-days/half_life)) -- an e-folding constant, not a
    literal half-life (see module docstring). days<0 (future/clock-skew
    dates) clamped to 0. Never decays below _FRESHNESS_FLOOR -- old evidence
    counts less, never nothing (see the floor's own comment above)."""
    days = max(0.0, days)
    return max(_FRESHNESS_FLOOR, math.exp(-days / half_life))


def freshness_multiplier_for_signal(signal_text: Optional[str]) -> float:
    """0-1 decay factor for a signal's text, from its MOST RECENT embedded
    date. No parseable date -> _UNDATED_FRESHNESS flat penalty (never treated
    as maximally fresh -- see module docstring). Never raises."""
    if not signal_text:
        return _UNDATED_FRESHNESS
    try:
        dates = _parse_dates(signal_text)
        if not dates:
            return _UNDATED_FRESHNESS
        newest = max(dates)
        age_days = (_utc_now() - newest).total_seconds() / 86400.0
        return _recency_decay(age_days)
    except Exception as exc:
        log.debug("freshness_multiplier_for_signal: parse failed (%s) -- using undated penalty", exc)
        return _UNDATED_FRESHNESS
