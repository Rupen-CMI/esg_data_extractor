"""
country_baselines.py -- real, frozen World Bank ESG country baselines for
the standalone ESG calculator. Deliberately NOT wired into
agentic_estimation/ at runtime -- see emission_factors.py's module
docstring for why this whole package stays self-contained.

SOURCE
    World Bank Sovereign ESG Data (esgdata_download-2026-05-01.xlsx),
    already used by the live pipeline's own agentic_estimation/layer_1/
    country_baseline_agent.py -- SAME computation, SAME source file, just
    run ONCE offline and frozen to a static JSON here instead of imported
    live. Reasons for freezing rather than importing that module directly:
      1. This calculator's hard constraint is zero imports from
         agentic_estimation/ (see plans/ESG_CALCULATOR_PLAN.md).
      2. country_baseline_agent.py's _compute_baselines() parses a 10MB
         Excel workbook and takes ~19s -- unacceptable for a per-request
         call in a tool whose whole design goal is sub-second responses.
      3. That module also creates a SQLAlchemy DB engine at import time
         for its caching/persistence path -- another hard constraint this
         calculator avoids entirely (no DB dependency at all).

METHODOLOGY (reproduced here for reference, not reimplemented -- computed
by the pipeline's own code, this file only stores the frozen OUTPUT):
    ~30 World Bank indicators (GDP growth, internet access, women in
    parliament, Worldwide Governance Indicators, health/poverty/education
    stats, an ILO labour-rights ratification count, etc.) are min-max
    normalised to 0-100 ACROSS ALL COUNTRIES per indicator, then averaged
    per pillar (E/S/G). A country needs at least half its pillar's
    indicators present to get a real score; otherwise that pillar falls
    back to a neutral 50.0. See country_baseline_agent.py's
    _E_INDICATORS/_S_INDICATORS/_G_INDICATORS for the exact indicator
    lists and directionality.

REGENERATING THIS FILE: run country_baseline_agent._compute_baselines()
(from a checkout with DB access configured, since importing that module
requires it even though this specific function doesn't touch the DB) and
export iso3/country/e_score/s_score/g_score/indicator_count/year per
country to country_baselines_2026.json. Done once on 2026-08-27, 210
countries with usable scores (of ~214 in the World Bank file).

USE IN THIS CALCULATOR: the country dropdown (GET /calculator/v2/countries)
is generated FROM this file's iso3 list, so a user can only ever submit a
country we actually have a real baseline for -- no silent "we don't
recognise this ISO3 but we validated the format anyway" gap. The baseline
itself contributes a small vote to each pillar's starting point (see
scoring.py) -- NOT a peer-comparison or industry-adjusted figure, just
"how does this country's overall ESG context compare to other countries",
same as the live pipeline's own country-baseline fallback tier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).parent / "country_baselines_2026.json"

_by_iso3: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _by_iso3
    if _by_iso3 is None:
        rows = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        _by_iso3 = {r["iso3"]: r for r in rows}
    return _by_iso3


def country_options() -> list[dict]:
    """[{iso3, country}, ...] sorted by country name -- for the dropdown."""
    rows = _load()
    return sorted(
        ({"iso3": r["iso3"], "country": r["country"]} for r in rows.values()),
        key=lambda r: r["country"],
    )


def baseline_for(iso3: str | None) -> dict | None:
    """Real e_score/s_score/g_score for this ISO3, or None if we have no
    baseline for it (caller should treat that as 'no country vote', not
    invent a number)."""
    if not iso3:
        return None
    return _load().get(iso3.upper())
