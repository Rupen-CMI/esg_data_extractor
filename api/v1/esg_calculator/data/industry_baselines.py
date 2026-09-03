"""
industry_baselines.py -- real, frozen EXIOBASE-derived ESG baselines for
the standalone ESG calculator's 11 closed industries (schema.INDUSTRY_OPTIONS).
Same freeze-once-offline discipline as country_baselines.py -- see that
module's docstring for why this calculator never imports agentic_estimation/
or touches a DB at request time.

SOURCE
    exio_sector_intensity (150 EXIOBASE sectors), already used by the live
    pipeline's own agentic_estimation/layer_3/exio_lookup.py -- real
    physical/structural data per M EUR of output (co2e/employees/lowskill/
    female intensity), independent of any single company's disclosed
    behavior. SAME percentile computation and E-inversion/G-as-proxy logic
    as exio_lookup.py, just precomputed offline instead of imported live.

METHODOLOGY (reproduced here for reference, not reimplemented): the
calculator's 11 industries are broad umbrella categories (e.g.
"Manufacturing"), while EXIOBASE's 150 labels are narrow physical/
production sectors (e.g. "Manufacture of textiles (17)"). A single fuzzy
best-match per umbrella category would be arbitrary -- "Manufacturing"
token-overlaps dozens of "Manufacture of X" rows with no principled way
to pick just one. Instead each EXIOBASE row was hand-classified (one-time
editorial decision, not automated) into whichever of the 11 categories it
genuinely belongs to, and each category's baseline is the MEDIAN of its
matched rows' E/S percentiles -- a real aggregate over real sub-sector
data. co2e_pct is inverted (100 - pct) so higher = better, same
convention as the rest of this calculator and exio_lookup.py. G has no
EXIOBASE column at all (confirmed zero governance columns in that table)
so, exactly as exio_lookup.exio_g_vote does it, G is estimated as the
mean of that category's own E and S medians -- a proxy of a proxy,
explicitly weaker than E or S.

REGENERATING THIS FILE: rerun the generator script (hand-curated
CATEGORY_KEYWORDS/EXACT_MATCHES mapping against a live DB connection,
see conversation history for the exact script) -- one-time, 2026-08-27,
11 of 11 categories matched at least one EXIOBASE row.

USE IN THIS CALCULATOR: the industry dropdown (GET /calculator/industries)
already only ever offers these 11 values (schema.INDUSTRY_OPTIONS), so
every submitted industry is guaranteed to have a baseline here -- no
silent "valid string, no data" gap, same guarantee country_baselines.py
already gives. Combines with the country vote in scoring.py -- when BOTH
are present, they are averaged into a single starting point per pillar
(see _add_country_vote/_add_industry_vote in scoring.py), rather than one
silently overriding the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).parent / "industry_baselines_2026.json"

_by_industry: Optional[dict[str, dict]] = None


def _load() -> dict[str, dict]:
    global _by_industry
    if _by_industry is None:
        rows = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
        _by_industry = {r["industry"]: r for r in rows}
    return _by_industry


def industry_baseline_for(industry: Optional[str]) -> Optional[dict]:
    """Real e_score/s_score/g_score for this industry (must be an exact
    schema.INDUSTRY_OPTIONS value), or None if unset."""
    if not industry:
        return None
    return _load().get(industry)
