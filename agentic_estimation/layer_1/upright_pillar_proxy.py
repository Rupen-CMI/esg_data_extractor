"""
upright_pillar_proxy.py — derives per-pillar E/S percentile proxies for
upright_lookup companies from Upright's raw impact sub-components.

WHY THIS EXISTS: upright_lookup has NO e_score/s_score/g_score columns like
bcorp_lookup does -- only a single net_impact_ratio_percentile (an overall,
all-pillars-combined score). But it DOES carry granular raw sub-components
across the same conceptual space as E/S (confirmed live, all ~10,086 rows,
near-full coverage on every column):
    E: e1_ghg, e2_non_ghg, e3_scarce_resources, e4_biodiversity, e5_waste
       (each split into _negative/_positive)
    S: h1_physical_diseases, h2_mental_diseases, h3_nutrition, h4_relationships,
       h5_meaning_joy (health/wellbeing) + s1_jobs, s2_taxes, s3_societal_infra,
       s4_societal_stability, s5_equality (societal) -- again _negative/_positive
       where both exist.
No usable G-analog exists (k1-k4 are human-capital/knowledge columns, not
governance -- board structure, compliance, litigation have no upright
equivalent at all), so this module deliberately produces E and S proxies
ONLY, never a G proxy.

HOW THE PROXY IS COMPUTED (percentile-averaging, not raw summation):
Each raw sub-component is on its OWN unrelated scale (confirmed live:
e1_ghg_negative ranges ~0.02-16.2, e3_scarce_resources_positive ranges
~0.00000004-1.08) -- these are impact-weighted ratios, not bounded 0-100 or
0-1 scores, and cannot be summed/averaged directly without fabricating
relative-importance weights Upright's own proprietary model uses internally
(which we do not have). Instead, EVERY sub-component is converted to its own
percentile RANK within the full upright_lookup distribution for that column
(same technique peer_anchor.py already uses for bcorp's impact_area_* columns
-- unit-free, no weight-guessing). "_negative" columns are inverted
(100 - raw_percentile) so higher always means better across every column,
consistent with the rest of this pipeline. The pillar proxy is then the
PLAIN MEAN of its component percentiles -- not a weighted combination, since
we have no principled basis to weight e.g. e1_ghg over e4_biodiversity.

This is an HONEST, LABELED PROXY, not a reconstruction of Upright's real
scoring methodology -- callers must treat it as such (method tag
"upright_proxy_e"/"upright_proxy_s" downstream, never presented as if it
were a genuine Upright-published per-pillar score).

CLI:
    python -m agentic_estimation.layer_1.upright_pillar_proxy score "Walmart"
"""

import bisect
from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("upright_pillar_proxy")

# Column -> True if "negative" (needs inversion so higher-is-better, consistent
# across every component). Grouped by which pillar proxy each feeds.
E_COLUMNS: dict[str, bool] = {
    "e1_ghg_negative": True,               "e1_ghg_positive": False,
    "e2_non_ghg_negative": True,            "e2_non_ghg_positive": False,
    "e3_scarce_resources_negative": True,   "e3_scarce_resources_positive": False,
    "e4_biodiversity_negative": True,       "e4_biodiversity_positive": False,
    "e5_waste_negative": True,              "e5_waste_positive": False,
}

S_COLUMNS: dict[str, bool] = {
    "h1_physical_diseases_negative": True,  "h1_physical_diseases_positive": False,
    "h2_mental_diseases_negative": True,    "h2_mental_diseases_positive": False,
    "h3_nutrition_positive": False,
    "h4_relationships_negative": True,      "h4_relationships_positive": False,
    "h5_meaning_joy_negative": True,        "h5_meaning_joy_positive": False,
    "s1_jobs_positive": False,
    "s2_taxes_positive": False,
    "s3_societal_infra_positive": False,
    "s4_societal_stability_negative": True, "s4_societal_stability_positive": False,
    "s5_equality_negative": True,           "s5_equality_positive": False,
}

# Cached once per process: for each raw column, the sorted list of every
# real (non-null) value across upright_lookup -- reused for every company's
# percentile lookup rather than re-queried per call.
_distribution_cache: dict[str, list[float]] = {}
# Cached once per process: company_id -> {column: raw_value}, loaded in one
# query rather than one round-trip per company.
_company_values_cache: Optional[dict[str, dict[str, float]]] = None


def _db_conn():
    from agentic_estimation.layer_1.peer_anchor_collector import _db_conn as _conn
    return _conn()


def _load_distribution(column: str) -> list[float]:
    if column not in _distribution_cache:
        conn = _db_conn()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {column} FROM upright_lookup WHERE {column} IS NOT NULL")
            vals = sorted(float(r[0]) for r in cur.fetchall())
        finally:
            conn.close()
        _distribution_cache[column] = vals
        log.info("loaded upright %s distribution: %d real values", column, len(vals))
    return _distribution_cache[column]


def _percentile_rank(value: float, sorted_dist: list[float]) -> float:
    """Midrank percentile: (count_below + 0.5*count_equal) / N * 100."""
    n = len(sorted_dist)
    if n == 0:
        return 50.0
    lo = bisect.bisect_left(sorted_dist, value)
    hi = bisect.bisect_right(sorted_dist, value)
    return (lo + 0.5 * (hi - lo)) / n * 100.0


def _all_company_values() -> dict[str, dict[str, float]]:
    global _company_values_cache
    if _company_values_cache is None:
        all_cols = list(E_COLUMNS) + list(S_COLUMNS)
        conn = _db_conn()
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT name, {', '.join(all_cols)} FROM upright_lookup")
            rows = cur.fetchall()
        finally:
            conn.close()
        cache: dict[str, dict[str, float]] = {}
        for row in rows:
            name = row[0]
            vals = {col: row[i + 1] for i, col in enumerate(all_cols) if row[i + 1] is not None}
            cache[name] = vals
        _company_values_cache = cache
        log.info("loaded raw upright component values for %d companies", len(cache))
    return _company_values_cache


@dataclass
class UprightPillarProxy:
    pillar: str                    # 'E' | 'S'
    percentile: Optional[float]    # 0-100, mean of component percentiles; None if no components available
    n_components: int              # how many of the pillar's raw columns had a real value for this company
    n_components_total: int        # how many columns this pillar proxy is defined over


def _pillar_proxy_for_values(pillar: str, columns: dict[str, bool], raw_values: dict[str, float]) -> UprightPillarProxy:
    component_pctiles = []
    for col, is_negative in columns.items():
        if col not in raw_values:
            continue
        dist = _load_distribution(col)
        pct = _percentile_rank(raw_values[col], dist)
        if is_negative:
            pct = 100.0 - pct
        component_pctiles.append(pct)

    if not component_pctiles:
        return UprightPillarProxy(pillar=pillar, percentile=None, n_components=0, n_components_total=len(columns))

    return UprightPillarProxy(
        pillar=pillar,
        percentile=sum(component_pctiles) / len(component_pctiles),
        n_components=len(component_pctiles),
        n_components_total=len(columns),
    )


def upright_pillar_proxy(company_name: str, pillar: str) -> UprightPillarProxy:
    """E or S percentile proxy for one upright_lookup company (matched by
    exact `name`). Returns n_components=0 (percentile=None) if the company
    isn't in upright_lookup, or has no non-null values for that pillar's
    columns. G is not supported -- raises ValueError (no upright column set
    maps to governance; callers must not silently treat a missing case as an
    abstain vote for a pillar this module was never designed to answer)."""
    if pillar not in ("E", "S"):
        raise ValueError(f"upright_pillar_proxy has no proxy for pillar {pillar!r} -- only 'E' and 'S' are supported")

    values = _all_company_values().get(company_name)
    if values is None:
        return UprightPillarProxy(pillar=pillar, percentile=None, n_components=0,
                                   n_components_total=len(E_COLUMNS if pillar == "E" else S_COLUMNS))

    columns = E_COLUMNS if pillar == "E" else S_COLUMNS
    return _pillar_proxy_for_values(pillar, columns, values)


def peer_group_pillar_proxy(company_names: list[str], pillar: str) -> Optional[float]:
    """Mean pillar-proxy percentile across a peer GROUP (e.g. all upright
    companies matched via sector_crosswalk for a given industry) -- used by
    peer_anchor.py the same way it uses peer_median() for bcorp fields.
    Returns None if no peer in the group has any usable component."""
    proxies = [upright_pillar_proxy(name, pillar) for name in company_names]
    valid = [p.percentile for p in proxies if p.percentile is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import sys
    if len(sys.argv) < 3 or sys.argv[1] != "score":
        print("Usage: python -m agentic_estimation.layer_1.upright_pillar_proxy score <company_name>")
        sys.exit(1)
    company_name = sys.argv[2]
    for pillar in ("E", "S"):
        proxy = upright_pillar_proxy(company_name, pillar)
        print(f"{pillar}: percentile={proxy.percentile} "
              f"({proxy.n_components}/{proxy.n_components_total} components available)")


if __name__ == "__main__":
    _cli()
