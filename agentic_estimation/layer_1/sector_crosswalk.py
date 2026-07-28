"""
sector_crosswalk.py — static, hand-built mapping between bcorp_lookup's and
upright_lookup's sector/industry vocabularies.

WHY THIS EXISTS: bcorp_lookup.industry_category (22 real values, e.g.
"Manufactured Goods", "Financial & insurance activities") and
upright_lookup.industry (30 real values, e.g. "Automotive", "Insurance") are
two DIFFERENT, independently-designed taxonomies with no shared vocabulary —
confirmed live (see peer_anchor.py, sector_matcher.py) that exact-string
matching between them almost never hits, and even TF-IDF fuzzy matching
misses pure-synonym pairs with zero shared tokens (e.g. "Automotive" has no
token in common with a hypothetical "Car Manufacturing" bcorp label).

Rather than matching these two vocabularies at query time via similarity
scoring, this module hand-maps them ONCE, offline: every one of upright's 30
industry labels is assigned to exactly one of bcorp's 22 industry_category
buckets (bcorp's categories are broader, so this is the natural direction —
several upright industries legitimately belong under one bcorp category,
e.g. "Computer Hardware" and "Computer Software" both fall under bcorp's
"Information, communication & technology"). This mapping was built by reading
both label sets side by side, not derived from string similarity — it is
exact and auditable: two companies with the SAME canonical bcorp category are
peers by construction, not because a similarity threshold happened to clear
some bar.

This is deliberately NOT a replacement for sector_matcher.py's fuzzy match --
it is a higher-confidence, exact-match tier that peer_anchor.py tries FIRST.
sector_matcher.py's fuzzy TF-IDF fallback still runs for any sector string
that isn't a recognized upright label or a recognized bcorp category (e.g. a
company's own free-text industry field from Wikidata/metadata, which matches
neither vocabulary exactly).

Coverage note: this crosswalk only standardizes the bcorp<->upright pair.
Climate TRACE's sector/subsector vocabulary is a SEPARATE, narrower taxonomy
(10 sectors / 64 subsectors, all physical/emissions-heavy activities -- power,
mining, agriculture, heavy manufacturing) with no category at all for most of
upright's 30 industries (banks, software, healthcare, retail, insurance,
telecom...). Measured live: of ~20,400 bcorp+upright companies, only 35 (all
upright-side) exact-name-match a Climate TRACE owner, and 0 bcorp-side do --
these are almost entirely disjoint company populations, not overlapping views
of the same universe. Climate TRACE is therefore NOT folded into this
crosswalk; its owner/subsector match stays a separate, rare, high-value bonus
signal (see climate_trace_anchor.py), applied independently of whichever
sector-standardization path a company's peer group used.

CLI:
    python -m agentic_estimation.layer_1.sector_crosswalk lookup "Automotive"
    python -m agentic_estimation.layer_1.sector_crosswalk list
"""

from typing import Optional

# Every one of upright_lookup's 30 distinct industry values (confirmed live),
# hand-mapped to exactly one of bcorp_lookup's 22 distinct industry_category
# values (also confirmed live). Read top-to-bottom as "this upright industry
# belongs under this bcorp category."
UPRIGHT_TO_BCORP_CATEGORY: dict[str, str] = {
    "Aerospace and Defense":                   "Manufactured Goods",
    "Agriculture and Forestry":                "Agriculture, forestry & fishing",
    "Automotive":                              "Manufactured Goods",
    "Banks":                                   "Financial & insurance activities",
    "Chemicals":                               "Manufactured Goods",
    "Civic, Non-Profit and Membership Groups": "Other services",
    "Computer Hardware":                       "Information, communication & technology",
    "Computer Software":                       "Information, communication & technology",
    "Construction and Building Materials":     "Construction",
    "Consumer Product Manufacturing":          "Manufactured Goods",
    "Consumer Services":                       "Other services",
    "Corporate Services":                      "Professional & technical services",
    "Electronics":                             "Manufactured Goods",
    "Energy and Environmental":                "Energy",
    "Financial Services":                      "Financial & insurance activities",
    "Food and Beverage":                       "Manufactured Goods",
    "Government":                              "Other services",
    "Holding Companies":                       "Financial & insurance activities",
    "Hospitals and Healthcare":                "Human health & social work",
    "Industrial Manufacturing and Services":   "Manufactured Goods",
    "Insurance":                               "Financial & insurance activities",
    "Leisure, Sports and Recreation":          "Arts, entertainment & recreation",
    "Media":                                   "Information, communication & technology",
    "Mining and Metals":                       "Manufactured Goods",
    "Pharmaceuticals and Biotechnology":       "Manufactured Goods",
    "Real Estate":                             "Real estate, design & building",
    "Retail":                                  "Retail",
    "Schools and Education":                   "Education",
    "Telecommunications":                      "Information, communication & technology",
    "Transportation":                          "Transportation & storage",
}

# Reverse index: bcorp category -> every upright industry mapped to it. Built
# once at import time from the table above (never hand-maintained separately,
# so the two directions can't drift out of sync).
BCORP_CATEGORY_TO_UPRIGHT: dict[str, list[str]] = {}
for _upright, _bcorp in UPRIGHT_TO_BCORP_CATEGORY.items():
    BCORP_CATEGORY_TO_UPRIGHT.setdefault(_bcorp, []).append(_upright)


def canonical_bcorp_category(upright_industry: Optional[str]) -> Optional[str]:
    """upright_lookup.industry -> its mapped bcorp_lookup.industry_category,
    or None if `upright_industry` isn't one of the 30 known labels."""
    if not upright_industry:
        return None
    return UPRIGHT_TO_BCORP_CATEGORY.get(upright_industry)


def upright_industries_for_bcorp_category(bcorp_category: Optional[str]) -> list[str]:
    """bcorp_lookup.industry_category -> every upright_lookup.industry label
    mapped to it (possibly empty if `bcorp_category` isn't one of the 22
    known labels, or none of upright's industries map to it)."""
    if not bcorp_category:
        return []
    return list(BCORP_CATEGORY_TO_UPRIGHT.get(bcorp_category, []))


def crosswalk_sector(sector: Optional[str]) -> Optional[str]:
    """Given ANY sector string that might be either an upright industry or a
    bcorp industry_category, return the canonical bcorp_lookup.industry_category
    bucket it belongs to -- the crosswalk's single canonical target vocabulary
    (bcorp's 22 categories are broader/pillar-scored, see module docstring).
    Returns None if `sector` matches neither known vocabulary exactly --
    caller should fall back to sector_matcher.py's fuzzy match."""
    if not sector:
        return None
    if sector in BCORP_CATEGORY_TO_UPRIGHT:
        return sector  # already a bcorp category
    return UPRIGHT_TO_BCORP_CATEGORY.get(sector)  # try as an upright industry


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m agentic_estimation.layer_1.sector_crosswalk {lookup <sector>|list}")
        sys.exit(1)

    if sys.argv[1] == "list":
        for bcorp_cat, upright_list in sorted(BCORP_CATEGORY_TO_UPRIGHT.items()):
            print(f"\n{bcorp_cat}:")
            for u in upright_list:
                print(f"  - {u}")
    elif sys.argv[1] == "lookup" and len(sys.argv) > 2:
        sector = sys.argv[2]
        canonical = crosswalk_sector(sector)
        print(f"crosswalk_sector({sector!r}) -> {canonical!r}")
    else:
        print("Usage: python -m agentic_estimation.layer_1.sector_crosswalk {lookup <sector>|list}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
