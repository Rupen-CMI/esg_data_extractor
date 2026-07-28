"""
country_crosswalk.py — resolve a caller's country string to the exact
vocabulary bcorp_lookup.country / upright_lookup.country actually use, so
peer_anchor_collector.py's `country = %s` SQL filter isn't silently
comparing incompatible strings.

WHY THIS EXISTS (DEFECT_FIX_PLAN.md 2.1): bcorp_lookup.country and
upright_lookup.country are TWO DIFFERENT vocabularies -- confirmed live via
direct query:
  - bcorp_lookup.country:   full country names ("France", "Germany") --
    88 of 103 distinct values match the World Bank Metadata sheet's own
    "Economy" column exactly (the same canonical form resolve_country_name()
    already produces); 15 do NOT (bcorp-specific spellings/quirks, see
    _BCORP_COUNTRY_QUIRKS below).
  - upright_lookup.country: ISO3 codes ("FRA", "DEU") -- 95 of 101 distinct
    values match the World Bank Metadata sheet's ISO3 column exactly; 6
    (including a NULL row) do not (small territories/regions the World Bank
    dataset itself omits, see _UPRIGHT_NO_WB_TERRITORY below).

peer_anchor.py's callers resolve a company's country via
country_baseline_agent.resolve_country_name() (or graph.py's
_resolve_state_country), which produces a World Bank Economy name --
correct for bcorp's SQL filter as-is for 88/103 cases, but ALWAYS wrong for
upright's ISO3 column, and wrong for bcorp's 15 quirky spellings too. Without
this crosswalk, every upright country+sector-tier peer lookup was silently
unreachable (0 peers, always falling through to the sector-only or
sector_matcher fuzzy tiers) regardless of how good the sector match was --
directly contributing to the baseline-tie collapse this fix targets.

Usage:
    from agentic_estimation.layer_1.country_crosswalk import (
        country_for_bcorp, country_for_upright,
    )
    bcorp_country = country_for_bcorp("Czech Republic")     # -> "Czechia"... no,
                                                             # see note below
"""
from typing import Optional

# Bcorp-specific country-string quirks, confirmed live (query above) --
# these 15 bcorp_lookup.country values do NOT match the World Bank Metadata
# sheet's "Economy" column, so resolve_country_name()'s WB-canonical output
# needs converting BACK to bcorp's own spelling before it can match bcorp's
# `country = %s` filter. Built by direct inspection, not string similarity --
# each entry is a real, confirmed 1:1 spelling difference for the SAME
# country/territory, keyed by the WB canonical Economy name (or
# resolve_country_name()'s alias target) so this composes with that
# function's output.
#
# 8 of these (Czech Republic/Czechia, Egypt, Russia/Russian Federation,
# South Korea/Korea Rep., Slovakia/Slovak Republic, Turkey/Turkiye,
# Venezuela/Venezuela RB, Vietnam/Viet Nam) are already the exact INVERSE of
# country_baseline_agent._COUNTRY_ALIASES -- i.e. bcorp uses the everyday
# name that resolve_country_name() ALIASES AWAY from. The other 7 have no WB
# equivalent at all (small territories, or a WB-side omission).
_WB_ECONOMY_TO_BCORP: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Egypt, Arab Rep.": "Egypt",
    "Russian Federation": "Russia",
    "Korea, Rep.": "South Korea",
    "Slovak Republic": "Slovakia",
    "Turkiye": "Turkey",
    "Venezuela, RB": "Venezuela",
    "Viet Nam": "Vietnam",
    "Croatia": "Croatia (Hrvatska)",
    "Netherlands": "Netherlands The",
    "Hong Kong SAR, China": "Hong Kong S.A.R.",
    "Taiwan, China": "Taiwan",
    # No World Bank Economy equivalent at all for these three (small
    # territories) -- included as direct raw-string entries below instead,
    # not through the WB-Economy-name path.
}

# Raw input strings (not WB Economy names) that map directly to a bcorp
# country value -- covers cases where the caller's raw/metadata-sourced
# country string IS bcorp's own spelling already, or where no WB Economy
# name exists to route through _WB_ECONOMY_TO_BCORP above.
_RAW_TO_BCORP: dict[str, str] = {
    "guernsey": "Guernsey and Alderney",
    "alderney": "Guernsey and Alderney",
    "jersey": "Jersey",
    "puerto rico": "Puerto Rico",
    "hong kong": "Hong Kong S.A.R.",
    "taiwan": "Taiwan",
}

# ISO3 codes confirmed live in upright_lookup.country that have NO row in the
# World Bank Metadata sheet at all (small territories the WB dataset omits,
# or Taiwan for the same political-status reason many international datasets
# omit it) -- economy_to_iso3() can never resolve these, so they're listed
# here as a direct raw-string -> ISO3 fallback for callers that already know
# the territory name.
_UPRIGHT_NO_WB_TERRITORY: dict[str, str] = {
    "guernsey": "GGY",
    "alderney": "GGY",
    "jersey": "JEY",
    "hong kong": "HKG",
    "macau": "MAC",
    "macao": "MAC",
    "taiwan": "TWN",
}


def country_for_bcorp(country: Optional[str]) -> Optional[str]:
    """Convert a resolved country string (WB Economy name, or a raw
    caller-supplied string) to the exact spelling bcorp_lookup.country uses.
    Falls through unchanged when no quirk applies -- covers the 88/103 bcorp
    countries that already match the WB Economy name as-is."""
    if not country:
        return None
    if country in _WB_ECONOMY_TO_BCORP:
        return _WB_ECONOMY_TO_BCORP[country]
    lower = country.strip().lower()
    if lower in _RAW_TO_BCORP:
        return _RAW_TO_BCORP[lower]
    return country


def country_for_upright(country: Optional[str]) -> Optional[str]:
    """Convert a resolved country string (WB Economy name, or a raw
    caller-supplied string) to the ISO3 code upright_lookup.country uses.
    Returns None if no ISO3 code can be determined (caller should treat this
    as "no country filter available for upright", same as country=None)."""
    if not country:
        return None
    lower = country.strip().lower()
    if lower in _UPRIGHT_NO_WB_TERRITORY:
        return _UPRIGHT_NO_WB_TERRITORY[lower]
    # Already an ISO3 code (3 alpha chars) -- pass through unchanged rather
    # than trying to look it up as an Economy name.
    if len(country.strip()) == 3 and country.strip().isalpha():
        return country.strip().upper()
    from agentic_estimation.layer_1.country_baseline_agent import economy_to_iso3
    return economy_to_iso3(country)
