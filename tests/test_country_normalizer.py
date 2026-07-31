"""
Tests for country_normalizer + its integration into country resolution.

The rejection tests matter as much as the resolution tests. Resolution failing
is visible (None -> global-average baseline, logged); a bad resolution is
silent and produces a wrong baseline, wrong peer group, and wrong localized
keywords with no signal that anything went wrong.

Three distinct behaviours are pinned here, because they are easy to conflate:
  * countries and their many spellings resolve (including ISO2/ISO3 codes and
    World Bank Economy names, which must round-trip)
  * ISO 3166-2 subdivisions resolve to their parent country ("Delhi" -> IND)
  * non-places do NOT resolve -- placeholders ("global"), continents/regions
    ("Europe", bare "America"), addresses, and scraped prose

Plain city names (Mumbai, Sydney) sit outside ISO 3166 entirely and cannot
resolve; that boundary is asserted explicitly so it reads as a known limit
rather than an inconsistency with Delhi.
"""

import pytest

from agentic_estimation.layer_1.country_normalizer import (
    to_iso3, iso3_to_common_name, is_country,
)
from agentic_estimation.layer_1.country_baseline_agent import (
    resolve_country_name, get_country_baseline_with_fallback,
)


# ── Resolution: every one of these must produce the right ISO3 ────────────────

@pytest.mark.parametrize("raw,expected", [
    # ISO codes: alpha-3, alpha-2 (alpha-2 was unsupported before).
    ("USA", "USA"), ("GBR", "GBR"), ("DEU", "DEU"), ("IND", "IND"),
    ("IN", "IND"), ("DE", "DEU"),
    # Canonical names.
    ("United States", "USA"), ("United Kingdom", "GBR"), ("Germany", "DEU"),
    # US colloquialisms.
    ("US", "USA"), ("U.S.", "USA"), ("U.S.A.", "USA"), ("usa", "USA"),
    ("united states of america", "USA"),
    # UK: "great britain" was aliased before but bare "britain" was not, and
    # the constituent countries have no pycountry entry at all.
    ("UK", "GBR"), ("Britain", "GBR"), ("Great Britain", "GBR"),
    ("England", "GBR"), ("Scotland", "GBR"), ("Wales", "GBR"),
    # Netherlands: "Netherlands The" is emitted verbatim by upright_lookup.
    ("Netherlands", "NLD"), ("Netherlands The", "NLD"),
    ("The Netherlands", "NLD"), ("Holland", "NLD"),
    # Territories with a real ISO identity but no World Bank economy -- the
    # identity must still resolve; the missing baseline is a separate concern.
    ("Taiwan", "TWN"), ("TWN", "TWN"),
    ("Hong Kong", "HKG"), ("HKG", "HKG"), ("Hong Kong S.A.R.", "HKG"),
    ("Guernsey", "GGY"), ("GGY", "GGY"),
    # World Bank Economy spellings must round-trip (they flow back in from our
    # own baseline table and from resolve_country_name output).
    ("Korea, Rep.", "KOR"), ("Egypt, Arab Rep.", "EGY"),
    ("Venezuela, RB", "VEN"), ("Lao PDR", "LAO"), ("Slovak Republic", "SVK"),
    ("Turkiye", "TUR"), ("Viet Nam", "VNM"), ("Russian Federation", "RUS"),
    # Colloquial forms.
    ("South Korea", "KOR"), ("Russia", "RUS"), ("Turkey", "TUR"),
    ("Vietnam", "VNM"), ("Czech Republic", "CZE"), ("Ivory Coast", "CIV"),
    # Whitespace / case / markdown bleed-through from the LLM extractor.
    ("  france  ", "FRA"), ("FRANCE", "FRA"), ("**Germany", "DEU"),
])
def test_resolves(raw, expected):
    assert to_iso3(raw) == expected


# ── Rejection: none of these may EVER resolve ────────────────────────────────

@pytest.mark.parametrize("raw", [
    # Non-answers / placeholders. "global" alone accounted for 1415 of 1523
    # companies.country rows before the extractor stopped guessing countries.
    "global", "worldwide", "international", "multinational", "various",
    "N/A", "n/a", "unknown", "none", "null", "not specified", "TBD",
    # Continents and supra-national regions -- real places, but not countries.
    # Bare "America" is ambiguous with the continents and must not silently
    # become the US; "United States of America" is the unambiguous form.
    "Europe", "Asia", "Africa", "APAC", "EMEA", "North America",
    "Latin America", "Southeast Asia", "European Union", "Caribbean",
    "America", "Americas",
    # Street addresses -- a leading digit is never a country.
    "4085 Sladeview Crescent",
    # Scraped marketing prose. Rejected structurally by the token cap, NOT by
    # blocklisting the city inside them -- which is why the city-name tests
    # below can safely expect resolution.
    "Sydney well equipped with the best of machines",
    "Riyadh with deep understanding of Saudi regulations and NCA",
    "the Dallas Fort Worth area", "Shanghai 7+ factories in China",
    "Singapore) was formed by Asian",
    "Hong Kong – 10+ years in the watch trade – Ships worldwide",
    # Empty / malformed.
    "", "   ", None,
])
def test_rejects(raw):
    assert to_iso3(raw) is None, f"{raw!r} must not resolve to a country"


@pytest.mark.parametrize("raw,expected", [
    # ISO 3166-2 subdivisions resolve to their parent country. Checked
    # explicitly (not left to country-level fuzzy search, which covers these
    # only by accident) so the boundary is a stated rule rather than a
    # coincidence of which names happen to collide.
    ("Delhi", "IND"), ("Maharashtra", "IND"), ("Gujarat", "IND"),
    ("Antwerp", "BEL"), ("New South Wales", "AUS"),
    ("California", "USA"), ("Ontario", "CAN"),
])
def test_subdivisions_resolve_to_parent_country(raw, expected):
    assert to_iso3(raw) == expected


@pytest.mark.parametrize("raw", [
    # Plain city names are absent from ISO 3166 entirely, so they cannot be
    # resolved. Documented as a known boundary: "Delhi" resolves (it is an
    # Indian union territory in ISO 3166-2) while "Mumbai" does not (a city,
    # not an administrative division). Callers treat None as "unknown country"
    # and fall back to registry resolution.
    "Mumbai", "Sydney", "Shenzhen", "Houston", "Espoo", "Suzhou",
])
def test_plain_cities_are_not_in_iso(raw):
    assert to_iso3(raw) is None


def test_rejects_non_string():
    assert to_iso3(123) is None
    assert to_iso3([]) is None


def test_ambiguous_refuses():
    """Bare "virgin islands" is genuinely ambiguous (VGB vs VIR) -- refusing
    beats picking one. The qualified forms are pinned and must still work."""
    assert to_iso3("virgin islands") is None
    assert to_iso3("Virgin Islands (U.S.)") == "VIR"
    assert to_iso3("British Virgin Islands") == "VGB"


def test_allow_fuzzy_false_still_resolves_exact():
    assert to_iso3("Germany", allow_fuzzy=False) == "DEU"
    assert to_iso3("DEU", allow_fuzzy=False) == "DEU"
    assert to_iso3("Britain", allow_fuzzy=False) == "GBR"  # alias, not fuzzy


def test_memoisation_is_consistent():
    for _ in range(3):
        assert to_iso3("Netherlands The") == "NLD"
        assert to_iso3("Mumbai") is None


def test_iso3_to_common_name():
    assert iso3_to_common_name("USA") == "United States"
    assert iso3_to_common_name("KOR") == "South Korea"
    assert iso3_to_common_name("TWN") == "Taiwan"
    assert iso3_to_common_name("ZZZ") is None
    assert iso3_to_common_name("") is None


def test_is_country():
    assert is_country("Britain")
    assert not is_country("Mumbai")


# ── Integration with the World Bank baseline layer ───────────────────────────

@pytest.mark.parametrize("raw,expected_economy", [
    ("Britain", "United Kingdom"),
    ("England", "United Kingdom"),
    ("Netherlands The", "Netherlands"),
    ("Holland", "Netherlands"),
    ("IN", "India"),
    ("Delhi", "India"),
])
def test_resolve_country_name_gap_cases(raw, expected_economy):
    """Cases that returned None before the normalizer was wired in."""
    assert resolve_country_name(raw) == expected_economy


@pytest.mark.parametrize("raw,expected_economy", [
    ("USA", "United States"), ("United States", "United States"),
    ("GBR", "United Kingdom"), ("CHN", "China"), ("JPN", "Japan"),
    ("South Korea", "Korea, Rep."), ("Russia", "Russian Federation"),
    ("Turkey", "Turkiye"), ("Egypt", "Egypt, Arab Rep."),
    ("Slovakia", "Slovak Republic"), ("India", "India"),
])
def test_resolve_country_name_no_regression(raw, expected_economy):
    """Inputs that already worked must be unchanged -- the normalizer is a
    final fallback stage, never a replacement for the earlier lookups."""
    assert resolve_country_name(raw) == expected_economy


@pytest.mark.parametrize("raw", ["Mumbai", "global", "Europe", "America",
                                 "4085 Sladeview Crescent",
                                 "Sydney well equipped with the best of machines"])
def test_resolve_country_name_still_rejects_junk(raw):
    assert resolve_country_name(raw) is None


def test_resolve_country_name_maps_subdivision_to_economy():
    """"Antwerp" is an ISO 3166-2 Belgian province, so it resolves through to
    Belgium's World Bank economy rather than being discarded."""
    assert resolve_country_name("Antwerp") == "Belgium"


def test_crown_dependencies_use_regional_fallback():
    """The World Bank publishes no economy for Guernsey/Jersey; both are
    British Crown Dependencies, so the UK baseline is a factual stand-in."""
    for raw in ("GGY", "Guernsey", "Jersey"):
        bl, source = get_country_baseline_with_fallback(raw)
        assert source == "regional"
        assert bl.country == "United Kingdom"


@pytest.mark.parametrize("raw", ["TWN", "Taiwan", "taiwan", "Chinese Taipei"])
def test_taiwan_uses_peer_median_not_global_average(raw):
    """Taiwan has no World Bank economy row. The global-average fallback was
    measurably wrong for it -- S 60.68 / G 46.11 against every advanced East
    Asian economy sitting at S 77-89 / G 59-64 -- so it uses a stated peer
    median instead, labelled distinctly so it is never mistaken for real data."""
    bl, source = get_country_baseline_with_fallback(raw)
    assert source == "regional_peer"
    # Median of Japan / Korea / Singapore, well above the global average.
    assert bl.s_score > 80, f"S={bl.s_score} -- expected the peer median, not the world mean"
    assert bl.g_score > 60, f"G={bl.g_score} -- expected the peer median, not the world mean"
    assert bl.indicator_count == 0, "derived baselines must not claim real indicator coverage"
    assert "peer median" in bl.country


def test_taiwan_peer_set_excludes_china():
    """China is Taiwan's largest trading partner but a poor match on what the
    baseline indicators measure (income tier, governance, energy mix). Its
    inclusion would move E by ~-14 and G by ~-4 for reasons that do not
    describe Taiwan."""
    from agentic_estimation.layer_1.country_baseline_agent import _PEER_BASELINE_SETS
    assert "China" not in _PEER_BASELINE_SETS["TWN"]


@pytest.mark.parametrize("raw", ["HKG", "Hong Kong", "MAC", "Macao"])
def test_hong_kong_and_macao_still_global_average(raw):
    """Deliberately NOT given peer sets yet: Hong Kong's governance profile
    diverged from Singapore's after 2020 on exactly the WGI dimensions used
    here, and Macao's gaming-monoculture economy has no structural comparator.
    Both keep the global-average fallback until sourced properly."""
    _, source = get_country_baseline_with_fallback(raw)
    assert source == "global_average"


def test_peer_median_never_overrides_real_country_data():
    """The peer path sits after exact/resolved/regional, so a country with its
    own World Bank baseline must never receive an estimate."""
    for raw in ("Japan", "Korea, Rep.", "Singapore", "USA", "Britain", "IN"):
        _, source = get_country_baseline_with_fallback(raw)
        assert source in ("exact", "resolved"), f"{raw} got {source}"


def test_wb_economy_names_round_trip():
    """Every World Bank Economy name must resolve back to an ISO3 code --
    these strings are produced by our own resolve_country_name and stored in
    country_esg_baseline, so a failure here breaks downstream crosswalks."""
    from agentic_estimation.layer_1.country_baseline_agent import (
        load_all_baselines, get_all_baselines,
    )
    load_all_baselines()
    economies = get_all_baselines()
    assert len(economies) > 200
    unresolved = [name for name in economies if to_iso3(name) is None]
    assert not unresolved, f"WB Economy names that do not resolve: {unresolved}"
