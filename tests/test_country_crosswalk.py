"""country_crosswalk.py -- DEFECT_FIX_PLAN.md 2.1's country-vocabulary fix.
bcorp_lookup.country stores full names (with some source-specific spelling
quirks); upright_lookup.country stores ISO3 codes. Both were being queried
with the SAME unconverted caller string, silently making every upright
country+sector-tier peer lookup unreachable."""
from agentic_estimation.layer_1.country_crosswalk import country_for_bcorp, country_for_upright


def test_bcorp_passthrough_for_matching_economy_name():
    """88/103 bcorp countries already match the WB Economy name as-is."""
    assert country_for_bcorp("Germany") == "Germany"
    assert country_for_bcorp("France") == "France"


def test_bcorp_quirk_conversion():
    assert country_for_bcorp("Czechia") == "Czech Republic"
    assert country_for_bcorp("Croatia") == "Croatia (Hrvatska)"
    assert country_for_bcorp("Netherlands") == "Netherlands The"
    assert country_for_bcorp("Hong Kong SAR, China") == "Hong Kong S.A.R."


def test_bcorp_raw_string_fallback():
    assert country_for_bcorp("jersey") == "Jersey"
    assert country_for_bcorp("Puerto Rico") == "Puerto Rico"


def test_bcorp_none_input():
    assert country_for_bcorp(None) is None
    assert country_for_bcorp("") is None


def test_upright_economy_name_converts_to_iso3():
    assert country_for_upright("France") == "FRA"
    assert country_for_upright("Germany") == "DEU"


def test_upright_already_iso3_passes_through():
    assert country_for_upright("FRA") == "FRA"
    assert country_for_upright("fra") == "FRA"


def test_upright_no_wb_territory_fallback():
    assert country_for_upright("Taiwan") == "TWN"
    assert country_for_upright("Hong Kong") == "HKG"
    assert country_for_upright("Jersey") == "JEY"


def test_upright_unresolvable_returns_none():
    assert country_for_upright("Nowhereland") is None


def test_upright_none_input():
    assert country_for_upright(None) is None
    assert country_for_upright("") is None
