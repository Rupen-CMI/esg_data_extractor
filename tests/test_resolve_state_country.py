"""graph.py's _resolve_state_country -- the single country-resolution point
added for DEFECT_FIX_PLAN.md 1.1 (C2+C3). Before this, only the legacy llm
scorer nodes ever wrote state['country']; the ensemble/formula paths just
read back whatever was passed into run_company_graph unchanged (often "" or
None), so every downstream consumer (baseline, peer anchor, holistic vote)
could silently run with no country context even when metadata had one."""
from agentic_estimation.graph import _resolve_state_country


def test_empty_input_falls_back_to_metadata_country():
    assert _resolve_state_country("", {"country": "Germany"}) == "Germany"


def test_none_input_falls_back_to_metadata_country():
    assert _resolve_state_country(None, {"country": "Germany"}) == "Germany"


def test_input_country_wins_over_metadata():
    assert _resolve_state_country("France", {"country": "Germany"}) == "France"


def test_no_country_anywhere_returns_none():
    assert _resolve_state_country("", {}) is None
    assert _resolve_state_country(None, {}) is None


def test_resolvable_alias_is_canonicalized():
    """"USA" resolves to the World Bank dataset's own Economy name via
    resolve_country_name -- confirms the alias/ISO3 resolution actually runs,
    not just a pass-through."""
    assert _resolve_state_country("USA", {}) == "United States"


def test_unresolvable_string_is_kept_raw_not_dropped():
    """A country string that doesn't resolve via alias/ISO3 lookup is kept
    as-is (not discarded to None) -- get_country_baseline_with_fallback has
    its own downstream regional/global-average chain that still needs SOME
    string to log/fall back from."""
    result = _resolve_state_country("Nowhereland", {})
    assert result == "Nowhereland"


def test_metadata_country_also_gets_canonicalized():
    assert _resolve_state_country(None, {"country": "usa"}) == "United States"
