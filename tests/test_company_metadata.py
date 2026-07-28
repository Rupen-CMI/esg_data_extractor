"""company_metadata.py's DEFECT_FIX_PLAN.md 2.3 (H3+H4) fixes:
- H3: Wikidata QID search accepting hits[0] unconditionally with no name
  similarity check, when no hit's description mentioned a company word.
- H4: revenue/total_assets fetched in their native Wikidata currency (often
  EUR, confirmed live for Bosch) but labeled "(USD)" unconditionally with no
  conversion -- a real value error, not a display quirk.
"""
from agentic_estimation.layer_1.company_metadata import _to_usd, _name_overlap


# ── _to_usd ───────────────────────────────────────────────────────────────

def test_usd_passthrough_rate_one():
    assert _to_usd(1000.0, "United States dollar", "TestCo") == 1000.0


def test_euro_converted():
    result = _to_usd(1000.0, "euro", "TestCo")
    assert abs(result - 1080.0) < 1e-6


def test_case_insensitive_currency_label():
    assert _to_usd(1000.0, "EURO", "TestCo") == _to_usd(1000.0, "euro", "TestCo")


def test_none_value_returns_none():
    assert _to_usd(None, "euro", "TestCo") is None


def test_no_currency_label_passes_through_unchanged():
    """Older Wikidata statements sometimes lack a quantityUnit -- treat as
    already-USD (prior behavior) rather than discarding a real number."""
    assert _to_usd(1000.0, None, "TestCo") == 1000.0
    assert _to_usd(1000.0, "", "TestCo") == 1000.0


def test_unrecognized_currency_drops_value_not_silently_passes_through():
    """An unrecognized NON-USD currency must not silently pass through
    unconverted -- that would be the exact original bug (EUR mislabeled as
    USD), just for a currency not yet in the table."""
    assert _to_usd(1000.0, "Bitcoin", "TestCo") is None


# ── _name_overlap (used by the new Wikidata QID guard) ───────────────────

def test_name_overlap_exact_match():
    assert _name_overlap("Apple Inc", "Apple Inc") == 1.0


def test_name_overlap_unrelated_names_score_zero():
    """The exact failure shape the QID guard exists to catch: an unrelated
    Wikidata entity (e.g. a fruit, a person) sharing no real tokens with the
    company name."""
    assert _name_overlap("Apple Inc", "edible fruit of the apple tree") == 0.0


def test_name_overlap_suffix_variant_still_matches():
    score = _name_overlap("Apple", "Apple Inc.")
    assert score > 0.6
