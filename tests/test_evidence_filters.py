"""
Tests for the entity gate and ground-truth leakage filter in evidence_filters.

Both guards were added after measuring the actual evidence in the frozen
calibration dumps: only ~51% of fetched signal text was both about the right
company and ESG-relevant. Two failure modes dominated, and each has a guard
here:

  * OFF-ENTITY: a site:/quoted-name query that finds nothing silently falls
    back to general results, returning either a different company's page
    (Goodyear's audit committee arriving as XinerLink's "board composition")
    or topic-generic filler ("an anti-bribery policy is a component of...").
    Joined into one blob, these were indistinguishable from real evidence.

  * LEAKAGE: bcorp/upright are the accuracy answer-key AND are published on
    the open web, so a plain company search returns the very score being
    predicted. Left in, it inflates tune-set accuracy while teaching nothing
    that generalizes -- the worst kind of bug, because it looks like progress.

The fail-open cases are pinned as deliberately as the rejections. Dropping a
company's every signal because its name is unusual is a worse outcome than
letting one weak snippet through, so entity matching hits on ANY identifying
token and returns True when a name yields no tokens at all.
"""

import pytest

from agentic_estimation.layer_1.evidence_filters import (
    _company_tokens,
    filter_search_results,
    has_ground_truth_leakage,
    mentions_company,
)


# ── tokenization ─────────────────────────────────────────────────────────────

def test_legal_suffixes_are_not_identifying():
    """"Ltd"/"Inc"/"Group" appear in most corporate pages, so matching on them
    would pass essentially every result. Only distinctive words survive."""
    assert _company_tokens("Coursera, Inc.") == ["coursera"]
    assert _company_tokens("Spectre Holding A/S") == ["spectre"]
    assert "limited" not in _company_tokens("Seetec Business Technology Centre Limited")


def test_short_and_punctuation_only_tokens_dropped():
    assert _company_tokens("ba&sh") == []          # both fragments <= 2 chars
    assert "s" not in _company_tokens("Sales: Untangled")


# ── entity gate ──────────────────────────────────────────────────────────────

def test_rejects_other_company_page():
    """The exact failure found in the dumps: Goodyear's proxy statement
    returned as XinerLink's board-composition evidence."""
    snippet = ("The Audit Committee assists the Board in fulfilling its "
               "responsibilities for oversight of Goodyear's financial statements")
    assert not mentions_company(snippet, "XinerLink")


def test_rejects_topic_generic_filler():
    """Returned verbatim for both Seetec and Coursera when the company-specific
    query found nothing -- a definition, not evidence about anyone."""
    snippet = ("An anti-bribery policy is a component of an overall compliance "
               "policy and helps an organization avoid the costs of bribery.")
    assert not mentions_company(snippet, "Coursera, Inc.")


def test_accepts_real_mention():
    assert mentions_company("Reviews from XINERLINK employees about culture",
                            "XinerLink")


def test_partial_name_still_matches():
    """Snippets truncate long names; requiring the full string would drop
    legitimate evidence."""
    assert mentions_company("Isigny-Sainte-Mere seeks to reassure employees",
                            "ISIGNY SAINTE-MERE")


def test_url_alone_is_sufficient():
    """A company's own domain often omits the name from the visible snippet."""
    assert mentions_company("Our 2024 impact report is now available.",
                            "Coursera", url="https://coursera.org/impact")


def test_unidentifiable_name_fails_open():
    """A name of pure legal-form filler yields no tokens. Returning False would
    drop every result for that company; returning True keeps it in the corpus."""
    assert mentions_company("any text at all", "The Group Ltd")


# ── leakage ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("snippet", [
    "Based on the B Impact assessment, Seetec earned an overall score of 136.4.",
    "The median score for ordinary businesses who complete the assessment is 50.9.",
    "Certified B Corporation. Certified Since 2019.",
])
def test_detects_benchmark_score_leakage(snippet):
    assert has_ground_truth_leakage(snippet)


def test_ordinary_esg_text_is_not_leakage():
    """The filter must not swallow genuine disclosure evidence."""
    assert not has_ground_truth_leakage(
        "The company published its 2024 sustainability report and disclosed "
        "Scope 1 and Scope 2 emissions to CDP."
    )


# ── per-result filtering ─────────────────────────────────────────────────────

def test_filters_results_individually_not_as_a_blob():
    """The core fix: one bad result among good ones is dropped on its own
    rather than contaminating a joined string."""
    results = [
        {"body": "Goodyear audit committee oversight", "href": "http://a.com"},
        {"body": "XinerLink employee reviews and culture", "href": "http://b.com"},
        {"body": "Based on the B Impact assessment, overall score of 92", "href": "http://c.com"},
    ]
    kept = filter_search_results(results, "XinerLink")
    assert [r["href"] for r in kept] == ["http://b.com"]


def test_empty_result_signals_no_evidence():
    """Callers must treat [] as "no signal". Falling back to the unfiltered
    text would make the whole gate moot."""
    results = [{"body": "Goodyear audit committee", "href": "http://a.com"}]
    assert filter_search_results(results, "XinerLink") == []


def test_require_entity_false_keeps_topic_results():
    """Country-level governance lookups are about a jurisdiction, not a named
    company; demanding the company name there would reject everything."""
    results = [{"body": "India's corporate governance code requires...", "href": "http://a.com"}]
    assert len(filter_search_results(results, "SomeCo", require_entity=False)) == 1


def test_leakage_dropped_even_when_entity_check_disabled():
    """Leakage is unconditional -- it is never acceptable input."""
    results = [{"body": "Certified B Corporation with an overall score of 101", "href": "http://a.com"}]
    assert filter_search_results(results, "SomeCo", require_entity=False) == []
