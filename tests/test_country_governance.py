"""
Tests for the country-level governance keyword vocabulary and its fetcher.

The relevance/bleed-through gates carry most of the weight here. A locale-routed
query returns whatever that country's news edition ranks highest, and DuckDuckGo
falls back to general reference pages when a query is sparse -- verified live,
that produced air-quality alerts, university sustainability awards, Taiwan
population statistics, and (worst) another country's governance regime. Any of
those reaching the claim extractor invites invented governance claims, so the
filters are the part that must not regress.

Network-touching tests are marked slow and assert on structure, not on specific
headlines (live feeds change daily).
"""

import pytest

from agentic_estimation.layer_1.country_governance_keywords import (
    COUNTRY_GOVERNANCE_PROFILES,
    generic_governance_terms,
    get_governance_profile,
    governance_terms_for,
    locales_for,
    supported_iso3,
)
from agentic_estimation.layer_1.signal_agent import (
    _is_governance_relevant,
    _mentions_country,
    _split_snippet_segments,
)


# ── Keyword vocabulary ────────────────────────────────────────────────────────

def test_profiles_cover_the_main_corpus_countries():
    """These account for ~86% of the 393-company benchmark corpus."""
    for iso3 in ("USA", "GBR", "FRA", "CHN", "ITA", "JPN", "NLD", "AUS",
                 "CHL", "COL", "FIN", "BRA", "DEU", "CAN", "SWE", "ESP",
                 "CHE", "BEL", "DNK", "MEX", "TWN", "KOR", "IND"):
        assert iso3 in COUNTRY_GOVERNANCE_PROFILES, f"missing profile: {iso3}"


def test_every_profile_has_locales_and_terms():
    for iso3, profile in COUNTRY_GOVERNANCE_PROFILES.items():
        assert profile.iso3 == iso3, f"{iso3}: iso3 field mismatch"
        assert profile.gn_locales, f"{iso3}: no Google News locale"
        assert profile.all_terms(), f"{iso3}: no governance terms"
        for locale in profile.gn_locales:
            # Locale strings are appended raw to a Google News query string.
            assert "hl=" in locale and "gl=" in locale and "ceid=" in locale, \
                f"{iso3}: malformed locale {locale!r}"


def test_named_regimes_present_for_countries_rupa_had_nothing_for():
    """rupa's ESG keyword sets had ZERO governance terms for Japan, Germany and
    South Korea despite each having a well-known named regime. That gap is the
    reason this module exists rather than reusing those keywords."""
    assert any("会社法" in t or "コーポレートガバナンス" in t
               for t in COUNTRY_GOVERNANCE_PROFILES["JPN"].all_terms())
    assert any("Kodex" in t or "Aufsichtsrat" in t
               for t in COUNTRY_GOVERNANCE_PROFILES["DEU"].all_terms())
    assert any("상법" in t or "기업지배구조" in t
               for t in COUNTRY_GOVERNANCE_PROFILES["KOR"].all_terms())
    assert any("SEBI" in t for t in COUNTRY_GOVERNANCE_PROFILES["IND"].all_terms())


def test_regime_terms_come_first():
    """Query builders truncate, so the country's own named instruments must
    survive truncation ahead of generic vocabulary."""
    profile = get_governance_profile("IND")
    assert profile is not None
    assert profile.all_terms()[0] in profile.regime_terms


def test_terms_topped_up_with_generic_vocabulary():
    terms = governance_terms_for("SGP", limit=8)
    assert len(terms) == 8
    assert any(t in generic_governance_terms(99) for t in terms)


def test_unmapped_country_falls_back_to_generic_terms():
    """None from get_governance_profile is not an error -- the fetch still runs
    with an English query rather than being skipped."""
    assert get_governance_profile("ZZZ") is None
    assert locales_for("ZZZ") == ()
    terms = governance_terms_for("ZZZ", limit=5)
    assert terms and all(t in generic_governance_terms(99) for t in terms)


def test_no_duplicate_terms_within_a_profile():
    for iso3, profile in COUNTRY_GOVERNANCE_PROFILES.items():
        terms = profile.all_terms()
        assert len(terms) == len(set(terms)), f"{iso3}: duplicate terms"


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_none_and_empty_are_safe(bad):
    assert get_governance_profile(bad) is None
    assert locales_for(bad) == ()


def test_supported_iso3_matches_profile_keys():
    assert supported_iso3() == frozenset(COUNTRY_GOVERNANCE_PROFILES)


# ── Relevance gate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line", [
    "SEBI Amends LODR Regulations; Listed Entities to Follow New Procedures",
    "Coal India fined for non-compliance with SEBI norms",
    "Corporate Governance Best-Practice Principles for TSE/GTSM Listed Companies",
    "Update fur den Aufsichtsrat - Deloitte",
    "Hyundai Motor Group Stake Purchase Draws Governance Debate",
    "Fraud in related party transactions: Can we spot the rot?",
    "Board Meetings Under India's Companies Act, 2013: Compliance, Procedures",
    "코스피 829개社, 기업지배구조 보고서 제출",
    "台灣與德拉瓦公司法的對話開啟公司治理新視野",
])
def test_governance_relevant_lines_pass(line):
    assert _is_governance_relevant(line)


@pytest.mark.parametrize("line", [
    # All observed live from the un-gated version of this fetch.
    "South Korea Air Quality Alert - IQAir",
    "NCHU Wins Double Honors at the 2025 TCSA Taiwan Corporate Sustainability Awards",
    "True south is one end of the axis about which the Earth rotates, called the South Pole.",
    "With around 23.9 million inhabitants, Taiwan is among the most densely populated countries.",
    "Starbucks Korea Workers Establish Company's First Labor Union",
])
def test_irrelevant_lines_are_filtered(line):
    assert not _is_governance_relevant(line)


# ── Country bleed-through guard ───────────────────────────────────────────────

def test_mentions_country_accepts_direct_and_adjectival_forms():
    assert _mentions_country("Taiwan Stock Exchange corporate governance", "Taiwan", "TWN")
    assert _mentions_country("German corporate governance code", "Germany", "DEU")
    assert _mentions_country("Japanese board reform", "Japan", "JPN")


def test_mentions_country_rejects_other_countries():
    """Observed live: a Taiwan governance query returned an India/CII passage
    and a US Sarbanes-Oxley passage. Attaching another country's regime is
    worse than returning nothing."""
    assert not _mentions_country(
        "CII formed a task force to develop corporate governance code for Indian Companies",
        "Taiwan", "TWN")
    assert not _mentions_country(
        "SOX included several corporate governance related provisions",
        "Taiwan", "TWN")


def test_split_snippet_segments_separates_results():
    blob = ("First snippet about governance <https://a.example/1> "
            "Second snippet about boards <https://b.example/2>")
    segments = _split_snippet_segments(blob)
    assert len(segments) == 2
    assert "First snippet" in segments[0] and "https://a.example/1" in segments[0]
    assert "Second snippet" in segments[1]


def test_split_snippet_segments_handles_no_urls():
    assert _split_snippet_segments("plain text") == ["plain text"]
    assert _split_snippet_segments("") == []


# ── Fetcher contract (no network) ──────────────────────────────────────────────

@pytest.mark.parametrize("country", [None, "", "global", "worldwide", "Mumbai"])
def test_fetch_returns_empty_for_non_countries(country):
    """A value that does not resolve to a country must short-circuit before any
    network call -- otherwise every polluted country string triggers a fetch."""
    from agentic_estimation.layer_1.signal_agent import _country_governance_signal
    assert _country_governance_signal(country, "steel") == ""


@pytest.mark.slow
def test_live_fetch_returns_governance_evidence_with_urls():
    from agentic_estimation.layer_1.signal_agent import _country_governance_signal
    text = _country_governance_signal("India", "cement")
    assert text, "expected governance evidence for India"
    assert "Country Governance Context" in text
    assert "<http" in text, "every signal must carry its source URL"
    body = text.splitlines()[1:]
    assert body, "expected at least one evidence line"
    assert all(_is_governance_relevant(line) for line in body), \
        "every line must pass the governance relevance gate"


@pytest.mark.slow
def test_live_fetch_is_cached_per_country_and_industry():
    """The result is identical for every company in a country, so it must be
    fetched once per (country, industry) rather than once per company -- the
    corpus gather was rate-limited off the free-tier gateway twice."""
    import time
    from agentic_estimation.layer_1.signal_agent import _country_governance_signal

    first = _country_governance_signal("Sweden", "manufacturing")
    started = time.perf_counter()
    second = _country_governance_signal("Sweden", "manufacturing")
    elapsed = time.perf_counter() - started

    assert first == second
    assert elapsed < 0.5, f"cache miss on repeat call ({elapsed:.2f}s)"
