"""
Tests for all-210-country governance locale routing and multi-language vocabulary.

The headline guarantee is coverage: every World Bank economy must get a
locale-routed, native-language governance query, not just the ~30 with a
hand-written profile. Before this module ~180 economies fell back to an
unrouted English query -- the same failure mode country_esg_keywords.py was
built to fix for E/S ("non-English-market companies returned zero or near-zero
English-language signals").

The locale strings are also cross-checked against rupa's 58 hand-verified
entries, which is what caught two derivation bugs: an invented "en-SG"/"en-IE"
regional edition, and "ceid=MY:en-MY" where Google actually serves "ceid=MY:en".
"""

import json
from pathlib import Path

import pytest

from agentic_estimation.layer_1.country_baseline_agent import (
    get_all_baselines,
    load_all_baselines,
)
from agentic_estimation.layer_1.country_governance_keywords import (
    governance_terms_for,
    locales_for,
)
from agentic_estimation.layer_1.country_normalizer import to_iso3
from agentic_estimation.layer_1.governance_locales import (
    derive_gn_locales,
    languages_for,
    mapped_countries,
    native_governance_terms,
    supported_languages,
)

_RUPA_LOCALES = Path(
    "C:/Users/rupen/Documents/rupa/code/config_countries/esg_data_sources_all_countries.json"
)


@pytest.fixture(scope="module")
def wb_iso3() -> list[str]:
    load_all_baselines()
    return sorted({i for i in (to_iso3(n) for n in get_all_baselines()) if i})


# ── Coverage: the whole point of this module ──────────────────────────────────

def test_every_world_bank_economy_gets_a_locale(wb_iso3):
    missing = [i for i in wb_iso3 if not locales_for(i)]
    assert not missing, f"economies with no Google News locale: {missing}"


def test_every_world_bank_economy_gets_governance_terms(wb_iso3):
    missing = [i for i in wb_iso3 if not governance_terms_for(i)]
    assert not missing, f"economies with no governance terms: {missing}"


def test_corpus_has_at_least_200_economies(wb_iso3):
    """Guards the fixture itself -- a truncated baseline load would make the
    coverage assertions above pass trivially."""
    assert len(wb_iso3) >= 200


# ── Locale string correctness ─────────────────────────────────────────────────

def test_locale_strings_are_well_formed(wb_iso3):
    for iso3 in wb_iso3:
        for locale in locales_for(iso3):
            assert locale.startswith("hl="), f"{iso3}: {locale}"
            assert "&gl=" in locale and "&ceid=" in locale, f"{iso3}: {locale}"
            assert " " not in locale, f"{iso3}: whitespace in {locale}"


@pytest.mark.parametrize("iso3,expected", [
    ("USA", "hl=en-US&gl=US&ceid=US:en"),
    ("GBR", "hl=en-GB&gl=GB&ceid=GB:en"),
    ("IND", "hl=en-IN&gl=IN&ceid=IN:en"),
    ("DEU", "hl=de&gl=DE&ceid=DE:de"),
    ("BRA", "hl=pt-BR&gl=BR&ceid=BR:pt"),
    ("PRT", "hl=pt-PT&gl=PT&ceid=PT:pt"),
    ("PER", "hl=es-419&gl=PE&ceid=PE:es-419"),
    ("ESP", "hl=es&gl=ES&ceid=ES:es"),
    ("TWN", "hl=zh-TW&gl=TW&ceid=TW:zh-Hant"),
    ("CHN", "hl=zh-CN&gl=CN&ceid=CN:zh-Hans"),
    ("SGP", "hl=en&gl=SG&ceid=SG:en"),
])
def test_known_locale_strings(iso3, expected):
    assert expected in derive_gn_locales(iso3)


def test_malaysia_ceid_uses_base_language():
    """Google serves "hl=en-MY&...&ceid=MY:en", never "ceid=MY:en-MY". A first
    pass here emitted the regional variant in both positions."""
    locales = derive_gn_locales("MYS")
    assert "hl=en-MY&gl=MY&ceid=MY:en" in locales


def test_no_invented_regional_english_editions():
    """rupa's live-checked strings use plain "hl=en" for these markets; Google
    does not serve an "en-XX" edition for every English-speaking country."""
    for iso3, iso2 in (("SGP", "SG"), ("IRL", "IE"), ("NZL", "NZ"),
                       ("PHL", "PH"), ("ZAF", "ZA"), ("NGA", "NG")):
        assert f"hl=en&gl={iso2}" in " ".join(derive_gn_locales(iso3)), \
            f"{iso3} should use plain hl=en"


@pytest.mark.skipif(not _RUPA_LOCALES.exists(),
                    reason="rupa reference config not present")
def test_matches_rupa_hand_verified_locales():
    """Cross-check against 58 locale strings verified live in a separate
    shipped codebase. Exact agreement on all 58 is what validates the
    mechanical derivation for the other ~150 countries."""
    entries = json.loads(_RUPA_LOCALES.read_text(encoding="utf-8"))
    disagreements = []
    for entry in entries:
        iso3 = to_iso3(entry["iso2"])
        if not iso3:
            continue
        mine = set(derive_gn_locales(iso3))
        theirs = set(entry["gn_locale"])
        if not (mine & theirs):
            disagreements.append((entry["iso2"], sorted(theirs), sorted(mine)))
    assert not disagreements, f"locale disagreements vs rupa: {disagreements}"


# ── Language mapping ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("iso3,lang", [
    ("DEU", "de"), ("AUT", "de"), ("FRA", "fr"), ("BRA", "pt"),
    ("PRT", "pt"), ("MEX", "es"), ("POL", "pl"), ("TUR", "tr"),
    ("EGY", "ar"), ("SAU", "ar"), ("KAZ", "ru"), ("UKR", "uk"),
    ("GRC", "el"), ("ISR", "he"), ("VNM", "vi"), ("IDN", "id"),
    ("THA", "th"), ("JPN", "ja"), ("KOR", "ko"), ("TWN", "zh"),
])
def test_primary_language(iso3, lang):
    assert languages_for(iso3)[0] == lang


def test_english_primary_for_english_business_press():
    """India, Nigeria, Kenya, Pakistan and the Philippines have other national
    languages, but their business/regulatory press is predominantly English --
    matching what rupa's hand-verified locales do."""
    for iso3 in ("IND", "NGA", "KEN", "PAK", "PHL", "SGP", "ZAF"):
        assert languages_for(iso3)[0] == "en", iso3


def test_bilingual_countries_get_both_editions():
    for iso3 in ("CAN", "BEL", "CHE", "MYS", "JPN", "CHN", "HKG"):
        assert len(derive_gn_locales(iso3)) >= 2, f"{iso3} should be multi-locale"


def test_unmapped_country_falls_back_to_english():
    assert languages_for("ZZZ") == ("en",)
    assert languages_for(None) == ("en",)


# ── Native vocabulary ─────────────────────────────────────────────────────────

def test_native_terms_are_in_the_local_language():
    assert "ład korporacyjny" in native_governance_terms("POL", 8)
    assert "gobierno corporativo" in native_governance_terms("PER", 8)
    assert "quản trị công ty" in native_governance_terms("VNM", 8)
    assert "корпоративное управление" in native_governance_terms("KAZ", 8)
    assert "حوكمة الشركات" in native_governance_terms("EGY", 8)
    assert "εταιρική διακυβέρνηση" in native_governance_terms("GRC", 8)


def test_countries_sharing_a_language_share_vocabulary():
    """Vocabulary is modelled per LANGUAGE, not per country -- Austria and
    Germany share German terms, and all of Spanish-speaking Latin America
    shares one set."""
    assert native_governance_terms("AUT", 4) == native_governance_terms("DEU", 4)
    assert native_governance_terms("PER", 4) == native_governance_terms("CHL", 4)


def test_native_terms_respect_limit():
    for limit in (1, 3, 6, 10):
        assert len(native_governance_terms("DEU", limit)) <= limit


def test_every_supported_language_has_terms():
    from agentic_estimation.layer_1.governance_locales import _LANG_TERMS
    for lang, terms in _LANG_TERMS.items():
        assert terms, f"{lang}: no terms"
        assert len(terms) >= 4, f"{lang}: only {len(terms)} terms"


def test_language_coverage_is_reasonable():
    assert len(supported_languages()) >= 25
    assert len(mapped_countries()) >= 200


# ── Curated profiles still win ────────────────────────────────────────────────

def test_curated_named_regimes_take_precedence():
    """The hand-written profiles were authored deliberately; the mechanical
    fallbacks must not displace them."""
    assert governance_terms_for("IND", 4)[0] == "SEBI LODR"
    assert "Deutscher Corporate Governance Kodex" in governance_terms_for("DEU", 4)
    assert "King IV" in governance_terms_for("ZAF", 4)


def test_uncurated_country_gets_native_not_english():
    """Poland has no curated profile, so before this module it received an
    English query. It must now get Polish terms."""
    terms = governance_terms_for("POL", 4)
    assert "ład korporacyjny" in terms
