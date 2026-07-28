"""
country_esg_keywords.py — per-country localized ESG query terms for
signal_agent.py's Google News RSS fetch.

WHY THIS EXISTS: today's signal gathering queries every company in English
only, with one fixed ESG keyword list. Measured live in the n=60 bcorp
backtest: several non-English-market companies (Spanish, Portuguese,
Malaysian) returned zero or near-zero signals, producing zero claims and
falling back to the undifferentiated country baseline. A production ESG
system (analyzed via the code-review-graph MCP tool against a separate,
already-shipped client codebase) solves this with ~60 CountryProfile
dataclasses carrying native-language ESG keywords and Google News locale
routing (hl=/gl=/ceid= params send the query to that country's actual news
edition, not just an English-language search).

This module is a scaled-down version of that idea, sized for a
per-company (not corpus-scale) pipeline: a curated set of major markets
with real localized ESG regulation vocabulary (the terms that matter --
CSRD, BRSR, ASG, GX -- an English-only query will never surface), keyed by
the SAME country-name vocabulary country_baseline_agent.py already uses
(World Bank "Economy" names, e.g. "Germany", "India", "Brazil"), so no new
country-resolution logic is needed anywhere else in the pipeline.

Countries not listed here fall back to the existing English-only keyword
set -- this is additive, not a replacement path.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CountryEsgQueryProfile:
    gn_locale: str            # Google News hl=/gl=/ceid= query-string suffix
    esg_keywords: tuple[str, ...] = ()   # native-language ESG/regulatory terms
    extra_gn_locale: Optional[str] = None  # second locale (e.g. native + English edition)


# Keyed by the exact country-name string World Bank's "Economy" column (and
# therefore country_baseline_agent._cache / companies.country) uses.
COUNTRY_ESG_PROFILES: dict[str, CountryEsgQueryProfile] = {
    "Germany": CountryEsgQueryProfile(
        gn_locale="hl=de&gl=DE&ceid=DE:de",
        extra_gn_locale="hl=en&gl=DE&ceid=DE:en",
        esg_keywords=(
            "Nachhaltigkeit", "CO2-Fußabdruck", "Klimaneutralität",
            "Dekarbonisierung", "Kreislaufwirtschaft", "CO2-Zertifikate",
            "Nachhaltigkeitsbericht", "CSRD", "Lieferkettensorgfaltspflicht",
            "Lieferkettengesetz", "EU-Taxonomie", "grüne Anleihe", "Biodiversität",
        ),
    ),
    "India": CountryEsgQueryProfile(
        gn_locale="hl=en-IN&gl=IN&ceid=IN:en",
        esg_keywords=(
            "BRSR", "CSR", "carbon credit", "green bond", "environment clearance",
            "pollution control board", "GRIHA", "plastic ban", "SEBI ESG",
        ),
    ),
    "United States": CountryEsgQueryProfile(
        gn_locale="hl=en-US&gl=US&ceid=US:en",
        esg_keywords=(
            "SEC climate disclosure", "Scope 3 emissions", "TCFD", "SBTi",
            "environmental justice", "DEI", "anti-ESG", "EPA violation",
        ),
    ),
    "United Kingdom": CountryEsgQueryProfile(
        gn_locale="hl=en-GB&gl=GB&ceid=GB:en",
        esg_keywords=(
            "TCFD", "Streamlined Energy and Carbon Reporting", "SECR",
            "Modern Slavery Act", "FCA ESG", "net zero strategy",
        ),
    ),
    "Brazil": CountryEsgQueryProfile(
        gn_locale="hl=pt-BR&gl=BR&ceid=BR:pt",
        esg_keywords=(
            "sustentabilidade", "neutralidade de carbono", "crédito de carbono",
            "mercado de carbono", "SBCE", "descarbonização", "ASG",
            "Resolução CVM 244", "desmatamento", "EUDR",
        ),
    ),
    "Japan": CountryEsgQueryProfile(
        gn_locale="hl=ja&gl=JP&ceid=JP:ja",
        extra_gn_locale="hl=en&gl=JP&ceid=JP:en",
        esg_keywords=(
            "サステナビリティ", "カーボンニュートラル", "脱炭素", "温室効果ガス",
            "ESG投資", "統合報告書", "人的資本", "女性活躍", "GX経済移行債",
        ),
    ),
    "France": CountryEsgQueryProfile(
        gn_locale="hl=fr&gl=FR&ceid=FR:fr",
        esg_keywords=(
            "durabilité", "neutralité carbone", "devoir de vigilance",
            "CSRD", "taxonomie verte", "bilan carbone", "économie circulaire",
        ),
    ),
    "Spain": CountryEsgQueryProfile(
        gn_locale="hl=es&gl=ES&ceid=ES:es",
        esg_keywords=(
            "sostenibilidad", "huella de carbono", "neutralidad climática",
            "economía circular", "informe de sostenibilidad", "CSRD",
        ),
    ),
    "China": CountryEsgQueryProfile(
        gn_locale="hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        extra_gn_locale="hl=en&gl=CN&ceid=CN:en",
        esg_keywords=(
            "碳中和", "双碳", "ESG披露", "可持续发展报告", "碳排放权交易",
        ),
    ),
    "Mexico": CountryEsgQueryProfile(
        gn_locale="hl=es-419&gl=MX&ceid=MX:es-419",
        esg_keywords=(
            "sostenibilidad", "huella de carbono", "responsabilidad social",
            "bono verde", "economía circular",
        ),
    ),
    "Italy": CountryEsgQueryProfile(
        gn_locale="hl=it&gl=IT&ceid=IT:it",
        esg_keywords=(
            "sostenibilità", "neutralità carbonica", "economia circolare",
            "rendicontazione di sostenibilità", "CSRD",
        ),
    ),
    "Netherlands": CountryEsgQueryProfile(
        gn_locale="hl=nl&gl=NL&ceid=NL:nl",
        esg_keywords=(
            "duurzaamheid", "klimaatneutraal", "circulaire economie", "CSRD",
        ),
    ),
    "South Korea": CountryEsgQueryProfile(
        gn_locale="hl=ko&gl=KR&ceid=KR:ko",
        extra_gn_locale="hl=en&gl=KR&ceid=KR:en",
        esg_keywords=("ESG경영", "탄소중립", "지속가능경영보고서", "RE100"),
    ),
    "Indonesia": CountryEsgQueryProfile(
        gn_locale="hl=id&gl=ID&ceid=ID:id",
        esg_keywords=("keberlanjutan", "netral karbon", "ESG", "deforestasi"),
    ),
    "Malaysia": CountryEsgQueryProfile(
        gn_locale="hl=en-MY&gl=MY&ceid=MY:en",
        esg_keywords=("Bursa ESG", "sustainability report", "carbon neutral", "MSCI ESG"),
    ),
    "Australia": CountryEsgQueryProfile(
        gn_locale="hl=en-AU&gl=AU&ceid=AU:en",
        esg_keywords=("modern slavery statement", "climate risk disclosure", "ASIC ESG", "net zero"),
    ),
    "Canada": CountryEsgQueryProfile(
        gn_locale="hl=en-CA&gl=CA&ceid=CA:en",
        esg_keywords=("greenwashing", "CSA climate disclosure", "net zero", "Indigenous consultation"),
    ),
    "Switzerland": CountryEsgQueryProfile(
        gn_locale="hl=de-CH&gl=CH&ceid=CH:de",
        extra_gn_locale="hl=en&gl=CH&ceid=CH:en",
        esg_keywords=("Nachhaltigkeit", "Klimaneutralität", "Sorgfaltspflicht", "CO2-Gesetz"),
    ),
    "Portugal": CountryEsgQueryProfile(
        gn_locale="hl=pt-PT&gl=PT&ceid=PT:pt",
        esg_keywords=("sustentabilidade", "neutralidade carbónica", "economia circular"),
    ),
}


def get_country_esg_profile(country: Optional[str]) -> Optional[CountryEsgQueryProfile]:
    """Case-insensitive lookup against country_baseline_agent's own vocabulary.
    Returns None for unmapped countries -- callers fall back to the existing
    English-only keyword set, never raise, never guess a locale."""
    if not country:
        return None
    for name, profile in COUNTRY_ESG_PROFILES.items():
        if name.lower() == country.strip().lower():
            return profile
    return None


def localized_esg_query_terms(country: Optional[str], max_terms: int = 6) -> tuple[str, ...]:
    """Top-N localized ESG keywords for a country, or () if unmapped."""
    profile = get_country_esg_profile(country)
    if not profile:
        return ()
    return profile.esg_keywords[:max_terms]


def gn_locales_for_country(country: Optional[str]) -> list[str]:
    """One or two Google News locale query-strings for a country. Falls back
    to the default US English edition when unmapped -- matches today's
    existing behavior exactly, so unmapped countries see zero change."""
    profile = get_country_esg_profile(country)
    if not profile:
        return ["hl=en-US&gl=US&ceid=US:en"]
    locales = [profile.gn_locale]
    if profile.extra_gn_locale:
        locales.append(profile.extra_gn_locale)
    return locales
