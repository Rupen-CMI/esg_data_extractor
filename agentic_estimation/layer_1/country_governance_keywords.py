"""
country_governance_keywords.py — per-country corporate-governance query terms
and Google News locale routing, for the country-level governance evidence fetch.

WHY THIS EXISTS: the G pillar is the pipeline's weakest. Measured across the
393-company benchmark corpus, only 53.4% of companies have ANY company-level
governance evidence -- 35.6% carry nothing but a peer anchor and 10.9% have no
contributions at all. The cause is structural: gov_board_sec and
gov_litigation_sec read SEC DEF 14A / 10-K filings and are therefore US-listed
only, returning "" for every company outside that set. Consequently all 11
scoring variants came back UNDECIDABLE for G in the Stage 1.1 benchmark.

Country-level governance evidence is the cheapest available fill for that gap:
a country's corporate-governance regime (its governance code, board-independence
rules, audit regulator, anti-bribery statute, disclosure regime) is public,
findable, and applies to every company domiciled there.

HONEST LIMITATION, stated up front: a country-level signal is CONSTANT across
every company in that country, so it adds no *within-country* ranking signal.
Ground-truth comparisons here are largely within-country, so this should be
expected to improve coverage and absolute calibration WITHOUT necessarily
moving Spearman. Where it can affect ranking is the industry-qualified query
(a country's banking-governance news differs from its mining-governance news),
which is why fetch functions accept an industry hint. Judge this feature on
coverage first, not on rho.

PROVENANCE OF THE LOCALE STRINGS: the gn_locale values were mined from a
separate, already-shipped codebase in this workspace (rupa,
code/config_countries/esg_data_sources_all_countries.json -- 58 CountryProfile
entries) rather than re-derived. That project's ESG keyword sets were NOT
reused: audited across all 58 countries they contain only 13 distinct
governance-related terms (essentially "corporate governance", "disclosure" and
translations), with zero governance terms for Japan, Germany or South Korea
despite each having a well-known named regime. The vocabulary below is
therefore written specifically for governance, naming the actual statutes and
regulators an English-only generic query never surfaces.

COVERAGE: the countries below account for ~86% of the benchmark corpus.
Unlisted countries fall back to _GENERIC_GOVERNANCE_TERMS with no locale
routing (plain English query) -- additive, never a replacement path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("country_governance_keywords")


@dataclass(frozen=True)
class CountryGovernanceProfile:
    """Governance query vocabulary + news locale routing for one country.

    gn_locales:  Google News hl=/gl=/ceid= query-string suffixes. Multiple
                 entries mean multi-language sourcing (e.g. Japan's native ja
                 edition plus its English edition); the fetcher queries each
                 and dedupes across them.
    regime_terms: the country's OWN named governance instruments -- codes,
                 statutes, regulators. These are the high-value terms: a query
                 for "Deutscher Corporate Governance Kodex" surfaces material
                 that "corporate governance Germany" never reaches.
    native_terms: generic governance vocabulary in the local language.
    regulators:  named bodies whose enforcement actions are governance events.
    """
    iso3: str
    gn_locales: tuple[str, ...] = ()
    regime_terms: tuple[str, ...] = ()
    native_terms: tuple[str, ...] = ()
    regulators: tuple[str, ...] = ()

    def all_terms(self) -> tuple[str, ...]:
        """Regime terms first -- they are the most specific and the query
        builders truncate, so ordering decides what survives truncation."""
        seen: dict[str, None] = {}
        for t in self.regime_terms + self.regulators + self.native_terms:
            if t and t not in seen:
                seen[t] = None
        return tuple(seen)


# Applied to every country, including those with no profile below. Kept in
# English because these terms appear in international coverage of any market.
_GENERIC_GOVERNANCE_TERMS: tuple[str, ...] = (
    "corporate governance",
    "board independence",
    "independent director",
    "audit committee",
    "shareholder rights",
    "minority shareholder",
    "executive compensation",
    "anti-corruption",
    "anti-bribery",
    "whistleblower protection",
    "related party transaction",
    "disclosure requirements",
    "insider trading",
    "accounting fraud",
    "regulatory fine",
)

# Keyed by ISO3 (country_normalizer.to_iso3 output), so no new country-
# resolution vocabulary is introduced anywhere in the pipeline.
COUNTRY_GOVERNANCE_PROFILES: dict[str, CountryGovernanceProfile] = {
    "USA": CountryGovernanceProfile(
        iso3="USA",
        gn_locales=("hl=en-US&gl=US&ceid=US:en",),
        regime_terms=("Sarbanes-Oxley", "Dodd-Frank", "SEC disclosure rules",
                      "say-on-pay", "Delaware corporate law", "SOX compliance",
                      "Foreign Corrupt Practices Act", "FCPA"),
        regulators=("SEC enforcement", "DOJ investigation", "PCAOB", "FINRA"),
    ),
    "GBR": CountryGovernanceProfile(
        iso3="GBR",
        gn_locales=("hl=en-GB&gl=GB&ceid=GB:en",),
        regime_terms=("UK Corporate Governance Code", "Stewardship Code",
                      "Companies Act 2006", "Bribery Act 2010",
                      "senior managers regime", "audit reform"),
        regulators=("Financial Reporting Council", "FRC", "FCA enforcement",
                    "Serious Fraud Office", "Companies House"),
    ),
    "DEU": CountryGovernanceProfile(
        iso3="DEU",
        gn_locales=("hl=de&gl=DE&ceid=DE:de", "hl=en&gl=DE&ceid=DE:en"),
        regime_terms=("Deutscher Corporate Governance Kodex", "DCGK",
                      "Aktiengesetz", "Lieferkettensorgfaltspflichtengesetz",
                      "Mitbestimmung", "Aufsichtsrat"),
        native_terms=("Unternehmensführung", "Vorstandsvergütung",
                      "Korruption", "Compliance-Verstoß", "Bilanzskandal"),
        regulators=("BaFin", "Bundeskartellamt"),
    ),
    "FRA": CountryGovernanceProfile(
        iso3="FRA",
        gn_locales=("hl=fr&gl=FR&ceid=FR:fr", "hl=en&gl=FR&ceid=FR:en"),
        regime_terms=("code AFEP-MEDEF", "loi Sapin II", "loi PACTE",
                      "devoir de vigilance", "conseil d'administration"),
        native_terms=("gouvernance d'entreprise", "administrateur indépendant",
                      "rémunération des dirigeants", "corruption", "fraude comptable"),
        regulators=("AMF", "Autorité des marchés financiers", "Parquet national financier"),
    ),
    "JPN": CountryGovernanceProfile(
        iso3="JPN",
        gn_locales=("hl=ja&gl=JP&ceid=JP:ja", "hl=en&gl=JP&ceid=JP:en"),
        regime_terms=("コーポレートガバナンス・コード", "Corporate Governance Code",
                      "会社法", "社外取締役", "スチュワードシップ・コード",
                      "政策保有株式"),
        native_terms=("企業統治", "取締役会", "内部統制", "不正会計", "コンプライアンス違反"),
        regulators=("金融庁", "証券取引等監視委員会", "Financial Services Agency"),
    ),
    "CHN": CountryGovernanceProfile(
        iso3="CHN",
        gn_locales=("hl=zh-CN&gl=CN&ceid=CN:zh-Hans", "hl=en&gl=US&ceid=US:en"),
        regime_terms=("公司法", "上市公司治理准则", "Company Law of China",
                      "state-owned enterprise reform", "VIE structure"),
        native_terms=("公司治理", "独立董事", "内部控制", "财务造假", "反腐败"),
        regulators=("中国证监会", "CSRC", "SAMR", "State Administration for Market Regulation"),
    ),
    "KOR": CountryGovernanceProfile(
        iso3="KOR",
        gn_locales=("hl=ko&gl=KR&ceid=KR:ko",),
        regime_terms=("상법", "기업지배구조", "Commercial Act", "chaebol governance",
                      "cumulative voting", "스튜어드십 코드"),
        native_terms=("사외이사", "감사위원회", "내부통제", "회계부정", "총수일가"),
        regulators=("금융감독원", "공정거래위원회", "Fair Trade Commission",
                    "Financial Supervisory Service"),
    ),
    "IND": CountryGovernanceProfile(
        iso3="IND",
        gn_locales=("hl=en-IN&gl=IN&ceid=IN:en",),
        regime_terms=("SEBI LODR", "Companies Act 2013", "SEBI listing obligations",
                      "related party transaction", "independent director",
                      "promoter pledge", "BRSR"),
        regulators=("SEBI", "Serious Fraud Investigation Office", "SFIO",
                    "Ministry of Corporate Affairs", "NCLT"),
    ),
    "ITA": CountryGovernanceProfile(
        iso3="ITA",
        gn_locales=("hl=it&gl=IT&ceid=IT:it",),
        regime_terms=("Codice di Autodisciplina", "Codice di Corporate Governance",
                      "Testo Unico della Finanza", "decreto legislativo 231"),
        native_terms=("governo societario", "amministratore indipendente",
                      "collegio sindacale", "corruzione", "frode contabile"),
        regulators=("CONSOB", "Banca d'Italia"),
    ),
    "NLD": CountryGovernanceProfile(
        iso3="NLD",
        gn_locales=("hl=nl&gl=NL&ceid=NL:nl", "hl=en&gl=NL&ceid=NL:en"),
        regime_terms=("Nederlandse Corporate Governance Code", "structuurregime",
                      "raad van commissarissen", "Wet bestuur en toezicht"),
        native_terms=("corporate governance", "onafhankelijke commissaris",
                      "beloningsbeleid", "omkoping", "boekhoudfraude"),
        regulators=("AFM", "Autoriteit Financiële Markten", "ACM"),
    ),
    "AUS": CountryGovernanceProfile(
        iso3="AUS",
        gn_locales=("hl=en-AU&gl=AU&ceid=AU:en",),
        regime_terms=("ASX Corporate Governance Principles", "Corporations Act 2001",
                      "two-strikes rule", "continuous disclosure"),
        regulators=("ASIC", "ACCC", "APRA"),
    ),
    "CAN": CountryGovernanceProfile(
        iso3="CAN",
        gn_locales=("hl=en-CA&gl=CA&ceid=CA:en", "hl=fr&gl=CA&ceid=CA:fr"),
        regime_terms=("National Instrument 58-101", "Canada Business Corporations Act",
                      "majority voting policy", "TSX governance disclosure"),
        regulators=("OSC", "Ontario Securities Commission", "Competition Bureau"),
    ),
    "BRA": CountryGovernanceProfile(
        iso3="BRA",
        gn_locales=("hl=pt-BR&gl=BR&ceid=BR:pt",),
        regime_terms=("Novo Mercado", "Lei das S.A.", "Lei Anticorrupção",
                      "governança corporativa", "IBGC"),
        native_terms=("conselho de administração", "conselheiro independente",
                      "corrupção", "fraude contábil", "acordo de leniência"),
        regulators=("CVM", "CADE", "Controladoria-Geral da União"),
    ),
    "CHL": CountryGovernanceProfile(
        iso3="CHL",
        gn_locales=("hl=es-419&gl=CL&ceid=CL:es-419",),
        regime_terms=("Norma de Carácter General 461", "Ley de Sociedades Anónimas",
                      "ley 20.393"),
        native_terms=("gobierno corporativo", "director independiente",
                      "comité de directores", "corrupción", "fraude contable"),
        regulators=("CMF", "Comisión para el Mercado Financiero", "FNE"),
    ),
    "COL": CountryGovernanceProfile(
        iso3="COL",
        gn_locales=("hl=es-419&gl=CO&ceid=CO:es-419",),
        regime_terms=("Código País", "Circular Externa 028", "ley 1778"),
        native_terms=("gobierno corporativo", "junta directiva",
                      "miembro independiente", "corrupción", "soborno"),
        regulators=("Superintendencia Financiera", "Superintendencia de Sociedades"),
    ),
    "FIN": CountryGovernanceProfile(
        iso3="FIN",
        gn_locales=("hl=fi&gl=FI&ceid=FI:fi",),
        regime_terms=("Hallinnointikoodi", "osakeyhtiölaki",
                      "Finnish Corporate Governance Code"),
        native_terms=("hyvä hallintotapa", "riippumaton hallituksen jäsen",
                      "tarkastusvaliokunta", "lahjonta", "kirjanpitorikos"),
        regulators=("Finanssivalvonta", "FIN-FSA"),
    ),
    "SWE": CountryGovernanceProfile(
        iso3="SWE",
        gn_locales=("hl=sv&gl=SE&ceid=SE:sv",),
        regime_terms=("Svensk kod för bolagsstyrning", "aktiebolagslagen",
                      "valberedning"),
        native_terms=("bolagsstyrning", "oberoende styrelseledamot",
                      "revisionsutskott", "muta", "bokföringsbrott"),
        regulators=("Finansinspektionen", "Ekobrottsmyndigheten"),
    ),
    "ESP": CountryGovernanceProfile(
        iso3="ESP",
        gn_locales=("hl=es&gl=ES&ceid=ES:es",),
        regime_terms=("Código de buen gobierno", "Ley de Sociedades de Capital",
                      "informe anual de gobierno corporativo"),
        native_terms=("gobierno corporativo", "consejero independiente",
                      "comisión de auditoría", "corrupción", "fraude contable"),
        regulators=("CNMV", "Banco de España"),
    ),
    "CHE": CountryGovernanceProfile(
        iso3="CHE",
        gn_locales=("hl=de&gl=CH&ceid=CH:de", "hl=fr&gl=CH&ceid=CH:fr"),
        regime_terms=("Swiss Code of Best Practice", "Obligationenrecht",
                      "Verordnung gegen übermässige Vergütungen", "VegüV"),
        native_terms=("Corporate Governance", "unabhängiger Verwaltungsrat",
                      "Vergütungsbericht", "Bestechung"),
        regulators=("FINMA", "WEKO"),
    ),
    "BEL": CountryGovernanceProfile(
        iso3="BEL",
        gn_locales=("hl=nl&gl=BE&ceid=BE:nl", "hl=fr&gl=BE&ceid=BE:fr"),
        regime_terms=("Belgische Corporate Governance Code",
                      "Code belge de gouvernance d'entreprise",
                      "Wetboek van vennootschappen"),
        native_terms=("corporate governance", "onafhankelijk bestuurder",
                      "administrateur indépendant", "omkoping", "corruption"),
        regulators=("FSMA",),
    ),
    "DNK": CountryGovernanceProfile(
        iso3="DNK",
        gn_locales=("hl=da&gl=DK&ceid=DK:da",),
        regime_terms=("Anbefalinger for god selskabsledelse", "selskabsloven"),
        native_terms=("selskabsledelse", "uafhængigt bestyrelsesmedlem",
                      "revisionsudvalg", "bestikkelse", "regnskabssvindel"),
        regulators=("Finanstilsynet",),
    ),
    "MEX": CountryGovernanceProfile(
        iso3="MEX",
        gn_locales=("hl=es-419&gl=MX&ceid=MX:es-419", "hl=en&gl=MX&ceid=MX:en"),
        regime_terms=("Código de Principios y Mejores Prácticas de Gobierno Corporativo",
                      "Ley del Mercado de Valores"),
        native_terms=("gobierno corporativo", "consejero independiente",
                      "comité de auditoría", "corrupción", "soborno"),
        regulators=("CNBV", "COFECE"),
    ),
    "TWN": CountryGovernanceProfile(
        iso3="TWN",
        gn_locales=("hl=zh-TW&gl=TW&ceid=TW:zh-Hant",),
        regime_terms=("公司治理守則", "公司法", "證券交易法",
                      "Corporate Governance Best Practice Principles",
                      "獨立董事"),
        native_terms=("公司治理", "審計委員會", "內部控制", "財報不實", "反貪腐"),
        regulators=("金融監督管理委員會", "Financial Supervisory Commission",
                    "公平交易委員會"),
    ),
    "HKG": CountryGovernanceProfile(
        iso3="HKG",
        gn_locales=("hl=en&gl=HK&ceid=HK:en", "hl=zh-TW&gl=HK&ceid=HK:zh-Hant"),
        regime_terms=("HKEX Corporate Governance Code", "Companies Ordinance",
                      "Listing Rules", "連續關連交易"),
        native_terms=("公司治理", "獨立非執行董事", "審核委員會"),
        regulators=("SFC", "Securities and Futures Commission", "ICAC", "HKEX"),
    ),
    "MYS": CountryGovernanceProfile(
        iso3="MYS",
        gn_locales=("hl=en&gl=MY&ceid=MY:en", "hl=ms&gl=MY&ceid=MY:ms"),
        regime_terms=("Malaysian Code on Corporate Governance", "MCCG",
                      "Companies Act 2016", "Bursa Listing Requirements"),
        native_terms=("tadbir urus korporat", "pengarah bebas", "rasuah"),
        regulators=("Securities Commission Malaysia", "MACC",
                    "Malaysian Anti-Corruption Commission", "Bursa Malaysia"),
    ),
    "ZAF": CountryGovernanceProfile(
        iso3="ZAF",
        gn_locales=("hl=en&gl=ZA&ceid=ZA:en",),
        regime_terms=("King IV", "King IV Report", "Companies Act 71 of 2008",
                      "JSE Listings Requirements", "B-BBEE"),
        regulators=("JSE", "Financial Sector Conduct Authority",
                    "Companies and Intellectual Property Commission"),
    ),
    "SGP": CountryGovernanceProfile(
        iso3="SGP",
        gn_locales=("hl=en&gl=SG&ceid=SG:en",),
        regime_terms=("Singapore Code of Corporate Governance", "Companies Act",
                      "SGX Listing Rules"),
        regulators=("MAS", "Monetary Authority of Singapore", "ACRA", "CPIB"),
    ),
    "THA": CountryGovernanceProfile(
        iso3="THA",
        gn_locales=("hl=th&gl=TH&ceid=TH:th", "hl=en&gl=TH&ceid=TH:en"),
        regime_terms=("CG Code", "Public Limited Companies Act",
                      "Thai Institute of Directors"),
        native_terms=("การกำกับดูแลกิจการ", "กรรมการอิสระ", "การทุจริต"),
        regulators=("SEC Thailand", "NACC"),
    ),
    "NOR": CountryGovernanceProfile(
        iso3="NOR",
        gn_locales=("hl=no&gl=NO&ceid=NO:no",),
        regime_terms=("Norsk anbefaling for eierstyring", "allmennaksjeloven",
                      "åpenhetsloven"),
        native_terms=("eierstyring", "uavhengig styremedlem", "korrupsjon"),
        regulators=("Finanstilsynet", "Økokrim"),
    ),
    "IRL": CountryGovernanceProfile(
        iso3="IRL",
        gn_locales=("hl=en&gl=IE&ceid=IE:en",),
        regime_terms=("Irish Corporate Governance Annex", "Companies Act 2014",
                      "UK Corporate Governance Code"),
        regulators=("Central Bank of Ireland", "CEA",
                    "Corporate Enforcement Authority"),
    ),
}


def get_governance_profile(iso3: Optional[str]) -> Optional[CountryGovernanceProfile]:
    """Profile for an ISO3 code, or None when the country has no curated entry.

    None is not an error: callers fall back to _GENERIC_GOVERNANCE_TERMS with
    no locale routing, which still produces a usable English query.
    """
    if not iso3:
        return None
    return COUNTRY_GOVERNANCE_PROFILES.get(iso3.strip().upper())


def governance_terms_for(iso3: Optional[str], limit: int = 6) -> tuple[str, ...]:
    """Highest-value governance query terms for a country.

    Resolution order, most specific first:
      1. hand-written regime terms + regulators from the curated profile above
      2. web-verified regime terms (governance_regimes_verified.py, produced by
         scripts/verify_governance_regimes.py) for countries with no curated
         profile
      3. native-language generic governance vocabulary for the country's own
         business-news language (governance_locales, ~29 languages)
      4. generic English vocabulary

    A query naming an actual statute or regulator massively outperforms a
    generic one, so stages 1-2 come first; stages 3-4 top the list up so a thin
    profile still yields a broad enough query. Every country reaches at least
    stage 3, so no country falls back to English-only unless its language is
    unmapped.
    """
    terms: list[str] = []

    profile = get_governance_profile(iso3)
    if profile:
        terms.extend(profile.all_terms()[:limit])

    if len(terms) < limit and iso3:
        for t in _verified_regimes_for(iso3):
            if len(terms) >= limit:
                break
            if t not in terms:
                terms.append(t)

    if len(terms) < limit:
        from agentic_estimation.layer_1.governance_locales import native_governance_terms
        for t in native_governance_terms(iso3, limit=limit):
            if len(terms) >= limit:
                break
            if t not in terms:
                terms.append(t)

    for t in _GENERIC_GOVERNANCE_TERMS:
        if len(terms) >= limit:
            break
        if t not in terms:
            terms.append(t)
    return tuple(terms[:limit])


def _verified_regimes_for(iso3: str) -> tuple[str, ...]:
    """Web-verified regime terms, or () when the module has not been generated.

    Imported lazily and tolerantly: governance_regimes_verified.py is produced
    by an optional discovery script, so its absence must not break governance
    queries for the curated countries.
    """
    try:
        from agentic_estimation.layer_1.governance_regimes_verified import VERIFIED_REGIMES
    except Exception:
        return ()
    return tuple(VERIFIED_REGIMES.get(iso3.strip().upper(), ()))


def locales_for(iso3: Optional[str]) -> tuple[str, ...]:
    """Google News locale suffixes for a country.

    Falls back to governance_locales.derive_gn_locales, which covers all 210
    World Bank economies by routing on the country's dominant business-news
    language. Previously this returned () for any country without a curated
    profile -- ~180 of them -- meaning an unrouted English query. An empty tuple
    now means only that the country has no ISO2 equivalent for Google's gl=
    parameter.
    """
    profile = get_governance_profile(iso3)
    if profile and profile.gn_locales:
        return profile.gn_locales
    from agentic_estimation.layer_1.governance_locales import derive_gn_locales
    return derive_gn_locales(iso3)


def generic_governance_terms(limit: int = 6) -> tuple[str, ...]:
    """The country-agnostic fallback vocabulary."""
    return _GENERIC_GOVERNANCE_TERMS[:limit]


def supported_iso3() -> frozenset[str]:
    """ISO3 codes with a curated governance profile."""
    return frozenset(COUNTRY_GOVERNANCE_PROFILES)
