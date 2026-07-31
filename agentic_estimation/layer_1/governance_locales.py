"""
governance_locales.py — Google News locale routing and translated governance
vocabulary for every country, not just the curated ones.

WHY THIS EXISTS: country_governance_keywords.py carries hand-written profiles
for 30 countries -- the ones covering ~86% of the benchmark corpus. The World
Bank baseline table has 210 economies, so ~180 countries were falling back to a
plain English query with no locale routing at all. That is the exact gap
country_esg_keywords.py was originally built to close for the E/S pillars
("measured live in the n=60 bcorp backtest: non-English-market companies
returned zero or near-zero English-language signals").

This module supplies the two parts of a governance query that CAN be produced
mechanically for all 210:

  1. gn_locale  -- Google News hl=/gl=/ceid= routing, from a country's dominant
                   language. Sends the query to that country's own news edition.
  2. native governance terms -- "corporate governance", "board independence",
                   "audit committee", "anti-corruption" etc. translated per
                   LANGUAGE (~25 languages) rather than per country, because
                   governance vocabulary is a property of the language. Austria,
                   Germany and Switzerland share German terms; all of Spanish-
                   speaking Latin America shares one set.

What it deliberately does NOT supply is named regimes (a country's actual
governance code / statute / regulator). Those are country-specific facts, not
derivable from language, and inventing them is actively harmful: a fabricated
statute name returns nothing, or returns another country's material. Named
regimes come from country_governance_keywords.py's curated profiles and from
the web-verified set built by scripts/verify_governance_regimes.py.

LANGUAGE SELECTION: dominant language per country is stated explicitly below
rather than derived from locale.locale_alias, which lists regional and minority
languages indiscriminately (Frisian for DE, Breton/Basque/Catalan for FR) --
correct as language data, wrong for news routing, where the goal is the edition
most business/regulatory coverage is published in. For countries whose business
press is predominantly English despite another national language (India,
Nigeria, Kenya, Pakistan, Philippines, Singapore, Malaysia), English is used or
paired, matching what rupa's 58 hand-verified locales do.
"""

from __future__ import annotations

from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("governance_locales")


# ── Generic governance vocabulary, per language ────────────────────────────────
# One entry per LANGUAGE (not per country). Terms are the language's standard
# corporate-governance vocabulary: the concept names a regulator, exchange, or
# business paper would use. Ordered most-to-least discriminating.
#
# Scope note: 25 languages, chosen to cover the large majority of the 210
# economies by company volume. Countries whose language is absent fall back to
# the English set -- still a working query, just less locally targeted. That is
# a deliberate limit: a mistranslated governance term matches nothing, so this
# only covers languages where the terminology is unambiguous.
_LANG_TERMS: dict[str, tuple[str, ...]] = {
    "en": ("corporate governance", "board independence", "independent director",
           "audit committee", "shareholder rights", "anti-corruption",
           "regulatory fine", "disclosure requirements"),
    "es": ("gobierno corporativo", "director independiente", "comité de auditoría",
           "derechos de los accionistas", "corrupción", "soborno",
           "sanción regulatoria", "fraude contable"),
    "pt": ("governança corporativa", "conselheiro independente",
           "comitê de auditoria", "direitos dos acionistas", "corrupção",
           "suborno", "fraude contábil"),
    "fr": ("gouvernance d'entreprise", "administrateur indépendant",
           "comité d'audit", "droits des actionnaires", "corruption",
           "sanction réglementaire", "fraude comptable"),
    "de": ("Corporate Governance", "Unternehmensführung", "Aufsichtsrat",
           "unabhängiger Verwaltungsrat", "Prüfungsausschuss", "Korruption",
           "Bestechung", "Bilanzskandal"),
    "it": ("governo societario", "amministratore indipendente",
           "collegio sindacale", "diritti degli azionisti", "corruzione",
           "frode contabile"),
    "nl": ("corporate governance", "onafhankelijke commissaris",
           "auditcommissie", "aandeelhoudersrechten", "omkoping",
           "boekhoudfraude"),
    "sv": ("bolagsstyrning", "oberoende styrelseledamot", "revisionsutskott",
           "aktieägarnas rättigheter", "muta", "bokföringsbrott"),
    "da": ("selskabsledelse", "uafhængigt bestyrelsesmedlem", "revisionsudvalg",
           "aktionærrettigheder", "bestikkelse", "regnskabssvindel"),
    "no": ("eierstyring", "uavhengig styremedlem", "revisjonsutvalg",
           "aksjonærrettigheter", "korrupsjon", "regnskapsjuks"),
    "fi": ("hyvä hallintotapa", "hallinnointikoodi",
           "riippumaton hallituksen jäsen", "tarkastusvaliokunta", "lahjonta",
           "kirjanpitorikos"),
    "pl": ("ład korporacyjny", "niezależny członek rady nadzorczej",
           "komitet audytu", "prawa akcjonariuszy", "korupcja",
           "oszustwo księgowe"),
    "cs": ("správa a řízení společnosti", "nezávislý člen dozorčí rady",
           "výbor pro audit", "práva akcionářů", "korupce"),
    "hu": ("társaságirányítás", "független igazgatósági tag",
           "auditbizottság", "részvényesi jogok", "korrupció"),
    "ro": ("guvernanță corporativă", "administrator independent",
           "comitet de audit", "drepturile acționarilor", "corupție"),
    "el": ("εταιρική διακυβέρνηση", "ανεξάρτητο μέλος διοικητικού συμβουλίου",
           "επιτροπή ελέγχου", "διαφθορά"),
    "tr": ("kurumsal yönetim", "bağımsız yönetim kurulu üyesi",
           "denetim komitesi", "ortak hakları", "yolsuzluk", "rüşvet"),
    "ru": ("корпоративное управление", "независимый директор",
           "комитет по аудиту", "права акционеров", "коррупция", "взятка"),
    "uk": ("корпоративне управління", "незалежний директор",
           "комітет з аудиту", "корупція"),
    "ar": ("حوكمة الشركات", "عضو مجلس إدارة مستقل", "لجنة المراجعة",
           "حقوق المساهمين", "فساد", "رشوة"),
    "he": ("ממשל תאגידי", "דירקטור חיצוני", "ועדת ביקורת", "שוחד"),
    "zh": ("公司治理", "独立董事", "审计委员会", "股东权利", "腐败", "财务造假"),
    "ja": ("企業統治", "コーポレートガバナンス", "社外取締役", "監査委員会",
           "内部統制", "不正会計"),
    "ko": ("기업지배구조", "사외이사", "감사위원회", "주주권리", "부정회계"),
    "th": ("การกำกับดูแลกิจการ", "กรรมการอิสระ", "คณะกรรมการตรวจสอบ",
           "การทุจริต"),
    "vi": ("quản trị công ty", "thành viên hội đồng quản trị độc lập",
           "ban kiểm soát", "tham nhũng"),
    "id": ("tata kelola perusahaan", "komisaris independen", "komite audit",
           "korupsi", "suap"),
    "ms": ("tadbir urus korporat", "pengarah bebas", "jawatankuasa audit",
           "rasuah"),
    "hi": ("कॉर्पोरेट गवर्नेंस", "स्वतंत्र निदेशक", "लेखा परीक्षा समिति", "भ्रष्टाचार"),
}

# ── Dominant business-news language per country (ISO3 -> language codes) ───────
# Ordered: the first entry is the primary edition to query. A second entry is
# added where a country's business/regulatory press is materially bilingual
# (Canada, Belgium, Switzerland, India, Malaysia, Hong Kong, ...).
#
# Countries absent from this map fall back to English routing. Every ISO3 here
# is a World Bank economy or a territory the pipeline resolves (see
# country_normalizer); the list is grouped by region for reviewability.
_COUNTRY_LANGS: dict[str, tuple[str, ...]] = {
    # ── English-primary ───────────────────────────────────────────────────────
    "USA": ("en",), "GBR": ("en",), "AUS": ("en",), "NZL": ("en",),
    "IRL": ("en",), "CAN": ("en", "fr"), "ZAF": ("en",), "SGP": ("en",),
    "IND": ("en", "hi"), "PAK": ("en",), "BGD": ("en",), "LKA": ("en",),
    "NGA": ("en",), "KEN": ("en",), "GHA": ("en",), "UGA": ("en",),
    "TZA": ("en",), "ZMB": ("en",), "ZWE": ("en",), "BWA": ("en",),
    "NAM": ("en",), "MWI": ("en",), "RWA": ("en",), "PHL": ("en",),
    "MYS": ("en", "ms"), "JAM": ("en",), "TTO": ("en",), "BRB": ("en",),
    "BHS": ("en",), "GUY": ("en",), "BLZ": ("en",), "MLT": ("en",),
    "CYP": ("en", "el"), "FJI": ("en",), "PNG": ("en",), "MUS": ("en", "fr"),
    "SYC": ("en", "fr"), "GMB": ("en",), "SLE": ("en",), "LBR": ("en",),
    "SSD": ("en",), "SWZ": ("en",), "LSO": ("en",), "BMU": ("en",),
    "CYM": ("en",), "VGB": ("en",), "GGY": ("en",), "JEY": ("en",),
    "IMN": ("en",), "GIB": ("en",), "HKG": ("en", "zh"), "MAC": ("zh", "en"),
    "ISL": ("en",), "LUX": ("fr", "de"), "BRN": ("ms", "en"),
    "MDV": ("en",), "BTN": ("en",), "NPL": ("en",), "ATG": ("en",),
    "DMA": ("en",), "GRD": ("en",), "KNA": ("en",), "LCA": ("en",),
    "VCT": ("en",), "SLB": ("en",), "VUT": ("en", "fr"), "WSM": ("en",),
    "TON": ("en",), "KIR": ("en",), "NRU": ("en",), "PLW": ("en",),
    "MHL": ("en",), "FSM": ("en",), "TUV": ("en",), "ASM": ("en",),
    "GUM": ("en",), "MNP": ("en",), "PRI": ("es", "en"), "VIR": ("en",),
    "SXM": ("nl", "en"), "CUW": ("nl", "en"), "ABW": ("nl", "en"),
    "TCA": ("en",), "AIA": ("en",), "MSR": ("en",), "FLK": ("en",),
    "SHN": ("en",), "COK": ("en",), "NIU": ("en",), "TKL": ("en",),
    "NFK": ("en",), "CXR": ("en",), "CCK": ("en",), "HMD": ("en",),
    "IOT": ("en",), "PCN": ("en",), "SGS": ("en",), "UMI": ("en",),
    # ── Spanish-primary ──────────────────────────────────────────────────────
    "ESP": ("es",), "MEX": ("es",), "ARG": ("es",), "CHL": ("es",),
    "COL": ("es",), "PER": ("es",), "URY": ("es",), "PRY": ("es",),
    "BOL": ("es",), "ECU": ("es",), "VEN": ("es",), "CRI": ("es",),
    "PAN": ("es",), "GTM": ("es",), "HND": ("es",), "SLV": ("es",),
    "NIC": ("es",), "DOM": ("es",), "CUB": ("es",), "GNQ": ("es",),
    # ── Portuguese ───────────────────────────────────────────────────────────
    "BRA": ("pt",), "PRT": ("pt",), "AGO": ("pt",), "MOZ": ("pt",),
    "CPV": ("pt",), "GNB": ("pt",), "STP": ("pt",), "TLS": ("pt", "en"),
    # ── French ───────────────────────────────────────────────────────────────
    "FRA": ("fr",), "BEL": ("nl", "fr"), "CHE": ("de", "fr"),
    "MCO": ("fr",), "SEN": ("fr",), "CIV": ("fr",), "CMR": ("fr",),
    "MLI": ("fr",), "BFA": ("fr",), "NER": ("fr",), "TCD": ("fr",),
    "GIN": ("fr",), "BEN": ("fr",), "TGO": ("fr",), "GAB": ("fr",),
    "COG": ("fr",), "COD": ("fr",), "CAF": ("fr",), "MDG": ("fr",),
    "COM": ("fr", "ar"), "DJI": ("fr", "ar"), "BDI": ("fr",),
    "HTI": ("fr",), "NCL": ("fr",), "PYF": ("fr",), "REU": ("fr",),
    "MTQ": ("fr",), "GLP": ("fr",), "GUF": ("fr",), "MYT": ("fr",),
    "SPM": ("fr",), "WLF": ("fr",), "BLM": ("fr",), "MAF": ("fr",),
    "AND": ("es", "fr"),
    # ── German / Dutch / Nordic ──────────────────────────────────────────────
    "DEU": ("de",), "AUT": ("de",), "LIE": ("de",), "NLD": ("nl",),
    "SWE": ("sv",), "DNK": ("da",), "NOR": ("no",), "FIN": ("fi",),
    "FRO": ("da",), "GRL": ("da",), "ALA": ("sv", "fi"),
    # ── Rest of Europe ───────────────────────────────────────────────────────
    "ITA": ("it",), "SMR": ("it",), "VAT": ("it",),
    "POL": ("pl",), "CZE": ("cs",), "SVK": ("cs",), "HUN": ("hu",),
    "ROU": ("ro",), "MDA": ("ro",), "GRC": ("el",), "TUR": ("tr",),
    "RUS": ("ru",), "BLR": ("ru",), "KAZ": ("ru",), "KGZ": ("ru",),
    "UZB": ("ru",), "TJK": ("ru",), "TKM": ("ru",), "ARM": ("ru",),
    "AZE": ("tr", "ru"), "GEO": ("ru", "en"), "UKR": ("uk",),
    "BGR": ("ru", "en"), "SRB": ("en",), "HRV": ("en",), "SVN": ("en",),
    "BIH": ("en",), "MKD": ("en",), "MNE": ("en",), "ALB": ("en",),
    "XKX": ("en",), "LTU": ("en",), "LVA": ("en",), "EST": ("en",),
    # ── Middle East / North Africa ───────────────────────────────────────────
    "SAU": ("ar",), "ARE": ("ar", "en"), "QAT": ("ar", "en"),
    "KWT": ("ar", "en"), "BHR": ("ar", "en"), "OMN": ("ar", "en"),
    "EGY": ("ar",), "JOR": ("ar",), "LBN": ("ar", "fr"), "IRQ": ("ar",),
    "SYR": ("ar",), "YEM": ("ar",), "LBY": ("ar",), "TUN": ("ar", "fr"),
    "DZA": ("ar", "fr"), "MAR": ("ar", "fr"), "MRT": ("ar", "fr"),
    "SDN": ("ar",), "SOM": ("ar", "en"), "PSE": ("ar",), "ISR": ("he", "en"),
    "IRN": ("en",), "AFG": ("en",),
    # ── Asia-Pacific ─────────────────────────────────────────────────────────
    "CHN": ("zh", "en"), "TWN": ("zh",), "JPN": ("ja", "en"),
    "KOR": ("ko",), "PRK": ("ko",), "THA": ("th", "en"), "VNM": ("vi",),
    "IDN": ("id",), "KHM": ("en",), "LAO": ("en",), "MMR": ("en",),
    "MNG": ("en",),
    # ── Sub-Saharan Africa (non-English/French above) ────────────────────────
    "ETH": ("en",), "ERI": ("ar", "en"),
}

# Google News accepts a small set of hl= values; these are the ones verified in
# rupa's 58 hand-checked locale strings plus the standard Google News editions.
# Mapping language -> the hl= token Google News actually serves.
_HL: dict[str, str] = {
    "en": "en", "es": "es", "pt": "pt-BR", "fr": "fr", "de": "de",
    "it": "it", "nl": "nl", "sv": "sv", "da": "da", "no": "no",
    "fi": "fi", "pl": "pl", "cs": "cs", "hu": "hu", "ro": "ro",
    "el": "el", "tr": "tr", "ru": "ru", "uk": "uk", "ar": "ar",
    "he": "he", "zh": "zh-CN", "ja": "ja", "ko": "ko", "th": "th",
    "vi": "vi", "id": "id", "ms": "ms", "hi": "hi",
}

# ceid= language token, where it differs from the bare language code.
# Verified against rupa's live-checked strings: Portuguese uses plain "pt" in
# ceid even when hl= is "pt-BR"/"pt-PT", while Latin-American Spanish carries the
# "es-419" variant through to ceid ("hl=es-419&gl=AR&ceid=AR:es-419"). Chinese
# uses a script tag rather than a language code. These are Google's own edition
# identifiers, so they are recorded, not inferred.
_CEID_LANG: dict[str, str] = {"zh": "zh-Hans"}

# (language, iso2) pairs whose ceid token differs from the base language.
_CEID_REGIONAL: dict[tuple[str, str], str] = {
    ("es", cc): "es-419" for cc in (
        "AR", "MX", "CO", "CL", "PE", "VE", "UY", "PY", "BO", "EC", "CR",
        "PA", "GT", "HN", "SV", "NI", "DO", "CU", "PR",
    )
}

# Territories where Google News serves traditional Chinese rather than simplified.
_ZH_HANT = frozenset({"TWN", "HKG", "MAC"})

# Regional hl= variants Google News serves for specific (language, country)
# pairs. Cross-checked against rupa's 58 hand-verified locale strings: of those,
# 48 agreed with the mechanical hl= derivation and 10 differed -- every
# difference being a regional refinement of exactly this kind ("en-US" not "en",
# "es-419" for Latin America, "pt-PT" for Portugal). Those are Google's own
# edition identifiers, so the verified variant is used where one exists and the
# bare language code elsewhere.
# Only the pairs rupa actually verified are listed. Google News does NOT serve a
# distinct regional English edition for every English-speaking market: rupa's
# live-checked strings use plain "hl=en" for Singapore, Ireland, New Zealand,
# the Philippines, South Africa, Nigeria and Bangladesh. Extrapolating an
# "en-XX" for those (as a first pass here did) invents editions that do not
# exist, so the verified data wins over the pattern.
_REGIONAL_HL: dict[tuple[str, str], str] = {
    ("en", "US"): "en-US", ("en", "GB"): "en-GB", ("en", "AU"): "en-AU",
    ("en", "IN"): "en-IN", ("en", "CA"): "en-CA", ("en", "MY"): "en-MY",
    ("pt", "PT"): "pt-PT",
    # Google News serves one shared Latin-American Spanish edition (es-419);
    # Spain keeps plain "es".
    ("es", "AR"): "es-419", ("es", "MX"): "es-419", ("es", "CO"): "es-419",
    ("es", "CL"): "es-419", ("es", "PE"): "es-419", ("es", "VE"): "es-419",
    ("es", "UY"): "es-419", ("es", "PY"): "es-419", ("es", "BO"): "es-419",
    ("es", "EC"): "es-419", ("es", "CR"): "es-419", ("es", "PA"): "es-419",
    ("es", "GT"): "es-419", ("es", "HN"): "es-419", ("es", "SV"): "es-419",
    ("es", "NI"): "es-419", ("es", "DO"): "es-419", ("es", "CU"): "es-419",
    ("es", "PR"): "es-419",
}


def _iso2_for(iso3: str) -> Optional[str]:
    try:
        import pycountry
        c = pycountry.countries.get(alpha_3=iso3.upper())
        return c.alpha_2 if c else None
    except Exception:
        return None


def languages_for(iso3: Optional[str]) -> tuple[str, ...]:
    """Business-news language codes for a country, primary first.

    Falls back to ("en",) for countries not in the map -- English coverage of
    any market exists, so an English query is a working query.
    """
    if not iso3:
        return ("en",)
    return _COUNTRY_LANGS.get(iso3.strip().upper(), ("en",))


def derive_gn_locales(iso3: Optional[str]) -> tuple[str, ...]:
    """Google News hl=/gl=/ceid= query-string suffixes for a country.

    Returns one entry per business-news language, primary first. Empty tuple
    only when the ISO3 has no ISO2 equivalent (Google News keys gl= on ISO2),
    which the caller treats as "issue an unrouted query".
    """
    if not iso3:
        return ()
    iso3 = iso3.strip().upper()
    iso2 = _iso2_for(iso3)
    if not iso2:
        return ()

    out: list[str] = []
    for lang in languages_for(iso3):
        hl = _HL.get(lang)
        if not hl:
            continue
        if lang == "zh":
            hl = "zh-TW" if iso3 in _ZH_HANT else "zh-CN"
            ceid = "zh-Hant" if iso3 in _ZH_HANT else "zh-Hans"
        else:
            # ceid= is derived independently of the regional hl= variant:
            # Google serves "hl=en-MY&...&ceid=MY:en" (base language) but
            # "hl=es-419&...&ceid=AR:es-419" (variant carried through).
            ceid = _CEID_REGIONAL.get((lang, iso2)) or _CEID_LANG.get(lang, lang)
            hl = _REGIONAL_HL.get((lang, iso2), hl)
        locale = f"hl={hl}&gl={iso2}&ceid={iso2}:{ceid}"
        if locale not in out:
            out.append(locale)
    return tuple(out)


def native_governance_terms(iso3: Optional[str], limit: int = 6) -> tuple[str, ...]:
    """Governance vocabulary in the country's own business-news language(s).

    Terms come from the per-language table, so every country sharing a language
    shares the vocabulary -- the property being modelled is the language, not
    the country. English terms are appended as a top-up so a short native list
    still yields a broad enough query.
    """
    terms: list[str] = []
    for lang in languages_for(iso3):
        for t in _LANG_TERMS.get(lang, ()):
            if t not in terms:
                terms.append(t)
            if len(terms) >= limit:
                return tuple(terms[:limit])
    for t in _LANG_TERMS["en"]:
        if len(terms) >= limit:
            break
        if t not in terms:
            terms.append(t)
    return tuple(terms[:limit])


def supported_languages() -> frozenset[str]:
    return frozenset(_LANG_TERMS)


def mapped_countries() -> frozenset[str]:
    return frozenset(_COUNTRY_LANGS)
