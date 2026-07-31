"""
country_normalizer.py — resolve an arbitrary incoming country string to a
canonical ISO 3166-1 alpha-3 code.

WHY THIS EXISTS: country strings reach this pipeline from several places that
each spell things differently -- the key_players LLM extractor
(api/v1/key_players/services.py, which writes companies.country), Wikidata /
GLEIF / OpenStreetMap via company_metadata, the bcorp and upright ground-truth
tables, and hand-written calibration corpora. Measured live across the 393-
company benchmark corpus: 69 distinct country strings representing only ~46
real countries -- "USA"/"United States" split the US into two buckets,
"United Kingdom"/"GBR" split the UK, and "Netherlands The"/"TWN"/"HKG"/"IN"/
"GGY" resolved to nothing at all.

country_baseline_agent.resolve_country_name() previously carried a 22-entry
hand-written _COUNTRY_ALIASES dict for this job. That approach cannot win:
it had "great britain" but not "britain", "south korea" but not "korea, south".
Country identity is a solved, standardized problem (ISO 3166) and does not
belong in a hand-maintained dict, so this module delegates it to pycountry
(offline, no API, no rate limit) and keeps only the genuinely domain-specific
parts -- which spellings our own upstream sources emit, and which strings must
be REJECTED.

DESIGN: identity resolution (this module) is deliberately separate from
baseline lookup (country_baseline_agent). This returns an ISO3 code and knows
nothing about World Bank Economy names; the caller maps ISO3 -> Economy name.
That split matters because ISO3 codes exist for territories the World Bank
publishes no economy for (Taiwan, Guernsey, Jersey) -- resolving identity
correctly and *then* failing to find a baseline is a much more debuggable
outcome than failing to resolve the name at all.

REJECTION IS A FEATURE. The single most important behaviour here is returning
None for a string that is not a country. companies.country is 93% polluted
with LLM-fabricated values ("global", 1415 rows) plus cities, street
addresses, and sentence fragments ("Sydney well equipped with the best of
machines"). pycountry's search_fuzzy() will confidently map "Delhi" -> IND and
"Antwerp" -> BEL, which would silently convert visible pollution into
invisible pollution -- strictly worse than leaving it broken, because a wrong
country produces a wrong baseline, wrong peer group, and wrong localized
keywords with no signal that anything failed. Fuzzy matching is therefore
gated behind an explicit reject list and a token-count cap.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

import pycountry

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("country_normalizer")


# ── Aliases pycountry's own lookup/fuzzy search does not cover ────────────────
# Deliberately small: only colloquial or upstream-specific spellings that
# pycountry misses. Anything pycountry already resolves (every ISO2/ISO3 code,
# official names, "Korea, Republic of", "Viet Nam") is NOT repeated here.
# Keys are lowercased and whitespace-collapsed.
_ALIASES: dict[str, str] = {
    # UK colloquialisms + constituent countries. pycountry has no entry for the
    # constituent countries, and "Britain" fuzzy-matches GBR *and* PNG (Papua
    # New Guinea, via "...New Guinea" token overlap), so it must be pinned.
    "britain": "GBR",
    "great britain": "GBR",
    "england": "GBR",
    "scotland": "GBR",
    "wales": "GBR",
    "northern ireland": "GBR",
    "uk": "GBR",
    # Netherlands. "Netherlands The" is emitted verbatim by the upright
    # ground-truth table (10 corpus records) and matches nothing.
    "holland": "NLD",
    "netherlands the": "NLD",
    "the netherlands": "NLD",
    # US colloquialisms beyond the ISO codes pycountry handles. Bare "America"
    # is deliberately NOT aliased -- it is ambiguous with the continent(s) and
    # is listed in _REJECT_TOKENS instead; "United States of America" and the
    # ISO codes cover the unambiguous forms.
    "usa": "USA",
    "us": "USA",
    # Other upstream spellings seen in bcorp/upright/Wikidata.
    "south korea": "KOR",
    "north korea": "PRK",
    "ivory coast": "CIV",
    "czech republic": "CZE",
    "turkey": "TUR",
    "russia": "RUS",
    "vietnam": "VNM",
    "laos": "LAO",
    "syria": "SYR",
    "iran": "IRN",
    "macau": "MAC",
    "hong kong sar": "HKG",
    "hong kong sar china": "HKG",
    "macao sar china": "MAC",
    "taiwan china": "TWN",
    "chinese taipei": "TWN",
    "swaziland": "SWZ",
    "burma": "MMR",
    "cape verde": "CPV",
    "east timor": "TLS",
    # World Bank "Economy" spellings. These flow back in from our own
    # country_esg_baseline table and from resolve_country_name output, so a
    # round-trip (Economy name -> ISO3) has to work. pycountry does not carry
    # the WB's "X, Arab Rep."/"X, RB"/"X PDR" forms; the comma-suffixed ones
    # survive _clean (which strips dots, not commas) and match nothing.
    "egypt arab rep": "EGY",
    "egypt, arab rep": "EGY",
    "venezuela, rb": "VEN",
    "venezuela rb": "VEN",
    "yemen, rep": "YEM",
    "gambia, the": "GMB",
    "bahamas, the": "BHS",
    "congo, dem rep": "COD",
    "congo, rep": "COG",
    "lao pdr": "LAO",
    "kyrgyz republic": "KGZ",
    "slovak republic": "SVK",
    "turkiye": "TUR",
    "cote d'ivoire": "CIV",
    "curacao": "CUW",
    "st lucia": "LCA",
    "st kitts and nevis": "KNA",
    "st vincent and the grenadines": "VCT",
    "brunei darussalam": "BRN",
    "micronesia, fed sts": "FSM",
    "iran, islamic rep": "IRN",
    "korea, rep": "KOR",
    "korea, dem people's rep": "PRK",
    "west bank and gaza": "PSE",
    # Bare "virgin islands" is genuinely ambiguous (VGB British vs VIR U.S.)
    # and correctly refuses to resolve, but these qualified forms are not.
    "virgin islands (us)": "VIR",
    "virgin islands us": "VIR",
    "virgin islands, us": "VIR",
    "british virgin islands": "VGB",
}


# ── Strings that must NEVER resolve, even though fuzzy search would match ─────
# Non-answers and supra-national regions only. A bare city name is NOT rejected:
# "Delhi" resolving to India is correct, and refusing it bought nothing once the
# key_players extractor stopped guessing countries at all (the prompt no longer
# requests the field and save_companies_to_db no longer writes it), so the only
# remaining callers pass real location strings from registries and ground-truth
# tables rather than model guesses.
#
# What still must be refused is anything that is not a place: the LLM's
# "global"/"worldwide" non-answers (1415 of 1523 companies.country rows before
# the extractor change), and continent/region names that span dozens of
# countries and would otherwise fuzzy-match an arbitrary member.
#
# Scraped marketing prose ("Sydney well equipped with the best of machines",
# "Riyadh with deep understanding of Saudi regulations and NCA") is handled
# structurally by _MAX_FUZZY_TOKENS and _LEADING_NUMBER_RE below rather than by
# blocklisting the city inside it -- a per-city list can never be complete, and
# the token cap catches the whole class.
_REJECT_TOKENS: frozenset[str] = frozenset({
    # LLM/API non-answers and placeholders.
    "global", "worldwide", "international", "multinational", "various",
    "n/a", "na", "none", "null", "nil", "unknown", "unspecified",
    "not specified", "not mentioned", "not available", "tbd",
    "multiple", "several", "other", "others",
    # Continents / supra-national regions -- real places, but not countries,
    # and fuzzy search maps several of them onto an arbitrary member state.
    "europe", "asia", "africa", "america", "americas", "oceania", "antarctica",
    "middle east", "north america", "south america", "central america",
    "latin america", "eastern europe", "western europe", "southeast asia",
    "south asia", "east asia", "central asia", "sub-saharan africa",
    "apac", "emea", "eu", "european union", "asean", "mena", "nordics",
    "scandinavia", "balkans", "caribbean", "gulf", "gcc",
})

# A country name is at most a few words ("Bosnia and Herzegovina" = 3,
# "Democratic Republic of the Congo" = 5). Anything longer is a sentence
# fragment from a failed extraction ("Sydney well equipped with the best of
# machines", "Riyadh with deep understanding of Saudi regulations and NCA")
# and must not reach fuzzy matching, which would latch onto one stray token.
_MAX_FUZZY_TOKENS = 5

# Street addresses ("4085 Sladeview Crescent") -- a leading number is never a
# country name.
_LEADING_NUMBER_RE = re.compile(r"^\s*\d")

_lock = threading.Lock()
_memo: dict[str, Optional[str]] = {}


def _clean(raw: str) -> str:
    """Lowercase, collapse whitespace, strip decorative punctuation.

    Leading "**" appears in companies.country ("**Houston", "**Mumbai") --
    markdown bleed-through from the LLM extractor's response.

    Interior dots are dropped so dotted abbreviations collapse onto their
    undotted form ("U.S." -> "us", "Hong Kong S.A.R." -> "hong kong sar"),
    which is what the alias table is keyed on. Stripping only *trailing*
    punctuation left "u.s" and "hong kong s.a.r" -- distinct from every alias
    key and resolving to nothing.
    """
    s = (raw or "").strip().strip("*").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower().strip(" .,;:-")
    return re.sub(r"\.", "", s).strip()


def to_iso3(raw: Optional[str], *, allow_fuzzy: bool = True) -> Optional[str]:
    """Resolve an arbitrary country string to an ISO 3166-1 alpha-3 code.

    Returns None when the string is not a country -- a city, a region, an
    LLM non-answer ("global"), an address, or a sentence fragment. Callers
    MUST treat None as "unknown country", never as a reason to guess.

    allow_fuzzy=False restricts resolution to exact code/name/alias matches.
    Use it when the input is machine-generated and a near-miss should fail
    loudly rather than resolve to something plausible.

    Resolution order (first hit wins):
      1. reject list        -- known non-countries, checked BEFORE anything
                               else so no later stage can rescue them
      2. alias table        -- colloquialisms pycountry misses
      3. pycountry.lookup   -- ISO2/ISO3/numeric codes, official + common names
      4. pycountry fuzzy    -- gated: rejected input and long strings never
                               reach here; an ambiguous multi-hit result is
                               accepted only if all hits agree
    """
    # Non-string input reaches here from JSON-parsed LLM output, where the
    # country field can come back as a number or list rather than text.
    # Rejecting it here keeps every caller from having to type-check first.
    if not raw or not isinstance(raw, str):
        return None

    key = _clean(raw)
    if not key:
        return None

    with _lock:
        if key in _memo:
            return _memo[key]

    result = _resolve(key, raw, allow_fuzzy=allow_fuzzy)

    with _lock:
        _memo[key] = result
    return result


def _resolve(key: str, raw: str, *, allow_fuzzy: bool) -> Optional[str]:
    # 1. Explicit rejects first -- must beat every other rule.
    if key in _REJECT_TOKENS:
        log.debug("country_normalizer: rejected %r (known non-country)", raw)
        return None

    # 2. Curated aliases.
    if key in _ALIASES:
        return _ALIASES[key]

    # 3. pycountry exact lookup: handles alpha-2, alpha-3, numeric, official
    #    name, common name, and the ISO "Korea, Republic of" style forms.
    try:
        return pycountry.countries.lookup(key).alpha_3
    except LookupError:
        pass

    # 3b. Strip a trailing parenthetical or comma qualifier and retry the exact
    #     lookup. World Bank Economy names carry sovereignty/disambiguation
    #     suffixes pycountry has no entry for -- "Puerto Rico (US)",
    #     "Virgin Islands (U.S.)", "Somalia, Fed. Rep." -- but the base name
    #     alone resolves cleanly. Exact-lookup only, so this cannot invent a
    #     match the way fuzzy would.
    base = re.sub(r"\s*\([^)]*\)\s*$", "", key).strip()
    base = re.sub(r",\s*(fed rep|fed sts|the)$", "", base).strip()
    if base and base != key:
        if base in _ALIASES:
            return _ALIASES[base]
        try:
            return pycountry.countries.lookup(base).alpha_3
        except LookupError:
            pass

    if not allow_fuzzy:
        return None

    # 4. Fuzzy, heavily gated.
    if _LEADING_NUMBER_RE.match(raw):
        log.debug("country_normalizer: rejected %r (leading number -- address)", raw)
        return None
    tokens = key.split()
    if len(tokens) > _MAX_FUZZY_TOKENS:
        log.debug("country_normalizer: rejected %r (%d tokens -- sentence fragment)",
                  raw[:60], len(tokens))
        return None
    # A rejected token anywhere in a short phrase poisons the whole string:
    # "Penang and Johor Bahru" and "Thailand and" both arrive this way. Any
    # city token present means the extractor captured a location, not a
    # country -- except when a real country name is also present, which the
    # exact lookup above would already have caught.
    if any(t in _REJECT_TOKENS for t in tokens):
        log.debug("country_normalizer: rejected %r (contains non-country token)", raw[:60])
        return None

    # 4b. ISO 3166-2 subdivision names (states, provinces, union territories)
    #     resolve to their parent country. This is checked explicitly because
    #     pycountry's country-level fuzzy search covers it only by accident --
    #     "Delhi" and "Antwerp" match (both are ISO subdivisions) while
    #     "Mumbai" and "Sydney" do not (cities, absent from ISO entirely).
    #     Relying on the accident produced exactly that inconsistency; doing
    #     the lookup deliberately at least makes the boundary a stated rule:
    #     administrative divisions resolve, plain city names do not.
    try:
        subs = pycountry.subdivisions.lookup(key)
    except LookupError:
        subs = None
    if subs is not None:
        alpha2 = getattr(subs, "country_code", None)
        if alpha2:
            try:
                return pycountry.countries.lookup(alpha2).alpha_3
            except LookupError:
                pass

    try:
        hits = pycountry.countries.search_fuzzy(key)
    except LookupError:
        return None
    if not hits:
        return None
    codes = {h.alpha_3 for h in hits}
    if len(codes) > 1:
        # Ambiguous -- e.g. "Britain" returns GBR and PNG. Refusing beats
        # picking the first arbitrarily; genuinely common cases are pinned in
        # _ALIASES instead.
        log.debug("country_normalizer: rejected %r (ambiguous: %s)", raw, sorted(codes))
        return None
    return hits[0].alpha_3


def iso3_to_common_name(iso3: str) -> Optional[str]:
    """ISO3 -> human-readable common name ("KOR" -> "South Korea").

    For display and for building web-search queries, NOT for World Bank
    baseline lookup -- WB uses its own Economy names ("Korea, Rep."), which
    country_baseline_agent maps separately.
    """
    if not iso3:
        return None
    try:
        c = pycountry.countries.lookup(iso3.strip().upper())
    except LookupError:
        return None
    return getattr(c, "common_name", None) or c.name


def is_country(raw: Optional[str]) -> bool:
    """True when `raw` resolves to a real country/territory."""
    return to_iso3(raw) is not None
