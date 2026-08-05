"""
company_metadata.py — Structured company metadata lookup for the ESG pipeline.

Resolves real, structured metadata for a company name from free public sources,
tried in order until one hits:

  Layer 1: Wikidata      — best for large/notable companies
  Layer 2: GLEIF         — best for regulated/financial entities, many mid-size private firms
  Layer 3: OpenStreetMap — location signal for companies mapped as a POI
  Layer 4: Name-inference — regex/keyword heuristic on the company name itself

No API keys required. Returns metadata only — no ESG scores computed here.

Key fields returned (where available):
  employees, revenue, total_assets, industry, country, hq_city,
  legal_form, subsidiary_count, stock_exchanges, inception_year,
  parent_org, board_member_count, lei, sec_cik, website

Usage:
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    meta = get_company_metadata("Patagonia")

CLI:
    python -m agentic_estimation.company_metadata "Patagonia" "BASF"
"""

import json
import os
import re
import time
import logging
import requests
from functools import lru_cache
from typing import Optional
from uuid import UUID

import psycopg2
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("company_metadata")

_HEADERS = {"User-Agent": "ESG-Data-Extractor/1.0 (research-pipeline)"}
_TIMEOUT = 15

# ── Per-source rate limiting ─────────────────────────────────────────────────
# REWRITTEN after an audit found this module was the weakest throttle in the
# pipeline, and it runs on EVERY company (calibration_harness and
# governance_collector both call get_company_metadata).
#
# Three defects in the previous implementation, all of which only bite under
# the concurrency the pipeline actually runs at:
#
#   1. NOT THREAD-SAFE. The read-modify-write of _LAST_CALL had no lock, so N
#      concurrent workers all read the same stale timestamp, all computed
#      wait<=0, and all fired at once -- the throttle silently did nothing
#      exactly when it was needed. This is the same class of bug that let a
#      burst of Google News requests through and got the pipeline blocked.
#   2. NO JITTER. Perfectly-spaced requests are a bot signature; every other
#      limiter in this codebase randomises its gap.
#   3. GAPS TOO TIGHT for the politeness policies of these specific hosts --
#      Wikimedia and OpenStreetMap both publish explicit guidance, and OSM's
#      Nominatim is an absolute maximum of 1 req/s SHARED across all clients.
#
# Now delegates to signal_agent._RateLimiter: one process-wide, lock-held,
# jittered limiter per host, identical to the mechanism protecting DDG,
# Google News, Wikipedia and SEC. One implementation, one place to tune.
from agentic_estimation.layer_1.signal_agent import _RateLimiter, RateLimitTripped

_LIMITERS: dict[str, _RateLimiter] = {
    # Wikimedia asks for serial, clearly-identified traffic from bots.
    "wikidata": _RateLimiter(min_gap=1.5, jitter=1.0),
    # GLEIF publishes no explicit limit; stay conservative, it is a small
    # public-good API and we query it once per company.
    "gleif":    _RateLimiter(min_gap=1.5, jitter=1.0),
    # Nominatim's usage policy is a hard ceiling of 1 request/second across
    # every client sharing an IP. 2.0s + jitter keeps us clearly under it.
    "osm":      _RateLimiter(min_gap=2.0, jitter=1.0),
}


def _throttle(source: str) -> None:
    """Block until this source's next request is allowed. Process-wide and
    thread-safe: concurrent workers queue here rather than bursting."""
    lim = _LIMITERS.get(source)
    if lim is not None:
        lim.wait()


def _check_throttled(r, url: str) -> None:
    """Feed a 429/503 response into the shared rate-limit tripwire.

    Every request in this module tests `r.ok` rather than calling
    raise_for_status(), so a throttling response was indistinguishable from
    "this company has no Wikidata entry" -- it silently became a missing
    metadata field. Worse, because these calls never went through
    signal_agent._get, their 429s were never counted at all, so the abort
    threshold could not see them.

    Raises RateLimitTripped once the shared threshold is crossed.
    """
    if r is not None and r.status_code in (429, 503):
        from agentic_estimation.layer_1.signal_agent import _note_rate_limit
        _note_rate_limit(url)


# ── Name normalisation ────────────────────────────────────────────────────────
# Suffix list lives in shared/company_name_utils.py -- see that module's
# docstring for why (two independent matchers must agree on what a "legal
# suffix" is, or the same company pair can match in one and not the other).

from agentic_estimation.shared.company_name_utils import LEGAL_SUFFIXES

_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b\.?",
    re.I,
)


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = _SUFFIX_RE.sub(" ", s)
    return " ".join(t for t in s.split() if len(t) > 1)


def _name_overlap(query: str, candidate: str) -> float:
    """Bidirectional token-overlap score 0–1 (suffix-stripped)."""
    q = set(_norm(query).split())
    c = set(_norm(candidate).split())
    if not q or not c:
        return 0.0
    fwd = len(q & c) / len(q)
    rev = len(q & c) / len(c)
    if fwd < 0.8 or rev < 0.5:
        return 0.0
    return (fwd + rev) / 2


# ── Layer 1: Wikidata ─────────────────────────────────────────────────────────

_WIKIDATA_API   = "https://www.wikidata.org/w/api.php"
_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

_COMPANY_WORDS = {
    "company", "manufacturer", "corporation", "business", "enterprise",
    "producer", "supplier", "firm", "group", "brand", "maker",
    "chemical", "pharmaceutical", "industrial", "multinational",
}

# DEFECT_FIX_PLAN.md 2.3 (H4): revenue/assets are Wikidata QUANTITY values
# with their OWN currency unit (confirmed live: Bosch's P2139 revenue is
# stated in euro, not USD -- psv:P2139 -> wikibase:quantityUnit gives the
# real unit). The query previously fetched only the bare numeric value with
# no currency, and format_for_prompt() labeled it "(USD)" unconditionally --
# silently mislabeling a real ~92B EUR figure as ~92B USD (an outright wrong
# value once the ~10% EUR/USD gap is folded in, not just a rounding issue).
# Now fetches each quantity's unit label (revUnitLabel/assetsUnitLabel) so
# _wikidata_lookup can convert to USD via _CURRENCY_TO_USD below.
_SPARQL = """
SELECT ?instanceLabel ?countryLabel ?hqCityLabel ?inception ?parentLabel
       ?website ?employees ?empYear ?revenue ?revYear ?revUnitLabel
       ?assets ?assetsUnitLabel ?industries
       ?products ?subs ?exchanges ?legalFormLabel ?boardSize ?lei ?cik
WHERE {{
  OPTIONAL {{ wd:{qid} wdt:P31  ?instance. }}
  OPTIONAL {{ wd:{qid} wdt:P17  ?country. }}
  OPTIONAL {{ wd:{qid} wdt:P159 ?hqCity. }}
  OPTIONAL {{ wd:{qid} wdt:P571 ?inception. }}
  OPTIONAL {{ wd:{qid} wdt:P749 ?parent. }}
  OPTIONAL {{ wd:{qid} wdt:P856 ?website. }}
  OPTIONAL {{ wd:{qid} wdt:P1454 ?legalForm. }}
  OPTIONAL {{ wd:{qid} wdt:P1278 ?lei. }}
  OPTIONAL {{ wd:{qid} wdt:P5531 ?cik. }}
  OPTIONAL {{
    SELECT ?employees ?empYear WHERE {{
      wd:{qid} p:P1128 ?est. ?est ps:P1128 ?employees.
      OPTIONAL {{ ?est pq:P585 ?ed. BIND(YEAR(?ed) AS ?empYear) }}
    }} ORDER BY DESC(?empYear) LIMIT 1
  }}
  OPTIONAL {{
    SELECT ?revenue ?revYear ?revUnit WHERE {{
      wd:{qid} p:P2139 ?rst. ?rst ps:P2139 ?revenue.
      OPTIONAL {{ ?rst pq:P585 ?rd. BIND(YEAR(?rd) AS ?revYear) }}
      OPTIONAL {{ ?rst psv:P2139 ?revNode. ?revNode wikibase:quantityUnit ?revUnit. }}
    }} ORDER BY DESC(?revYear) LIMIT 1
  }}
  OPTIONAL {{
    SELECT ?assets ?assetsUnit WHERE {{
      wd:{qid} p:P2403 ?ast. ?ast ps:P2403 ?assets.
      OPTIONAL {{ ?ast psv:P2403 ?astNode. ?astNode wikibase:quantityUnit ?assetsUnit. }}
    }} LIMIT 1
  }}
  OPTIONAL {{
    SELECT (GROUP_CONCAT(DISTINCT ?indL; separator=", ") AS ?industries) WHERE {{
      wd:{qid} wdt:P452 ?ind. ?ind rdfs:label ?indL. FILTER(LANG(?indL)="en")
    }}
  }}
  OPTIONAL {{
    SELECT (GROUP_CONCAT(DISTINCT ?prodL; separator=", ") AS ?products) WHERE {{
      wd:{qid} wdt:P1056 ?prod. ?prod rdfs:label ?prodL. FILTER(LANG(?prodL)="en")
    }}
  }}
  OPTIONAL {{
    SELECT (COUNT(DISTINCT ?sub) AS ?subs) WHERE {{ wd:{qid} wdt:P355 ?sub. }}
  }}
  OPTIONAL {{
    SELECT (GROUP_CONCAT(DISTINCT ?exL; separator=", ") AS ?exchanges) WHERE {{
      wd:{qid} wdt:P414 ?ex. ?ex rdfs:label ?exL. FILTER(LANG(?exL)="en")
    }}
  }}
  OPTIONAL {{
    SELECT (COUNT(DISTINCT ?board) AS ?boardSize) WHERE {{ wd:{qid} wdt:P3320 ?board. }}
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 1
"""

# Static, fixed conversion rates to USD -- NOT live-updated (a live FX API
# would add a network dependency + failure mode to a metadata lookup that's
# otherwise pure SPARQL). Approximate but far better than treating every
# non-USD figure as if it were already USD (the prior bug: a real ~10-40%+
# error depending on currency, not a rounding difference). Covers the
# currencies Wikidata's P2139/P2403 statements actually use for the
# companies this pipeline has looked up in practice; extend as new
# currencies are observed live (see the "unknown currency" log below).
_CURRENCY_TO_USD: dict[str, float] = {
    "united states dollar": 1.0,
    "euro": 1.08,
    "pound sterling": 1.27,
    "japanese yen": 0.0067,
    "swiss franc": 1.13,
    "canadian dollar": 0.73,
    "australian dollar": 0.65,
    "chinese yuan": 0.14,
    "renminbi": 0.14,
    "south korean won": 0.00072,
    "indian rupee": 0.012,
    "swedish krona": 0.094,
    "norwegian krone": 0.091,
    "danish krone": 0.145,
    "hong kong dollar": 0.128,
}


def _to_usd(value: Optional[float], currency_label: Optional[str], company: str) -> Optional[float]:
    """Convert a Wikidata quantity value to USD using its own stated
    currency unit. Returns the value UNCHANGED if no currency label is
    present at all (older statements sometimes lack a quantityUnit) --
    treating "unlabeled" the same as "already USD" preserves prior
    behavior for the common case rather than discarding a real number.
    Returns None (degrades to "unavailable" rather than a wrong number)
    only when a currency label IS present but not in _CURRENCY_TO_USD --
    an unrecognized non-USD currency should never silently pass through
    unconverted."""
    if value is None:
        return None
    if not currency_label:
        return value
    rate = _CURRENCY_TO_USD.get(currency_label.strip().lower())
    if rate is None:
        log.warning("[%s] unrecognized Wikidata currency %r -- dropping value rather than mislabeling it",
                    company, currency_label)
        return None
    return value * rate


@lru_cache(maxsize=256)
def _wikidata_qid(company: str) -> Optional[str]:
    """DEFECT_FIX_PLAN.md 2.3 (H3): previously fell through to hits[0]["id"]
    UNCONDITIONALLY when no hit's description mentioned a company word --
    accepting a completely unrelated entity (a person, a place, a common
    noun -- e.g. searching "Apple" without a company-word hit could accept
    the "apple" (fruit) or "Apple" (given name) QID) with zero name
    similarity check. Now applies the same _name_overlap >= 0.6 guard
    _gleif_candidates already uses, against each hit's own label, before
    ever falling through to hits[0]."""
    try:
        _throttle("wikidata")
        r = requests.get(
            _WIKIDATA_API,
            params={
                "action": "wbsearchentities", "search": company,
                "language": "en", "type": "item", "format": "json", "limit": 5,
            },
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        _check_throttled(r, _WIKIDATA_API)
        hits = r.json().get("search", []) if r.ok else []
        if not hits:
            return None
        for h in hits:
            if any(w in (h.get("description") or "").lower() for w in _COMPANY_WORDS):
                if _name_overlap(company, h.get("label") or "") >= 0.6:
                    return h["id"]
        best_hit, best_score = None, 0.0
        for h in hits:
            score = _name_overlap(company, h.get("label") or "")
            if score > best_score:
                best_hit, best_score = h, score
        if best_hit is not None and best_score >= 0.6:
            return best_hit["id"]
        log.debug("Wikidata: no hit for '%s' cleared the 0.6 name-overlap guard -- falling through", company)
        return None
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.debug("Wikidata QID search failed for '%s': %s", company, exc)
        return None


@lru_cache(maxsize=256)
def _wikidata_lookup(company: str) -> dict:
    qid = _wikidata_qid(company)
    if not qid:
        return {}
    try:
        _throttle("wikidata")
        r = requests.get(
            _WIKIDATA_SPARQL,
            params={"query": _SPARQL.format(qid=qid), "format": "json"},
            headers=_HEADERS, timeout=_TIMEOUT + 10,
        )
        _check_throttled(r, _WIKIDATA_API)
        rows = r.json().get("results", {}).get("bindings", []) if r.ok else []
        if not rows:
            return {}

        b = rows[0]

        def val(k):
            v = b.get(k, {}).get("value")
            return v if v else None

        def num(k):
            v = val(k)
            try:
                return float(v) if v else None
            except ValueError:
                return None

        inception = val("inception")
        facts = {
            "source":             "wikidata",
            "qid":                qid,
            "instance_of":        val("instanceLabel"),
            "industry":           val("industries"),
            "products":           val("products"),
            "legal_form":         val("legalFormLabel"),
            "country":            val("countryLabel"),
            "hq_city":            val("hqCityLabel"),
            "inception_year":     inception[:4] if inception else None,
            "employees":          int(num("employees")) if num("employees") else None,
            "employees_year":     val("empYear"),
            "revenue":            _to_usd(num("revenue"), val("revUnitLabel"), company),
            "revenue_year":       val("revYear"),
            "total_assets":       _to_usd(num("assets"), val("assetsUnitLabel"), company),
            "subsidiary_count":   int(num("subs")) if num("subs") else None,
            "parent_org":         val("parentLabel"),
            "board_member_count": int(num("boardSize")) if num("boardSize") else None,
            "lei":                val("lei"),
            "sec_cik":            val("cik"),
            "stock_exchanges":    val("exchanges"),
            "website":            val("website"),
        }

        _substantive = (
            "employees", "revenue", "total_assets", "industry",
            "stock_exchanges", "subsidiary_count", "board_member_count",
        )
        if not any(facts[k] for k in _substantive):
            return {}
        return facts
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.debug("Wikidata SPARQL failed for '%s' (%s): %s", company, qid, exc)
        return {}


# ── Layer 2: GLEIF ────────────────────────────────────────────────────────────

_GLEIF_BASE    = "https://api.gleif.org/api/v1"
_GLEIF_HEADERS = {**_HEADERS, "Accept": "application/vnd.api+json"}


def _gleif_candidates(company: str) -> list[tuple[str, float]]:
    """Return [(lei, score)] from fuzzy completion, sorted best-first."""
    try:
        _throttle("gleif")
        r = requests.get(
            f"{_GLEIF_BASE}/fuzzycompletions",
            params={"field": "entity.legalName", "q": _norm(company) or company},
            headers=_GLEIF_HEADERS, timeout=_TIMEOUT,
        )
        _check_throttled(r, _GLEIF_BASE)
        if not r.ok:
            return []
        out = []
        for d in r.json().get("data", []):
            name = d.get("attributes", {}).get("value", "")
            lei  = ((d.get("relationships") or {}).get("lei-records") or {}) \
                       .get("data", {}).get("id")
            if not lei:
                continue
            score = _name_overlap(company, name)
            if score >= 0.6:
                out.append((lei, score))
        return sorted(out, key=lambda t: -t[1])
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.debug("GLEIF fuzzy search failed for '%s': %s", company, exc)
        return []


def _gleif_record(lei: str) -> Optional[dict]:
    try:
        _throttle("gleif")
        r = requests.get(
            f"{_GLEIF_BASE}/lei-records/{lei}",
            headers=_GLEIF_HEADERS, timeout=_TIMEOUT,
        )
        _check_throttled(r, _GLEIF_BASE)
        return r.json().get("data") if r.ok else None
    except RateLimitTripped:
        raise
    except Exception:
        return None


@lru_cache(maxsize=256)
def _gleif_lookup(company: str) -> dict:
    try:
        best, best_score = None, 0.0

        candidates = _gleif_candidates(company)
        if candidates:
            lei, score = candidates[0]
            rec = _gleif_record(lei)
            if rec:
                best, best_score = rec, score

        # Fallback: full-text search
        if not best:
            _throttle("gleif")
            r = requests.get(
                f"{_GLEIF_BASE}/lei-records",
                params={"filter[fulltext]": company, "page[size]": 10},
                headers=_GLEIF_HEADERS, timeout=_TIMEOUT,
            )
            _check_throttled(r, _GLEIF_BASE)
            if r.ok:
                for rec in r.json().get("data", []):
                    ent   = rec.get("attributes", {}).get("entity", {})
                    score = _name_overlap(company, (ent.get("legalName") or {}).get("name", ""))
                    for other in ent.get("otherNames") or []:
                        score = max(score, _name_overlap(company, other.get("name", "")))
                    if score > best_score:
                        best, best_score = rec, score

        if not best or best_score < 0.6:
            return {}

        attrs      = best.get("attributes", {})
        ent        = attrs.get("entity", {})
        reg        = attrs.get("registration", {})
        hq         = ent.get("headquartersAddress") or {}
        legal_addr = ent.get("legalAddress") or {}
        parent_known = bool(
            ((best.get("relationships") or {}).get("direct-parent") or {})
            .get("links", {}).get("lei-record")
        )

        return {
            "source":              "gleif",
            "lei":                 attrs.get("lei"),
            "legal_name":          (ent.get("legalName") or {}).get("name"),
            "jurisdiction":        ent.get("jurisdiction"),
            "country":             hq.get("country") or legal_addr.get("country"),
            "hq_city":             hq.get("city") or legal_addr.get("city"),
            "legal_form":          (ent.get("legalForm") or {}).get("id"),
            "entity_status":       ent.get("status"),
            "entity_category":     ent.get("category"),
            "registration_status": reg.get("status"),
            "parent_org_known":    parent_known,
        }
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.debug("GLEIF lookup failed for '%s': %s", company, exc)
        return {}


# ── Layer 3: OpenStreetMap ────────────────────────────────────────────────────
# Only used as a last-resort location signal — requires exactly one unambiguous
# business POI match. Low hit rate for non-local companies, but zero cost.

_OSM_URL             = "https://nominatim.openstreetmap.org/search"
_OSM_BUSINESS_CLASSES = {"office", "shop", "craft", "industrial", "commercial"}
_OSM_BUSINESS_TYPES   = {"company", "advertising_agency", "office"}


@lru_cache(maxsize=256)
def _osm_lookup(company: str) -> dict:
    try:
        _throttle("osm")
        r = requests.get(
            _OSM_URL,
            params={"q": company, "format": "jsonv2", "limit": 5, "addressdetails": 1},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        _check_throttled(r, _OSM_URL)
        results = r.json() if r.ok else []

        # Require exactly one unambiguous business hit to avoid false positives
        hits = [
            item for item in results
            if (item.get("class") in _OSM_BUSINESS_CLASSES
                or item.get("type") in _OSM_BUSINESS_TYPES)
            and set(_norm(company).split()).issubset(
                set(_norm(item.get("display_name", "").split(",")[0]).split())
            )
        ]
        if len(hits) != 1:
            return {}

        addr = hits[0].get("address", {})
        return {
            "source":       "openstreetmap",
            "country":      addr.get("country"),
            "country_code": (addr.get("country_code") or "").upper() or None,
            "hq_city":      addr.get("city") or addr.get("town") or addr.get("village"),
        }
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.debug("OSM lookup failed for '%s': %s", company, exc)
        return {}


# ── Layer 4: Name-inference ───────────────────────────────────────────────────

_CHINA_TOKENS = {
    "anhui", "beijing", "chongqing", "fujian", "gansu", "guangdong", "guangxi",
    "guizhou", "hainan", "hebei", "heilongjiang", "henan", "hubei", "hunan",
    "jiangsu", "jiangxi", "jilin", "liaoning", "ningxia", "qinghai", "shaanxi",
    "shandong", "shanghai", "shanxi", "sichuan", "tianjin", "xinjiang", "yunnan",
    "zhejiang", "guangzhou", "shenzhen", "nanjing", "hangzhou", "wuhan",
    "chengdu", "qingdao", "dalian", "xiamen", "suzhou", "ningbo", "wuxi",
    "dongguan", "foshan", "changsha", "zhengzhou", "jinan", "harbin", "shenyang",
    "kunming", "nanchang", "hefei", "taiyuan", "nanning", "guiyang", "lanzhou",
}

# (keyword, country) — multi-word entries matched as substrings, single words as tokens
_COUNTRY_HINTS: list[tuple[str, str]] = [
    ("china", "China"), ("chinese", "China"),
    ("deutschland", "Germany"), ("germany", "Germany"), ("gmbh", "Germany"),
    ("france", "France"), ("sas", "France"),
    ("italia", "Italy"), ("italy", "Italy"), ("spa", "Italy"),
    ("nederland", "Netherlands"), ("netherlands", "Netherlands"), ("bv", "Netherlands"),
    ("espana", "Spain"), ("spain", "Spain"),
    ("schweiz", "Switzerland"), ("switzerland", "Switzerland"), ("swiss", "Switzerland"),
    ("sverige", "Sweden"), ("sweden", "Sweden"),
    ("suomi", "Finland"), ("finland", "Finland"),
    ("norge", "Norway"), ("norway", "Norway"),
    ("danmark", "Denmark"), ("denmark", "Denmark"),
    ("belgium", "Belgium"), ("belgie", "Belgium"),
    ("poland", "Poland"), ("polska", "Poland"),
    ("austria", "Austria"),
    ("india", "India"), ("bharat", "India"), ("pvt", "India"),
    ("japan", "Japan"), ("kabushiki", "Japan"),
    ("korea", "South Korea"), ("korean", "South Korea"),
    ("taiwan", "Taiwan"),
    ("malaysia", "Malaysia"), ("sdn", "Malaysia"),
    ("singapore", "Singapore"),
    ("thailand", "Thailand"),
    ("vietnam", "Vietnam"),
    ("indonesia", "Indonesia"),
    ("brazil", "Brazil"), ("brasil", "Brazil"),
    ("mexico", "Mexico"),
    ("canada", "Canada"),
    ("australia", "Australia"), ("pty", "Australia"),
    ("turkey", "Turkey"), ("turkiye", "Turkey"),
    ("russia", "Russia"),
    ("saudi", "Saudi Arabia"),
    ("emirates", "United Arab Emirates"), ("uae", "United Arab Emirates"),
    ("israel", "Israel"),
    ("united kingdom", "United Kingdom"), ("britain", "United Kingdom"),
    ("usa", "United States"),
]

_WORD_RE = re.compile(r"[a-z]+")


@lru_cache(maxsize=512)
def _infer_country(company_name: str) -> Optional[str]:
    low   = company_name.lower()
    words = set(_WORD_RE.findall(low))
    if words & _CHINA_TOKENS:
        return "China"
    for kw, country in _COUNTRY_HINTS:
        if " " in kw:
            if kw in low:
                return country
        elif kw in words:
            return country
    return None


# ── Manufacturing vs. services classification ─────────────────────────────────
# Keyword lookup over free-text industry strings (same pattern as
# wikirate_fetcher.WIKIRATE_KEYWORD_MAP), NOT an LLM guess — this is Tier-1
# structured classification, so it must be deterministic and auditable.
# Order matters: more specific/telling keywords first within each list.

_MANUFACTURING_KEYWORDS = (
    "automotive", "steel", "textile", "apparel", "chemical", "pharmaceutical",
    "semiconductor", "electronics manufactur", "machinery", "industrial",
    "aerospace", "shipbuilding", "food industry", "food processing",
    "beverage", "packaging", "mining", "metal", "plastics", "paper",
    "cement", "glass manufactur", "furniture manufactur", "appliance",
    "consumer electronics", "battery industry", "solar industry",
    "manufactured goods", "manufacturing",
)

_SERVICES_KEYWORDS = (
    "software", "saas", "it service", "consulting", "financial service",
    "insurance", "banking", "retail", "e-commerce", "advertising",
    "media", "publishing", "telecommunications service", "healthcare service",
    "education", "hospitality", "real estate", "logistics service",
    "professional & technical service", "administrative & support service",
    "accommodation & food service", "wholesale", "rental & repair",
    "waste management", "other services",
)


def classify_manufacturing_vs_services(industry_text: Optional[str]) -> dict:
    """
    Classify a company as manufacturing-heavy or services-heavy from its
    free-text industry description (Wikidata `industry`, or sasb_sector /
    industry_category strings from bcorp_lookup-style sources).

    Deterministic keyword match, NOT an LLM call — this is Tier-1
    classification and must be auditable. Returns "unknown" (not a forced
    binary guess) when the text gives no signal, consistent with the
    "no fabricated confidence" principle used throughout Layer 1.

    Returns: {"classification": "manufacturing"|"services"|"mixed"|"unknown",
              "confidence": float, "matched_keywords": list[str]}
    """
    if not industry_text:
        return {"classification": "unknown", "confidence": 0.0, "matched_keywords": []}

    text = industry_text.lower()
    mfg_hits = [kw for kw in _MANUFACTURING_KEYWORDS if kw in text]
    svc_hits = [kw for kw in _SERVICES_KEYWORDS if kw in text]

    if mfg_hits and not svc_hits:
        return {"classification": "manufacturing", "confidence": 0.8, "matched_keywords": mfg_hits}
    if svc_hits and not mfg_hits:
        return {"classification": "services", "confidence": 0.8, "matched_keywords": svc_hits}
    if mfg_hits and svc_hits:
        # Both present (e.g. a company with retail + manufacturing arms) — real
        # ambiguity, not a data gap, so still a positive-confidence finding.
        return {"classification": "mixed", "confidence": 0.5, "matched_keywords": mfg_hits + svc_hits}
    return {"classification": "unknown", "confidence": 0.0, "matched_keywords": []}


# ── Brand-name alias resolution ────────────────────────────────────────────────
# Cheap, explicit fix for well-known brands tracked under a different legal
# entity name in Wikidata (e.g. "Zara" -> "Inditex"). Extend this table as new
# misses are found — deliberately a small curated list, not a heuristic guesser,
# since a wrong alias would silently misattribute a company's whole metadata.

_BRAND_ALIASES: dict[str, str] = {
    "zara": "Inditex",
    "bershka": "Inditex",
    "pull&bear": "Inditex",
    "massimo dutti": "Inditex",
}


def _resolve_brand_alias(company: str) -> Optional[str]:
    return _BRAND_ALIASES.get(company.strip().lower())


# ── Public API ────────────────────────────────────────────────────────────────

def get_company_metadata(company: str) -> dict:
    """
    Resolve structured metadata for a company name.
    Tries Wikidata → GLEIF → OpenStreetMap → name-inference in order,
    returning on the first meaningful hit. Before falling through to
    "no match", tries a known brand-name→legal-entity alias (e.g.
    "Zara"->"Inditex") so well-known consumer brands tracked under a
    different legal name in Wikidata still resolve.

    Always returns a dict with at minimum:
      query, matched (bool), source (str or None), country, hq_city,
      manufacturing_classification (dict, see classify_manufacturing_vs_services)

    Additional fields depend on the source — see each layer above.
    """
    log.debug("Metadata lookup: '%s'", company)

    def _finalize(result: dict) -> dict:
        result["manufacturing_classification"] = classify_manufacturing_vs_services(result.get("industry"))
        return result

    wd = _wikidata_lookup(company)
    if wd:
        log.debug("'%s' resolved via Wikidata", company)
        return _finalize({"query": company, "matched": True, **wd})

    gl = _gleif_lookup(company)
    if gl:
        log.debug("'%s' resolved via GLEIF", company)
        return _finalize({"query": company, "matched": True, **gl})

    osm = _osm_lookup(company)
    if osm:
        log.debug("'%s' resolved via OpenStreetMap", company)
        return _finalize({"query": company, "matched": True, **osm})

    alias = _resolve_brand_alias(company)
    if alias:
        log.debug("'%s' resolved via brand alias -> '%s'", company, alias)
        wd_alias = _wikidata_lookup(alias)
        if wd_alias:
            return _finalize({
                "query": company, "matched": True, "resolved_via_alias": alias, **wd_alias,
            })
        gl_alias = _gleif_lookup(alias)
        if gl_alias:
            return _finalize({
                "query": company, "matched": True, "resolved_via_alias": alias, **gl_alias,
            })

    country = _infer_country(company)
    if country:
        log.debug("'%s' country inferred from name: %s", company, country)
        return _finalize({
            "query": company, "matched": True, "source": "name_inference",
            "country": country, "hq_city": None,
        })

    log.debug("'%s' — no metadata found", company)
    return _finalize({"query": company, "matched": False, "source": None,
                       "country": None, "hq_city": None})


def format_for_prompt(meta: dict) -> str:
    """
    Render metadata as a compact text block for injection into the scoring prompt.
    Only includes fields that are actually present.
    """
    if not meta.get("matched"):
        return "No structured metadata found for this company."

    lines = []
    field_labels = [
        ("source",             "Metadata source"),
        ("country",            "Country"),
        ("hq_city",            "HQ city"),
        ("industry",           "Industry"),
        ("legal_form",         "Legal form"),
        ("inception_year",     "Founded"),
        ("employees",          "Employees"),
        ("revenue",            "Revenue (USD)"),
        ("total_assets",       "Total assets (USD)"),
        ("subsidiary_count",   "Subsidiaries"),
        ("parent_org",         "Parent org"),
        ("board_member_count", "Board size"),
        ("stock_exchanges",    "Listed on"),
        ("entity_status",      "Entity status"),
        ("entity_category",    "Entity category"),
    ]
    for key, label in field_labels:
        v = meta.get(key)
        if v is not None:
            lines.append(f"  {label}: {v}")

    return "\n".join(lines) if lines else "Metadata matched but no substantive fields found."


# ── DB cache ─────────────────────────────────────────────────────────────────

def _db_conn():
    """Open a psycopg2 connection using the same URL conversion as country_baseline_agent."""
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    db_url = os.getenv("ASYNC_DB_URL", os.getenv("DB_URL", ""))
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    sslmode = qs.pop("ssl", [None])[0]
    clean_qs = urlencode({k: v[0] for k, v in qs.items()})
    if sslmode:
        clean_qs = (clean_qs + "&" if clean_qs else "") + f"sslmode={sslmode}"
    return psycopg2.connect(urlunparse(parsed._replace(query=clean_qs)))


def _load_from_db(company_id: UUID) -> Optional[dict]:
    """Return cached metadata dict for company_id, or None if not in DB."""
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT data FROM company_metadata WHERE company_id = %s",
            (str(company_id),),
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return None
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.warning("company_metadata DB load failed: %s", exc)
        return None


def _save_to_db(company_id: UUID, meta: dict) -> None:
    """Upsert metadata into company_metadata table."""
    try:
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO company_metadata (company_id, source, data)
            VALUES (%s, %s, %s)
            ON CONFLICT (company_id) DO UPDATE
                SET source = EXCLUDED.source,
                    data   = EXCLUDED.data,
                    fetched_at = now()
            """,
            (str(company_id), meta.get("source"), json.dumps(meta)),
        )
        conn.commit()
        conn.close()
        log.debug("Metadata saved to DB for company_id=%s", company_id)
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.warning("company_metadata DB save failed: %s", exc)


def get_or_fetch_metadata(company_id: UUID, company_name: str) -> dict:
    """
    Return metadata for a company, using DB cache when available.
    If not in DB, fetches from Wikidata/GLEIF/OSM/name-inference and saves the result.
    Safe to call with any company_id — DB errors are non-fatal.
    """
    cached = _load_from_db(company_id)
    if cached:
        log.debug("Metadata cache hit for '%s' (source: %s)", company_name, cached.get("source"))
        return cached

    log.info("Metadata cache miss for '%s' — fetching from web", company_name)
    meta = get_company_metadata(company_name)
    if meta.get("matched"):
        _save_to_db(company_id, meta)
    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    names = sys.argv[1:] or ["Patagonia", "BASF", "Totally Unknown Widget Co"]
    for name in names:
        print("=" * 60)
        print(f"  {name}")
        print("=" * 60)
        meta = get_company_metadata(name)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"\nPrompt block:\n{format_for_prompt(meta)}\n")
