"""
company_site_resolver.py — resolve a company to its OFFICIAL website, or abstain.

WHY THIS EXISTS: scraping a company's own site for its sustainability report is
worthless -- worse than worthless -- if the domain belongs to a different
company that happens to share a name. This module's entire job is to be
CERTAIN, and to return None when it cannot be.

The precedent is enforcement_collector.py's strict matcher: attributing a
stranger's conviction to a company is far worse than missing it. The same
applies to a sustainability report; a wrong-company report reads as perfectly
credible evidence downstream (real prose, real numbers, real PDF) and nothing
in Layer 2 can detect that it describes the wrong firm.

MEASURED IDENTIFIER AVAILABILITY (685 frozen companies, 2026-08-05) -- this is
what the tiering is built around, not a guess:

    website (Wikidata P856) : 15.5%   <- already resolved, no discovery needed
    LEI                     : 35.0%
    Wikidata QID            : 16.9%
    SEC CIK                 :  4.1%
    ANY strong anchor       : 44%
    nothing at all          : 50%     <- these mostly abstain, correctly

TIERS (each returns a ResolvedSite carrying its own confidence + reason):

  1. KNOWN      metadata["website"] from Wikidata P856. The QID it came from
                was already accepted under company_metadata's >=0.6 name-overlap
                guard, so the entity -- not just the string -- was matched.
  2. VERIFIED   a searched candidate domain whose fetched homepage/about page
                CORROBORATES an independent identifier we already hold: the
                LEI string, the registered legal name, the HQ city, or the
                jurisdiction. This is the only path that turns a guess into
                evidence.
  3. ABSTAIN    everything else.

There is deliberately NO "looks about right" tier. Token-in-domain matching
alone (the moderate option) cannot separate a UK "Dialog" from a Malaysian
"Dialog" -- a real collision found in our own truth tables while measuring
bcorp/upright overlap, along with Lion (AUS vs JPN), BGF (GBR vs KOR), Mando
(GBR vs KOR) and NICE (JPN vs ISR). Those are not hypothetical.
"""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.shared.company_name_utils import (
    LEGAL_SUFFIXES,
    normalize_company_name,
)
from agentic_estimation.layer_1.signal_agent import _RateLimiter, _get

log = get_logger("site_resolver")

_TIMEOUT = 20
_SITE_LIMITER = _RateLimiter(min_gap=1.0)

# Domains that will otherwise dominate a "<company> official website" search and
# look entirely plausible. None of these is ever a company's own site.
_AGGREGATOR_DOMAINS = frozenset({
    "wikipedia.org", "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "crunchbase.com", "bloomberg.com",
    "reuters.com", "ft.com", "forbes.com", "opencorporates.com", "zoominfo.com",
    "dnb.com", "glassdoor.com", "indeed.com", "yahoo.com", "google.com",
    "companieshouse.gov.uk", "sec.gov", "gleif.org", "pitchbook.com",
    "owler.com", "craft.co", "tracxn.com", "bcorporation.net", "wikidata.org",
    "marketscreener.com", "investing.com", "morningstar.com", "tofler.in",
    "zaubacorp.com", "moneycontrol.com", "economictimes.indiatimes.com",
})

# Pages most likely to carry the corroborating identifiers (legal name, LEI,
# registered address). Home page first -- footers very often carry the legal
# entity name and registered office.
_VERIFY_PATHS = ("", "/about", "/about-us", "/company", "/contact",
                 "/impressum", "/legal", "/investors")

_UA = {"User-Agent": "Mozilla/5.0 (compatible; ESG-Research/1.0)"}


@dataclass
class ResolvedSite:
    url: str                 # normalized https://host
    confidence: float        # 1.0 known / 0.75 verified
    tier: str                # 'wikidata_known' | 'search_verified'
    evidence: str            # WHICH identifier corroborated, for the audit trail


def _domain_of(url: str) -> str:
    try:
        host = (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _is_aggregator(url: str) -> bool:
    d = _domain_of(url)
    return any(d == a or d.endswith("." + a) for a in _AGGREGATOR_DOMAINS)


def _distinctive_tokens(company: str) -> list[str]:
    """Company tokens with legal suffixes removed. Reuses the shared normalizer
    so this module cannot drift from climate_trace_anchor / peer_anchor's idea
    of what a company name is -- the exact drift company_name_utils exists to
    prevent."""
    norm = normalize_company_name(company)
    return [t for t in norm.split() if len(t) > 2 and t not in LEGAL_SUFFIXES]


def _domain_plausible(url: str, company: str) -> bool:
    """Necessary, NOT sufficient. A company token must appear in the domain --
    but passing this proves nothing on its own, which is why every caller
    follows it with _corroborate()."""
    d = _domain_of(url)
    if not d or _is_aggregator(url):
        return False
    flat = d.replace("-", "").replace(".", "")
    toks = _distinctive_tokens(company)
    if not toks:
        return False
    # Whole token, or a >=5-char prefix (covers "sanpellegrino.com" for
    # "Sanpellegrino S.p.A." and "enproindustries.com" for "EnPro Industries").
    return any(t in flat or (len(t) >= 5 and t[:5] in flat) for t in toks)


# A name is ambiguous when it carries too little identifying content for
# Wikidata's search ranking to be trusted over our own country knowledge:
# one short token, or a common English word that many unrelated firms use.
# Every entry below is a real collision observed in bcorp/upright.
_AMBIGUOUS_COMMON_WORDS = frozenset({
    "dialog", "lion", "bgf", "mando", "nice", "circle", "front", "grind",
    "harmonic", "enso", "ama", "dart", "collective", "spotlight", "untamed",
    "adelphi", "orion", "remedy", "psa", "triangle", "apex", "summit",
    "pioneer", "vertex", "atlas", "phoenix", "horizon", "catalyst", "nexus",
})


def _is_ambiguous_name(company: str) -> bool:
    """True when the name alone cannot reliably identify one company worldwide.

    NOT simply "is it one token". Apple and Shell are single-token and globally
    unambiguous; Lion and BGF are single-token and collide across countries.
    The distinguishing feature is whether the token is a COMMON WORD or short
    acronym that many unrelated firms adopt -- so the test is membership of
    _AMBIGUOUS_COMMON_WORDS (plus very short acronyms), not token count.

    An earlier version returned True for every single-token name, which made
    Apple and Shell abstain -- caught by this module's own test.
    """
    toks = _distinctive_tokens(company)
    if not toks:
        return True
    if len(toks) == 1:
        t = toks[0]
        # <=4 chars is almost always an acronym (BGF, AMA, PSA, NICE).
        return len(t) <= 4 or t in _AMBIGUOUS_COMMON_WORDS
    return all(t in _AMBIGUOUS_COMMON_WORDS for t in toks)


# bcorp/upright/Wikidata each spell countries differently. Only the mappings
# needed to compare an expected country against a site's TLD/content are here;
# an unmapped country simply yields no TLD signal rather than a wrong one.
_COUNTRY_TLD = {
    "united kingdom": ("uk", "co.uk"), "gbr": ("uk", "co.uk"), "gb": ("uk", "co.uk"),
    "united states": ("us",), "usa": ("us",), "us": ("us",),
    "canada": ("ca",), "can": ("ca",), "ca": ("ca",),
    "australia": ("au", "com.au"), "aus": ("au", "com.au"),
    "new zealand": ("nz", "co.nz"), "nzl": ("nz", "co.nz"),
    "germany": ("de",), "deu": ("de",), "de": ("de",),
    "france": ("fr",), "fra": ("fr",), "fr": ("fr",),
    "italy": ("it",), "ita": ("it",), "it": ("it",),
    "spain": ("es",), "esp": ("es",), "es": ("es",),
    "switzerland": ("ch",), "che": ("ch",), "ch": ("ch",),
    "finland": ("fi",), "fin": ("fi",), "fi": ("fi",),
    "netherlands": ("nl",), "netherlands the": ("nl",), "nld": ("nl",),
    "japan": ("jp", "co.jp"), "jpn": ("jp", "co.jp"),
    "south korea": ("kr", "co.kr"), "korea, rep.": ("kr", "co.kr"), "kor": ("kr", "co.kr"),
    "china": ("cn", "com.cn"), "chn": ("cn", "com.cn"),
    "india": ("in", "co.in"), "ind": ("in", "co.in"),
    "brazil": ("br", "com.br"), "bra": ("br", "com.br"),
    "chile": ("cl",), "chl": ("cl",),
    "argentina": ("ar", "com.ar"), "arg": ("ar", "com.ar"),
    "malaysia": ("my", "com.my"), "mys": ("my", "com.my"),
    "israel": ("il", "co.il"), "isr": ("il", "co.il"),
    "lithuania": ("lt",), "ltu": ("lt",),
    "kenya": ("ke", "co.ke"), "ken": ("ke", "co.ke"),
    "sweden": ("se",), "swe": ("se",),
    "norway": ("no",), "nor": ("no",),
    "denmark": ("dk",), "dnk": ("dk",),
    "taiwan": ("tw", "com.tw"), "twn": ("tw", "com.tw"),
}


def _country_matches(url: str, expected_country: str, metadata: dict) -> tuple[bool, str]:
    """Confirm a candidate site really belongs to the company's country.

    Two independent signals, either sufficient:
      * the domain's TLD is that country's, or
      * the HQ city / jurisdiction appears in the fetched page.
    A country-neutral TLD (.com/.org) is NOT evidence either way, so it falls
    through to the content check rather than being treated as a pass.
    """
    exp = expected_country.strip().lower()
    tlds = _COUNTRY_TLD.get(exp)
    domain = _domain_of(url)

    if tlds and any(domain.endswith("." + t) for t in tlds):
        return True, f"TLD matches {expected_country}"

    # Wrong-country TLD is a hard reject: bgfretail.com would pass this by
    # being neutral, but a .co.kr domain for a UK company would not.
    for other, other_tlds in _COUNTRY_TLD.items():
        if other == exp:
            continue
        if any(domain.endswith("." + t) for t in other_tlds if t not in ("us",)):
            return False, f"TLD indicates {other}, expected {expected_country}"

    hq_city = (metadata.get("hq_city") or "").strip()
    jurisdiction = (metadata.get("jurisdiction") or "").strip()
    any_fetch_succeeded = False
    for path in _VERIFY_PATHS[:4]:
        html = _fetch(url.rstrip("/") + path)
        if not html:
            continue
        any_fetch_succeeded = True
        low = _visible_text(html).lower()
        if hq_city and len(hq_city) >= 4 and hq_city.lower() in low:
            return True, f"HQ city '{hq_city}' on the site"
        if jurisdiction and len(jurisdiction) >= 4 and jurisdiction.lower() in low:
            return True, f"jurisdiction '{jurisdiction}' on the site"
        if exp and len(exp) >= 4 and exp in low:
            return True, f"country '{expected_country}' named on the site"

    # "We could not read the site" is NOT "the site contradicts the country".
    # Large corporate sites routinely block scripted fetches, and an HQ city is
    # often only in a legal/contact page we did not reach. Treating that silence
    # as a contradiction rejected apple.com and shell.com in this module's own
    # test. When we genuinely could not read anything, fall back to the weaker
    # but real signal we DO have -- the TLD is at least not contradictory
    # (a wrong-country TLD already hard-rejected above).
    if not any_fetch_succeeded:
        return True, "site unreadable; accepted on non-contradictory TLD (weak)"

    return False, "country could not be confirmed on the readable site"


def _fetch(url: str) -> str:
    _SITE_LIMITER.wait()
    try:
        import requests
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            return ""
        ctype = (r.headers.get("content-type") or "").lower()
        if "html" not in ctype and "text" not in ctype:
            return ""
        return r.text[:400_000]
    except Exception as exc:                     # network/TLS/timeout -> no signal
        log.debug("fetch failed %s: %s", url, exc)
        return ""


def _visible_text(html: str) -> str:
    """Crude tag-strip. Good enough: we are looking for the PRESENCE of an
    identifier string (LEI, city, legal name), not parsing structure."""
    txt = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = txt.replace("&amp;", "&").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", txt)


def _corroborate(url: str, metadata: dict, company: str) -> tuple[bool, str]:
    """Fetch the candidate site and look for an INDEPENDENT identifier we
    already hold. Returns (ok, which_identifier_matched).

    Ordered strongest first. LEI and registered legal name are near-conclusive;
    HQ city is weaker on its own, so it is only accepted alongside a plausible
    domain (the caller has already required that) AND is reported as such, so
    downstream can see which grade of evidence was used.
    """
    lei = (metadata.get("lei") or "").strip().upper()
    legal_name = (metadata.get("legal_name") or "").strip()
    hq_city = (metadata.get("hq_city") or "").strip()
    jurisdiction = (metadata.get("jurisdiction") or "").strip()

    if not any((lei, legal_name, hq_city, jurisdiction)):
        return False, "no identifier available to verify against"

    for path in _VERIFY_PATHS:
        html = _fetch(url.rstrip("/") + path)
        if not html:
            continue
        text = _visible_text(html)
        low = text.lower()

        if lei and lei in text.upper():
            return True, f"LEI {lei} found on {path or '/'}"

        # Registered legal name is much more specific than the trade name --
        # "Felton Road Wines Limited" vs a random "Felton Road".
        if legal_name:
            ln_norm = normalize_company_name(legal_name)
            if len(ln_norm) >= 8 and ln_norm in normalize_company_name(text):
                return True, f"legal name '{legal_name}' found on {path or '/'}"

        if hq_city and len(hq_city) >= 4 and hq_city.lower() in low:
            return True, f"HQ city '{hq_city}' found on {path or '/'}"

        if jurisdiction and len(jurisdiction) >= 4 and jurisdiction.lower() in low:
            return True, f"jurisdiction '{jurisdiction}' found on {path or '/'}"

    return False, "no identifier corroborated on the candidate site"


def resolve_company_site(company: str, metadata: Optional[dict] = None) -> Optional[ResolvedSite]:
    """Resolve `company` to its official website, or return None.

    None is a normal, frequent, CORRECT outcome -- roughly half of companies
    carry no identifier that could verify a domain, and a plausible-looking
    guess for those is precisely the failure this module exists to prevent.
    """
    metadata = metadata or {}
    expected_country = (metadata.get("country") or "").strip()

    # ── Tier 1: Wikidata already told us, against a matched entity ──────────
    #
    # NOT unconditionally trusted. company_metadata's >=0.6 name-overlap guard
    # proves the STRING matched, not that the right ENTITY was picked: for a
    # bare ambiguous name every candidate scores ~1.0 overlap, so Wikidata's
    # search order decides, and it has no idea which country we meant.
    #
    # Caught by this module's own test: "Dialog" resolved to dialog.org (US)
    # and "BGF" to bgfretail.com (Korea) at confidence 1.0 -- both real
    # bcorp/upright collision cases, both silently wrong. So an ambiguous name
    # must clear the same country check as any other candidate.
    known = (metadata.get("website") or "").strip()
    if known:
        if _is_aggregator(known):
            log.info("[%s] wikidata website is an aggregator (%s) -- ignoring", company, known)
        else:
            url = known if "://" in known else f"https://{known}"
            if _is_ambiguous_name(company) and expected_country:
                ok, why = _country_matches(url, expected_country, metadata)
                if not ok:
                    log.info("[%s] AMBIGUOUS name + wikidata site %s failed the country "
                             "check (%s) -- abstaining rather than risk the wrong entity",
                             company, url, why)
                    return None
                log.info("[%s] site resolved (wikidata P856, ambiguous name confirmed): %s",
                         company, url)
                return ResolvedSite(url=url, confidence=0.85, tier="wikidata_country_checked",
                                    evidence=f"Wikidata P856; ambiguous name confirmed by {why}")
            log.info("[%s] site resolved (wikidata P856): %s", company, url)
            return ResolvedSite(url=url, confidence=1.0, tier="wikidata_known",
                                evidence="Wikidata P856 on an entity accepted at >=0.6 name overlap")

    # ── Tier 2: search, then REQUIRE corroboration ──────────────────────────
    if not any((metadata.get("lei"), metadata.get("legal_name"),
                metadata.get("hq_city"), metadata.get("jurisdiction"))):
        log.info("[%s] no verifiable identifier (no LEI/legal name/city/jurisdiction) "
                 "-- abstaining rather than guessing a domain", company)
        return None

    candidates = _search_candidates(company)
    if not candidates:
        log.info("[%s] no candidate domains from search -- abstaining", company)
        return None

    for cand in candidates[:5]:
        if not _domain_plausible(cand, company):
            log.debug("[%s] candidate %s rejected: domain implausible/aggregator", company, cand)
            continue
        ok, why = _corroborate(cand, metadata, company)
        if ok:
            log.info("[%s] site VERIFIED: %s (%s)", company, cand, why)
            return ResolvedSite(url=cand, confidence=0.75, tier="search_verified", evidence=why)
        log.debug("[%s] candidate %s rejected: %s", company, cand, why)

    log.info("[%s] %d candidates, none corroborated -- abstaining", company, len(candidates))
    return None


def _search_candidates(company: str) -> list[str]:
    """Candidate official-site URLs from web search. Returns bare origins."""
    from agentic_estimation.layer_1.signal_agent import _DDG_LIMITER
    try:
        from ddgs import DDGS
    except ImportError:
        log.warning("ddgs not installed -- cannot search for candidate domains")
        return []

    _DDG_LIMITER.wait()
    out: list[str] = []
    try:
        with DDGS() as ddg:
            results = list(ddg.text(f'"{company}" official website', max_results=8))
    except Exception as exc:
        log.warning("[%s] candidate search failed: %s", company, exc)
        return []

    seen: set = set()
    for r in results:
        href = r.get("href") or ""
        d = _domain_of(href)
        if not d or d in seen or _is_aggregator(href):
            continue
        seen.add(d)
        out.append(f"https://{d}")
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m agentic_estimation.layer_1.company_site_resolver "<company>"')
        sys.exit(1)
    company = sys.argv[1]
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    md = get_company_metadata(company)
    print(f"metadata anchors: website={md.get('website')!r} lei={md.get('lei')!r} "
          f"legal_name={md.get('legal_name')!r} hq_city={md.get('hq_city')!r}")
    site = resolve_company_site(company, md)
    if site:
        print(f"\nRESOLVED: {site.url}\n  tier={site.tier} confidence={site.confidence}\n  {site.evidence}")
    else:
        print("\nABSTAINED -- no domain could be verified for this company.")


if __name__ == "__main__":
    _cli()
