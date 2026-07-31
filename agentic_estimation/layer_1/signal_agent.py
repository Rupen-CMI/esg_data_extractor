"""
signal_agent.py — ESG signal gathering for a single company.

Fetches real-time and static signals from multiple free/low-cost sources
in parallel and returns a structured dict of source → text evidence.

Sources (Tier 1 — real-time news):
  - NewsAPI          : 80k+ sources, full article snippets (title or desc match)
  - Google News RSS  : Latest headlines per company, no key needed, past-year window
  - Reuters, Bloomberg, Financial Times : high-credibility business press,
                        via Google News RSS site: restriction (no DDG dependency)
  - ESG Today, GreenBiz : dedicated sustainability trade press, same site:
                        restriction mechanism -- better hit-rate specifically
                        on pledges, certifications, disclosure-report coverage
  - Localized ESG    : native-language query against the company's home
                        country's own Google News edition (see country_esg_keywords.py)

Sources (Tier 2 — specialist ESG & corporate):
  - Business & Human Rights Resource Centre : Human rights / labor incidents
  - UN Global Compact                       : Signatory status
  - SBTi                                    : Science-based targets (web)
  - B Corp                                  : Certification status (web)
  - CDP                                     : Climate disclosure score (web)
  - ISO certifications                      : 14001, 45001, SA8000 (web)

Sources (Tier 3 — databases & reports):
  - GRI database     : Published sustainability reports
  - Wikipedia        : Company overview, ownership, size
  - Web searches     : Sustainability reports, net-zero pledges, controversies

Usage:
    signals = fetch_company_signals("Tata Steel", "Steel Manufacturing")
    # → {"gdelt": "...", "news_api": "...", "wikipedia": "...", ...}
"""

import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.evidence_filters import (
    dedup_and_filter_lines,
    filter_search_results,
    has_ground_truth_leakage,
)
log = get_logger("signal_agent")

_HEADERS = {"User-Agent": "ESG-Signal-Agent/1.0 (research@example.com)"}
_TIMEOUT = 12

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # newsapi.org key (optional)

# ── Rate limiters ─────────────────────────────────────────────────────────────
# Each limiter ensures a minimum gap between consecutive calls to that source,
# even when multiple threads fire simultaneously.

class _RateLimiter:
    """Thread-safe minimum-gap enforcer with optional jitter."""

    def __init__(self, min_gap: float, jitter: float = 0.0):
        self._min_gap = min_gap
        self._jitter  = jitter
        self._lock     = threading.Lock()
        self._last     = 0.0

    def wait(self) -> None:
        with self._lock:
            gap     = self._min_gap + random.uniform(0, self._jitter)
            elapsed = time.time() - self._last
            if elapsed < gap:
                time.sleep(gap - elapsed)
            self._last = time.time()


# DDG is the most sensitive — serialize ALL DDG calls through one limiter.
# 2s base + up to 1.5s jitter keeps us well under DDG's ~1 req/s soft limit
# and makes the request pattern look human rather than robotic.
_DDG_LIMITER      = _RateLimiter(min_gap=2.0, jitter=1.5)

# Wikipedia REST API: polite crawling policy — 1 req/s is fine, add small jitter.
_WIKIPEDIA_LIMITER = _RateLimiter(min_gap=1.0, jitter=0.5)

# GDELT: no documented rate limit but slow server; 1 req per call, no concern.
# SEC EDGAR: 10 req/s limit per their fair-use policy.
# Google News RSS / ESG Today RSS / Reuters RSS: standard RSS fetch, once per run.
# NewsAPI: 100 req/day free tier — one call per company is fine.

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, timeout: int = _TIMEOUT) -> requests.Response | None:
    try:
        r = requests.get(url, headers=_HEADERS, params=params, timeout=timeout)
        r.raise_for_status()
        log.debug("GET %s → HTTP %s (%d bytes)", url, r.status_code, len(r.content))
        return r
    except requests.HTTPError as e:
        # A 404 means "resource doesn't exist" — an expected outcome when probing
        # slug variants (e.g. Wikipedia), not a real error. Keep it at debug.
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            log.debug("GET %s → 404 Not Found", url)
        else:
            log.warning("GET %s → HTTP error: %s", url, e)
    except requests.Timeout:
        log.warning("GET %s → timed out after %ss", url, timeout)
    except Exception as e:
        log.warning("GET %s → %s: %s", url, type(e).__name__, e)
    return None



# ── Tier 1: Real-time news ───────────────────────────────────────────────────



def _newsapi_signal(company: str) -> str:
    """
    NewsAPI.org: 80k+ sources, full article content, date-filtered.
    Free tier: 100 req/day. Set NEWS_API_KEY env var to enable.
    """
    if not NEWS_API_KEY:
        log.debug("[%s] news_api → skipped (NEWS_API_KEY not set)", company)
        return ""
    log.info("[%s] news_api → querying NewsAPI.org", company)
    r = _get(
        "https://newsapi.org/v2/everything",
        params={
            "q": (
                f'"{company}" AND ('
                'ESG OR sustainability OR climate OR emissions OR carbon OR renewable OR pollution OR waste OR recycling OR environmental '
                'OR "net zero" OR "carbon neutral" '
                'OR workers OR labor OR labour OR employees OR lawsuit OR "human rights" OR discrimination OR safety OR "supply chain" OR community OR controversy OR fine OR penalty '
                'OR governance OR board OR executive OR fraud OR compliance OR transparency OR scandal OR corruption OR "data breach"'
                ')'
            ),
            "sortBy": "publishedAt",
            "pageSize": "10",
            "language": "en",
            "apiKey": NEWS_API_KEY,
        },
    )
    if not r:
        log.warning("[%s] news_api → no response", company)
        return ""
    try:
        articles = r.json().get("articles", [])
        if not articles:
            log.info("[%s] news_api → 0 articles found", company)
            return ""
        company_lower = company.lower()
        snippets = []
        for a in articles:
            title = a.get("title", "") or ""
            desc  = a.get("description", "") or ""
            # Company must appear in title or description
            if company_lower not in title.lower() and company_lower not in desc.lower():
                log.debug("[%s] news_api → skipping (company not in title/desc): %s", company, title[:80])
                continue
            date   = (a.get("publishedAt") or "")[:10]
            source = (a.get("source") or {}).get("name", "")
            snippets.append(f"[{date}] {source}: {title} — {desc[:200]}")
        log.info("[%s] news_api → %d articles returned", company, len(snippets))
        return "NewsAPI:\n" + "\n".join(snippets)
    except Exception as e:
        log.warning("[%s] news_api → parse error: %s", company, e)
        return ""


# Same ESG topic breadth as the NewsAPI query (news_api, above) -- Google News
# RSS search doesn't support boolean AND/OR grouping, but OR-joining the same
# keyword set as a plain search string still broadens recall well beyond the
# original fixed "ESG sustainability" phrase, which returned only headlines
# containing that exact phrase.
_GOOGLE_RSS_KEYWORDS = (
    "ESG OR sustainability OR climate OR emissions OR carbon OR renewable OR pollution OR waste OR recycling OR environmental "
    'OR "net zero" OR "carbon neutral" '
    # Social terms were entirely negative/neutral ("lawsuit", "discrimination",
    # "controversy", "fine", "penalty" -- no positive term at all), so the query
    # could only ever surface bad Social news. A company with genuinely strong
    # labor practices and a company that simply avoided press coverage were
    # indistinguishable -- "no bad news found" and "no evidence found" produced
    # the same (lack of) signal, leaving Social a pure-penalty pillar with no
    # way to ever score a company UP. Positive terms added below so a workplace
    # award or DEI program has a chance of being found at all.
    'OR workers OR labor OR labour OR employees OR lawsuit OR "human rights" OR discrimination OR safety OR "supply chain" OR community OR controversy OR fine OR penalty '
    'OR "best workplace" OR "great place to work" OR "employer of choice" OR diversity OR inclusion OR "gender pay gap" OR "living wage" OR "parental leave" OR "employee wellbeing" OR "union agreement" OR "labor certification" OR "workplace safety award" '
    # Governance terms had the SAME one-sided defect just fixed for Social:
    # "fraud", "scandal", "corruption", "data breach" are all negative, and the
    # only positive-leaning word was "transparency" (which mostly means "a
    # company disclosed something," not "a company was commended"). Verified
    # live: Boeing and Volkswagen -- both companies with real governance news
    # cycles -- returned ZERO positive-governance headlines under this query,
    # while Wells Fargo surfaced one only by accident (a "Sustainable
    # Leadership Award" story matched via "sustainability", not any G term).
    # Positive terms added so a governance-rating upgrade, an independent-board
    # milestone, or an ethics certification has a chance of being found.
    'OR governance OR board OR executive OR fraud OR compliance OR transparency OR scandal OR corruption OR "data breach" '
    'OR "board diversity" OR "independent director" OR "governance rating" OR "ethics award" OR "shareholder rights" OR "ESG leadership" OR "whistleblower protection" OR "audit committee" OR "say on pay" OR "proxy access"'
)


def _google_news_rss_query(company: str, extra: str = "", site: str = "", locale: str = "hl=en-US&gl=US&ceid=US:en", when_days: int = 365) -> Optional[str]:
    """Shared Google News RSS fetch+parse -- used for the general feed, the
    outlet-restricted feeds (Reuters, Bloomberg, Financial Times, ESG Today,
    GreenBiz -- site restriction via Google's site: operator, same trick as
    the DDG site-search helpers use), and the localized-ESG feed (locale
    routes the query to that country's actual Google News edition).

    site: MUST come first in the query string -- found live that Google News
    RSS silently ignores/ranks-away a trailing "site:x.com" once it comes
    after a large parenthesized OR-group (verified: identical query with
    site: at the end returned zero reuters.com articles; moved to the front,
    the exact same terms returned 100/100 real reuters.com results).

    when_days: Google News RSS's own "when:Nd" freshness operator, appended
    directly to the query string (not a URL param) -- restricts results to
    the past N days. Default 365 (past year): ESG evidence (pledges,
    certifications, disclosed incidents) is meaningfully relevant over a
    longer horizon than typical news search, but an unbounded query pulls in
    decade-old, no-longer-representative coverage. Set to 0 to omit the
    restriction entirely (unbounded, prior behavior)."""
    q = f"site:{site} " if site else ""
    q += f'"{company}" ({_GOOGLE_RSS_KEYWORDS})'
    if extra:
        q += f" {extra}"
    if when_days > 0:
        q += f" when:{when_days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&{locale}"
    r = _get(url)
    if not r:
        return None
    try:
        root  = ET.fromstring(r.content)
        items = root.findall(".//item")
        hits  = []
        for item in items[:8]:
            title = (item.findtext("title") or "").strip()
            pub   = (item.findtext("pubDate") or "")[:16]
            link  = (item.findtext("link") or "").strip()
            # Found live testing a site:-restricted query against a
            # well-known company on a sparse outlet (ft.com): Google News RSS
            # can return the outlet's generic homepage listing as a filler
            # result when it doesn't have enough real matches -- title reads
            # "Home - {Outlet} - {Outlet}" with a stale/arbitrary pubDate
            # (observed: 2006, decades outside any when:Nd restriction).
            # Real articles never repeat the outlet name twice; cheap to
            # detect and drop before it enters the evidence pool.
            if title.lower().startswith("home - ") and title.count(" - ") >= 2:
                continue
            # Article URL appended so every headline entering the evidence pool
            # stays traceable to the page it came from -- claims cite a
            # source_tag, and a source_tag without a link can't be audited by
            # anyone reviewing a score. Kept inline in the signal text (rather
            # than a parallel structure) so it survives every downstream
            # consumer unchanged: frozen corpus dumps, DB persistence, and the
            # LLM claim extractor all pass the text through verbatim.
            hits.append(f"[{pub}] {title}" + (f" <{link}>" if link else ""))
        return "\n".join(hits) if hits else None
    except Exception:
        return None


def _localized_esg_signal(company: str, country: Optional[str]) -> str:
    """Native-language ESG query against the company's home country's own
    Google News edition. Found live in the n=60 bcorp backtest: non-English-
    market companies (Spanish/Portuguese/Malaysian) returned zero or
    near-zero English-language signals and fell back to an undifferentiated
    country baseline. A localized query surfaces local ESG regulation terms
    (e.g. Germany's CSRD/Lieferkettengesetz, India's BRSR, Brazil's ASG) an
    English-only search never matches. Unmapped countries return "" --
    additive, never a replacement for the general feed."""
    from agentic_estimation.layer_1.country_esg_keywords import get_country_esg_profile

    profile = get_country_esg_profile(country)
    if not profile:
        return ""

    kw_query = " OR ".join(f'"{kw}"' for kw in profile.esg_keywords[:6])
    locales = [profile.gn_locale] + ([profile.extra_gn_locale] if profile.extra_gn_locale else [])

    all_hits: list[str] = []
    for locale in locales:
        log.info("[%s] localized_esg → querying %s Google News edition", company, locale)
        hits = _google_news_rss_query(company, extra=f"({kw_query})" if kw_query else "", locale=locale)
        if hits:
            all_hits.extend(hits.splitlines())

    if not all_hits:
        log.info("[%s] localized_esg → no matching articles", company)
        return ""
    # Cross-locale fingerprint dedup (same story often appears in both the
    # native and English editions of a country's Google News). Query-targeted
    # (the localized keywords are already baked into the query string), so no
    # keyword pre-gate needed here -- same rule as the general/reuters feeds.
    deduped = dedup_and_filter_lines("\n".join(all_hits), require_keyword=False)
    hit_count = len(deduped.splitlines())
    log.info("[%s] localized_esg → %d headlines (%d locales)", company, hit_count, len(locales))
    return "Localized ESG News:\n" + "\n".join(deduped.splitlines()[:10])


# ── Country-level governance evidence ─────────────────────────────────────────
# Cached per (iso3, industry) rather than per company: the result is IDENTICAL
# for every company in a country, so fetching it per company would issue ~390
# requests to obtain ~30 distinct answers. That waste is not hypothetical -- the
# corpus gather was rate-limited off the free-tier gateway twice, so avoidable
# duplicate fetching is a real cost, not a micro-optimisation.
_country_gov_cache: dict[tuple[str, str], str] = {}
_country_gov_lock = threading.Lock()

# Governance-relevance gate. A locale-routed query on a native-language term
# still returns whatever that country's news edition ranks highest, which
# verified live as air-quality alerts and university awards. A headline only
# enters the evidence pool if it carries at least one governance word -- the
# alternative (accepting everything the feed returns) hands the claim extractor
# unrelated national news and invites it to invent governance claims from it.
_GOV_RELEVANCE_TERMS: tuple[str, ...] = (
    # English
    "governance", "board", "director", "audit", "shareholder", "disclosure",
    "corrupt", "bribe", "fraud", "complian", "regulat", "enforcement", "fine",
    "penalt", "sanction", "whistleblow", "insider", "transparen", "accounting",
    "oversight", "stewardship", "proxy", "executive pay", "remuneration",
    "minority", "related party", "conflict of interest", "money laundering",
    # Native-language governance roots for the locales we route to
    "governan", "gouvernance", "gobierno corporativo", "governo societario",
    "unternehmensführung", "aufsichtsrat", "vorstand", "korruption",
    "bestechung", "bolagsstyrning", "selskabsledelse", "eierstyring",
    "hallinnointi", "tadbir urus", "rasuah",
    "公司治理", "企業統治", "取締役", "監査", "不正", "汚職",
    "기업지배구조", "사외이사", "감사", "부정", "공정거래",
    "董事", "審計", "审计", "腐败", "貪腐", "監管",
    "การกำกับดูแล", "กรรมการ", "ทุจริต",
)


def _is_governance_relevant(line: str) -> bool:
    low = line.lower()
    return any(t in low for t in _GOV_RELEVANCE_TERMS)


def _mentions_country(segment: str, country_name: str, iso3: str) -> bool:
    """True when a snippet actually refers to the target country.

    Matches the common name, its adjectival forms, and the ISO3 code. Kept
    deliberately simple -- this is a bleed-through guard, not entity linking;
    a false negative just drops one snippet, while a false positive imports
    another country's governance regime.
    """
    low = segment.lower()
    name_low = country_name.lower()
    if name_low in low or iso3.lower() in low.split():
        return True
    # "Germany" -> "german", "Japan" -> "japan(ese)", "Taiwan" -> "taiwan(ese)".
    stem = name_low.split(",")[0].split(" ")[-1]
    if len(stem) >= 5 and stem[:-1] in low:
        return True
    return False


def _split_snippet_segments(text: str) -> list[str]:
    """Split a concatenated DDG result blob back into per-result segments.

    _ddg_fallback joins every result's snippet into one string, each followed by
    its "<url>" marker. Splitting on those markers recovers the individual
    results so relevance can be judged per result -- filtering the whole blob
    would either keep one irrelevant snippet because a sibling matched, or drop
    a good snippet because the blob as a whole looked off-topic.
    """
    segments: list[str] = []
    for chunk in re.split(r"(?<=>)\s+", text or ""):
        chunk = chunk.strip()
        if chunk:
            segments.append(chunk)
    return segments or ([text.strip()] if text and text.strip() else [])


def _governance_rss_query(term: str, country_name: str, locale: str,
                          when_days: int = 730) -> Optional[str]:
    """Google News RSS for one governance term, scoped to a country.

    Unlike _google_news_rss_query this does NOT inject the general ESG keyword
    group -- the term IS the query. Results are gated on governance relevance
    before being returned.
    """
    q = f'"{term}" {country_name}'
    if when_days > 0:
        q += f" when:{when_days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&{locale}"
    r = _get(url)
    if not r:
        return None
    try:
        root = ET.fromstring(r.content)
        hits: list[str] = []
        for item in root.findall(".//item")[:6]:
            title = (item.findtext("title") or "").strip()
            pub = (item.findtext("pubDate") or "")[:16]
            link = (item.findtext("link") or "").strip()
            if not title or not _is_governance_relevant(title):
                continue
            if title.lower().startswith("home - ") and title.count(" - ") >= 2:
                continue
            hits.append(f"[{pub}] {title}" + (f" <{link}>" if link else ""))
        return "\n".join(hits) if hits else None
    except Exception:
        return None


def _country_governance_signal(country: Optional[str], industry: str = "") -> str:
    """Country-level corporate-governance evidence: the governance regime that
    applies to every company domiciled in `country`.

    WHY: gov_board_sec / gov_litigation_sec read SEC filings and are therefore
    US-listed only. Measured on the 393-company benchmark corpus, only 53.4% of
    companies have ANY company-level governance evidence, and all 11 scoring
    variants came back UNDECIDABLE for G. A country's governance code, board
    rules, audit regulator and anti-bribery statute are public and apply
    universally, so they fill that gap for the ~46% with nothing.

    SCOPE CAVEAT: this signal is CONSTANT for all companies in a country, so it
    adds no within-country ranking signal. It improves coverage and absolute
    calibration; the industry-qualified variant is the part that can vary within
    a country. Claims extracted from it are tagged country_governance so
    downstream consumers can treat them as context, not company-specific
    evidence.
    """
    from agentic_estimation.layer_1.country_normalizer import to_iso3, iso3_to_common_name
    from agentic_estimation.layer_1.country_governance_keywords import (
        governance_terms_for, locales_for,
    )

    if not country:
        return ""
    iso3 = to_iso3(country)
    if not iso3:
        log.info("country_governance → %r did not resolve to a country, skipping", country)
        return ""

    # Industry participates in the cache key: "banking governance" and "mining
    # governance" are genuinely different queries for the same country.
    sector = (industry or "").strip().lower()[:40]
    key = (iso3, sector)
    with _country_gov_lock:
        if key in _country_gov_cache:
            log.info("country_governance → cache hit for %s/%s", iso3, sector or "-")
            return _country_gov_cache[key]

    name = iso3_to_common_name(iso3) or country
    terms = governance_terms_for(iso3, limit=6)
    locales = locales_for(iso3)

    hits: list[str] = []

    # 1. Locale-routed news, queried on the governance terms THEMSELVES.
    #    Deliberately not routed through _google_news_rss_query: that helper is
    #    company-shaped -- it wraps its subject in quotes and always injects the
    #    broad _GOOGLE_RSS_KEYWORDS ESG OR-group (~40 climate/labour/waste
    #    terms). Verified live that reusing it returned air-quality alerts and
    #    university sustainability awards for Korea/Taiwan: the generic ESG
    #    group swamped the governance terms. Querying the named instruments
    #    directly ("공정거래위원회", "公司治理守則", "SEBI LODR") is the whole
    #    point of having them.
    for locale in (locales or ("hl=en&gl=US&ceid=US:en",)):
        for term in terms[:4]:
            rss = _governance_rss_query(term, name, locale, when_days=730)
            if rss:
                hits.extend(rss.splitlines())

    # 2. Web search for the regime itself -- codes/statutes/regulator actions
    #    live on regulator and law-firm sites, not in news feeds.
    #
    #    Each returned snippet is gated on governance relevance individually.
    #    Verified live that a country-name query pulls in encyclopaedia filler
    #    ("True south is one end of the axis about which the Earth rotates...",
    #    Taiwan population statistics) because DuckDuckGo falls back to
    #    general reference pages when the specific query is sparse. Unfiltered,
    #    that text reaches the claim extractor as "governance evidence".
    sector_hint = f" {industry}" if industry else ""
    for query in (
        f'"{name}" corporate governance code{sector_hint} board independence disclosure requirements',
        f'"{name}" securities regulator enforcement{sector_hint} corporate governance violation fine 2024 2025',
    ):
        raw = _ddg_fallback(query, prefix="", min_len=80)
        if not raw:
            continue
        # Also require the country to be named in the snippet. Verified live:
        # a Taiwan governance query returned an India/CII task-force passage and
        # a US Sarbanes-Oxley passage -- DuckDuckGo matches "corporate
        # governance code" globally and the country term ranks weakly. Attaching
        # another country's regime to this country's context would be worse than
        # returning nothing, since the extractor cannot tell them apart.
        kept = [seg for seg in _split_snippet_segments(raw)
                if _is_governance_relevant(seg) and _mentions_country(seg, name, iso3)]
        if kept:
            hits.extend(kept)

    if not hits:
        log.info("country_governance → no results for %s", iso3)
        with _country_gov_lock:
            _country_gov_cache[key] = ""
        return ""

    body = dedup_and_filter_lines("\n".join(hits), require_keyword=False)
    text = (f"Country Governance Context ({name}"
            f"{' / ' + industry if industry else ''}):\n"
            + "\n".join(body.splitlines()[:12]))
    log.info("country_governance → %s: %d lines (%d locales, %d terms)",
             iso3, len(body.splitlines()), len(locales or (1,)), len(terms))
    with _country_gov_lock:
        _country_gov_cache[key] = text
    return text


def _google_news_rss_signal(company: str) -> str:
    """Google News RSS — latest headlines, no API key needed. Broadened to
    the same ESG keyword breadth as news_api (was a single fixed phrase,
    "ESG sustainability", missing most labor/governance/controversy coverage).
    Query-targeted (the ESG OR-group is baked into the query string), so
    fingerprint-dedup only, no keyword pre-gate."""
    log.info("[%s] google_news_rss → fetching RSS feed", company)
    hits = _google_news_rss_query(company)
    if hits is None:
        log.warning("[%s] google_news_rss → no response or parse error", company)
        return ""
    hits = dedup_and_filter_lines(hits, require_keyword=False)
    log.info("[%s] google_news_rss → %d headlines", company, len(hits.splitlines()))
    return "Google News RSS:\n" + hits


def _reuters_signal(company: str) -> str:
    """Reuters coverage via Google News RSS restricted to site:reuters.com --
    no DDG dependency (the earlier DDG-based Reuters site search was
    unreliable). Same keyword breadth as the general Google News RSS feed."""
    log.info("[%s] reuters → fetching Google News RSS restricted to reuters.com", company)
    hits = _google_news_rss_query(company, site="reuters.com")
    if hits is None:
        log.info("[%s] reuters → no matching articles", company)
        return ""
    hits = dedup_and_filter_lines(hits, require_keyword=False)
    log.info("[%s] reuters → %d headlines", company, len(hits.splitlines()))
    return "Reuters (via Google News RSS):\n" + hits


# Same site-restricted Google News RSS trick as Reuters, extended to more
# credible sources -- general high-credibility business/financial press
# (litigation, regulatory fines, executive scandals get real coverage there)
# plus dedicated sustainability trade press (better hit-rate specifically on
# pledges, certifications, and disclosure-report coverage than general news).
_OUTLET_SOURCES = {
    "bloomberg":       ("bloomberg.com", "Bloomberg"),
    "financial_times": ("ft.com", "Financial Times"),
    "esg_today":       ("esgtoday.com", "ESG Today"),
    "greenbiz":        ("greenbiz.com", "GreenBiz"),
}


def _outlet_signal(company: str, domain: str, label: str) -> str:
    """Generic site-restricted Google News RSS fetch for one outlet -- same
    mechanism as _reuters_signal, parameterised so each new outlet doesn't
    need its own near-duplicate function."""
    log.info("[%s] %s → fetching Google News RSS restricted to %s", company, label, domain)
    hits = _google_news_rss_query(company, site=domain)
    if hits is None:
        log.info("[%s] %s → no matching articles", company, label)
        return ""
    hits = dedup_and_filter_lines(hits, require_keyword=False)
    log.info("[%s] %s → %d headlines", company, label, len(hits.splitlines()))
    return f"{label} (via Google News RSS):\n" + hits


# ── Tier 2: Specialist ESG & corporate databases ─────────────────────────────

def _bhrrc_signal(company: str) -> str:
    """Business & Human Rights Resource Centre — labor/human rights incidents via DDG."""
    log.info("[%s] bhrrc → searching business-humanrights.org via DDG", company)
    result = _ddg_fallback(
        f'site:business-humanrights.org "{company}"',
        prefix="B&HR Resource Centre", min_len=60, reject_wikipedia=True,
        company=company,
    )
    log.info("[%s] bhrrc → %s", company, "hit" if result else "no result")
    return result



def _sbti_signal(company: str) -> str:
    """SBTi — Science Based Targets commitment via DDG site search."""
    log.info("[%s] sbti → searching sciencebasedtargets.org via DDG", company)
    result = _ddg_fallback(
        f'site:sciencebasedtargets.org "{company}"',
        prefix="SBTi", reject_wikipedia=True,
        company=company,
    )
    log.info("[%s] sbti → %s", company, "hit" if result else "no result")
    return result




def _cdp_signal(company: str) -> str:
    """CDP Climate Disclosure Project — scores and disclosure status via DDG."""
    log.info("[%s] cdp → searching CDP disclosure data via DDG", company)
    result = _ddg_fallback(
        f'"{company}" CDP score climate disclosure carbon 2023 2024',
        prefix="CDP Climate Disclosure", reject_wikipedia=True,
        company=company,
    )
    log.info("[%s] cdp → %s", company, "hit" if result else "no result")
    return result





# ── Tier 3: Databases & reports ──────────────────────────────────────────────

def _gri_signal(company: str) -> str:
    """GRI Sustainability Disclosure Database — published sustainability reports via DDG."""
    log.info("[%s] gri → searching globalreporting.org via DDG", company)
    result = _ddg_fallback(
        f'site:globalreporting.org "{company}" sustainability report',
        prefix="GRI Database", min_len=60, reject_wikipedia=True,
        company=company,
    )
    log.info("[%s] gri → %s", company, "hit" if result else "no result")
    return result


def _wikipedia_signal(company: str) -> str:
    """
    Wikipedia — company overview, ownership, size, notable incidents.
    Tries multiple slugs and rejects disambiguation pages.
    """
    log.info("[%s] wikipedia → fetching Wikipedia REST API summary", company)

    # Build candidate slugs: exact name, then with common suffixes
    base = company.replace(" ", "_")
    suffixes = ["", "_(company)", "_(corporation)", "_(brand)", ",_Inc."]
    candidates = [f"{base}{s}" for s in suffixes]

    # Keywords that confirm a page is about a company, not a place/person/concept
    _COMPANY_SIGNALS = (
        "founded", "headquartered", "ceo", "revenue", "employees", "subsidiary",
        "company", "corporation", "inc.", "gmbh", "ltd", "plc", "manufacturer",
        "brand", "products", "services", "private", "publicly traded",
    )

    for slug in candidates:
        _WIKIPEDIA_LIMITER.wait()
        r = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}")
        if not r:
            continue
        try:
            data      = r.json()
            page_type = data.get("type", "")
            extract   = data.get("extract", "")

            # Skip disambiguation pages
            if page_type == "disambiguation" or "may refer to" in extract[:120]:
                log.info("[%s] wikipedia → slug '%s' is a disambiguation page, trying next", company, slug)
                continue

            # Skip pages that don't look like a company (geography, person, concept)
            extract_lower = extract.lower()
            if not any(sig in extract_lower for sig in _COMPANY_SIGNALS):
                log.info("[%s] wikipedia → slug '%s' doesn't look like a company page, trying next", company, slug)
                continue

            extract = extract[:1200]
            if not extract:
                continue

            log.info("[%s] wikipedia → hit on slug '%s' (%d chars)", company, slug, len(extract))
            # Canonical page URL from the REST summary payload, so the
            # Wikipedia signal is auditable like every other source.
            page_url = (
                (data.get("content_urls", {}).get("desktop", {}) or {}).get("page")
                or f"https://en.wikipedia.org/wiki/{slug}"
            )
            return f"Wikipedia: {extract} <{page_url}>"
        except Exception as e:
            log.warning("[%s] wikipedia → parse error on slug '%s': %s", company, slug, e)

    log.warning("[%s] wikipedia → no non-disambiguation page found", company)
    return ""


def _net_zero_signal(company: str) -> str:
    """Net-zero and carbon neutrality commitments via DDG."""
    log.info("[%s] net_zero → searching net-zero commitments via DDG", company)
    result = _ddg_fallback(
        f'"{company}" net zero carbon neutral 2030 2040 2050 pledge climate target',
        prefix="Net Zero Commitment", min_len=60, reject_wikipedia=True,
        company=company,
    )
    log.info("[%s] net_zero → %s", company, "hit" if result else "no result")
    return result


# ── DuckDuckGo fallback (used internally by several sources) ─────────────────

# Phrases that indicate DDG returned a Wikipedia page instead of the intended source.
# When a site: query returns no hits, DDG silently falls back to general results
# and Wikipedia is almost always the top hit for company names.
_WIKIPEDIA_BLEED = (
    "may refer to:",
    "is a german multinational engineering",
    "is an american multinational",
    "is a japanese multinational",
    "commonly known as",  # Wikipedia intro boilerplate: "X, commonly known as Y, is a..."
    "en.wikipedia.org",
    "wikipedia",
)


# Per-source character budget for DDG snippet text. Raised from the original
# inline 1200 when source URLs began being appended inline, so adding links
# doesn't silently evict snippet text the claim extractor previously saw.
_DDG_TEXT_BUDGET = 1800


def _ddg_fallback(
    query: str,
    prefix: str = "",
    min_len: int = 0,
    reject_wikipedia: bool = False,
    company: str = "",
    require_entity: bool = True,
) -> str:
    """
    DuckDuckGo search serialized through _DDG_LIMITER.
    All DDG-based sources call this, so concurrent threads queue here
    rather than firing simultaneously — preventing rate-limit / IP bans.
    Retries once with exponential back-off on 202/429 responses.

    reject_wikipedia: if True, discards results that look like Wikipedia bleed-through
                      (happens when site: queries return 0 results and DDG falls back).

    company: when given, each result is gated individually on whether it
             actually mentions this company (and on ground-truth leakage)
             BEFORE the results are joined -- see evidence_filters.
             filter_search_results. Without this, one off-entity result
             contaminated the whole joined blob invisibly.

    require_entity: set False for queries that are inherently about a topic
                    rather than a named company (e.g. country-level governance
                    regime lookups), where demanding the company name would
                    reject every legitimate result.
    """
    from ddgs import DDGS, exceptions as ddg_exc

    log.debug("DDG query [%s]: %s", prefix or "raw", query[:120])
    for attempt in range(2):
        _DDG_LIMITER.wait()
        try:
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=5))

            if not results:
                log.debug("DDG [%s] → 0 results", prefix)
                return ""

            # Check if the top result is a Wikipedia URL — dead giveaway of bleed-through
            top_url = (results[0].get("href") or "").lower()
            if reject_wikipedia and "wikipedia.org" in top_url:
                log.info("DDG [%s] → top result is Wikipedia, discarding (site query returned nothing)", prefix)
                return ""

            # Per-result gate BEFORE joining. A site: query that finds nothing
            # falls back to general results, which for a small/private company
            # are topic-generic pages ("an anti-bribery policy is a component
            # of...") or another company entirely. Those look identical to real
            # evidence once concatenated, so they must be dropped here.
            if company:
                results = filter_search_results(
                    results, company, prefix=prefix, require_entity=require_entity
                )
                if not results:
                    log.info("DDG [%s] → all results filtered out for '%s' (no on-entity evidence)",
                             prefix, company)
                    return ""

            # Each result carries its own source URL, appended inline after its
            # snippet so a reviewer can open the exact page a claim came from.
            # Previously only r["body"] was kept and r["href"] was discarded,
            # leaving every DDG-sourced signal (CDP, SBTi, gov_*, facility,
            # sustainability report) unauditable.
            text = " ".join(
                (r.get("body", "") + (f" <{r.get('href')}>" if r.get("href") else ""))
                for r in results if r.get("body")
            ).strip()

            # Secondary check: text body looks like a Wikipedia extract
            if reject_wikipedia:
                text_lower = text.lower()
                if any(phrase in text_lower for phrase in _WIKIPEDIA_BLEED):
                    log.info("DDG [%s] → result looks like Wikipedia bleed-through, discarding", prefix)
                    return ""

            if len(text) < min_len:
                log.debug("DDG [%s] → result too short (%d < %d chars), discarding", prefix, len(text), min_len)
                return ""

            log.debug("DDG [%s] → %d chars returned", prefix, len(text))
            # Budget raised from 1200 to accommodate the appended <url> markers
            # without evicting snippet text that previously fit. Truncation
            # backs off to the last completed "<...>" marker so a cut never
            # lands mid-URL and emits a broken, unopenable link.
            if len(text) > _DDG_TEXT_BUDGET:
                text = text[:_DDG_TEXT_BUDGET]
                cut_open = text.rfind("<")
                if cut_open > text.rfind(">"):
                    text = text[:cut_open].rstrip()
            return f"{prefix}: {text}" if prefix else text
        except ddg_exc.RatelimitException:
            log.warning("DDG rate-limited on query [%s] (attempt %d) — backing off", prefix, attempt + 1)
            if attempt == 0:
                time.sleep(8 + random.uniform(0, 4))
        except Exception as e:
            log.warning("DDG error on query [%s]: %s", prefix, e)
            break
    return ""


# ── Main entry point ─────────────────────────────────────────────────────────

def fetch_company_signals(company: str, industry: str = "", country: Optional[str] = None) -> dict[str, str]:
    """
    Fetch all ESG signals for one company in parallel.

    Args:
        company:  Company name (e.g. "Tata Steel")
        industry: Market/sector context (e.g. "Steel Manufacturing") — used
                  to sharpen controversy and sustainability report searches.
        country:  Company's home country (World Bank "Economy" name, e.g.
                  "Germany") — when mapped in country_esg_keywords.py, adds a
                  localized-language ESG query against that country's own
                  Google News edition. Unmapped/None countries: no change.

    Returns:
        Dict of source_name → evidence text. Empty sources are omitted.
    """
    # Industry context tightens sector-specific searches where relevant
    sector_hint = f" {industry}" if industry else ""

    tasks: dict[str, callable] = {
        # Tier 1 — real-time news
        "news_api":         lambda: _newsapi_signal(company),
        "google_news_rss":  lambda: _google_news_rss_signal(company),
        "reuters":          lambda: _reuters_signal(company),
        "bloomberg":        lambda: _outlet_signal(company, *_OUTLET_SOURCES["bloomberg"]),
        "financial_times":  lambda: _outlet_signal(company, *_OUTLET_SOURCES["financial_times"]),
        "esg_today":        lambda: _outlet_signal(company, *_OUTLET_SOURCES["esg_today"]),
        "greenbiz":         lambda: _outlet_signal(company, *_OUTLET_SOURCES["greenbiz"]),
        "localized_esg":    lambda: _localized_esg_signal(company, country),
        # Country-level governance regime -- fills the G-pillar gap for the
        # ~46% of companies with no company-level governance evidence (the
        # SEC-backed gov_* sources are US-listed only). Cached per
        # (country, industry), so this costs one fetch per country, not per
        # company.
        "country_governance": lambda: _country_governance_signal(country, industry),
        # Tier 2 — specialist ESG & corporate
        "bhrrc":            lambda: _bhrrc_signal(company),
        "sbti":             lambda: _sbti_signal(company),
        "cdp":              lambda: _cdp_signal(company),
        # Tier 3 — filings & databases
        "gri":              lambda: _gri_signal(company),
        "wikipedia":        lambda: _wikipedia_signal(company),
        "sustainability_report": lambda: _ddg_fallback(
            f'"{company}"{sector_hint} sustainability report 2024 2025 ESG annual disclosure',
            prefix="Sustainability Report", min_len=80, reject_wikipedia=True,
            company=company,
        ),
        "net_zero":         lambda: _net_zero_signal(company),
        "controversies":    lambda: _ddg_fallback(
            f'"{company}"{sector_hint} environmental violation labor controversy scandal fine 2023 2024 2025 -site:wikipedia.org',
            prefix="ESG Controversies", min_len=60, reject_wikipedia=True,
            company=company,
        ),
    }

    signals: dict[str, str] = {}

    log_header(log, "Signal Agent", company=company, industry=industry or "N/A", sources=len(tasks))
    log.info("[%s] starting signal fetch (%d sources)", company, len(tasks))

    with ThreadPoolExecutor(max_workers=18) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = (fut.result() or "").strip()
                if result:
                    signals[name] = result
                    log.info("[%s] %s → OK (%d chars)", company, name, len(result))
                    log.debug("[%s] %s OUTPUT:\n%s", company, name, result[:800])
                else:
                    log.info("[%s] %s → empty", company, name)
            except Exception as e:
                log.warning("[%s] %s → exception: %s", company, name, e)

    log.info("[%s] done — %d/%d sources returned data", company, len(signals), len(tasks))

    # Cross-source fingerprint dedup: the same headline can independently
    # surface via google_news_rss, reuters, localized_esg, and news_api --
    # each is deduped internally, but not against each other. A duplicate
    # controversy mention counted 3x across sources looks like 3x the
    # evidence to the pillar extractor for no real reason. Each of these
    # feeds emits "Header:\nline1\nline2..." -- keep the header line as-is
    # (never fingerprinted) and dedup only the headline lines beneath it.
    from agentic_estimation.layer_1.evidence_filters import _fingerprint
    _HEADLINE_FEEDS = ("news_api", "google_news_rss", "reuters", "bloomberg",
                       "financial_times", "esg_today", "greenbiz", "localized_esg")
    seen_fp: set = set()
    for name in _HEADLINE_FEEDS:
        if name not in signals:
            continue
        lines = signals[name].splitlines()
        if not lines:
            continue
        header, body = lines[0], lines[1:]
        kept_lines = []
        for line in body:
            stripped = line.strip()
            if not stripped:
                continue
            fp = _fingerprint(stripped)
            if fp in seen_fp:
                continue
            seen_fp.add(fp)
            kept_lines.append(line)
        if kept_lines:
            signals[name] = "\n".join([header] + kept_lines)
        else:
            # Every headline in this feed was a cross-source duplicate of an
            # earlier feed's headlines -- drop the now-header-only source.
            del signals[name]

    return signals


# ── DB signal cache (company_esg_signals) ────────────────────────────────────
# Signals are expensive (many external calls, some rate-limited). Once gathered
# for a company we persist them and reuse on later runs (gap-fill re-runs, the
# full pipeline vs metrics-only path, etc.), so we don't re-hammer the web.
# No TTL for now — a company's signals are reused until explicitly refreshed.

def _signals_db_conn():
    """psycopg2 connection using the same URL conversion used across the pipeline."""
    import psycopg2
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


def _load_signals_from_db(company_id) -> dict[str, str]:
    """Return cached {source: signal_text} for a company, or {} if none."""
    try:
        conn = _signals_db_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT source, signal_text FROM company_esg_signals WHERE company_id = %s",
            (str(company_id),),
        )
        rows = cur.fetchall()
        conn.close()
        return {src: txt for src, txt in rows if txt}
    except Exception as exc:
        log.warning("signal cache DB load failed: %s", exc)
        return {}


def _save_signals_to_db(company_id, signals: dict[str, str]) -> None:
    """Upsert gathered signals into company_esg_signals (one row per source)."""
    if not signals:
        return
    try:
        conn = _signals_db_conn()
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO company_esg_signals (company_id, source, signal_text, gathered_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (company_id, source) DO UPDATE
                SET signal_text = EXCLUDED.signal_text,
                    gathered_at = now()
            """,
            [(str(company_id), src, txt) for src, txt in signals.items()],
        )
        conn.commit()
        conn.close()
        log.debug("Cached %d signals to DB for company_id=%s", len(signals), company_id)
    except Exception as exc:
        log.warning("signal cache DB save failed: %s", exc)


def get_or_fetch_signals(company_id, company: str, industry: str = "",
                          country: Optional[str] = None) -> dict[str, str]:
    """
    Return ESG signals for a company, using the DB cache when available.
    On a cache miss, gathers from the web and persists the result.
    DB errors are non-fatal — falls back to a live fetch.

    country: forwarded to fetch_company_signals's localized_esg source (see
    that function's docstring). Previously dropped on the floor here (this
    is the path every production run actually takes, since every real
    pipeline call has a company_id) -- country=None silently disabled the
    localized-language ESG query for every company, even ones with a known
    country, regardless of what the caller passed. Cache-hit companies still
    won't refetch with country added retroactively (acceptable -- new/
    cache-miss companies benefit; see DEFECT_FIX_PLAN.md 1.4).
    """
    cached = _load_signals_from_db(company_id)
    if cached:
        log.info("[%s] signal cache hit — %d sources (skipping web fetch)", company, len(cached))
        return cached

    log.info("[%s] signal cache miss — gathering from web", company)
    signals = fetch_company_signals(company, industry, country=country)
    _save_signals_to_db(company_id, signals)
    return signals


def fetch_signals_for_companies(
    companies: list[str],
    industry: str = "",
    on_progress: callable | None = None,
    countries: dict[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    """
    Fetch signals for multiple companies with bounded concurrency.

    Args:
        companies:   List of company names
        industry:    Shared market/sector context
        on_progress: Optional callback(msg: str) for progress updates
        countries:   Optional {company_name: country} map -- forwarded into
            fetch_company_signals so the localized_esg source can fire on this
            batch call too (PHASE_5_PLAN.md 0.3b / DEFECT_FIX_PLAN.md 1.4:
            this function previously dropped country entirely, silently
            skipping localization for every company on the batch path even
            when the caller had it). None (default) preserves prior behavior
            -- run_signals.py, the only current caller, has no per-company
            country data to give it.

    Returns:
        Dict of company_name → signals dict
    """
    results: dict[str, dict[str, str]] = {}

    def _fetch_one(company: str) -> tuple[str, dict]:
        country = (countries or {}).get(company)
        return company, fetch_company_signals(company, industry, country=country)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_one, c): c for c in companies}
        done = 0
        for fut in as_completed(futures):
            company = futures[fut]
            try:
                name, signals = fut.result()
                results[name] = signals
            except Exception:
                results[company] = {}
            done += 1
            msg = f"[signal_agent] {done}/{len(companies)} — {company} ({len(results.get(company, {}))} sources)"
            if on_progress:
                on_progress(msg)

    return results
