"""
esg_press_collector.py — full-text ESG trade press via RSS.

WHY THIS EXISTS: our news sources return SEARCH SNIPPETS (~200 chars each,
five joined into one blob), and the tier-1 outlets we route through Google
News RSS -- reuters, bloomberg, financial_times -- measured 0.01-0.12 kept
claims per fetch because a paywalled article yields nothing but a headline.
Reuters' own sustainability feed answers 401, so there is no fix for those.

These four publishers put the FULL ARTICLE BODY in <content:encoded>, free,
no key, no rate limit. Measured live:

    trellis.net           28.7 KB max body   (formerly GreenBiz -- note that
                                              greenbiz.com/rss.xml still
                                              answers 200 with ZERO items)
    corporateknights.com  22.2 KB
    esgtoday.com           7.7 KB
    news.mongabay.com      2.1 KB, 32 items  (deforestation / supply chain)

ARCHITECTURE NOTE: these are CORPUS-WIDE feeds, not per-company endpoints.
There is no company parameter -- each feed is fetched ONCE per process and
cached, then every company is matched against the pooled articles. Fetching
per company would mean 336x the same HTTP request.

MEASURED YIELD IS LOW, AND THAT IS THE HONEST RESULT. Across 62 live articles
pooled from all four feeds, exactly ONE was about a specific company
("Walmart delivers mixed results on mid-decade sustainability goals"). The
rest are industry commentary: standards updates, think-pieces, book lists,
climate features. These outlets cover the ESG PROFESSION more than they cover
individual companies.

Consequently this source contributes evidence for a handful of large,
newsworthy companies and nothing for the rest of the corpus. It is kept
because when it does hit, it returns multi-KB real article prose rather than
a search snippet -- but it is NOT a coverage fix, and it should not be
expected to move pillar rankings on a corpus of small private firms.

Two traps found while building this, both preserved in the matching rules:
  * "Amazon" in these feeds usually means the RAINFOREST, not the company;
    2 of the 3 company-named headlines were Mongabay rainforest pieces.
  * Roundup articles ("Sustainability tools to use in 2026") name many
    companies as vendors. Body-mention matching -- including requiring
    repeated mentions -- returned identical generic prose for Toyota and
    Microsoft. Only headline authorship tracks what an article is about.
"""

import re
import threading
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.evidence_filters import (
    _company_tokens,
    has_ground_truth_leakage,
    matches_esg_keywords,
)
from agentic_estimation.layer_1.signal_agent import _RateLimiter, _get

log = get_logger("esg_press")

_TIMEOUT = 30
_MAX_CHARS_PER_SIGNAL = 2600
_MAX_ARTICLES_PER_COMPANY = 3
# Window of article body kept around a company mention. The whole article is
# usually about something broader; the passage naming the company is the part
# that is actually evidence about it.
_CONTEXT_CHARS = 900

_FEED_LIMITER = _RateLimiter(min_gap=2.0, jitter=1.0)

# WordPress-style feeds; all verified live to carry <content:encoded>.
_FEEDS: tuple[tuple[str, str], ...] = (
    ("trellis", "https://trellis.net/feed/"),
    ("corporate_knights", "https://www.corporateknights.com/feed/"),
    ("esg_today", "https://www.esgtoday.com/feed/"),
    ("mongabay", "https://news.mongabay.com/feed/"),
)

_feed_cache: Optional[list[dict]] = None
_feed_lock = threading.Lock()


def _clean(raw: str) -> str:
    """CDATA-unwrap, strip tags, collapse whitespace."""
    raw = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw, flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = (raw.replace("&amp;", "&").replace("&#8217;", "'").replace("&#8216;", "'")
              .replace("&#8220;", '"').replace("&#8221;", '"').replace("&nbsp;", " ")
              .replace("&#8211;", "-").replace("&#039;", "'").replace("&quot;", '"'))
    return re.sub(r"\s+", " ", raw).strip()


def _tag(block: str, name: str) -> str:
    m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", block, re.DOTALL)
    return _clean(m.group(1)) if m else ""


def _parse_feed(source: str, xml: str) -> list[dict]:
    out: list[dict] = []
    for block in re.findall(r"<item>(.*?)</item>", xml, re.DOTALL):
        body = _tag(block, "content:encoded") or _tag(block, "description")
        if not body:
            continue
        out.append({
            "source": source,
            "title": _tag(block, "title"),
            "link": _tag(block, "link"),
            "date": _tag(block, "pubDate")[:16],
            "body": body,
        })
    return out


def _load_feeds() -> list[dict]:
    """Fetch every feed once per process and pool the articles."""
    global _feed_cache
    with _feed_lock:
        if _feed_cache is not None:
            return _feed_cache
        pooled: list[dict] = []
        for source, url in _FEEDS:
            _FEED_LIMITER.wait()
            r = _get(url, timeout=_TIMEOUT)
            if not r:
                log.warning("ESG press feed unavailable: %s", source)
                continue
            items = _parse_feed(source, r.text)
            log.info("%s → %d articles", source, len(items))
            pooled.extend(items)
        _feed_cache = pooled
        log.info("ESG press pool: %d articles from %d feeds", len(pooled), len(_FEEDS))
        return _feed_cache


# Words that are part of company names but occur constantly in ESG prose, so
# matching on them finds articles about the TOPIC rather than the company.
# Measured: without this, "Green Tank", "Chemicals Limited", "Worlds Better"
# and "Halo" all matched articles that never mention those companies.
_GENERIC_IN_ESG_PROSE = frozenset((
    "green", "clean", "climate", "carbon", "energy", "solar", "wind", "water",
    "earth", "eco", "environment", "environmental", "sustainable",
    "sustainability", "impact", "future", "better", "world", "worlds",
    "global", "planet", "nature", "natural", "organic", "renewable", "power",
    "resource", "resources", "chemicals", "materials", "development", "bank",
    "media", "business", "community", "health", "food", "farm", "forest",
    "tank", "halo", "map", "match", "capital", "partners", "ventures",
    "industries", "products", "brands", "works", "labs", "studio", "digital",
))

# Total characters of distinctive tokens required before a name can match a
# long article body. Short names produce chance collisions in 4,000 characters
# of prose.
_MIN_MATCH_CHARS = 6


def _distinctive_tokens(company: str) -> list[str]:
    """Company tokens minus words that saturate ESG writing."""
    return [t for t in _company_tokens(company) if t not in _GENERIC_IN_ESG_PROSE]


def _mentions_in_body(body: str, company: str) -> Optional[int]:
    """Index of the first WHOLE-WORD mention of the company, or None.

    Whole-word matching matters more here than in search snippets: an article
    body is long enough that a short token will appear inside some unrelated
    word by chance. Requires every DISTINCTIVE token to be present, and the
    returned index is the earliest of them so the extracted window covers the
    passage that actually discusses the company.

    Names made entirely of ESG-generic words ("Green Tank", "Worlds Better")
    return None rather than matching every climate article in the pool.
    """
    toks = _distinctive_tokens(company)
    if not toks:
        return None
    # Short names are chance-collision prone in 4,000 characters of prose --
    # EXCEPT all-caps brands ("IKEA", "BASF", "SAP"), where the capitalization
    # is itself the identifying signal and the length floor would wrongly
    # discard them.
    is_acronym = company.isupper() and len(company.replace(" ", "")) >= 3
    if not is_acronym and sum(len(t) for t in toks) < _MIN_MATCH_CHARS:
        return None

    positions: list[int] = []
    for t in toks:
        # Case-SENSITIVE match against the capitalized form. This is what
        # separates a company from a common noun: "Toyota"/"Amazon" are real
        # one-word companies, while corpus names like "Strategy", "Booking"
        # and "Acquisition" are ordinary words that appear lowercase in prose.
        # A lowercase-insensitive match treated those as mentions and pulled
        # in articles that had nothing to do with the company.
        m = re.search(rf"\b{re.escape(t.capitalize())}\b", body)
        if not m:
            # Also accept all-caps styling (e.g. "IKEA", "BASF").
            m = re.search(rf"\b{re.escape(t.upper())}\b", body)
        if not m:
            return None
        positions.append(m.start())
    return min(positions)


# Company names that are also places, ecosystems or common referents in ESG
# writing. Matching these on the name alone attributes geography to a company
# -- verified live: "Amazon" matched two Mongabay articles about the RAINFOREST
# ("New fire satellites offer hope for Amazon and other biomes", "The global
# gold rush consuming the Amazon"). Each needs a corroborating corporate term
# in the same headline before the mention counts.
_AMBIGUOUS_NAMES = frozenset((
    "amazon", "shell", "total", "orange", "delta", "gap", "apple", "target",
    "sun", "eagle", "jaguar", "puma", "columbia", "everest", "alpine",
))
_CORPORATE_CONTEXT = (
    "inc", "corp", "company", "ceo", "shareholder", "earnings", "revenue",
    "retailer", "supplier", "employees", "workers", "brand", "store",
    "warehouse", "logistics", "e-commerce", "cloud", "aws", "plc", "group",
)


def _name_is_ambiguous(company: str) -> bool:
    toks = _distinctive_tokens(company)
    return len(toks) == 1 and toks[0] in _AMBIGUOUS_NAMES


def _is_substantive_mention(article: dict, company: str) -> bool:
    """True only if the company is named in the HEADLINE.

    Body-mention rules were tried and all failed. Requiring recurrence in the
    body still admitted "Sustainability tools to use in 2026", which names
    Amazon and Google repeatedly as vendors -- the extracted window came back
    as "a single interoperable framework" and "changes to flight paths",
    identical generic prose for different companies.

    Headline authorship is the only signal here that tracks what an article is
    ABOUT. It is deliberately strict: measured across 62 live articles, just
    3 headlines named any large company at all, and 2 of those were the Amazon
    RAINFOREST rather than the company. Better to return nothing than to
    attribute an industry think-piece to whichever companies it happens to
    list.
    """
    if _mentions_in_body(article["title"], company) is None:
        return False
    if _name_is_ambiguous(company):
        # Name doubles as a place/ecosystem: require corporate vocabulary in
        # the headline or the article's opening, else it is geography.
        context = f"{article['title']} {article['body'][:400]}".lower()
        if not any(re.search(rf"\b{re.escape(t)}\b", context) for t in _CORPORATE_CONTEXT):
            return False
    return True


def fetch_esg_press_signals(company: str) -> dict[str, str]:
    """Full-text ESG trade-press coverage mentioning `company`.

    Returns {"esg_press": text} or {} when no pooled article names the
    company -- the normal outcome for small private firms, and NOT evidence
    of anything either way.
    """
    articles = _load_feeds()
    if not articles:
        return {}

    log_header(log, "ESG Press", company=company, pool=len(articles))
    chunks: list[str] = []
    for art in articles:
        idx = _mentions_in_body(f"{art['title']} {art['body']}", company)
        if idx is None:
            continue
        body = art["body"]
        # Leakage guard: these outlets cover B Corp certifications, so an
        # article can restate the benchmark score we are trying to predict.
        if has_ground_truth_leakage(body):
            log.info("[%s] esg_press → dropping leaking article from %s", company, art["source"])
            continue
        # Being NAMED in an ESG article is not the same as the article being
        # ABOUT the company. Verified on live feed data: "Sustainability tools
        # to use in 2026" and "This tracker ranks AI data center leaders"
        # each name several companies in passing, and the extracted window
        # came back as generic prose about climate standards -- identical text
        # returned for Toyota and Microsoft. Require the company to be a
        # subject of the piece: named in the headline, or mentioned repeatedly.
        if not _is_substantive_mention(art, company):
            log.debug("[%s] esg_press → passing mention only in %s", company, art["link"])
            continue
        start = max(0, idx - _CONTEXT_CHARS // 3)
        window = body[start:start + _CONTEXT_CHARS]
        # The company can be named in passing in an article about something
        # else entirely; require ESG vocabulary in the extracted window.
        if not matches_esg_keywords(window):
            continue
        chunks.append(f"[{art['source']} {art['date']}] {art['title']}. {window} <{art['link']}>")
        if len(chunks) >= _MAX_ARTICLES_PER_COMPANY:
            break

    if not chunks:
        log.info("[%s] esg_press → no matching articles", company)
        return {}
    body = " | ".join(chunks)[:_MAX_CHARS_PER_SIGNAL]
    log.info("[%s] esg_press → %d article(s), %d chars", company, len(chunks), len(body))
    return {"esg_press": f"ESG trade press: {body}"}


if __name__ == "__main__":  # manual probe
    import sys
    target = " ".join(sys.argv[1:]) or "Microsoft"
    out = fetch_esg_press_signals(target)
    if not out:
        print(f"(no ESG press coverage for {target!r})")
    for k, v in out.items():
        print(f"\n=== {k} ===\n{v[:900]}")
