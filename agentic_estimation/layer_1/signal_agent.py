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

import math
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote_plus, urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1 import http_cache as _http_cache
from agentic_estimation.layer_1.evidence_filters import (
    dedup_and_filter_lines,
    filter_search_results,
    has_ground_truth_leakage,
)
log = get_logger("signal_agent")

_HEADERS = {"User-Agent": "ESG-Signal-Agent/1.0 (research@example.com)"}
_TIMEOUT = 12

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # newsapi.org key (optional)

# NewsAPI's free tier is 100 requests PER DAY, and we spend one per company.
# That is a daily QUOTA, not a rate: once it is gone every further call returns
# 429 no matter how slowly we ask, so a limiter cannot help and retrying only
# burns the tripwire budget. A 150-company run exceeded it immediately --
# 5 workers each got 429 on their first company and aborted the run at 1/150.
#
# Treated as a hard per-run budget so the quota is spent on the first N
# companies and skipped silently after, instead of aborting the whole run.
# 0 disables the source entirely.
_NEWSAPI_DAILY_BUDGET = int(os.getenv("ESG_NEWSAPI_BUDGET", "0"))
_newsapi_spent = 0
_newsapi_lock = threading.Lock()


def _newsapi_take_budget() -> bool:
    """True if this call may spend one of the day's NewsAPI requests."""
    global _newsapi_spent
    with _newsapi_lock:
        if _newsapi_spent >= _NEWSAPI_DAILY_BUDGET:
            return False
        _newsapi_spent += 1
        return True

# ── Rate limiters ─────────────────────────────────────────────────────────────
# Each limiter ensures a minimum gap between consecutive calls to that source,
# even when multiple threads fire simultaneously.

# Adaptive-backoff tuning. A throttle doubles the offending limiter's gaps; the
# multiplier decays back to normal over _PENALTY_DECAY_S so one bad minute does
# not slow the rest of a multi-hour run forever.
#
# MEASURED COST of the pacing changes, on DDG (the busiest limiter, 9 sources
# per company): mean gap 2.75s -> 2.98s from the log-normal draw, plus ~0.75s
# per request once long pauses are amortised. That is ~34s per company against
# ~25s before, i.e. roughly 36% slower on a COLD gather.
#
# It is not 36% slower in practice, because http_cache is consulted before the
# limiter -- a warm gather costs 0.03s and skips pacing entirely. The slowdown
# is paid only on genuinely new fetches, which is exactly where throttle risk
# lives. Set ESG_LIMITER_PAUSES=0 to opt out for quick interactive runs.
_PENALTY_MAX     = float(os.getenv("ESG_LIMITER_PENALTY_MAX", "8.0"))
_PENALTY_DECAY_S = float(os.getenv("ESG_LIMITER_PENALTY_DECAY", "600"))

# Long pauses can be disabled for quick interactive runs, where a 25s pause in
# a 6-request script is pure cost with no throttle to avoid.
_LONG_PAUSES = os.getenv("ESG_LIMITER_PAUSES", "1") != "0"

class _RateLimiter:
    """Thread-safe minimum-gap enforcer with optional jitter.

    RESERVE-THEN-SLEEP. The obvious implementation sleeps while holding the
    lock:

        with self._lock:
            if elapsed < gap:
                time.sleep(gap - elapsed)     # lock held for the whole sleep
            self._last = time.time()

    That enforces the right rate but serialises the WAITING as well as the
    requests: with N threads queued on one limiter, thread N blocks for the
    sum of everyone else's sleeps before it can even compute its own. On
    _DDG_LIMITER -- which 9 sources and 18 workers share -- that turned a 2.75s
    average gap into a queue where late threads waited tens of seconds while
    holding nothing useful.

    Here each caller instead RESERVES the next slot under the lock (cheap,
    no I/O) and sleeps outside it. Same emitted rate -- slots are still handed
    out one gap apart -- but threads wait concurrently, and a thread that
    arrives when the limiter is idle proceeds immediately.

    self._next is the timestamp of the next free slot, not the time of the
    last call; that is what makes the reservation safe against two threads
    reading the same "last" value and both deciding they may go now.

    PACING FEATURES (all process-wide, configured here so no call site has to
    repeat them):

      * LOG-NORMAL gaps, not uniform. A uniform jitter still produces a flat,
        bounded distribution -- gaps never exceed min+jitter, which is itself a
        signature. Log-normal gives mostly-short gaps with an occasional long
        one, matching how request spacing actually varies, and it has no hard
        upper bound.

      * PERIODIC LONG PAUSES. Every _burst_len requests (itself randomised) the
        limiter inserts a multi-second pause. Beyond looking less mechanical,
        this genuinely helps: most throttles are enforced over a sliding
        window, so a pause lets the window drain instead of riding the limit
        continuously.

      * ADAPTIVE BACKOFF. On a throttle, penalise() multiplies this limiter's
        gaps and the multiplier decays back to 1.0 over time. This is the part
        that most reduces repeat throttling -- reacting to the host's actual
        signal beats any fixed schedule.

    HONEST SCOPE: this makes TIMING less mechanical. It does not disguise the
    traffic -- request ORDER, query shape, absent referrers and un-fetched page
    assets all still read as automated. Treat it as throttle avoidance, not as
    evasion; the real reduction in exposure came from caching (http_cache),
    which removes requests entirely.
    """

    def __init__(self, min_gap: float, jitter: float = 0.0,
                 name: str = "", pause_every: tuple = (14, 30),
                 pause_len: tuple = (8.0, 25.0)):
        self._min_gap = min_gap
        self._jitter  = jitter
        self._name    = name
        self._lock     = threading.Lock()
        self._next     = 0.0
        # Adaptive penalty, driven by penalise()/_decay.
        self._penalty      = 1.0
        self._penalty_set  = 0.0
        # Long-pause scheduling.
        self._pause_every = pause_every
        self._pause_len   = pause_len
        self._count       = 0
        self._until_pause = random.randint(*pause_every)

    def _sample_gap(self) -> float:
        """Log-normal gap with mean ~= min_gap + jitter/2.

        sigma=0.35 keeps the bulk within roughly +-40% of the mean while still
        producing an occasional long tail. Clamped below at min_gap so the
        configured floor -- which is what the host tolerance was tuned to -- is
        never violated by a low sample.
        """
        centre = self._min_gap + self._jitter * 0.5
        if centre <= 0:
            return 0.0
        gap = random.lognormvariate(math.log(max(centre, 1e-3)), 0.35)
        return max(self._min_gap, min(gap, centre * 4.0))

    def _current_penalty(self) -> float:
        """Penalty multiplier, decaying linearly to 1.0 over _PENALTY_DECAY_S."""
        if self._penalty <= 1.0:
            return 1.0
        age = time.time() - self._penalty_set
        if age >= _PENALTY_DECAY_S:
            self._penalty = 1.0
            return 1.0
        frac = 1.0 - (age / _PENALTY_DECAY_S)
        return 1.0 + (self._penalty - 1.0) * frac

    def penalise(self, factor: float = 2.0) -> None:
        """Slow this limiter down after a throttle. Compounds, capped."""
        with self._lock:
            self._penalty = min(self._current_penalty() * factor, _PENALTY_MAX)
            self._penalty_set = time.time()
        log.warning("limiter[%s] backing off x%.1f after throttle",
                    self._name or "?", self._penalty)

    def wait(self) -> None:
        with self._lock:
            gap = self._sample_gap() * self._current_penalty()

            # Long pause on a randomised schedule. Added to the reserved slot
            # inside the lock so that concurrent callers queue behind it rather
            # than all pausing independently.
            self._count += 1
            if _LONG_PAUSES and self._count >= self._until_pause:
                self._count = 0
                self._until_pause = random.randint(*self._pause_every)
                pause = random.uniform(*self._pause_len)
                gap += pause
                log.debug("limiter[%s] long pause %.1fs (next in %d reqs)",
                          self._name or "?", pause, self._until_pause)

            now  = time.time()
            slot = self._next if self._next > now else now
            self._next = slot + gap
        delay = slot - time.time()
        if delay > 0:
            time.sleep(delay)


# DDG is the most sensitive — serialize ALL DDG calls through one limiter.
#
# RAISED 2026-08-06 from 2.0+1.5 after DDG stopped answering mid-run: the html
# endpoint began returning HTTP 202 with an empty results shell, and it persisted
# through a 90s cooldown. That is an anti-bot stall rather than a per-second rate
# limit, so the fix is a slower, less regular cadence rather than a retry.
#
# RAISED AGAIN 2026-08-19 from 4.0+3.0 to 5.5+3.5 (~7.25s mean vs ~5.5s, ~0.14
# req/s vs ~0.18) -- precautionary, not in response to a new DDG block (the
# same-day BHRRC failures were traced to our OWN Wikipedia-bleed-through
# filtering discarding real, correctly-fetched results, not to DDG itself --
# see _render_ddg_results). Only ONE call site in this file (_bhrrc_signal)
# still uses a site: restriction; every other DDG call is already a pure
# keyword search with no site-native alternative to move to, so the real
# lever left for reducing DDG exposure is pacing, not fewer/different
# queries -- this is that lever, used a bit more conservatively.
#
# Overridable for runs that need to go slower still.
_DDG_LIMITER      = _RateLimiter(name="ddg",
    min_gap=float(os.getenv("ESG_DDG_MIN_GAP", "5.5")),
    jitter=float(os.getenv("ESG_DDG_JITTER", "3.5")),
)

# Wikipedia REST API: polite crawling policy — 1 req/s is fine, add small jitter.
# Wikipedia's REST API is the strictest host we touch. Measured during a
# 10-worker corpus run: 140 of 145 total HTTP 429s came from en.wikipedia.org
# (Google News 3, EPA 2, DuckDuckGo 0). The old 1.0s gap looks safe per
# thread, but _wikipedia_signal tries up to 5 slug variants per company, so
# N workers burst up to 5N requests before any of them waits.
_WIKIPEDIA_LIMITER = _RateLimiter(min_gap=3.0, jitter=1.5, name="wikipedia")

# Google News RSS had NO limiter until an IP-level block proved it needs one.
# Batching the keyword group (see _RSS_BATCH_SIZE) multiplied requests from
# 1 to 12 per news source, and with 6 news sources across 5 workers that is
# ~360 concurrent requests -- Google responded with a blanket HTTP 503
# "Sorry..." page for every query, for hours. Earlier research reporting "no
# rate limiting" tested SEQUENTIAL requests only; concurrency is what trips it.
#
# 3.0+2.0 (mean 4.0s) is NOT a tuned value -- we have never observed where
# Google's real threshold sits, only that ~360 concurrent requests is past it.
# The reference point for choosing it: at ~10.6 Google requests per company
# this yields 1.41 companies/min, against the 1.66 companies/min the pipeline
# reaches when it is latency-bound with Google News disabled entirely. So
# Google is still marginally the bottleneck here -- it costs roughly 15% of
# throughput versus not fetching it at all, which is the price paid for
# staying well back from a host that has issued two IP-level blocks.
# Re-derive from that latency-bound figure rather than lowering it by feel.
_GOOGLE_NEWS_LIMITER = _RateLimiter(min_gap=3.0, jitter=2.0, name="google_news")

# KILL SWITCH -- currently ON. We tripped an IP-level Google News block, and
# every further request EXTENDS it rather than waiting it out (a 3-company
# smoke test issued 28 more blocked requests and pushed the block further
# out). Until the block clears, all six Google-News-backed sources
# (google_news_rss, reuters, bloomberg, financial_times, esg_today, greenbiz,
# localized_esg) must issue zero requests.
#
# Re-enable with ESG_GOOGLE_NEWS=1 once a SINGLE manual probe returns 200 --
# probe by hand, never in a loop, and never from a worker pool.
_GOOGLE_NEWS_DISABLED = os.getenv("ESG_GOOGLE_NEWS", "0") != "1"

# GDELT: no documented rate limit but slow server; 1 req per call, no concern.
# SEC EDGAR: 10 req/s limit per their fair-use policy.
# Google News RSS / ESG Today RSS / Reuters RSS: standard RSS fetch, once per run.
# NewsAPI: 100 req/day free tier — one call per company is fine.

# ── Rate-limit tripwire ──────────────────────────────────────────────────────
# A 429 means we are taking more than a host is willing to give. Continuing
# past it produces a corpus with silent holes -- the affected companies look
# like "no evidence found" rather than "we were blocked" -- which is worse
# than stopping, because the gap is invisible in the output.
#
# Counting rather than failing on the first one: a single 429 can be a
# momentary burst, but a run of them means we are being throttled in earnest.
# Once the threshold trips, _get raises RateLimitTripped, which propagates out
# of every collector and aborts the run.

class RateLimitTripped(RuntimeError):
    """Raised when too many HTTP 429s are seen -- abort rather than collect a
    corpus full of invisible gaps."""


# Abort after THREE throttling responses, not ten. Ten was already too many:
# by the time the counter reached it, Google had issued an IP-level block that
# outlived the run. Three is low enough that a run stops while the damage is
# still recoverable, and any legitimate transient blip costs only a restart.
_RATE_LIMIT_THRESHOLD = int(os.getenv("ESG_RATE_LIMIT_ABORT", "3"))
_rate_limit_hits: Counter = Counter()
_rate_limit_lock = threading.Lock()


def _limiter_for_host(host: str):
    """The limiter governing `host`, so a throttle can slow the right one.

    Without this the backoff would be global: one DDG throttle would also slow
    Wikipedia and SEC, which are unrelated hosts with their own budgets.
    """
    h = (host or "").lower()
    if "duckduckgo" in h or h == "https://duckduckgo.com":
        return _DDG_LIMITER
    if "wikipedia" in h:
        return _WIKIPEDIA_LIMITER
    # Bing before Google: calibration.bing_search registers its limiter here at
    # import time (it lives in calibration/, which this module must not import).
    # Without this entry a Bing throttle would penalise nothing -- the lookup
    # would fall through and return None -- and Bing is now the primary search
    # source, so it is the host most worth reacting to.
    if "bing" in h and _EXTRA_LIMITERS.get("bing") is not None:
        return _EXTRA_LIMITERS["bing"]
    if "google" in h:
        return _GOOGLE_NEWS_LIMITER
    return None


# Limiters owned by modules this one cannot import (calibration/ depends on
# agentic_estimation, never the reverse). register_limiter() lets those modules
# opt into the same throttle-reaction path as the built-in hosts.
_EXTRA_LIMITERS: dict = {}


def register_limiter(key: str, limiter: "_RateLimiter") -> None:
    """Route throttles for hosts matching `key` to `limiter`."""
    _EXTRA_LIMITERS[key.lower()] = limiter


def _note_rate_limit(url: str) -> None:
    host = urlparse(url).netloc or url
    # ADAPTIVE BACKOFF: react to the host's own signal before anything else.
    # A fixed schedule cannot know where a host's threshold sits; a throttle is
    # the only direct evidence we ever get, so it is worth acting on
    # immediately rather than only counting toward the abort threshold.
    lim = _limiter_for_host(host)
    if lim is not None:
        lim.penalise()
    with _rate_limit_lock:
        _rate_limit_hits[host] += 1
        total = sum(_rate_limit_hits.values())
        n_host = _rate_limit_hits[host]
    log.warning("RATE LIMITED (429) by %s — %d from this host, %d total",
                host, n_host, total)
    if total >= _RATE_LIMIT_THRESHOLD:
        breakdown = ", ".join(f"{h}={c}" for h, c in _rate_limit_hits.most_common())
        raise RateLimitTripped(
            f"aborting: {total} HTTP 429s (threshold {_RATE_LIMIT_THRESHOLD}) — {breakdown}. "
            f"Reduce workers or raise the limiter gap for the offending host."
        )


def rate_limit_report() -> dict:
    """429 counts per host so far, for run summaries."""
    with _rate_limit_lock:
        return dict(_rate_limit_hits)


# ── Helpers ──────────────────────────────────────────────────────────────────

class _CachedResponse:
    """Minimal stand-in for requests.Response on a cache hit.

    Callers of _get use .content (XML/feed parsing), .text, .status_code and
    .ok. Reconstructing a real Response is not worth it; this exposes exactly
    that surface, so a cached fetch is indistinguishable at the call site.
    """

    __slots__ = ("text", "status_code", "url")

    def __init__(self, text: str, url: str = ""):
        self.text = text
        self.status_code = 200
        self.url = url

    @property
    def content(self) -> bytes:
        return self.text.encode("utf-8", errors="replace")

    @property
    def ok(self) -> bool:
        return True

    def json(self):
        import json as _json
        return _json.loads(self.text)


def _get(url: str, params: dict | None = None, timeout: int = _TIMEOUT,
         cache: bool = True) -> requests.Response | None:
    # Checked BEFORE any limiter the caller may have waited on is irrelevant --
    # what matters is that a hit costs no request at all, so the host never
    # sees it and the tripwire budget is untouched.
    if cache:
        hit = _http_cache.get("get", url, params)
        if hit is not None:
            log.debug("GET %s → cache hit (%d chars)", url, len(hit))
            return _CachedResponse(hit, url)
    try:
        r = requests.get(url, headers=_HEADERS, params=params, timeout=timeout)
        r.raise_for_status()
        log.debug("GET %s → HTTP %s (%d bytes)", url, r.status_code, len(r.content))
        if cache:
            # Successes only. A cached 429 would turn a transient throttle into
            # a permanent gap AND hide it from _note_rate_limit.
            _http_cache.put("get", url, r.text, params)
        return r
    except requests.HTTPError as e:
        # A 404 means "resource doesn't exist" — an expected outcome when probing
        # slug variants (e.g. Wikipedia), not a real error. Keep it at debug.
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            log.debug("GET %s → 404 Not Found", url)
        elif status in (429, 503):
            # 503 matters as much as 429 here: Google News RSS answers a
            # throttled client with 503 and an HTML "Sorry..." interstitial,
            # never 429. Watching only for 429 let an IP-level Google block
            # run silently through an entire corpus gather -- every news
            # source returned nothing, and the output looked like "these
            # companies have no news" rather than "we were blocked".
            _note_rate_limit(url)      # may raise RateLimitTripped
        else:
            log.warning("GET %s → HTTP error: %s", url, e)
    except RateLimitTripped:
        # MUST be re-raised. It is raised from inside the HTTPError handler
        # above, so without this clause the bare `except Exception` below
        # catches it, logs it as an ordinary fetch failure, and returns None
        # -- which is exactly what happened during a corpus run that logged
        # 8,466 blocked Google requests and never aborted. The tripwire
        # existed, counted correctly, and was then swallowed one frame later.
        raise
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
    if not _newsapi_take_budget():
        log.debug("[%s] news_api → skipped (daily budget %d exhausted)",
                  company, _NEWSAPI_DAILY_BUDGET)
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


# Headlines kept per feed after recency filtering. Raised from the original
# inline 8: Google returns ~100 items per query, and taking the first 8 by
# relevance threw away most of the recent coverage along with the old.
_MAX_RSS_ITEMS = 20

# QUERY OVERFLOW -- the reason this module batches its keyword group.
#
# Google News RSS silently DISCARDS trailing operators once a query exceeds a
# complexity threshold, then falls back to a plain relevance search. It does
# not error; it just quietly stops honouring `when:`/`after:`/`site:`.
# Measured live, holding `when:30d` fixed and varying only the OR-group size:
#
#     OR-terms   oldest article returned
#         8          5 days   -- filter honoured
#        12         25 days   -- filter honoured
#        15       5774 days   -- filter DROPPED
#        20       5774 days   -- filter DROPPED
#
# _GOOGLE_RSS_KEYWORDS carries ~60 terms, so `when:Nd` has never actually been
# applied in production -- which is why decade-old articles reached the claim
# extractor as if they were current evidence. The company name survives the
# overflow (85/92 titles still matched it); only the operators are lost.
#
# Fix: split the keyword group into small batches, query each with the date
# operator intact, and union the results. Verified to both filter correctly
# AND return more articles than the single long query:
#     Nestle    100 items (oldest 1571d)  ->  279 items (oldest 364d)
#     Shell     100 items (oldest 2028d)  ->  245 items (oldest 363d)
# Batching is free: 40 rapid sequential and 25 concurrent requests all
# returned 200 with no rate limiting and no required User-Agent.
# RAISED from 5 to 20 after batch-size 5 (12 requests/company) caused an
# IP-level Google block. 20 terms per batch means 3 requests per company
# instead of 12 -- a 4x reduction in request volume.
#
# The trade-off is real and accepted: 20 OR-terms is above the ~12-term
# threshold where Google starts silently dropping the `when:` operator, so
# server-side date filtering may not apply to these queries. The client-side
# pubDate filter in _filter_items_by_recency catches that -- it is why the
# belt-and-braces filter exists. Losing some server-side filtering is a much
# cheaper price than losing the source entirely.
_RSS_BATCH_SIZE = 20


def _keyword_batches(keywords: str, size: int = _RSS_BATCH_SIZE) -> list[str]:
    """Split an 'A OR B OR "C D"' string into OR-groups of at most `size`
    terms, keeping quoted phrases intact."""
    terms, buf, in_quotes = [], [], False
    for tok in keywords.split():
        if tok == "OR" and not in_quotes:
            if buf:
                terms.append(" ".join(buf))
                buf = []
            continue
        buf.append(tok)
        if tok.count('"') % 2 == 1:
            in_quotes = not in_quotes
    if buf:
        terms.append(" ".join(buf))
    return [" OR ".join(terms[i:i+size]) for i in range(0, len(terms), size)]


def _filter_items_by_recency(items: list, when_days: int) -> list:
    """Drop RSS items older than `when_days`, newest first.

    Google's own `when:Nd` operator does not work on the RSS endpoint (see
    _google_news_rss_query), so recency has to be enforced here. Items whose
    pubDate cannot be parsed are KEPT rather than dropped -- an unparseable
    date is not evidence that the article is old, and silently discarding
    them would lose real coverage.

    when_days <= 0 disables filtering entirely (still sorts newest-first).
    """
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone, timedelta

    if not items:
        return items
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=when_days) if when_days > 0 else None

    dated: list[tuple[Optional[object], object]] = []
    for it in items:
        raw = it.findtext("pubDate") or ""
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            dt = None
        if cutoff is not None and dt is not None and dt < cutoff:
            continue
        dated.append((dt, it))

    # Newest first; undated items sort last but are retained.
    dated.sort(key=lambda pair: (pair[0] is not None, pair[0] or now), reverse=True)
    return [it for _dt, it in dated]


def _google_news_rss_query(company: str, extra: str = "", site: str = "", locale: str = "hl=en-US&gl=US&ceid=US:en", when_days: int = 730) -> Optional[str]:
    """Shared Google News RSS fetch+parse -- used for the general feed, the
    outlet-restricted feeds (Reuters, Bloomberg, Financial Times, ESG Today,
    GreenBiz -- site restriction via Google's site: operator, same trick as
    the DDG site-search helpers use), and the localized-ESG feed (locale
    routes the query to that country's actual Google News edition).

    The keyword group is sent as SEVERAL small queries rather than one large
    one, and the results are unioned -- see _RSS_BATCH_SIZE for the measured
    reason (a long OR-group makes Google silently drop `when:`/`site:`).

    site: goes first in each query. The original note here said position was
    the cause of trailing `site:` being ignored; it is really the same
    overflow -- with a short enough query the operator is honoured wherever
    it sits. Kept first regardless, since it costs nothing.

    when_days: recency window in days. Enforced BOTH by Google's `when:Nd`
    operator (which does work, once the query is short enough) and again
    client-side against each item's pubDate. The client-side pass is a
    deliberate belt-and-braces: it is what caught the overflow bug in the
    first place, and it keeps a future query-length regression from silently
    re-admitting decade-old articles. pubDates are reliable -- 400/400 parsed,
    always GMT, and reflect original publication rather than index time.

    Default 730 (two years): ESG evidence (pledges, certifications, disclosed
    incidents) stays relevant longer than typical news, but a resolved
    five-year-old controversy should not still be moving a score today.
    Set to 0 to disable the window (results are still sorted newest-first)."""
    # Batch ONLY the general feed. A site:-restricted outlet has already
    # narrowed the result set so hard that one broad query returns everything
    # available, and those feeds measured 0.01-0.12 kept claims per fetch --
    # so batching them multiplied request volume sixfold for no evidence, and
    # was the direct cause of an IP-level Google block.
    batches = _keyword_batches(_GOOGLE_RSS_KEYWORDS) if not site else [_GOOGLE_RSS_KEYWORDS]
    seen_links: set[str] = set()
    merged: list = []
    for batch in batches:
        items = _rss_fetch_items(company, batch, extra, site, locale, when_days)
        for it in items:
            link = (it.findtext("link") or "").strip()
            key = link or (it.findtext("title") or "").strip()
            if key and key in seen_links:
                continue
            if key:
                seen_links.add(key)
            merged.append(it)
    if not merged:
        return None
    return _render_rss_items(_filter_items_by_recency(merged, when_days))


def _rss_fetch_items(company: str, keywords: str, extra: str, site: str,
                      locale: str, when_days: int) -> list:
    """One Google News RSS request. Returns raw <item> elements (possibly
    empty); never raises."""
    q = f"site:{site} " if site else ""
    q += f'"{company}" ({keywords})'
    if extra:
        q += f" {extra}"
    if when_days > 0:
        q += f" when:{when_days}d"
    if _GOOGLE_NEWS_DISABLED:
        return []
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&{locale}"
    _GOOGLE_NEWS_LIMITER.wait()
    r = _get(url)
    if not r:
        return []
    try:
        return ET.fromstring(r.content).findall(".//item")
    except RateLimitTripped:
        raise   # never swallow the abort signal
    except Exception:
        return []


def _render_rss_items(items: list) -> Optional[str]:
    """Format already-filtered, newest-first <item> elements into the
    "[date] headline <url>" lines the claim extractor consumes."""
    try:
        hits  = []
        for item in items[:_MAX_RSS_ITEMS]:
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
    # SECOND Google News path, easy to miss: this does NOT route through
    # _google_news_rss_query, so for a while it honoured neither the kill
    # switch nor _GOOGLE_NEWS_LIMITER. Being a nested loop (locales x terms)
    # it was in fact the HEAVIER of the two paths -- it issued 8,466 blocked
    # requests during a 20-company run while the guarded path was behaving.
    # Any new Google News call site must take these two lines with it.
    if _GOOGLE_NEWS_DISABLED:
        return None
    q = f'"{term}" {country_name}'
    if when_days > 0:
        q += f" when:{when_days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&{locale}"
    _GOOGLE_NEWS_LIMITER.wait()
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
    except RateLimitTripped:
        raise   # never swallow the abort signal
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


# S-pillar labor/HR trade press, fetched DIRECTLY (their own real RSS
# feeds), NOT via the Google News site: trick every _OUTLET_SOURCES entry
# above uses. Added 2026-08-19: confirmed 6 of the 9 news-shaped sources in
# this file (google_news_rss, reuters, bloomberg, financial_times,
# esg_today, greenbiz) all route through the SAME _google_news_rss_query()
# function -- one real dependency wearing six labels. Real S-evidence
# motivation: measured live 2026-08-19 that a 100-company well-known-
# public-firm sample got mostly-zero S-pillar claims -- the only S-specific
# source that existed was BHRRC (incident-focused).
#
# search=True vs search=False -- checked live 2026-08-19, don't assume:
#   Personnel Today: REAL search+RSS combo, WordPress-standard
#     ?s=<query>&feed=rss2 -- confirmed genuinely filters (title literally
#     reads "You searched for Kroger - Personnel Today", 2 real matching
#     items, not the generic feed).
#   HR Grapevine: rss?q=<query> returns HTTP 200 but is BYTE-IDENTICAL to
#     the unfiltered feed -- confirmed the param is silently ignored, not a
#     working filter. Kept as a passive latest-~20-articles snapshot.
#   HR Dive: has a real search PAGE (/search/?q=, 200, real per-company
#     results) but it's JS-rendered with no discoverable RSS/API combo from
#     a plain fetch -- not worth reverse-engineering an undocumented XHR
#     endpoint for one source. Kept passive, same as HR Grapevine.
# A passive source only hits when a company happens to be in that outlet's
# ~20-30 MOST RECENT articles at fetch time (confirmed: 5/100 companies hit
# across all 3 sources in a one-time snapshot test) -- real, but low, single-
# fetch coverage; the honest expectation is it accumulates across REPEATED
# production runs over time, not that it saturates on one run. Personnel
# Today's real search fixes this for itself; the other two remain
# snapshot-only until/unless a similar real search+feed combo is found.
_DIRECT_RSS_SOURCES = {
    "hr_dive":         ("https://www.hrdive.com/feeds/news/", "HR Dive", False),
    "hr_grapevine":    ("https://www.hrgrapevine.com/rss", "HR Grapevine", False),
    "personnel_today": ("https://www.personneltoday.com/", "Personnel Today", True),
}

_DIRECT_RSS_LIMITER = _RateLimiter(name="direct_rss", min_gap=1.5, jitter=1.0)

# Strips HTML/markup before company-name matching -- see the false-positive
# bug this fixed, documented at the match site below.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _direct_rss_signal(company: str, feed_url: str, label: str, search: bool = False) -> str:
    """Fetch one outlet's RSS -- either a real per-company SEARCH+RSS
    combo (search=True, feed_url is the site root, `?s=<company>&feed=rss2`
    is appended) or the outlet's own latest-articles snapshot (search=False,
    feed_url is the feed itself, filtered client-side for company mentions
    since these feeds aren't company-scoped -- see _DIRECT_RSS_SOURCES for
    which sources support real search, verified live, not assumed).
    Shares _DIRECT_RSS_LIMITER across all three direct feeds (same host-
    independence reasoning as _limiter_for_host: these are unrelated hosts
    from Google/DDG/Bing, so a throttle here should not slow anything else,
    and vice versa) -- NOT registered in _limiter_for_host since
    RateLimitTripped propagation isn't needed here (see except block: a
    429/parse failure on a low-volume trade feed degrades to "no evidence"
    for this one source, not an abort)."""
    if search:
        from urllib.parse import quote_plus
        feed_url = f"{feed_url.rstrip('/')}/?s={quote_plus(company)}&feed=rss2"
        log.info("[%s] %s → fetching search-scoped RSS", company, label)
    else:
        log.info("[%s] %s → fetching direct RSS feed", company, label)
    _DIRECT_RSS_LIMITER.wait()
    try:
        r = _get(feed_url, timeout=_TIMEOUT)
        if not r or not r.ok:
            log.info("[%s] %s → fetch failed or non-200", company, label)
            return ""
        items = ET.fromstring(r.content).findall(".//item")
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.info("[%s] %s → fetch/parse error: %s", company, label, exc)
        return ""

    # Kept even for search=True: WordPress's own search can return loosely-
    # relevant results (word-overlap matches, not just the exact company
    # name), so this still guards against noise -- not redundant with the
    # server-side search, a second real filter on top of it.
    #
    # REAL BUG FOUND AND FIXED 2026-08-19: matching against the raw
    # <description> field directly was a false-positive generator, not a
    # filter. HR Dive's description is CDN image-proxy markup + a short
    # prose snippet (e.g. `<img src="https://imgproxy.divecdn.com/
    # fjcQB3TBK.../g:ce/rs:fill:1600:900:1/Z3M6Ly9k...webp"/>`), and a short
    # company name like "BP" matches as a substring INSIDE that base64/URL
    # gibberish almost every time -- confirmed live: 100% of one feed's 20
    # items "matched" a query for "BP" even though the real prose text in
    # every one of them had nothing to do with BP. Strip HTML tags first
    # (_HTML_TAG_RE, module-level) so only real prose is searched.
    name_lower = company.lower()
    matched = [
        it for it in items
        if name_lower in (it.findtext("title") or "").lower()
        or name_lower in _HTML_TAG_RE.sub(" ", it.findtext("description") or "").lower()
    ]
    if not matched:
        log.info("[%s] %s → 0/%d articles mention %s", company, label, len(items), company)
        return ""

    hits = _render_rss_items(_filter_items_by_recency(matched, when_days=730))
    if not hits:
        return ""
    hits = dedup_and_filter_lines(hits, require_keyword=False)
    mode = "search RSS" if search else "direct RSS"
    log.info("[%s] %s → %d matching headlines", company, label, len(hits.splitlines()))
    return f"{label} ({mode}):\n" + hits


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



# REMOVED 2026-08-07: _sbti_signal, _cdp_signal, _gri_signal.
#
# All three were DuckDuckGo site: searches wearing the name of a database, not
# reads of the database. That mattered in two ways:
#
#   1. When DDG blocked this IP they went silent -- returning "" for every
#      company. That is indistinguishable in the output from "this company has
#      no SBTi target", so an entire evidence category could vanish from a
#      corpus run while every log line looked normal.
#   2. Even unblocked, a search snippet is weak evidence for a binary fact that
#      the source publishes authoritatively.
#
# Replacements, verified 2026-08-07:
#   SBTi -- files.sciencebasedtargets.org/production/files/companies-excel.xlsx
#           (2.2 MB, weekly refresh, no auth). A bulk join beats one search per
#           company: complete coverage, zero ban surface, no silent zeros.
#   CDP  -- company-level scores are license-only; no free source exists. Do not
#           re-add a search-based stand-in.
#   GRI  -- the Sustainability Disclosure Database is DISCONTINUED (last data
#           2020). There is nothing to query; report discovery is handled by
#           srn_report_source / report_collector instead.

# ── Tier 3: Databases & reports ──────────────────────────────────────────────

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
#
# Wikipedia-bleed-through detection lives in _render_ddg_results now, as a
# PER-RESULT filter (drop the individual off-site/Wikipedia results, keep
# the rest) rather than a whole-batch phrase match on the joined text --
# see that function for why (a real bug: the old whole-text _WIKIPEDIA_BLEED
# phrase list discarded an entire good result set because ONE unrelated
# Wikipedia result in the batch happened to contain a trigger phrase).


# Per-source character budget for DDG snippet text. Raised from the original
# inline 1200 when source URLs began being appended inline, so adding links
# doesn't silently evict snippet text the claim extractor previously saw.
_DDG_TEXT_BUDGET = 1800


def _render_ddg_results(
    results: list,
    query: str = "",
    prefix: str = "",
    min_len: int = 0,
    reject_wikipedia: bool = False,
    company: str = "",
    require_entity: bool = True,
) -> str:
    """Filter + format raw DDG results into evidence text.

    Split out of _ddg_fallback so the live path and the cache-hit path run the
    IDENTICAL filtering. Keeping two copies would let the cached path drift --
    and a drift here is invisible, because both return plausible-looking text.

    Everything in here is ours (entity gate, leakage gate, Wikipedia
    bleed-through, budget). Nothing in here touches the network, which is why
    the cache stores the raw list and re-runs this each time.
    """
    if not results:
        log.debug("DDG [%s] → 0 results", prefix)
        return ""

    # Check for Wikipedia bleed-through -- but only when the query was
    # site:-restricted AND genuinely NONE of the results are on that site.
    # REAL BUG FOUND AND FIXED 2026-08-19: the original check only looked at
    # results[0] -- if the TOP-RANKED result happened to be Wikipedia, the
    # ENTIRE result set was discarded, even when 9 of 10 other results were
    # genuinely on-site. Confirmed live: `site:business-humanrights.org
    # "Saudi Aramco"` returned 5 real business-humanrights.org hits about
    # genuine labor-abuse stories (a real racism-against-a-migrant-worker
    # case, a real exploitative-subcontracting report) alongside 2 unrelated
    # Wikipedia pages that also matched the bare company-name query -- the
    # OLD all-or-nothing checks (reject if results[0] is Wikipedia; reject
    # if the phrase "wikipedia" appears ANYWHERE in the joined text) threw
    # away all 5 real hits because of the 2 unrelated ones, returning "" --
    # indistinguishable from "DDG found nothing" or "we're banned" from the
    # caller's side, when DDG had answered correctly and our own logic
    # discarded the real answer.
    #
    # THE FIX: drop off-site/Wikipedia results INDIVIDUALLY, before joining,
    # rather than reject-or-keep the whole batch. This is strictly better
    # than an all-or-nothing gate on either signal (top-result heuristic or
    # whole-text phrase match) -- it removes exactly the bad results and
    # keeps the good ones, rather than being held hostage by whichever one
    # bad result happens to rank first or contribute a trigger phrase.
    if reject_wikipedia:
        site_match = re.search(r"site:([^\s\"]+)", query)
        target_site = site_match.group(1).lower() if site_match else None

        def _is_wikipedia_bleed(r: dict) -> bool:
            url = (r.get("href") or "").lower()
            if "wikipedia.org" in url:
                return True
            if target_site and target_site not in url:
                # site:-restricted query, but this one result isn't on that
                # site -- real bleed-through for THIS result specifically.
                return True
            return False

        before = len(results)
        results = [r for r in results if not _is_wikipedia_bleed(r)]
        if before != len(results):
            log.info("DDG [%s] → dropped %d/%d off-site/Wikipedia result(s)",
                      prefix, before - len(results), before)
        if not results:
            log.info("DDG [%s] → 0 results left after Wikipedia/off-site filtering, discarding", prefix)
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
    import json as _json

    from ddgs import DDGS, exceptions as ddg_exc

    log.debug("DDG query [%s]: %s", prefix or "raw", query[:120])

    # Cached on the QUERY ONLY, and on the RAW result list -- deliberately
    # before entity/leakage filtering. Those filters are ours and change; the
    # search results are the external thing worth not re-fetching. Caching the
    # filtered string instead would freeze today's filter behaviour into the
    # cache, so a later filter fix would appear to do nothing on a cached run.
    #
    # This is the highest-value cache entry in the pipeline: 9 sources share
    # _DDG_LIMITER at a ~2.75s average gap, so every avoided query is ~2.75s of
    # serialized wall-clock AND one less chance of a RatelimitException.
    _cached = _http_cache.get("ddg", query)
    if _cached is not None:
        try:
            results = _json.loads(_cached)
            log.debug("DDG [%s] → cache hit (%d results)", prefix, len(results))
            return _render_ddg_results(
                results, query=query, prefix=prefix, min_len=min_len,
                reject_wikipedia=reject_wikipedia, company=company,
                require_entity=require_entity)
        except RateLimitTripped:
            # _render_ddg_results can reach _note_rate_limit via entity
            # verification, which fetches. The broad handler below exists for
            # corrupt cache entries and would otherwise swallow the abort and
            # silently refetch -- the exact defect this file has shipped twice.
            raise
        except Exception:
            pass                       # corrupt entry -> fall through and refetch

    for attempt in range(2):
        _DDG_LIMITER.wait()
        try:
            with DDGS() as ddg:
                results = list(ddg.text(query, max_results=5))

            if results:
                try:
                    _http_cache.put("ddg", query, _json.dumps(results))
                except Exception:
                    pass               # caching must never break a fetch

            return _render_ddg_results(
                results, query=query, prefix=prefix, min_len=min_len,
                reject_wikipedia=reject_wikipedia, company=company,
                require_entity=require_entity)
        except ddg_exc.RatelimitException:
            # Feed the SHARED tripwire. DDG raises its own exception type
            # rather than surfacing an HTTP status, so these throttles were
            # invisible to _note_rate_limit -- DDG backs 8 of our sources, and
            # it could have been throttled for an entire corpus run while the
            # abort counter read zero. Counted BEFORE the back-off sleep so a
            # sustained block aborts instead of retrying into it.
            _note_rate_limit("https://duckduckgo.com")   # may raise RateLimitTripped
            log.warning("DDG rate-limited on query [%s] (attempt %d) — backing off", prefix, attempt + 1)
            if attempt == 0:
                time.sleep(8 + random.uniform(0, 4))
        except RateLimitTripped:
            raise      # same swallow trap as _get -- must outrun `except Exception`
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
        # news_api REMOVED: NewsAPI's free tier is a 100-request DAILY QUOTA and
        # we spend one per company. Once exhausted every call returns 429
        # regardless of pacing, so no limiter helps -- it aborted a 150-company
        # run at 1/150 when 5 workers each got 429 on their first company.
        # Re-enable only with a paid key or a per-run budget smaller than the
        # remaining daily allowance.
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
        # S-pillar labor/HR trade press -- direct RSS, NOT the Google News
        # site: trick (see _DIRECT_RSS_SOURCES docstring for why that
        # distinction matters). Added 2026-08-19 to close a measured gap:
        # 32% of well-known public companies got zero S-pillar claims from
        # the sources above.
        "hr_dive":          lambda: _direct_rss_signal(company, *_DIRECT_RSS_SOURCES["hr_dive"]),
        "hr_grapevine":     lambda: _direct_rss_signal(company, *_DIRECT_RSS_SOURCES["hr_grapevine"]),
        "personnel_today":  lambda: _direct_rss_signal(company, *_DIRECT_RSS_SOURCES["personnel_today"]),
        # "sbti"/"cdp"/"gri" REMOVED 2026-08-07 -- see the note where their
        # functions were defined. They were DDG searches, not database reads:
        # they returned "" for every company whenever DDG blocked us, which is
        # indistinguishable from "no target/score/report exists". SBTi is now
        # served by a bulk file; CDP is license-only; GRI's database is
        # discontinued.
        # "wikipedia" DISABLED -- measured on the frozen corpus (tune+holdout,
        # n=393): 74 fetches produced 1 kept claim (0.01 claims/fetch), the
        # worst yield of any source. It also dominates our rate-limit exposure:
        # during a 10-worker run it produced 140 of 145 total HTTP 429s, because
        # _wikipedia_signal probes up to 5 slug variants per company. Pacing it
        # safely costs ~15s per company (3s limiter gap x 5 slugs) -- roughly a
        # third of total fetch time -- to obtain almost no evidence.
        # KNOWN COST, measured rather than assumed: scoring_agent.py uses the
        # wikipedia signal as a THIRD-tier country fallback (after metadata and
        # the caller's own value). Across 1,856 frozen records, 874 had no
        # country in metadata and only 34 of those (1.8% of all records) had a
        # wikipedia signal that could have rescued them -- and country still
        # falls through to the baseline's alias -> regional -> global chain, so
        # none of them lose a baseline entirely. Backtests pass country
        # explicitly from the truth row, so they are unaffected.
        # Re-enable by uncommenting if a use case for the summary text appears;
        # _wikipedia_signal itself is left intact.
        # "wikipedia":      lambda: _wikipedia_signal(company),
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

    # Multi-signal collectors -- unlike the single-string tasks above, each of
    # these returns its OWN {signal_name: text} dict (enforcement_collector
    # queries 3 sources, sec_fulltext_collector up to 15 phrases), so they run
    # as their own futures and get merged in rather than assigned to one key.
    # Both are read-only, adjudicated-fact sources (regulator/court records,
    # SEC filings) -- the "risk events" signal this pipeline was missing:
    # almost every other source here is company self-disclosure, which is
    # ~97% positive-polarity on the frozen corpus and cannot separate a good
    # company from a bad one. See enforcement_collector.py's module docstring.
    #
    # Imported locally, not at module level: both modules import back from
    # THIS file (_get, RateLimitTripped, _RateLimiter), so a top-level import
    # here would be circular.
    from agentic_estimation.layer_1.enforcement_collector import fetch_enforcement_signals
    from agentic_estimation.layer_1.sec_fulltext_collector import fetch_sec_fulltext_signals
    from agentic_estimation.layer_1.report_collector import fetch_report_signals

    # report_collector: REAL sustainability-report PDFs (SRN index, on-demand
    # download + OpenDataLoader/pdfplumber parse), not the "sustainability_report"
    # key above -- that one is a DuckDuckGo search snippet mislabeled as a
    # report (confirmed live 2026-08-18: for Strandberg Guitars it returned a
    # guitar-forum thread, not an actual disclosure document). Until this
    # wiring, fetch_report_signals was ONLY ever called from
    # calibration_harness.py -- the real production graph never fetched a
    # single real PDF for any company, backtest or live. One company, at
    # most 2 reports, on-demand only when this company is actually being
    # scored (see fetch_report_signals docstring) -- bounded, not a bulk
    # crawl. Never raises (report_collector.py's own discipline -- an
    # unreachable publisher host is a missing signal, not a failed run).
    multi_tasks: dict[str, callable] = {
        "enforcement": lambda: fetch_enforcement_signals(company),
        "sec_fulltext": lambda: fetch_sec_fulltext_signals(company),
        "report_pdf": lambda: fetch_report_signals(company),
    }

    signals: dict[str, str] = {}

    log_header(log, "Signal Agent", company=company, industry=industry or "N/A",
               sources=len(tasks) + len(multi_tasks))
    log.info("[%s] starting signal fetch (%d sources)", company, len(tasks) + len(multi_tasks))

    with ThreadPoolExecutor(max_workers=18) as pool:
        futures = {pool.submit(fn): name for name, fn in tasks.items()}
        multi_futures = {pool.submit(fn): name for name, fn in multi_tasks.items()}

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
            except RateLimitTripped:
                raise   # never swallow the abort signal
            except Exception as e:
                log.warning("[%s] %s → exception: %s", company, name, e)

        for fut in as_completed(multi_futures):
            group = multi_futures[fut]
            try:
                got = fut.result() or {}
                for sub_name, text in got.items():
                    signals[sub_name] = text
                    log.info("[%s] %s → OK (%d chars)", company, sub_name, len(text))
                if not got:
                    log.info("[%s] %s → empty", company, group)
            except RateLimitTripped:
                raise   # never swallow the abort signal
            except Exception as e:
                log.warning("[%s] %s → exception: %s", company, group, e)

    log.info("[%s] done — %d/%d sources returned data", company,
              len(signals), len(tasks) + len(multi_tasks))

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
            except RateLimitTripped:
                raise   # never swallow the abort signal
            except Exception:
                results[company] = {}
            done += 1
            msg = f"[signal_agent] {done}/{len(companies)} — {company} ({len(results.get(company, {}))} sources)"
            if on_progress:
                on_progress(msg)

    return results
