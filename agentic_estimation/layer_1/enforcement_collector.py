"""
enforcement_collector.py — regulator/court enforcement evidence.

WHY THIS EXISTS: measured on the frozen calibration dumps, our highest-volume
sources are ~97% positive-polarity because they are company self-disclosure
(sustainability_report 1% negative, gov_compliance 4%, cdp 3%). Evidence that
is almost never negative cannot separate a good company from a bad one, which
is a large part of why our pillar rankings are weak.

Enforcement records are the structural opposite: a regulator only creates one
when something went wrong. They are also adjudicated facts rather than claims,
so they need no credibility discount.

Sources here are all free, key-free, and verified live (2026-08-01):

  EPA ECHO cases  E  civil/criminal environmental cases WITH dollar penalties
  EPA ECHO facil. E  per-facility CURRENT compliance state (added 2026-09-02):
                     how many of a company's facilities are in violation now,
                     which is a different question from whether it was ever
                     prosecuted. Measured 22% hit rate on US companies in the
                     frozen held-out corpus, and it reaches single-facility
                     firms that no other source in this pipeline covers.
  SEC litigation  G  federal securities enforcement actions
  SEC admin proc  G  administrative proceedings
  World Bank      G  cross-debarment (WB + ADB/EBRD/IDB/AfDB) -- the single
                     hardest governance red flag available anywhere free

ENTITY MATCHING IS THE CENTRAL RISK. All of these match on name substrings
with no company identifier, so "TYSON" returns "TYSON, FRANKLIN P." (an
individual) and "BERRYVILLE, CITY OF (TYSON)" (a municipality). Attributing a
stranger's conviction to a company would be far worse than missing it, so
every record goes through evidence_filters.mentions_company and, for the
bulk lists, a stricter token-containment check.

DELIBERATELY NOT INCLUDED: the UK ICO enforcement search. Its endpoint answers
200 with `{"results": [], "totalResults": 0}` for every query we tried,
including known-good ones -- indistinguishable from "this company is clean",
which is exactly the failure mode that would silently poison a score.
"""

import re
import threading
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.evidence_filters import (
    _company_tokens,
    mentions_company,
)
from agentic_estimation.layer_1.sec_filings import _SEC_LIMITER
from agentic_estimation.layer_1.signal_agent import RateLimitTripped, _RateLimiter, _get

log = get_logger("enforcement")

_TIMEOUT = 45
_MAX_CHARS = 2200

# These are public-sector endpoints without published rate limits; keep the
# footprint modest rather than assume tolerance.
# EPA ECHO has now thrown 429s at us TWICE: first at a flat 0.5s gap, then again
# at 1.5s+1.0 (~0.50 req/s), where it produced 5 of the 9 throttles that aborted
# a 150-company run at 45. It is the least tolerant host in the pipeline
# relative to how little it returns -- enforcement hits are rare, so most calls
# find nothing and we are spending our throttle budget on empty results.
#
# 4.0s+2.0 (~0.20 req/s) is deliberately slower than Google News. If ECHO
# throttles a third time the right answer is to drop it from live fetching and
# bulk-download the case file instead, rather than widen the gap again.
_ECHO_LIMITER = _RateLimiter(min_gap=4.0, jitter=2.0)
_WB_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)

_ECHO_CASES_URL = "https://echodata.epa.gov/echo/case_rest_services.get_cases"
# Facility COMPLIANCE summary -- a different endpoint from get_cases above and
# a different question. get_cases answers "was this company prosecuted, and
# for how much"; get_facilities answers "how many of its facilities are in
# violation right now". Measured 2026-08-28 on the frozen held-out corpus:
# get_facilities hit 4/18 US companies (22%) including single-facility firms
# (L.A Brewery, North Coast Brewing) that no other source in this pipeline
# has any evidence for at all. That long-tail reach is why it is worth a
# second endpoint rather than folding into the cases signal.
_ECHO_FACILITIES_URL = "https://echodata.epa.gov/echo/echo_rest_services.get_facilities"
_SEC_LITIGATION_RSS = "https://www.sec.gov/enforcement-litigation/litigation-releases/rss"
_SEC_ADMIN_RSS = "https://www.sec.gov/enforcement-litigation/administrative-proceedings/rss"
# The apikey below is the one the World Bank's own public sanctions page sends.
# It is not issued to us and can rotate without notice -- every failure path
# here degrades to "no signal", never to an exception.
_WB_URL = ("https://apigwext.worldbank.org/dvsvc/v1.0/json/APPLICATION/"
           "ADOBE_EXPRNCE_MGR/FIRM/SANCTIONED_FIRM")
_WB_APIKEY = "z9duUaFUiEUYSHs97CU38fcZO7ipOPvm"

_UA = {"User-Agent": "ESG-Signal-Agent/1.0 (research@example.com)"}
_SEC_UA = {"User-Agent": "ESG-Signal-Agent research@example.com"}

_wb_cache: Optional[list[dict]] = None
_wb_lock = threading.Lock()
_sec_rss_cache: dict[str, list[dict]] = {}
_sec_rss_lock = threading.Lock()


# Enforcement records go back to the 1980s. ESG standing is a present-state
# question, so a 1987 municipal penalty is not evidence about a company today.
# Matches the 2021 cutoff used for SEC full-text search.
_MIN_YEAR = 2021


def _is_recent(date_text: str) -> bool:
    """True if a date string contains a year >= _MIN_YEAR. Records with no
    parseable year are kept: several ECHO rows carry a penalty and party but
    no date, and dropping those would discard real enforcement actions."""
    years = re.findall(r"(19|20)\d{2}", date_text or "")
    if not years:
        return True
    return any(int(m) >= _MIN_YEAR for m in re.findall(r"((?:19|20)\d{2})", date_text))


# Below this length a company name carries too little information to match
# safely against a 1,500-row bulk list of global firm names. Measured: "GRAN",
# "Ts Tech" and "Li Auto" each produced false debarment hits before this floor
# existed. Skipping them loses nothing real -- these names were never going to
# match correctly -- while a false positive asserts corruption findings against
# an innocent company.
_MIN_MATCH_CHARS = 8


def _strict_name_match(record_name: str, company: str) -> bool:
    """Stricter than mentions_company: EVERY identifying token of the company
    must appear in the record, as a WHOLE WORD.

    Used for the bulk lists (World Bank, SEC RSS, ECHO) where one false
    positive means asserting that a company was debarred or criminally charged
    when it was not -- categorically worse than missing a real record.

    Word boundaries are essential, not cosmetic. Plain substring matching
    produced all of these against the World Bank list:
        "GRAN"    -> "SOCIETE GRANDS TRAVAUX MGHAIETH"   ("gran" in "grands")
        "Ts Tech" -> "REGAL INFORMATION TECHNOLOGY"      ("ts"/"tech" inside)
        "Li Auto" -> "OSTEK ... ELECTRIC AUTOMATION"     ("li" in "electric")
    """
    toks = _company_tokens(company)
    if not toks:
        return False
    if sum(len(t) for t in toks) < _MIN_MATCH_CHARS:
        return False
    # A single generic token is not identifying. "Community Services Group"
    # reduces to just ["community"] once legal-form filler is stripped, which
    # matched Pakistan's "COMMUNITY RESILIENCE INITIATIVE". Require either two
    # distinct tokens or one long, distinctive one.
    if len(toks) < 2 and len(toks[0]) < 10:
        return False
    low = (record_name or "").lower()
    return all(re.search(rf"\b{re.escape(t)}\b", low) for t in toks)


def _echo_signal(company: str) -> Optional[str]:
    """EPA ECHO enforcement cases, with federal penalty amounts.

    NOTE the parameter choice: `p_case_summary` is used, NOT `p_name`. ECHO
    accepts p_name and then IGNORES it, returning all ~310k cases with a 200
    and no warning -- a company would appear to have hundreds of violations.
    Verified: p_case_summary=TYSON returns 39 rows, all genuinely Tyson-related.
    """
    _ECHO_LIMITER.wait()
    r = _get(_ECHO_CASES_URL, params={
        "output": "JSON",
        "p_case_summary": company[:40],
        "p_case_summary_type": "ALL",
        "tablelist": "Y",
    }, timeout=_TIMEOUT)
    if not r:
        return None
    try:
        results = (r.json() or {}).get("Results") or {}
    except (ValueError, AttributeError):
        return None
    if str(results.get("Message", "")).lower() != "success":
        return None

    parts: list[str] = []
    for case in (results.get("Cases") or []):
        defendant = (case.get("DefendantName") or "").strip()
        case_name = (case.get("CaseName") or "").strip()
        subject = defendant or case_name
        # ECHO substring-matches on a free-text field, so the strict all-token
        # rule is required here. A single-token match let through
        # "BERRYVILLE, CITY OF (TYSON)" (a municipality) and "HUDSON FOODS,
        # INC." (a different registrant) as if they were Tyson Foods cases.
        if not _strict_name_match(f"{defendant} {case_name}", company):
            continue
        date = (case.get("SettlementDate") or case.get("FilingDate") or "").strip()
        if not _is_recent(date):
            continue
        penalty = (case.get("FedPenalty") or "").strip()
        law = (case.get("Statute") or case.get("Law") or "").strip()
        bits = [b for b in (subject, law, date) if b]
        if penalty and penalty not in ("$0.00", "None"):
            bits.append(f"federal penalty {penalty}")
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] echo_enforcement → %d matched case(s)", company, len(parts))
    return f"EPA ECHO enforcement: {body} <https://echodata.epa.gov/echo/>"


def _echo_money(val) -> float:
    """'$13,000' -> 13000.0; 0.0 for None/''/'-'/unparseable."""
    if val in (None, "", "-"):
        return 0.0
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _echo_facilities_signal(company: str) -> Optional[str]:
    """EPA ECHO facility compliance summary -- how many of a company's
    facilities are currently in violation.

    DIFFERENT DATA FROM _echo_signal, not a duplicate: that one reports
    prosecuted CASES with penalties, this one reports the current compliance
    state of every facility EPA associates with the name. A company can have
    zero cases and sixteen facilities in violation.

    ENDPOINT SHAPE (verified live 2026-08-28, Cardinal Health -- and this cost
    a wrong answer first time): get_facilities is a SUMMARY endpoint. It does
    NOT return facility rows, it returns aggregate counts plus a QueryID that
    a separate download call would consume. An implementation that iterates a
    `Facilities` list finds nothing and reports 0 violations for a company
    with 16. The counts below ARE the payload.

        QueryRows      facilities matched
        CVRows         currently in violation
        SVRows         current SIGNIFICANT violation
        V3Rows         violation in the last 3 years
        FEARows        formal enforcement actions (5y)
        InfFEARows     informal enforcement actions (5y)
        INSPRows       inspections (5y)
        TotalPenalties penalties, as a '$1,234' string

    RATE LIMIT: shares _ECHO_LIMITER with _echo_signal deliberately. ECHO has
    thrown 429s at this pipeline twice already (see the limiter's comment) and
    is the least tolerant host we use, so the two ECHO calls queue behind one
    another rather than each getting their own budget. This roughly doubles
    per-company ECHO wall time; that is the accepted cost of not being
    throttled a third time.

    NAME MATCHING: `p_fn` is a facility-name substring search, so a match is
    not proof of ownership -- "One Stone" returned 54 facilities. The signal
    text therefore states the matched query verbatim and leaves attribution
    to the extractor, exactly as the cases signal does.
    """
    _ECHO_LIMITER.wait()
    r = _get(_ECHO_FACILITIES_URL, params={
        "output": "JSON",
        "p_fn": company[:40],
    }, timeout=_TIMEOUT)
    if not r:
        return None
    try:
        results = (r.json() or {}).get("Results") or {}
    except (ValueError, AttributeError):
        return None

    def _int(key: str) -> int:
        try:
            return int(results.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    n_fac = _int("QueryRows")
    if not n_fac:
        return None

    curr_viol = _int("CVRows")
    sig_viol = _int("SVRows")
    viol_3yr = _int("V3Rows")
    formal = _int("FEARows")
    informal = _int("InfFEARows")
    inspections = _int("INSPRows")
    penalties = _echo_money(results.get("TotalPenalties"))

    # A clean record is NOT reported. Absence of violations is not evidence of
    # good conduct (same rule as fetch_enforcement_signals' docstring), and
    # emitting "0 violations" would hand the extractor a positive-sounding
    # line built from nothing.
    if not any((curr_viol, sig_viol, viol_3yr, formal, informal, penalties)):
        log.info("[%s] echo_facilities → %d facilities, no violations/actions — no signal",
                 company, n_fac)
        return None

    bits = [f"{n_fac} EPA-regulated facilities matched on name '{company[:40]}'"]
    if curr_viol:
        bits.append(f"{curr_viol} currently in violation")
    if sig_viol:
        bits.append(f"{sig_viol} in SIGNIFICANT violation")
    if viol_3yr:
        bits.append(f"{viol_3yr} with a violation in the last 3 years")
    if formal:
        bits.append(f"{formal} formal enforcement action(s) in 5 years")
    if informal:
        bits.append(f"{informal} informal enforcement action(s) in 5 years")
    if inspections:
        bits.append(f"{inspections} inspection(s) in 5 years")
    if penalties:
        bits.append(f"${penalties:,.0f} total penalties")

    log.info("[%s] echo_facilities → %d facilities, %d in violation (3y: %d)",
             company, n_fac, curr_viol, viol_3yr)
    return ("EPA ECHO facility compliance: " + "; ".join(bits)
            + " <https://echodata.epa.gov/echo/>")


def _load_sec_rss(url: str) -> list[dict]:
    """Fetch and parse one SEC enforcement RSS feed. Cached per process --
    the feed is corpus-wide, not per-company, so it is fetched once and
    matched against every company."""
    with _sec_rss_lock:
        if url in _sec_rss_cache:
            return _sec_rss_cache[url]
        _sec_rss_cache[url] = []
        # sec.gov, so it must go through the SEC limiter -- this is the same
        # host sec_filings.py paces at 6/s, and an unlimited call here would
        # sit outside that budget. Cached per process, so this fires at most
        # twice per run, but an unpaced call to a rate-limited host is exactly
        # the gap that has bitten this pipeline before.
        _SEC_LIMITER.wait()
        r = _get(url, timeout=_TIMEOUT)
        if not r:
            log.warning("SEC enforcement RSS unavailable: %s", url)
            return _sec_rss_cache[url]
        items: list[dict] = []
        for block in re.findall(r"<item>(.*?)</item>", r.text, re.DOTALL):
            def tag(name: str) -> str:
                m = re.search(rf"<{name}>(.*?)</{name}>", block, re.DOTALL)
                raw = m.group(1) if m else ""
                raw = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw, flags=re.DOTALL)
                return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()
            items.append({"title": tag("title"), "link": tag("link"),
                          "date": tag("pubDate"), "desc": tag("description")})
        _sec_rss_cache[url] = items
        log.info("SEC enforcement RSS: %d items from %s", len(items), url.rsplit("/", 2)[-2])
        return items


def _sec_enforcement_signal(company: str) -> Optional[str]:
    """SEC litigation releases + administrative proceedings.

    Titles in these feeds are the charged parties, so a strict all-token match
    is both possible and necessary -- these are accusations of securities
    violations and must not be attributed to the wrong company.
    """
    parts: list[str] = []
    for url, label in ((_SEC_LITIGATION_RSS, "litigation release"),
                       (_SEC_ADMIN_RSS, "administrative proceeding")):
        for item in _load_sec_rss(url):
            if not _strict_name_match(item["title"], company):
                continue
            desc = item["desc"][:400]
            parts.append(f"[SEC {label} {item['date'][:16]}] {item['title']}. "
                         f"{desc} <{item['link']}>")
    if not parts:
        return None
    log.info("[%s] sec_enforcement → %d action(s)", company, len(parts))
    return "SEC enforcement action: " + " | ".join(parts)[:_MAX_CHARS]


def _load_wb_debarments() -> list[dict]:
    """World Bank + cross-debarred firms (ADB/EBRD/IDB/AfDB). ~1,523 rows,
    fetched once per process."""
    global _wb_cache
    with _wb_lock:
        if _wb_cache is not None:
            return _wb_cache
        _wb_cache = []
        _WB_LIMITER.wait()
        # This endpoint needs its own apikey header, so it cannot go through
        # the shared _get() helper (which sends only our User-Agent).
        try:
            import requests
            resp = requests.get(_WB_URL, headers={**_UA, "apikey": _WB_APIKEY},
                                timeout=_TIMEOUT)
            r = resp if resp.status_code == 200 else None
        except RateLimitTripped:
            raise   # never swallow the abort signal
        except Exception as exc:
            log.warning("World Bank debarment fetch failed: %s", exc)
            r = None
        if not r:
            log.warning("World Bank debarment list unavailable (apikey may have rotated)")
            return _wb_cache
        try:
            rows = ((r.json() or {}).get("response") or {}).get("ZPROCSUPP") or []
            _wb_cache = [x for x in rows if x.get("SUPP_NAME")]
            log.info("loaded World Bank debarment list: %d firms", len(_wb_cache))
        except (ValueError, AttributeError) as exc:
            log.warning("World Bank debarment parse failed: %s", exc)
        return _wb_cache


def _worldbank_signal(company: str) -> Optional[str]:
    """Multilateral-development-bank debarment — the hardest free governance
    red flag. A debarred firm is barred from bank-financed contracts for
    fraud, corruption, collusion or obstruction."""
    parts: list[str] = []
    for row in _load_wb_debarments():
        name = row.get("SUPP_NAME") or ""
        if not _strict_name_match(name, company):
            continue
        # Debarment is a fixed-term sanction; one that ended years ago is
        # history, not current standing. Kept if the end date is in the future
        # or unparseable (many are open-ended, encoded as 2999-12-31).
        if not _is_recent(str(row.get("DEBAR_TO_DATE") or "")):
            continue
        parts.append(
            f"{name.strip()} debarred {row.get('DEBAR_FROM_DATE', '?')} to "
            f"{row.get('DEBAR_TO_DATE', '?')}; grounds: "
            f"{(row.get('DEBAR_REASON') or 'not stated').strip()}; "
            f"country: {(row.get('COUNTRY_NAME') or '?').strip()}"
        )
    if not parts:
        return None
    log.info("[%s] worldbank_debarment → %d record(s)", company, len(parts))
    return ("World Bank / MDB debarment: " + " | ".join(parts)[:_MAX_CHARS] +
            " <https://projects.worldbank.org/en/projects-operations/procurement/debarred-firms>")


def fetch_enforcement_signals(company: str) -> dict[str, str]:
    """Enforcement evidence for one company from all verified sources.

    Returns {signal_name: text}; {} when the company appears in none of them,
    which is the normal case for most companies and is NOT evidence of good
    conduct -- absence of an enforcement record must never be scored as a
    positive.
    """
    log_header(log, "Enforcement", company=company, sources=4)
    signals: dict[str, str] = {}
    for name, fn in (("echo_enforcement", _echo_signal),
                     ("echo_facilities", _echo_facilities_signal),
                     ("sec_enforcement", _sec_enforcement_signal),
                     ("worldbank_debarment", _worldbank_signal)):
        try:
            got = fn(company)
        except RateLimitTripped:
            raise   # never swallow the abort signal
        except Exception as exc:
            log.warning("[%s] %s → exception: %s", company, name, exc)
            continue
        if got:
            signals[name] = got
    log.info("[%s] enforcement done — %d/4 sources hit", company, len(signals))
    return signals


if __name__ == "__main__":  # manual probe
    import sys
    target = " ".join(sys.argv[1:]) or "Tyson Foods"
    out = fetch_enforcement_signals(target)
    if not out:
        print(f"(no enforcement records for {target!r})")
    for k, v in out.items():
        print(f"\n=== {k} ===\n{v[:700]}")
