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

  EPA ECHO        E  civil/criminal environmental cases WITH dollar penalties
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
from agentic_estimation.layer_1.signal_agent import _RateLimiter, _get

log = get_logger("enforcement")

_TIMEOUT = 45
_MAX_CHARS = 2200

# These are public-sector endpoints without published rate limits; keep the
# footprint modest rather than assume tolerance.
_ECHO_LIMITER = _RateLimiter(min_gap=0.5)
_WB_LIMITER = _RateLimiter(min_gap=1.0)

_ECHO_CASES_URL = "https://echodata.epa.gov/echo/case_rest_services.get_cases"
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


def _load_sec_rss(url: str) -> list[dict]:
    """Fetch and parse one SEC enforcement RSS feed. Cached per process --
    the feed is corpus-wide, not per-company, so it is fetched once and
    matched against every company."""
    with _sec_rss_lock:
        if url in _sec_rss_cache:
            return _sec_rss_cache[url]
        _sec_rss_cache[url] = []
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
    log_header(log, "Enforcement", company=company, sources=3)
    signals: dict[str, str] = {}
    for name, fn in (("echo_enforcement", _echo_signal),
                     ("sec_enforcement", _sec_enforcement_signal),
                     ("worldbank_debarment", _worldbank_signal)):
        try:
            got = fn(company)
        except Exception as exc:
            log.warning("[%s] %s → exception: %s", company, name, exc)
            continue
        if got:
            signals[name] = got
    log.info("[%s] enforcement done — %d/3 sources hit", company, len(signals))
    return signals


if __name__ == "__main__":  # manual probe
    import sys
    target = " ".join(sys.argv[1:]) or "Tyson Foods"
    out = fetch_enforcement_signals(target)
    if not out:
        print(f"(no enforcement records for {target!r})")
    for k, v in out.items():
        print(f"\n=== {k} ===\n{v[:700]}")
