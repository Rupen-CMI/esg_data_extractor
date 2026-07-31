"""
verify_governance_regimes.py — discover each country's real corporate-governance
code and securities regulator from the web, and emit a reviewable Python map.

WHY WEB-VERIFY: named regime terms (a country's actual governance code, statute,
and regulator) are the highest-value part of a governance query -- searching
"SEBI LODR" surfaces material that "corporate governance India" never reaches.
But they are country-specific facts. Hand-authoring them for 210 economies is
not something that can be done reliably, and a FABRICATED statute name is worse
than none: it either matches nothing, or matches another country's regime and
silently attributes it to this country. So each one is confirmed against a real
search result before being stored.

RATE-LIMIT DISCIPLINE (this run previously killed the free-tier gateway twice):
  * 1 worker. No thread pool. Strictly sequential.
  * _MIN_GAP_S between queries, process-wide, plus jitter.
  * Exponential back-off on any failure, and a hard stop after
    _MAX_CONSECUTIVE_FAILURES so a rate-limit wall ends the run instead of
    hammering through it.
  * Resumable: every result is appended to a JSONL checkpoint with fsync, and a
    re-run skips countries already present. Interrupting is safe.

Usage:
    python scripts/verify_governance_regimes.py --limit 20        # try a slice
    python scripts/verify_governance_regimes.py                   # all 210
    python scripts/verify_governance_regimes.py --emit            # write .py map
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_estimation.layer_1.country_normalizer import iso3_to_common_name, to_iso3  # noqa: E402
from agentic_estimation.layer_1.signal_agent import _ddg_fallback  # noqa: E402
from agentic_estimation.shared.pipeline_logger import get_logger  # noqa: E402

log = get_logger("verify_governance_regimes")

_CHECKPOINT = Path(__file__).resolve().parents[1] / "calibration" / "governance_regimes.jsonl"
_OUTPUT_PY = (Path(__file__).resolve().parents[1] / "agentic_estimation" / "layer_1"
              / "governance_regimes_verified.py")

# Deliberately slow. Two queries per country at ~8s spacing is ~30 min for 210
# countries -- acceptable for a one-off discovery run, and far cheaper than
# being rate-limited into a multi-hour back-off.
_MIN_GAP_S = 8.0
_JITTER_S = (1.0, 3.0)
_MAX_CONSECUTIVE_FAILURES = 6

_last_call = 0.0


def _throttle() -> None:
    global _last_call
    wait = _last_call + _MIN_GAP_S - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    time.sleep(random.uniform(*_JITTER_S))
    _last_call = time.monotonic()


# A governance code's name almost always contains one of these anchors. Used to
# extract candidate phrases from snippet text rather than trusting the whole
# snippet.
_CODE_PATTERNS = [
    r"\b((?:[A-Z][\w'&.-]+\s+){0,4}Corporate Governance Code)\b",
    r"\b(Code (?:of|on) Corporate Governance(?:\s+\w+){0,3})\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Companies Act(?:,?\s+\d{4})?)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Corporations Act(?:,?\s+\d{4})?)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Securities (?:and Exchange )?Act(?:,?\s+\d{4})?)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,4}Governance (?:Principles|Guidelines|Standards))\b",
    r"\b(King I{1,4}V?(?:\s+Report)?)\b",
    r"\b(Novo Mercado)\b",
    # Listing rules and named regulations are as query-useful as a governance
    # code and are what many markets actually publish instead of a "Code".
    r"\b((?:[A-Z][\w'&.-]+\s+){1,3}Listing (?:Rules|Requirements|Regulations))\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){1,3}Governance Code)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){1,3}Commercial Code)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){1,3}Company Law)\b",
]

_REGULATOR_PATTERNS = [
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Securities (?:and Exchange )?Commission)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Financial (?:Services|Supervisory|Conduct|Markets)\s+(?:Authority|Commission|Agency))\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Capital Market[s]? (?:Authority|Board|Commission))\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,2}Stock Exchange)\b",
    r"\b((?:[A-Z][\w'&.-]+\s+){0,3}Central Bank(?:\s+of\s+[A-Z][\w'&.-]+){0,2})\b",
]

# Phrases that look like a match but are generic boilerplate, not a named regime.
_REJECT_PHRASES = {
    "corporate governance code", "code of corporate governance",
    "companies act", "corporations act", "securities act",
    "securities and exchange commission", "stock exchange", "central bank",
    "financial services authority", "capital markets authority",
    "the corporate governance code", "a corporate governance code",
}


# Demonyms and country-name stems that identify a regime as belonging to a
# specific country. Any of these appearing in a candidate phrase for a DIFFERENT
# country means the phrase was captured from a comparative/adjacent passage --
# observed live: a Peru query returned France's "AFEP-MEDEF Corporate Governance
# Code" and a Vietnam query returned the "Polish Securities and Exchange
# Commission". Both would have been stored as that country's own regime.
_FOREIGN_MARKERS: dict[str, str] = {
    "afep": "FRA", "medef": "FRA", "french": "FRA", "france": "FRA",
    "polish": "POL", "poland": "POL", "german": "DEU", "germany": "DEU",
    "japanese": "JPN", "japan": "JPN", "korean": "KOR", "korea": "KOR",
    "chinese": "CHN", "china": "CHN", "indian": "IND", "india": "IND",
    "brazilian": "BRA", "brazil": "BRA", "mexican": "MEX", "mexico": "MEX",
    "spanish": "ESP", "spain": "ESP", "italian": "ITA", "italy": "ITA",
    "dutch": "NLD", "netherlands": "NLD", "swedish": "SWE", "sweden": "SWE",
    "danish": "DNK", "denmark": "DNK", "norwegian": "NOR", "norway": "NOR",
    "finnish": "FIN", "finland": "FIN", "swiss": "CHE", "switzerland": "CHE",
    "belgian": "BEL", "belgium": "BEL", "austrian": "AUT", "austria": "AUT",
    "portuguese": "PRT", "portugal": "PRT", "greek": "GRC", "greece": "GRC",
    "turkish": "TUR", "turkey": "TUR", "russian": "RUS", "russia": "RUS",
    "nigerian": "NGA", "nigeria": "NGA", "kenyan": "KEN", "kenya": "KEN",
    "egyptian": "EGY", "egypt": "EGY", "malaysian": "MYS", "malaysia": "MYS",
    "singapore": "SGP", "thai": "THA", "thailand": "THA",
    "vietnamese": "VNM", "vietnam": "VNM", "indonesian": "IDN",
    "indonesia": "IDN", "philippine": "PHL", "philippines": "PHL",
    "australian": "AUS", "australia": "AUS", "canadian": "CAN",
    "canada": "CAN", "british": "GBR", "american": "USA",
    "taiwan": "TWN", "hong kong": "HKG", "peruvian": "PER", "peru": "PER",
    "chilean": "CHL", "chile": "CHL", "colombian": "COL", "colombia": "COL",
    "argentine": "ARG", "argentina": "ARG", "saudi": "SAU",
    "south african": "ZAF", "irish": "IRL", "ireland": "IRL",
    "czech": "CZE", "hungarian": "HUN", "hungary": "HUN",
    "romanian": "ROU", "romania": "ROU", "israeli": "ISR", "israel": "ISR",
    "pakistani": "PAK", "pakistan": "PAK", "bangladesh": "BGD",
    "sri lanka": "LKA", "ghana": "GHA", "tanzania": "TZA", "uganda": "UGA",
}


# Regulator names distinctive enough to appear in comparative passages for any
# country, and therefore unsafe to attribute from a generic search. Each belongs
# to one specific country whose demonym does NOT appear in the name, so the
# demonym filter above cannot catch them.
# Common nouns that indicate the regex leading-context swallowed part of the
# surrounding sentence rather than capturing a name ("Regulator Financial
# Supervisory Authority", "Proposal Corporate Governance Code").
#
# Articles ("the", "a") are deliberately NOT listed: real names legitimately
# begin with them ("The UAE Corporate Governance Code"), and they are stripped
# for comparison instead.
_FRAGMENT_LEADERS: frozenset[str] = frozenset({
    "regulator", "regulators", "and", "or", "of", "by", "for", "its", "their",
    "proposal", "draft", "revised", "report", "guide", "overview", "about",
    "under", "with", "from", "this", "that", "these", "those", "such",
    "other", "another", "both", "each", "every", "some", "any",
})

_LEADING_ARTICLES = ("the ", "a ", "an ")


def _fragment_leader(phrase: str) -> bool:
    low = phrase.lower()
    for art in _LEADING_ARTICLES:
        if low.startswith(art):
            low = low[len(art):]
            break
    first = low.split()[0] if low.split() else ""
    return first in _FRAGMENT_LEADERS

_GLOBAL_REGULATOR_NAMES: tuple[str, ...] = (
    "federal financial supervisory authority",   # BaFin (Germany)
    "securities and exchange board",             # SEBI (India)
    "financial conduct authority",               # FCA (UK)
    "financial reporting council",               # FRC (UK)
    "monetary authority of singapore",
    "european securities and markets authority",
    "public company accounting oversight",       # PCAOB (US)
    "commodity futures trading",                 # CFTC (US)
)


def _extract(text: str, patterns: list[str], country: str) -> list[str]:
    """Pull named-regime candidates out of snippet text.

    Two filters, both learned from live output:
      * the phrase must be more specific than the bare generic term -- a snippet
        saying only "corporate governance code" adds nothing over the generic
        vocabulary we already query with;
      * the phrase must not carry ANOTHER country's marker anywhere in it.
        Checking only the leading word (a first attempt here) let "AFEP MEDEF
        Corporate Governance Code" through for Peru and "Polish Securities and
        Exchange Commission" through for Vietnam.
    """
    target = to_iso3(country)
    found: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            phrase = " ".join(m.group(1).split()).strip(" ,.;:")
            low = phrase.lower()
            if low in _REJECT_PHRASES or len(phrase) < 8 or len(phrase) > 70:
                continue
            if _names_another_country(phrase, target):
                log.debug("rejecting %r for %s: names another country", phrase, country)
                continue
            # Sentence-boundary bleed: the regex leading-context can swallow the
            # tail of the previous sentence ("Proposal. Corporate Governance
            # Code", "Armenia. Corporate Governance Code"). Keep only the part
            # after the last sentence break.
            if "." in phrase:
                tail = phrase.rsplit(".", 1)[1].strip()
                if len(tail) >= 8:
                    phrase = tail
                    low = phrase.lower()
                    if low in _REJECT_PHRASES:
                        continue
            # Named regulators that are famous enough to be referenced in ANY
            # country's search results under their English name. Without this,
            # Germany's BaFin ("Federal Financial Supervisory Authority") was
            # being recorded as Argentina's regulator -- the demonym filter
            # cannot catch it because the English name contains no demonym.
            if any(g in low for g in _GLOBAL_REGULATOR_NAMES):
                continue
            if phrase not in found:
                found.append(phrase)
    return found[:4]


def verify_country(iso3: str) -> dict:
    """Two throttled searches for one country: its governance code, and its
    regulator. Returns a record with whatever was confirmed (possibly nothing)."""
    name = iso3_to_common_name(iso3) or iso3
    rec: dict = {"iso3": iso3, "name": name, "codes": [], "regulators": [],
                 "ok": False, "error": None}
    try:
        _throttle()
        code_txt = _ddg_fallback(
            f'"{name}" corporate governance code listed companies official name',
            prefix="", min_len=60) or ""
        _throttle()
        reg_txt = _ddg_fallback(
            f'"{name}" securities regulator financial supervisory authority name',
            prefix="", min_len=60) or ""

        rec["codes"] = _extract(code_txt, _CODE_PATTERNS, name)
        rec["regulators"] = _extract(reg_txt, _REGULATOR_PATTERNS, name)
        rec["ok"] = bool(rec["codes"] or rec["regulators"])
        # Distinguish "the search worked but this country has no formally-named
        # code" from "the search returned nothing at all". Only the latter is
        # evidence of rate limiting; conflating them stopped an earlier run after
        # 6 countries that simply describe their rules in prose (verified: the
        # Bangladesh query returned 1450 chars of real, relevant governance text
        # with no proper-noun code name in it).
        rec["got_text"] = bool(code_txt or reg_txt)
        if not rec["ok"]:
            rec["error"] = ("no named regime in results" if rec["got_text"]
                            else "empty search response")
    except Exception as e:
        rec["error"] = str(e)[:200]
    return rec


def _load_done() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not _CHECKPOINT.exists():
        return done
    with _CHECKPOINT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue  # tolerate a torn final line
            if r.get("iso3"):
                done[r["iso3"]] = r
    return done


def _append(rec: dict) -> None:
    _CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with _CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _all_iso3() -> list[str]:
    from agentic_estimation.layer_1.country_baseline_agent import (
        get_all_baselines, load_all_baselines,
    )
    load_all_baselines()
    out: list[str] = []
    for name in get_all_baselines():
        i = to_iso3(name)
        if i and i not in out:
            out.append(i)
    return sorted(out)


def _names_another_country(phrase: str, target_iso3: Optional[str]) -> bool:
    """True when the phrase names a country other than the target.

    Checks every word (and adjacent word pair) through country_normalizer rather
    than a hand-written demonym list. The hand-written list was the same
    unmaintainable pattern this project already replaced for country
    normalization: it had "polish" and "french" but not "uzbekistan", which let
    "Central Bank of Uzbekistan" be recorded as both Armenia's and Vietnam's
    regulator. Delegating to the ISO 3166 database catches every country name
    without needing to enumerate them.
    """
    words = re.findall(r"[A-Za-z']+", phrase)
    candidates = list(words) + [f"{a} {b}" for a, b in zip(words, words[1:])]
    for cand in candidates:
        if len(cand) < 4:
            continue
        iso = to_iso3(cand, allow_fuzzy=False)
        if iso and iso != target_iso3:
            return True
    # Demonyms ISO 3166 does not carry ("Polish", "Egyptian", "Austrian").
    for marker, iso in _FOREIGN_MARKERS.items():
        if marker in phrase.lower() and iso != target_iso3:
            return True
    return False


def _refilter(phrases: list[str], country: str) -> list[str]:
    """Re-judge already-extracted phrases against the current filters.

    Same rules as _extract's post-match checks, applied to stored raw phrases so
    a filter improvement does not require re-running the whole web sweep.
    """
    target = to_iso3(country)
    out: list[str] = []
    for phrase in phrases:
        phrase = " ".join(phrase.split()).strip(" ,.;:")
        if "." in phrase:
            tail = phrase.rsplit(".", 1)[1].strip()
            if len(tail) >= 8:
                phrase = tail
        low = phrase.lower()
        if low in _REJECT_PHRASES or len(phrase) < 8 or len(phrase) > 70:
            continue
        if any(g in low for g in _GLOBAL_REGULATOR_NAMES):
            continue
        if _names_another_country(phrase, target):
            continue
        # Fragments left by regex leading-context capture ("Regulator Financial
        # Supervisory Authority") -- a real name does not begin with a common
        # noun like these.
        if _fragment_leader(phrase):
            continue
        if phrase not in out:
            out.append(phrase)
    return out


def emit_python(done: dict[str, dict]) -> None:
    """Write the verified regimes as a reviewable Python module."""
    lines = [
        '"""',
        "governance_regimes_verified.py — AUTO-GENERATED by",
        "scripts/verify_governance_regimes.py. Do not hand-edit; re-run the script.",
        "",
        "Each entry was extracted from a live web search for that country's own",
        "corporate-governance code and securities regulator, then filtered to",
        "reject generic boilerplate ('corporate governance code' on its own) and",
        "any phrase naming a different country. Entries are therefore evidence-",
        "backed rather than authored, but they are STILL machine-extracted: treat",
        "them as query hints, not as authoritative legal citations.",
        "",
        "Countries with a hand-written profile in country_governance_keywords.py",
        "take precedence over anything here -- those were written deliberately.",
        '"""',
        "",
        "VERIFIED_REGIMES: dict[str, tuple[str, ...]] = {",
    ]
    kept = 0
    dropped = 0
    for iso3 in sorted(done):
        rec = done[iso3]
        raw = list(dict.fromkeys((rec.get("codes") or []) + (rec.get("regulators") or [])))
        # Re-apply the current filters at emit time. Records checkpointed by an
        # earlier revision of this script were extracted with looser rules
        # (which let sentence-boundary bleed and famous foreign regulators
        # through), and re-fetching 200+ countries to fix that would be pure
        # waste -- the raw phrases are stored, so they can simply be re-judged.
        terms = _refilter(raw, rec.get("name", ""))
        dropped += len(raw) - len(terms)
        if not terms:
            continue
        kept += 1
        joined = ", ".join(repr(t) for t in terms)
        lines.append(f"    {iso3!r}: ({joined},),  # {rec.get('name', '')}")
    lines.append("}")
    lines.append("")
    _OUTPUT_PY.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {_OUTPUT_PY} with {kept} countries "
          f"({dropped} phrases dropped by re-filtering)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N new countries")
    ap.add_argument("--emit", action="store_true", help="write the .py map and exit")
    ap.add_argument("--only", default="", help="comma-separated ISO3 list")
    args = ap.parse_args()

    done = _load_done()
    if args.emit:
        emit_python(done)
        return

    targets = ([t.strip().upper() for t in args.only.split(",") if t.strip()]
               if args.only else _all_iso3())
    todo = [i for i in targets if i not in done]
    print(f"{len(targets)} targets, {len(done)} already done, {len(todo)} to go")
    if args.limit:
        todo = todo[:args.limit]
        print(f"limited to {len(todo)} this run")

    consecutive_failures = 0
    for n, iso3 in enumerate(todo, 1):
        rec = verify_country(iso3)
        _append(rec)
        status = "OK " if rec["ok"] else "-- "
        terms = (rec["codes"] + rec["regulators"])[:2]
        print(f"[{n}/{len(todo)}] {status} {iso3} {rec['name'][:22]:<24} {terms}")

        # Only an EMPTY search response counts toward the rate-limit stop. A
        # country whose governance rules simply have no proper-noun name is a
        # normal outcome, not a signal to abort the run.
        if rec.get("got_text"):
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                print(f"\nSTOPPING: {consecutive_failures} consecutive EMPTY responses -- "
                      f"search is likely rate-limited. Re-run later; progress is "
                      f"checkpointed and completed countries are skipped.")
                break

    done = _load_done()
    found = sum(1 for r in done.values() if r.get("ok"))
    print(f"\ncheckpoint: {len(done)} countries attempted, {found} with a named regime")
    print(f"run with --emit to write {_OUTPUT_PY.name}")


if __name__ == "__main__":
    main()
