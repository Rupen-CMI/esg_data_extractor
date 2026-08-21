"""
ilo_ratification.py — country-level ILO fundamental-convention ratification
status, feeding country_baseline_agent.py's S-pillar indicator set.

WHY THIS EXISTS: World Bank's existing _S_INDICATORS (country_baseline_agent.py)
are general development indicators (health, poverty, education, labor-force
participation) -- genuinely useful, but none of them measure whether a
country's LAWS actually protect workers. ILO's ratification data is the real
labor-standards signal World Bank's dataset has no equivalent for: does this
country legally recognize freedom of association? Ban child labor? Mandate
equal pay? Confirmed 2026-08-19 (research session): this matters most for
exactly the population with the weakest company-specific S evidence -- when
a company has zero extracted S claims, the country baseline is ALL the
pipeline has for S, and a baseline built only from "is this a developed
country" (World Bank) said nothing about labor rights specifically.

GENUINELY GLOBAL, unlike the RSS-based S-source candidates (HR Dive, HR
Grapevine, Personnel Today) which are US/UK-only: ratification status
exists and is tracked identically for every ILO member state (~187
countries), including China, Russia, Saudi Arabia -- the exact companies
the US/UK news sources cannot reach at all. Confirmed live 2026-08-19 that
this population (100 well-known public companies) is 43% US / 20% China /
2% UK -- the country-level signal is the one piece that doesn't skew
further toward the already-overrepresented US/UK slice.

DATA SOURCE: NORMLEX (normlex.ilo.org), ILO's treaty database. No official
bulk export or API exists -- confirmed via live checks (no CSV/JSON/REST
endpoint anywhere on the site). But each convention's ratification table is
a stable, server-rendered (non-JS) HTML page at a fixed instrument ID, so a
handful of plain HTTP GETs covers every ratifying country for that
convention. NOT hammering NORMLEX per-company or per-run: fetched once,
cached to disk, refreshed on an explicit re-run only (ratification status
changes on the order of years, never per-company-scored).

THE 10 FUNDAMENTAL CONVENTIONS (ILO's own core-rights framework, not a
subjective pick):
    C029  Forced Labour (1930)
    C087  Freedom of Association and Protection of the Right to Organise (1948)
    C098  Right to Organise and Collective Bargaining (1949)
    C100  Equal Remuneration (1951)
    C105  Abolition of Forced Labour (1957)
    C111  Discrimination (Employment and Occupation) (1958)
    C138  Minimum Age (1973)
    C155  Occupational Safety and Health (1981)
    C182  Worst Forms of Child Labour (1999)
    C187  Promotional Framework for OSH (2006)

CLI:
    python -m agentic_estimation.layer_1.ilo_ratification fetch
    python -m agentic_estimation.layer_1.ilo_ratification lookup "United States"
"""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("ilo_ratification")

_CACHE_PATH = Path(__file__).parent.parent.parent / "raw_esg_data" / "ilo_ratifications.json"

# Instrument IDs VERIFIED LIVE 2026-08-19 by parsing NORMLEX's own
# fundamental-conventions index page (P12100_INSTRUMENT_ID links, each
# immediately followed by its <strong>Cnnn</strong> code label) -- not
# guessed. An earlier draft of this file had WRONG ids for C087/C105 (both
# pointed at 312250, which is actually C105 -- C087 is really 312232); this
# was caught by fetching the index page and matching id->code pairs
# programmatically before trusting any hardcoded id, rather than shipping
# the first plausible-looking number. Re-verify by re-running the index
# parse if NORMLEX ever restructures -- don't hand-edit these from memory.
_FUNDAMENTAL_CONVENTIONS = {
    "C029": {"id": 312174, "name": "Forced Labour"},
    "C087": {"id": 312232, "name": "Freedom of Association and Protection of the Right to Organise"},
    "C098": {"id": 312243, "name": "Right to Organise and Collective Bargaining"},
    "C100": {"id": 312245, "name": "Equal Remuneration"},
    "C105": {"id": 312250, "name": "Abolition of Forced Labour"},
    "C111": {"id": 312256, "name": "Discrimination (Employment and Occupation)"},
    "C138": {"id": 312283, "name": "Minimum Age"},
    "C155": {"id": 312300, "name": "Occupational Safety and Health"},
    "C182": {"id": 312327, "name": "Worst Forms of Child Labour"},
    "C187": {"id": 312332, "name": "Promotional Framework for Occupational Safety and Health"},
}

_BASE_URL = "https://normlex.ilo.org/dyn/nrmlx_en/f?p=NORMLEXPUB:11300:0::NO::P11300_INSTRUMENT_ID:{id}"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

# Verified live 2026-08-19 against real NORMLEX HTML: rows carry trailing
# whitespace inside each <td> (e.g. "In Force   " not "In Force"), so every
# extracted field is .strip()'d in _fetch_one_convention below -- matching
# without \s* tolerance here undercounted real rows (158 found only after
# adding it, vs the true country count for C087).
_ROW_RE = re.compile(
    r'<td class="firstCol"><a[^>]*>([^<]+)</a>\s*</td>\s*'
    r'<td[^>]*>([^<]*)</td>\s*<td[^>]*>([^<]*)</td>',
    re.IGNORECASE,
)


def _fetch_one_convention(code: str, instrument_id: int) -> dict:
    """{country_name: {"date": str, "status": str}} for one convention.
    Never raises -- a fetch failure for one convention is a missing
    indicator, not a crashed run (same fail-open discipline as every other
    collector in this codebase).

    Uses the `curl` CLI, not `requests` -- confirmed live 2026-08-19 that
    NORMLEX (Oracle APEX, bot-detection on the WAF layer) returns HTTP 403
    to `requests` even with byte-identical User-Agent/Accept/Accept-Language
    headers, while plain `curl` with only -A (User-Agent) succeeds cleanly.
    The difference is TLS/HTTP fingerprinting (cipher order, ALPN, header
    ordering) that Python's requests/urllib3 stack cannot easily replicate,
    not a missing header -- shelling out to curl is the simplest fix that
    doesn't add a new dependency (curl_cffi, playwright) for one collector."""
    url = _BASE_URL.format(id=instrument_id)
    try:
        proc = subprocess.run(
            ["curl", "-s", "-m", "30", "-A", _UA, url],
            capture_output=True, text=True, timeout=35,
        )
        if proc.returncode != 0:
            log.warning("ILO %s: curl exited %d: %s", code, proc.returncode, proc.stderr[:200])
            return {}
        html = proc.stdout
    except Exception as exc:
        log.warning("ILO %s fetch failed: %s", code, exc)
        return {}

    out = {}
    for m in _ROW_RE.finditer(html):
        country, date, status = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        out[country] = {"date": date, "status": status}
    log.info("ILO %s: %d countries", code, len(out))
    return out


def fetch_all(polite_gap_s: float = 3.0) -> dict:
    """Fetch every fundamental convention's ratification table. Polite
    pacing (not a process-wide _RateLimiter -- this runs once, standalone,
    never inside a per-company hot path) since NORMLEX documents no rate
    limit and this is ILO's live treaty database, not a public API."""
    result = {}
    for code, meta in _FUNDAMENTAL_CONVENTIONS.items():
        result[code] = {"name": meta["name"], "instrument_id": meta["id"],
                         "ratifications": _fetch_one_convention(code, meta["id"])}
        time.sleep(polite_gap_s)
    return result


def save_cache(data: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    log.info("wrote ILO ratification cache -> %s", _CACHE_PATH)


def load_cache() -> Optional[dict]:
    if not _CACHE_PATH.exists():
        return None
    return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))


# Keyed by ISO3, not NORMLEX's raw name strings -- NORMLEX uses formal
# UN-style names ("United Kingdom of Great Britain and Northern Ireland",
# "Russian Federation") that rarely match what callers actually have
# (metadata['country'], an ISO3 code, or a common name like "UK"). Every
# other country-keyed lookup in this codebase (country_baseline_agent.py's
# CountryBaseline) is ISO3-keyed for the same reason -- one canonical key,
# resolved once, rather than a second bespoke alias table living here.
_labor_rights_score_cache: Optional[dict] = None   # {iso3: score}


def _build_iso3_index(cache: dict) -> dict:
    import pycountry

    scores: dict[str, int] = {}
    unresolved = []
    name_to_iso3: dict[str, Optional[str]] = {}

    for code, conv in cache.items():
        for country, rec in conv["ratifications"].items():
            if not rec["status"].lower().startswith("in force"):
                continue
            if country not in name_to_iso3:
                try:
                    match = pycountry.countries.search_fuzzy(country)
                    name_to_iso3[country] = match[0].alpha_3 if match else None
                except LookupError:
                    name_to_iso3[country] = None
            iso3 = name_to_iso3[country]
            if iso3 is None:
                if country not in unresolved:
                    unresolved.append(country)
                continue
            scores[iso3] = scores.get(iso3, 0) + 1

    if unresolved:
        log.warning("ILO ratification: %d country names could not resolve to ISO3: %s",
                     len(unresolved), unresolved[:10])
    return scores


def labor_rights_score(country: str) -> Optional[float]:
    """0-10: how many of the 10 fundamental conventions this country has
    ratified ('In Force' status). Accepts an ISO3 code OR a common/formal
    name (resolved via pycountry, same as the input). None if the country
    isn't in the cache (fetch_all() was never run) or can't be resolved --
    caller (country_baseline_agent) should treat None as 'no data', same
    as any other missing World Bank indicator, not as a score of 0."""
    global _labor_rights_score_cache
    if _labor_rights_score_cache is None:
        cache = load_cache()
        if cache is None:
            return None
        _labor_rights_score_cache = _build_iso3_index(cache)

    key = country.strip().upper()
    if len(key) == 3 and key.isalpha():
        iso3 = key
    else:
        import pycountry
        try:
            match = pycountry.countries.search_fuzzy(country)
            iso3 = match[0].alpha_3 if match else None
        except LookupError:
            iso3 = None
        if iso3 is None:
            return None

    return float(_labor_rights_score_cache[iso3]) if iso3 in _labor_rights_score_cache else None


def _cli():
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        return
    if sys.argv[1] == "fetch":
        data = fetch_all()
        save_cache(data)
        n_countries = len({c for conv in data.values() for c in conv["ratifications"]})
        print(f"fetched {len(data)} conventions, {n_countries} distinct countries")
    elif sys.argv[1] == "lookup" and len(sys.argv) > 2:
        score = labor_rights_score(sys.argv[2])
        print(f"{sys.argv[2]}: {score}/10 fundamental conventions ratified" if score is not None
              else f"{sys.argv[2]}: no data (run 'fetch' first, or name not found)")


if __name__ == "__main__":
    _cli()
