"""
country_baseline_agent.py — Agent 2: Country ESG Baseline

Reads the World Bank ESG dataset (esgdata_download-2026-05-01.xlsx), computes
per-country E/S/G baseline scores (0–100), and persists them to country_esg_baseline.
Results are cached in-process so the heavy Excel parse only runs once per session.

Public API:
    get_country_baseline(country: str) -> CountryBaseline | None
    load_all_baselines()  — precompute + upsert all 214 countries into DB

Country resolution:
    extract_country_from_wikipedia(text: str) -> str | None
    — scans Wikipedia signal text for a known country name via regex
"""

import logging
import os
import re
import statistics
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
log = get_logger("country_baseline_agent")


# ── Paths ─────────────────────────────────────────────────────────────────────

_EXCEL_PATH = Path(__file__).parent.parent.parent / "raw_esg_data" / "esgdata_download-2026-05-01.xlsx"

# ── Indicators to use (≥80% major-country coverage, direction-aware) ──────────
# direction: +1 = higher is better (score as-is), -1 = lower is better (invert)

_E_INDICATORS: dict[str, int] = {
    "EN.ATM.PM25.MC.M3":          -1,  # PM2.5 pollution — lower better
    "EN.GHG.CO2.PC.CE.AR5":       -1,  # CO2 per capita — lower better
    "EN.GHG.ALL.PC.CE.AR5":       -1,  # Total GHG per capita — lower better
    "EN.GHG.CH4.MT.CE.AR5":       -1,  # Methane total — lower better
    "EN.GHG.N2O.MT.CE.AR5":       -1,  # N2O total — lower better
    "EG.ELC.COAL.ZS":             -1,  # Coal electricity % — lower better
    "EG.IMP.CONS.ZS":             -1,  # Energy imports — lower better (energy independence)
    "EG.EGY.PRIM.PP.KD":          -1,  # Energy intensity — lower better
    "EG.USE.COMM.FO.ZS":          -1,  # Fossil fuel % — lower better
    "EG.ELC.RNEW.ZS":             +1,  # Renewable electricity % — higher better
    "EG.FEC.RNEW.ZS":             +1,  # Renewable energy % — higher better
    "ER.H2O.FWST.ZS":             -1,  # Water stress — lower better
    "EN.H2O.BDYS.ZS":             +1,  # Water quality bodies — higher better
    "AG.LND.FRST.ZS":             +1,  # Forest area % — higher better
    "NY.ADJ.DFOR.GN.ZS":          -1,  # Forest depletion — lower better
    "NY.ADJ.DRES.GN.ZS":          -1,  # Natural resource depletion — lower better
    "ER.H2O.FWTL.ZS":             -1,  # Freshwater withdrawals — lower better
    "ER.PTD.TOTL.ZS":             +1,  # Protected areas % — higher better
    "EN.MAM.THRD.NO":             -1,  # Threatened mammals — lower better
    "AG.LND.FRLS.HA":             -1,  # Tree cover loss — lower better
    "EN.CLC.SPEI.XD":             -1,  # Drought index — lower better (more negative = drier)
    "EN.LND.LTMP.DC":             -1,  # Land surface temperature — lower better
    "EN.CLC.CDDY.XD":             -1,  # Cooling degree days — lower better
    "EN.CLC.HDDY.XD":             -1,  # Heating degree days — lower better
    "EN.CLC.CSTP.ZS":             +1,  # Coastal protection — higher better
    "EN.CLC.HEAT.XD":             -1,  # Heat index — lower better
    "AG.LND.AGRI.ZS":             -1,  # Agricultural land % — lower better (intensity proxy)
    "NV.AGR.TOTL.ZS":             -1,  # Ag value added % GDP — lower better (industrialisation)
    "AG.PRD.FOOD.XD":             +1,  # Food production index — higher better
    "EN.GHG.CO2.LU.MT.CE.AR5":   -1,  # CO2 from land use — lower better
    "EN.POP.DNST":                -1,  # Population density — lower better (land pressure)
    "EG.USE.PCAP.KG.OE":         -1,  # Energy use per capita — lower better
    "EN.GHG.CO2.MT.CE.AR5":      -1,  # Total CO2 (abs) — lower better
    "EN.GHG.ALL.MT.CE.AR5":      -1,  # Total GHG (abs) — lower better
}

_S_INDICATORS: dict[str, int] = {
    # NOT a World Bank code -- ILO fundamental-convention ratification count
    # (0-10), injected into the pivot below before normalisation (see
    # _inject_ilo_indicator). Added 2026-08-19: every other _S_INDICATORS
    # entry measures general development (health/poverty/education/labor-
    # force participation) -- none measures whether a country's LAWS
    # actually protect workers (freedom of association, child/forced labor
    # bans, equal pay). Genuinely global (~187 countries via ILO/NORMLEX,
    # same coverage class as every real World Bank indicator here), unlike
    # the RSS-based S-evidence sources (HR Dive etc.) which are US/UK-only.
    "ILO.LABOR.RIGHTS": +1,  # fundamental conventions ratified (0-10) — higher better
    "EG.CFT.ACCS.ZS":    +1,  # Clean cooking access — higher better
    "EG.ELC.ACCS.ZS":    +1,  # Electricity access — higher better
    "SH.H2O.SMDW.ZS":    +1,  # Safe drinking water — higher better
    "SH.STA.SMSS.ZS":    +1,  # Safe sanitation — higher better
    "SP.DYN.TFRT.IN":    -1,  # Fertility rate — lower better (development proxy)
    "SP.DYN.LE00.IN":    +1,  # Life expectancy — higher better
    "SP.POP.65UP.TO.ZS":  +1,  # Elderly population % — higher better (longevity)
    "SE.XPD.TOTL.GB.ZS":  +1,  # Education spending — higher better
    "SE.PRM.ENRR":        +1,  # Primary enrollment — higher better
    "SL.TLF.ACTI.ZS":    +1,  # Labour force participation — higher better
    "SL.UEM.NEET.ME.ZS":  -1,  # Youth NEET — lower better
    "SL.UEM.TOTL.ZS":    -1,  # Unemployment — lower better
    "SL.EMP.WORK.ZS":    +1,  # Wage workers % — higher better (formal economy)
    "SH.DTH.COMM.ZS":    -1,  # Communicable disease deaths — lower better
    "SH.MED.BEDS.ZS":    +1,  # Hospital beds — higher better
    "SH.DYN.MORT":       -1,  # Child mortality — lower better
    "SH.STA.OWAD.ZS":    -1,  # Overweight adults — lower better
    "SN.ITK.DEFC.ZS":    -1,  # Undernourishment — lower better
    "SI.POV.GINI":        -1,  # Gini inequality — lower better
    "SI.DST.FRST.20":    +1,  # Bottom 20% income share — higher better
    "SI.POV.DDAY":        -1,  # Poverty $3/day — lower better
    "SI.POV.UMIC":        -1,  # Poverty $8.30/day — lower better
    "SI.SPR.PGAP":        -1,  # Prosperity gap — lower better
}

_G_INDICATORS: dict[str, int] = {
    "NY.GDP.MKTP.KD.ZG":   +1,  # GDP growth — higher better
    "IT.NET.USER.ZS":      +1,  # Internet access — higher better
    "SG.GEN.PARL.ZS":      +1,  # Women in parliament — higher better
    "SL.TLF.CACT.FM.ZS":   +1,  # Female/male LFP ratio — higher better
    "SE.ENR.PRSC.FM.ZS":   +1,  # Gender parity in school — higher better
    "GOV_WGI_GE.EST":      +1,  # Government effectiveness — higher better
    "GOV_WGI_RQ.EST":      +1,  # Regulatory quality — higher better
    "SD.ESR.PERF.XQ":      +1,  # Social rights performance — higher better
    "GOV_WGI_VA.EST":      +1,  # Voice & accountability — higher better
    "IP.PAT.RESD":         +1,  # Patents — higher better
    "GB.XPD.RSDV.GD.ZS":   +1,  # R&D spending — higher better
    "IP.JRN.ARTC.SC":      +1,  # Scientific articles — higher better
    "GOV_WGI_CC.EST":      +1,  # Control of corruption — higher better
    "SM.POP.NETM":         +1,  # Net migration — higher better (stability signal)
    "GOV_WGI_PV.EST":      +1,  # Political stability — higher better
    "GOV_WGI_RL.EST":      +1,  # Rule of law — higher better
}

_ALL_INDICATORS = {**_E_INDICATORS, **_S_INDICATORS, **_G_INDICATORS}

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class CountryBaseline:
    country: str
    iso3: str
    year: int
    e_score: float
    s_score: float
    g_score: float
    indicator_count: int


# ── In-process cache ──────────────────────────────────────────────────────────

_cache: dict[str, CountryBaseline] = {}        # economy name → baseline
_iso3_to_name: dict[str, str] = {}             # ISO3 → economy name
_cache_lock = threading.Lock()
_cache_loaded = False


def _load_from_db() -> dict[str, CountryBaseline]:
    """
    Fetch all country baselines from country_esg_baseline table.
    Much faster than parsing the Excel — single DB round-trip.
    """
    import psycopg2
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    db_url = os.getenv("ASYNC_DB_URL", os.getenv("DB_URL", ""))
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    # Strip query params psycopg2 doesn't understand (e.g. sslmode via ?ssl=require)
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    # map ?ssl=require → sslmode=require which psycopg2 does understand
    sslmode = qs.pop("ssl", [None])[0]
    clean_qs = urlencode({k: v[0] for k, v in qs.items()})
    if sslmode:
        clean_qs = (clean_qs + "&" if clean_qs else "") + f"sslmode={sslmode}"
    db_url = urlunparse(parsed._replace(query=clean_qs))

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT country, year, e_score, s_score, g_score, indicator_count FROM country_esg_baseline")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    baselines = {}
    for country, year, e_score, s_score, g_score, indicator_count in rows:
        baselines[country] = CountryBaseline(
            country=country,
            iso3="",
            year=year,
            e_score=float(e_score),
            s_score=float(s_score),
            g_score=float(g_score),
            indicator_count=int(indicator_count),
        )
    log.info("Loaded %d country baselines from DB", len(baselines))
    return baselines


# ── DB connection ─────────────────────────────────────────────────────────────

def _get_engine():
    url = os.getenv("DB_URL")
    return create_engine(url)


# ── Core computation ──────────────────────────────────────────────────────────

def _inject_ilo_indicator(pivot: "pd.DataFrame") -> None:
    """Add the ILO.LABOR.RIGHTS pseudo-column to `pivot` (ISO3-indexed,
    mutated in place) from the cached ILO ratification data (see
    ilo_ratification.py). Fail-open: if the cache doesn't exist yet
    (ilo_ratification.py's `fetch` CLI was never run), this indicator is
    simply absent from the pivot -- _S_INDICATORS' existing "need at least
    half the indicators" floor means baselines still compute normally from
    the World Bank indicators alone, same as before this was added."""
    try:
        from agentic_estimation.layer_1.ilo_ratification import load_cache, _build_iso3_index
    except Exception as exc:
        log.warning("ILO indicator injection skipped (import failed): %s", exc)
        return

    cache = load_cache()
    if cache is None:
        log.info("ILO indicator injection skipped: no cache (run `python -m "
                  "agentic_estimation.layer_1.ilo_ratification fetch` first)")
        return

    scores = _build_iso3_index(cache)
    if not scores:
        log.warning("ILO indicator injection skipped: cache loaded but produced 0 ISO3 scores")
        return

    pivot["ILO.LABOR.RIGHTS"] = pd.Series(scores)
    log.info("injected ILO.LABOR.RIGHTS for %d countries", len(scores))


def _compute_baselines() -> dict[str, CountryBaseline]:
    """
    Read the Excel file, normalise each indicator per pillar, and return
    a dict of economy_name → CountryBaseline.
    """
    log.info("Loading Excel: %s", _EXCEL_PATH)
    fw   = pd.read_excel(_EXCEL_PATH, sheet_name="Framework")
    meta = pd.read_excel(_EXCEL_PATH, sheet_name="Metadata")
    data = pd.read_excel(_EXCEL_PATH, sheet_name="Data")

    # Map ISO3 → economy name
    iso_to_economy = dict(zip(meta["ISO3 code"], meta["Economy"]))

    # Year columns present in the file (2015 onward only)
    all_year_cols = [c for c in data.columns if str(c).isdigit()]
    recent_cols   = [c for c in all_year_cols if int(str(c)) >= 2015]

    log.info("Using years %s–%s (%d cols), %d countries",
             recent_cols[0], recent_cols[-1], len(recent_cols), len(iso_to_economy))

    # Wide pivot: ISO3 × indicator_code → latest non-null value in recent years
    def _latest_value(row: pd.Series) -> float:
        vals = row[recent_cols].dropna()
        return float(vals.iloc[-1]) if len(vals) else np.nan

    data["_val"] = data.apply(_latest_value, axis=1)
    pivot = data.pivot_table(
        index="ISO3 code", columns="Indicator code", values="_val", aggfunc="first"
    )

    _inject_ilo_indicator(pivot)

    # Keep only indicators we care about
    available = [c for c in _ALL_INDICATORS if c in pivot.columns]
    missing   = [c for c in _ALL_INDICATORS if c not in pivot.columns]
    if missing:
        log.warning("Indicators not found in data: %s", missing)
    pivot = pivot[available]

    log.info("%d/%d indicators available in dataset", len(available), len(_ALL_INDICATORS))

    # Normalise each indicator to 0–100 across all countries (min-max)
    normalised = pd.DataFrame(index=pivot.index)
    for code in available:
        direction = _ALL_INDICATORS[code]
        col = pivot[code]
        lo, hi = col.min(), col.max()
        if hi == lo:
            normalised[code] = 50.0
            continue
        if direction == +1:
            normalised[code] = (col - lo) / (hi - lo) * 100
        else:
            normalised[code] = (hi - col) / (hi - lo) * 100

    # Compute pillar scores per country
    e_codes = [c for c in _E_INDICATORS if c in normalised.columns]
    s_codes = [c for c in _S_INDICATORS if c in normalised.columns]
    g_codes = [c for c in _G_INDICATORS if c in normalised.columns]

    results: dict[str, CountryBaseline] = {}

    for iso3 in normalised.index:
        economy = iso_to_economy.get(iso3, iso3)

        row = normalised.loc[iso3]

        e_vals = row[e_codes].dropna()
        s_vals = row[s_codes].dropna()
        g_vals = row[g_codes].dropna()

        # Need at least half the indicators per pillar to produce a reliable score
        if len(e_vals) < len(e_codes) // 2:
            e_score = np.nan
        else:
            e_score = float(e_vals.mean())

        if len(s_vals) < len(s_codes) // 2:
            s_score = np.nan
        else:
            s_score = float(s_vals.mean())

        if len(g_vals) < len(g_codes) // 2:
            g_score = np.nan
        else:
            g_score = float(g_vals.mean())

        if np.isnan(e_score) and np.isnan(s_score) and np.isnan(g_score):
            log.debug("Skipping %s (%s) — insufficient data", economy, iso3)
            continue

        # Median year of the actual data used (per-indicator latest non-null year)
        # Much more honest than "latest year any single indicator was updated"
        raw_row = data[(data["ISO3 code"] == iso3) & (data["Indicator code"].isin(available))]
        per_indicator_years = []
        for _, ind_row in raw_row.iterrows():
            for yr in sorted(recent_cols, reverse=True):
                if pd.notna(ind_row[yr]):
                    per_indicator_years.append(int(str(yr)))
                    break
        latest_year = int(np.median(per_indicator_years)) if per_indicator_years else 2020

        indicator_count = int(e_vals.count() + s_vals.count() + g_vals.count())

        bl = CountryBaseline(
            country=economy,
            iso3=iso3,
            year=latest_year,
            e_score=round(e_score if not np.isnan(e_score) else 50.0, 2),
            s_score=round(s_score if not np.isnan(s_score) else 50.0, 2),
            g_score=round(g_score if not np.isnan(g_score) else 50.0, 2),
            indicator_count=indicator_count,
        )
        results[economy] = bl
        log.debug("%s (%s): E=%.1f S=%.1f G=%.1f (%d indicators)",
                  economy, iso3, bl.e_score, bl.s_score, bl.g_score, indicator_count)

    log.info("Computed baselines for %d countries", len(results))
    return results, iso_to_economy


def _ensure_cache() -> None:
    global _cache, _iso3_to_name, _cache_loaded
    with _cache_lock:
        if _cache_loaded:
            return
        try:
            baselines = _load_from_db()
            if baselines:
                _cache = baselines
                _cache_loaded = True
                return
            log.warning("DB returned 0 baselines — falling back to Excel")
        except Exception as exc:
            log.warning("DB baseline load failed (%s) — falling back to Excel", exc)
        # Fallback: parse Excel (slow but always works)
        baselines, iso_to_economy = _compute_baselines()
        _cache = baselines
        _iso3_to_name = {iso: name for iso, name in iso_to_economy.items() if name in baselines}
        _cache_loaded = True


# ── Country name extraction from Wikipedia signal text ───────────────────────

# Built lazily from the economy names in the dataset — all 214 countries.
_COUNTRY_REGEX: Optional[re.Pattern] = None
_COUNTRY_NAMES: list[str] = []


def _build_country_regex() -> None:
    global _COUNTRY_REGEX, _COUNTRY_NAMES
    _ensure_cache()
    # Sort longest-first so "United Arab Emirates" matches before "United"
    names = sorted(_cache.keys(), key=len, reverse=True)
    _COUNTRY_NAMES = names
    pattern = "|".join(re.escape(n) for n in names)
    _COUNTRY_REGEX = re.compile(pattern, re.IGNORECASE)
    log.debug("Country regex built with %d country names", len(names))


def extract_country_from_wikipedia(text: str) -> Optional[str]:
    """
    Scan Wikipedia signal text for a known World Bank country name.
    Returns the canonical economy name (as it appears in the dataset), or None.

    Example:
        "Robert Bosch GmbH ... headquartered in Gerlingen, Baden-Württemberg, Germany"
        → "Germany"
    """
    global _COUNTRY_REGEX
    if _COUNTRY_REGEX is None:
        _build_country_regex()

    # Strip the "Wikipedia: " prefix if present
    body = text
    if body.startswith("Wikipedia:"):
        body = body[len("Wikipedia:"):].strip()

    m = _COUNTRY_REGEX.search(body)
    if not m:
        return None

    matched = m.group(0)
    # Return canonical casing from the cache
    for name in _COUNTRY_NAMES:
        if name.lower() == matched.lower():
            return name
    return matched


# ── Public API ────────────────────────────────────────────────────────────────

def get_country_baseline(country: str) -> Optional[CountryBaseline]:
    """
    Return the ESG baseline for a country name (case-insensitive).
    Loads and caches data on first call (~3–5s for Excel parse).

    Args:
        country: Country name as returned by extract_country_from_wikipedia()
                 or stored in companies.country, e.g. "Germany", "India"

    Returns:
        CountryBaseline or None if country not found in dataset.
    """
    _ensure_cache()
    # Exact match first
    if country in _cache:
        return _cache[country]
    # Case-insensitive fallback
    lower = country.lower()
    for name, bl in _cache.items():
        if name.lower() == lower:
            return bl
    log.warning("Country not found in baseline cache: %r", country)
    return None


def load_all_baselines() -> int:
    """
    Compute baselines for all countries and upsert into country_esg_baseline.
    Returns the number of rows upserted.
    Safe to call multiple times — uses ON CONFLICT DO UPDATE.
    """
    log_header(log, "Country Baseline Agent", action="load all baselines")
    _ensure_cache()
    engine = _get_engine()
    Session = sessionmaker(bind=engine)

    upsert_sql = text("""
        INSERT INTO country_esg_baseline
            (country, year, e_score, s_score, g_score, indicator_count, computed_at)
        VALUES
            (:country, :year, :e_score, :s_score, :g_score, :indicator_count, now())
        ON CONFLICT (country) DO UPDATE SET
            year            = EXCLUDED.year,
            e_score         = EXCLUDED.e_score,
            s_score         = EXCLUDED.s_score,
            g_score         = EXCLUDED.g_score,
            indicator_count = EXCLUDED.indicator_count,
            computed_at     = now()
    """)

    count = 0
    with Session() as session:
        for bl in _cache.values():
            session.execute(upsert_sql, {
                "country":         bl.country,
                "year":            bl.year,
                "e_score":         bl.e_score,
                "s_score":         bl.s_score,
                "g_score":         bl.g_score,
                "indicator_count": bl.indicator_count,
            })
            count += 1
        session.commit()

    log.info("Upserted %d country baselines into DB", count)
    return count


def get_all_baselines() -> dict[str, CountryBaseline]:
    """Return the full in-process cache of all country baselines."""
    _ensure_cache()
    return dict(_cache)


# ── Country-name resolution (ISO3 codes + common aliases) ───────────────────
# Found live during Phase 2 calibration: company_metadata.py's Wikidata lookup
# returns whatever label Wikidata has set for the country entity -- sometimes
# a full name ("United States"), sometimes a colloquial short form ("USA"),
# sometimes a bare ISO3 code ("USA"/"FRA"/"CHN"/"JPN"), inconsistently, since
# that's just how Wikidata's label data is. None of these matched the World
# Bank dataset's own bureaucratic naming ("Korea, Rep.", "Russian Federation",
# "Egypt, Arab Rep.") -- so real, large, well-known companies (Johnson &
# Johnson, Christian Dior, Cigna) were silently falling all the way through to
# the global-average fallback despite their country being perfectly known.
#
# This is a lookup problem, not a fuzzy-matching problem: country identity is
# binary (Iran and Iraq are NOT "80% the same country"), so this is an exact
# alias/code table, not a string-similarity score.
#
# TYPO TOLERANCE, added 2026-09-18: the above is still the right rule for
# ALIASES ("Britain" -> "United Kingdom" is a real, deliberate mapping
# choice, not a similarity score). But a genuine KEYBOARD TYPO ("Untied
# States", "Grmany") is a different failure mode -- the string is close to
# exactly one real country, not ambiguous between two real countries -- and
# today it falls all the way through resolve_country_name to None, landing
# the company on the global_average baseline instead of its own real one.
# _fuzzy_typo_match() below adds ONE more stage, deliberately narrow:
#   - ratio() >= 0.90 only. Measured against every geographically-confusable
#     WB economy pair (Iran/Iraq 0.750, Niger/Nigeria 0.833 -- the worst
#     case found -- Mali/Malawi 0.800, Guinea/Guinea-Bissau 0.632, Congo
#     variants 0.815, Chad/Chile 0.444) -- none reach 0.90, while real
#     single-character typos do (United States/Untied States 0.923,
#     Germany/Grmany 0.923, Brazil/Brasil 0.833 -- kept above the danger
#     ceiling despite being a genuine typo shape).
#   - the match must be UNIQUE at that threshold: if two different real
#     countries both score >=0.90 against the input, this is ambiguous by
#     definition and must return None (global_average), never guess.
#   - _TYPO_BLOCKLIST is a second, independent safety net: named pairs that
#     must NEVER fuzzy-match each other regardless of computed ratio, in
#     case a country pair not covered by the 0.90 analysis above turns out
#     to score higher than expected.
_TYPO_BLOCKLIST: set[frozenset] = {
    frozenset({"iran, islamic rep.", "iraq"}),
    frozenset({"niger", "nigeria"}),
    frozenset({"mali", "malawi"}),
    frozenset({"chad", "chile"}),
    frozenset({"guinea", "guinea-bissau"}),
    frozenset({"guinea", "equatorial guinea"}),
    frozenset({"guinea-bissau", "equatorial guinea"}),
    frozenset({"dominica", "dominican republic"}),
    frozenset({"sudan", "south sudan"}),
    frozenset({"congo, dem. rep.", "congo, rep."}),
    frozenset({"slovak republic", "slovenia"}),
}
_FUZZY_TYPO_THRESHOLD = 0.90


def _fuzzy_typo_match(raw: str) -> Optional[str]:
    """Last-resort typo tolerance for resolve_country_name -- see the module
    note above this function for the safety reasoning and threshold
    calibration. Returns the single real Economy name the input is almost
    certainly a typo of, or None if there's no match, the match isn't
    unique, or the pair is on the blocklist."""
    import difflib

    _ensure_cache()
    target = raw.strip().lower()
    if not target:
        return None

    best_name: Optional[str] = None
    best_ratio = 0.0
    runner_up_ratio = 0.0
    for name in _cache:
        ratio = difflib.SequenceMatcher(None, target, name.lower()).ratio()
        if ratio > best_ratio:
            runner_up_ratio = best_ratio
            best_ratio, best_name = ratio, name
        elif ratio > runner_up_ratio:
            runner_up_ratio = ratio

    if best_name is None or best_ratio < _FUZZY_TYPO_THRESHOLD:
        return None
    if runner_up_ratio >= _FUZZY_TYPO_THRESHOLD:
        log.warning("fuzzy typo match for %r ambiguous (top two both >=%.2f) -- refusing to guess",
                    raw, _FUZZY_TYPO_THRESHOLD)
        return None
    if frozenset({target, best_name.lower()}) in _TYPO_BLOCKLIST:
        log.warning("fuzzy typo match for %r -> %r blocked (known-confusable pair)", raw, best_name)
        return None

    log.info("resolve_country_name: %r -> %r via typo tolerance (ratio=%.3f)", raw, best_name, best_ratio)
    return best_name

_iso3_lookup_cache: Optional[dict[str, str]] = None  # ISO3 (upper) -> Economy name
_economy_to_iso3_cache: Optional[dict[str, str]] = None  # Economy name -> ISO3 (upper)


def _iso3_to_economy(iso3: str) -> Optional[str]:
    """ISO3 code -> the World Bank's own Economy name for it, via the same
    Metadata sheet climate_trace_anchor.py already uses for the reverse
    direction. Covers the "USA"/"FRA"/"CHN"/"JPN"-as-country-value case."""
    global _iso3_lookup_cache
    if _iso3_lookup_cache is None:
        import pandas as pd
        meta = pd.read_excel(_EXCEL_PATH, sheet_name="Metadata")
        _iso3_lookup_cache = {str(iso).upper(): name for iso, name in zip(meta["ISO3 code"], meta["Economy"])}
    return _iso3_lookup_cache.get(iso3.strip().upper())


def economy_to_iso3(economy: str) -> Optional[str]:
    """Reverse of _iso3_to_economy: the World Bank's own Economy name -> its
    ISO3 code. Added for peer_anchor_collector.py's country crosswalk
    (DEFECT_FIX_PLAN.md 2.1) -- upright_lookup.country stores ISO3 codes
    while bcorp_lookup.country stores full names, so a caller's resolved
    country string (a WB Economy name, via resolve_country_name) needs
    converting before it can match upright's column."""
    global _economy_to_iso3_cache
    if _economy_to_iso3_cache is None:
        import pandas as pd
        meta = pd.read_excel(_EXCEL_PATH, sheet_name="Metadata")
        _economy_to_iso3_cache = {name: str(iso).upper() for iso, name in zip(meta["ISO3 code"], meta["Economy"])}
    return _economy_to_iso3_cache.get(economy)


# Colloquial/common names -> the World Bank dataset's own (often bureaucratic)
# Economy name, for the cases verified live where they differ. Not exhaustive
# by design -- add here only when a real mismatch is found, same discipline
# as _REGIONAL_FALLBACK below.
_COUNTRY_ALIASES: dict[str, str] = {
    "usa": "United States", "us": "United States", "u.s.": "United States",
    "u.s.a.": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "great britain": "United Kingdom",
    "south korea": "Korea, Rep.", "korea": "Korea, Rep.",
    "north korea": "Korea, Dem. People's Rep.",
    "russia": "Russian Federation",
    "vietnam": "Viet Nam",
    "turkey": "Turkiye",
    "egypt": "Egypt, Arab Rep.",
    "iran": "Iran, Islamic Rep.",
    "czech republic": "Czechia",
    "ivory coast": "Cote d'Ivoire",
    "laos": "Lao PDR",
    "syria": "Syrian Arab Republic",
    "venezuela": "Venezuela, RB",
    "slovakia": "Slovak Republic",
}


def resolve_country_name(raw: str) -> Optional[str]:
    """
    Resolve an arbitrary incoming country string (ISO3 code, colloquial name,
    or already-correct World Bank Economy name) to the exact Economy name
    used by this dataset's cache keys, or None if it can't be resolved.
    Exact/alias/code lookup first (see module note above for why country
    identity is a lookup problem, not a similarity score); a narrow, guarded
    typo-tolerance stage (_fuzzy_typo_match) runs ONLY as the final fallback,
    after every exact path has already failed -- see _TYPO_BLOCKLIST's
    module note for the threshold/safety reasoning.
    """
    if not raw:
        return None
    stripped = raw.strip()

    _ensure_cache()
    if stripped in _cache:
        return stripped
    lower = stripped.lower()
    for name in _cache:
        if name.lower() == lower:
            return name

    alias = _COUNTRY_ALIASES.get(lower)
    if alias and alias in _cache:
        return alias

    if len(stripped) == 3 and stripped.isalpha():
        economy = _iso3_to_economy(stripped)
        if economy and economy in _cache:
            return economy

    # Final stage: full ISO 3166 resolution via country_normalizer, then map
    # the resulting ISO3 back to this dataset's Economy name. The exact/alias/
    # ISO3 stages above stay first so behaviour for already-working inputs is
    # bit-identical; this only catches what previously returned None.
    #
    # The hand-written _COUNTRY_ALIASES above cannot cover the space (it had
    # "great britain" but not "britain", and nothing for "Netherlands The",
    # ISO2 codes, or "Hong Kong S.A.R."). country_normalizer delegates identity
    # to pycountry's ISO 3166 database and, critically, REFUSES to resolve
    # cities/regions/LLM non-answers -- so this cannot turn "Mumbai" into India.
    iso3 = _normalize_to_iso3(stripped)
    if iso3:
        economy = _iso3_to_economy(iso3)
        if economy and economy in _cache:
            return economy
        # Real country, but the World Bank publishes no economy row for it
        # (Taiwan, Guernsey, Jersey) -- or it has too few indicators to have
        # been cached. Caller falls back to the global average; logged so the
        # gap is visible rather than silent.
        log.debug("resolve_country_name: %r -> ISO3 %s has no WB economy baseline",
                  raw, iso3)

    # Last resort: genuine keyboard-typo tolerance -- see _fuzzy_typo_match's
    # own docstring and the module note above _TYPO_BLOCKLIST for the safety
    # reasoning. Stays last so every exact/alias/ISO3 path above (bit-
    # identical behaviour for already-working inputs) is tried first.
    typo_match = _fuzzy_typo_match(stripped)
    if typo_match:
        return typo_match

    return None


def _normalize_to_iso3(raw: str) -> Optional[str]:
    """country_normalizer.to_iso3, imported lazily and never fatal.

    Isolated in a helper so a missing pycountry dependency degrades to the
    previous alias-only behaviour instead of breaking country resolution for
    the whole pipeline.
    """
    try:
        from agentic_estimation.layer_1.country_normalizer import to_iso3
    except Exception as e:  # pragma: no cover - dependency-missing path
        log.warning("country_normalizer unavailable (%s); alias-only resolution", e)
        return None
    try:
        return to_iso3(raw)
    except Exception as e:
        log.warning("country_normalizer failed on %r: %s", raw, e)
        return None


# ── Fallback for countries/territories absent from the World Bank dataset ────
# A small, fixed list of real sovereignty/administrative relationships for
# territories that appear in the ESG dataset's own Metadata sheet but have too
# few indicators to compute a baseline (verified: 4 cases as of the 2026-05-01
# dataset -- St. Martin, Sint Maarten, Isle of Man, Channel Islands). This is
# NOT a general nearest-country algorithm -- the dataset's own "Geographic
# region" column is far too coarse (e.g. "Latin America & Caribbean" spans
# dozens of countries) to derive this from data. Each entry below is a real,
# well-known constitutional/administrative fact, not a guess:
_REGIONAL_FALLBACK: dict[str, str] = {
    "isle of man": "United Kingdom",           # British Crown Dependency
    "channel islands": "United Kingdom",        # British Crown Dependency (Jersey/Guernsey)
    "sint maarten (dutch part)": "Netherlands",  # constituent country, Kingdom of the Netherlands
    "st. martin (french part)": "France",        # French overseas collectivity
    # Crown Dependencies the World Bank omits entirely (no Metadata row at
    # all, unlike the four above). Same constitutional basis as
    # "channel islands", reached now that country_normalizer resolves the
    # individual island names and ISO3 codes.
    "guernsey": "United Kingdom",               # British Crown Dependency
    "jersey": "United Kingdom",                 # British Crown Dependency
}

# Territories with a real ISO 3166 identity that the World Bank publishes NO
# economy for, and where no sovereignty fallback is factually available.
#
# Taiwan and Hong Kong are the consequential entries: the World Bank omits
# Taiwan from its country list entirely, and its Hong Kong economy row is
# absent from the ESG dataset's Metadata sheet. Between them they cover real
# manufacturers in the benchmark corpus (Foxconn Technology, Formosa Plastics).
# Unlike the Crown Dependencies above there is no uncontested administrative
# parent whose baseline could stand in.
#
# Entries WITHOUT a _PEER_BASELINE_SETS entry below fall through to the
# global-average baseline; they are listed here so the gap is documented and
# greppable rather than looking like an unhandled normalization failure.
_NO_WB_BASELINE_ISO3: dict[str, str] = {
    "TWN": "Taiwan -- absent from the World Bank country list",
    "HKG": "Hong Kong SAR -- no row in the ESG dataset Metadata sheet",
    "MAC": "Macao SAR -- no row in the ESG dataset Metadata sheet",
}

# Peer-median baselines for territories with no World Bank economy row.
#
# WHY: the global-average fallback is measurably wrong for these. Measured
# against the 2026-05-01 dataset, the global average is E 70.72 / S 60.68 /
# G 46.11, while every advanced East Asian economy sits at S 77-89 and G 59-64.
# Taiwan's two corpus companies (Foxconn Technology, Formosa Plastics) were
# therefore starting ~20 points low on S and ~16 low on G purely because of a
# missing upstream row. Since the baseline is what evidence contributions move
# *from* (and net contributions are typically only a few points), a wrong
# baseline is a wrong score no amount of evidence corrects.
#
# WHY NOT the dataset's own region x income grouping: Taiwan/Hong Kong/Macao
# have no Metadata row at all, so neither region nor income group can be
# derived for them -- the peer set has to be stated. And the nearest
# auto-derived group ("East Asia & Pacific" + "High income") is polluted with
# Pacific microstates (American Samoa, Nauru, Palau, Guam, New Caledonia) that
# share a region with Taiwan and nothing else; their inclusion pulls the S
# median from 81.8 down to 66.2.
#
# Peers are chosen on the attributes the baseline indicators actually measure
# -- income level, industrial structure, and institutional quality -- not
# geographic adjacency. MEDIAN (not mean) so a single outlier cannot skew a
# small set. Baselines produced this way are labelled "regional_peer", never
# "exact", so a score built on an estimate is never presented as that
# territory's own real data.
#
# China is deliberately EXCLUDED from Taiwan's peer set. It is Taiwan's largest
# trading partner, but on what these indicators measure it is a poor match:
# upper-middle vs high income, materially different governance (WGI voice &
# accountability / rule of law), and a much more coal-weighted energy mix.
# Including it moves E by ~-14 and G by ~-4 for reasons that do not describe
# Taiwan.
#
# Hong Kong and Macao are intentionally NOT given peer sets yet: Hong Kong's
# governance profile diverged sharply from Singapore's after 2020 (National
# Security Law, electoral changes) on exactly the WGI dimensions in _G_INDICATORS,
# so Singapore alone would overstate its G, and Macao's gaming-monoculture
# economy (~50% of GDP) has no good structural comparator. Both keep the
# global-average fallback until sourced properly.
_PEER_BASELINE_SETS: dict[str, tuple[str, ...]] = {
    # High-income, export-manufacturing, mature democratic institutions.
    # Korea is the closest single match (comparable scale, conglomerate-heavy
    # corporate structure, late-1980s democratization).
    "TWN": ("Japan", "Korea, Rep.", "Singapore"),
}

_peer_baseline_cache: dict[str, Optional[CountryBaseline]] = {}


def _peer_median_baseline(iso3: str) -> Optional[CountryBaseline]:
    """Median E/S/G across a stated peer set, or None if no set is defined.

    Returns None (not a partial result) when fewer than two peers have real
    baselines -- a "median" over one country is just that country's data
    wearing a misleading label.
    """
    if iso3 in _peer_baseline_cache:
        return _peer_baseline_cache[iso3]

    peers = _PEER_BASELINE_SETS.get(iso3)
    if not peers:
        _peer_baseline_cache[iso3] = None
        return None

    _ensure_cache()
    found = [_cache[name] for name in peers if name in _cache]
    if len(found) < 2:
        log.warning("peer baseline for %s: only %d/%d peers have baselines -- "
                    "falling back to global average", iso3, len(found), len(peers))
        _peer_baseline_cache[iso3] = None
        return None

    bl = CountryBaseline(
        country=f"(peer median: {', '.join(b.country for b in found)})",
        iso3=iso3,
        year=max(b.year for b in found),
        e_score=round(statistics.median(b.e_score for b in found), 2),
        s_score=round(statistics.median(b.s_score for b in found), 2),
        g_score=round(statistics.median(b.g_score for b in found), 2),
        # 0 marks this as derived rather than computed from real indicator
        # counts, matching _global_average_baseline's convention.
        indicator_count=0,
    )
    _peer_baseline_cache[iso3] = bl
    return bl

_global_average_cache: Optional[CountryBaseline] = None


def _global_average_baseline() -> CountryBaseline:
    """Mean E/S/G across every country that DOES have a real baseline -- a
    genuine computed statistic (not a flat invented number), used only as the
    last-resort fallback for a country string that resolves to nothing at all,
    not even via _REGIONAL_FALLBACK."""
    global _global_average_cache
    if _global_average_cache is not None:
        return _global_average_cache

    _ensure_cache()
    all_bl = list(_cache.values())
    n = len(all_bl)
    _global_average_cache = CountryBaseline(
        country="(global average)", iso3="", year=max(b.year for b in all_bl),
        e_score=round(sum(b.e_score for b in all_bl) / n, 2),
        s_score=round(sum(b.s_score for b in all_bl) / n, 2),
        g_score=round(sum(b.g_score for b in all_bl) / n, 2),
        indicator_count=0,
    )
    return _global_average_cache


def get_country_baseline_with_fallback(country: str) -> tuple[CountryBaseline, str]:
    """
    Like get_country_baseline(), but never returns None. Returns
    (baseline, source) where source is one of:
      "exact"     -- the country's own real World Bank baseline
      "resolved"  -- matched via resolve_country_name (ISO3 code or a known
                     alias like "USA"/"South Korea") to its real baseline --
                     still the country's OWN real data, just reached through
                     a name-normalisation step first. See resolve_country_name's
                     docstring for why this is an exact lookup, not fuzzy matching.
      "regional"  -- a real parent/administrative country's baseline
                     (see _REGIONAL_FALLBACK -- a fixed, hand-verified list)
      "regional_peer" -- MEDIAN across a stated comparator set, for territories
                     the World Bank publishes no economy for (see
                     _PEER_BASELINE_SETS). An ESTIMATE, not the territory's own
                     data -- the distinct label exists so a score built on one
                     is never presented as real country data.
      "global_average" -- mean across all 210 countries with real data
                     (a genuine statistic, never an arbitrary flat number)

    Use this (not get_country_baseline) wherever a caller needs a baseline
    unconditionally, e.g. formula_estimator.py -- callers that need to
    distinguish "no data" from "has data" should keep using
    get_country_baseline() and check for None.
    """
    exact = get_country_baseline(country)
    if exact is not None:
        return exact, "exact"

    resolved_name = resolve_country_name(country)
    if resolved_name:
        resolved_bl = get_country_baseline(resolved_name)
        if resolved_bl is not None:
            log.info("resolved %r -> %r via alias/ISO3 lookup", country, resolved_name)
            return resolved_bl, "resolved"

    # _REGIONAL_FALLBACK is keyed on canonical lowercase country names, so a
    # raw ISO3 ("GGY") or variant spelling never matched it before. Resolving
    # the identity to a common name first lets a Crown Dependency arriving as a
    # code reach its documented constitutional parent.
    lookup_keys = [country.strip().lower()]
    iso3 = _normalize_to_iso3(country)
    if iso3:
        from agentic_estimation.layer_1.country_normalizer import iso3_to_common_name
        common = iso3_to_common_name(iso3)
        if common:
            lookup_keys.append(common.strip().lower())

    for key in lookup_keys:
        parent = _REGIONAL_FALLBACK.get(key)
        if parent:
            parent_bl = get_country_baseline(parent)
            if parent_bl is not None:
                log.info("no baseline for %r -- using regional fallback %r", country, parent)
                return parent_bl, "regional"

    # Peer-median stand-in, for territories the World Bank publishes no economy
    # for but where a defensible comparator set exists (see _PEER_BASELINE_SETS).
    # Ordered before the global average because a regional peer median is
    # measurably closer than a world mean for these -- and after the exact /
    # resolved / regional paths, so it can never override real country data.
    if iso3:
        peer_bl = _peer_median_baseline(iso3)
        if peer_bl is not None:
            log.info("no World Bank baseline for %r (%s) -- using peer median %s",
                     country, _NO_WB_BASELINE_ISO3.get(iso3, "no WB economy row"),
                     peer_bl.country)
            return peer_bl, "regional_peer"

    if iso3 and iso3 in _NO_WB_BASELINE_ISO3:
        # Known, documented upstream gap -- not a normalisation failure. Logged
        # distinctly so these don't get debugged as broken country resolution.
        log.info("no World Bank baseline exists for %r (%s) -- using global average",
                 country, _NO_WB_BASELINE_ISO3[iso3])
    else:
        log.warning("no baseline for %r (including alias/regional fallback) -- using global average", country)
    return _global_average_baseline(), "global_average"


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Country ESG Baseline Agent")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("load", help="Compute and upsert all country baselines into DB")

    lookup = sub.add_parser("lookup", help="Look up a single country baseline")
    lookup.add_argument("country", help="Country name, e.g. 'Germany'")

    extract = sub.add_parser("extract", help="Extract country from Wikipedia text")
    extract.add_argument("text", help="Wikipedia signal text")

    top = sub.add_parser("top", help="Show top/bottom N countries per pillar")
    top.add_argument("pillar", choices=["E", "S", "G"])
    top.add_argument("--n", type=int, default=10)

    args = parser.parse_args()

    if args.cmd == "load":
        n = load_all_baselines()
        print(f"Loaded {n} country baselines into DB.")

    elif args.cmd == "lookup":
        bl = get_country_baseline(args.country)
        if bl:
            print(f"\n{bl.country} ({bl.iso3}) — data year: {bl.year}")
            print(f"  E score: {bl.e_score:.1f}/100")
            print(f"  S score: {bl.s_score:.1f}/100")
            print(f"  G score: {bl.g_score:.1f}/100")
            print(f"  Indicators used: {bl.indicator_count}")
        else:
            print(f"Country not found: {args.country!r}")

    elif args.cmd == "extract":
        country = extract_country_from_wikipedia(args.text)
        print(f"Extracted country: {country!r}")

    elif args.cmd == "top":
        all_bl = get_all_baselines()
        key = {"E": "e_score", "S": "s_score", "G": "g_score"}[args.pillar]
        ranked = sorted(all_bl.values(), key=lambda b: getattr(b, key), reverse=True)
        print(f"\nTop {args.n} — {args.pillar} score:")
        for i, b in enumerate(ranked[:args.n], 1):
            print(f"  {i:>3}. {b.country:<35} {getattr(b, key):.1f}")
        print(f"\nBottom {args.n}:")
        for i, b in enumerate(ranked[-args.n:], 1):
            print(f"  {i:>3}. {b.country:<35} {getattr(b, key):.1f}")

    else:
        parser.print_help()
