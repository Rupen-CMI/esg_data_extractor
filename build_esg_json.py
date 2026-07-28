"""
Builds the ESG compliance JSON for any market in the DB.

Scoring is peer-rank based (0-100) per metric, direction-aware, over the
CORE (star) metric set only. Every company is expected to carry the full core
set — real disclosed values where available, agentic estimates
(source='agentic_metrics_v1') where not. Real data ALWAYS beats an estimate,
regardless of reporting year.

Absolute-quantity metrics (emissions, energy, water, waste) are normalised by
employee count before ranking, so large firms aren't penalised purely for size.

Run: venv/Scripts/python.exe build_esg_json.py "Textile And Apparel Market"
"""
import sys
import json
import logging
from database import Sessionlocal
from api.v1.models import Market, Company, CompanyMetricValue, ESGMetricDefinition, MarketCompanyLink
from agentic_estimation.layer_3.metric_estimation_agent import (
    CORE_METRICS, CORE_METRIC_KEYS, ESTIMATE_SOURCE, CORRECTION_SOURCE, _display_value,
)

log = logging.getLogger(__name__)

# ── Core metric metadata (single source of truth = metric_estimation_agent) ───
_CORE = {m["key"]: m for m in CORE_METRICS}
CORE_KEYS = set(CORE_METRIC_KEYS)
SCORED_CORE = {k for k, m in _CORE.items() if m["direction"] != "neutral"}

# Agentic pillar score keys — the actual pillar score for every company that's
# been through the estimation pipeline (produced by the ensemble scorer, or
# the legacy scoring/evaluator agents for older runs). Core-metric bands are
# only a fallback when no agentic pillar row exists at all (see _pillar()).
AGENTIC_E_KEY = "esg_e_score"
AGENTIC_S_KEY = "esg_s_score"
AGENTIC_G_KEY = "esg_g_score"
_AGENTIC_PILLAR = {"E": AGENTIC_E_KEY, "S": AGENTIC_S_KEY, "G": AGENTIC_G_KEY}
AGENTIC_SUMMARY_KEY = "esg_summary"

# Provenance tiers. Agent CORRECTIONS of implausible reported data beat real
# disclosures; plain gap-fill ESTIMATES lose to real disclosures.
_ESTIMATE_SOURCES = {ESTIMATE_SOURCE}
_CORRECTION_SOURCES = {CORRECTION_SOURCE}
# Agent-produced (not a real disclosure) — for the per-metric `estimated` flag.
_AGENTIC_SOURCES = _ESTIMATE_SOURCES | _CORRECTION_SOURCES
# Among same-tier rows, higher priority wins on a year tie. agentic_ensemble_v1
# (Phase 6 cutover: Tier-0 + v5 formula + v8 reconcile + Confidence Gate,
# proven better than the single-shot scorer on fixed-evidence backtests --
# see EQUATION_CHANGES.md / PHASE_4_PLAN.md) outranks both legacy pillar
# sources. All three land in the SAME tier as pillar-score keys (esg_e/s/g_
# score have no real-disclosure rows, ever -- see _row_rank's tier logic
# below), so this priority number is what actually decides which one wins.
_SOURCE_PRIORITY = {"agentic_ensemble_v1": 3, "agentic_evaluator_v1": 2, "agentic_scoring_v1": 1}

# Pillar weights for the total ESG score
_PILLAR_WEIGHTS = {"E": 0.40, "S": 0.35, "G": 0.25}


def _row_rank(source: str, year: int) -> tuple:
    """
    Deduplication rank for competing metric rows. Higher tuple wins:
      1. tier: correction (2, beats real) > real disclosure (1) > estimate (0)
      2. then newer reporting year
      3. then source priority (evaluator > scoring)
    Tier dominates year, so a correction/real always beats an older-or-newer
    lower tier.
    """
    if source in _CORRECTION_SOURCES:
        tier = 2
    elif source in _ESTIMATE_SOURCES:
        tier = 0
    else:
        tier = 1  # real disclosed data
    return (tier, year, _SOURCE_PRIORITY.get(source, 0))


def derive_risk(score: float) -> str:
    if score >= 75: return "Low"
    if score >= 50: return "Medium"
    if score >= 25: return "High"
    return "Extreme"


def derive_rating(score: float) -> str:
    if score >= 75: return "Leader"
    if score >= 50: return "Follower"
    return "Laggard"


def _band_score(x: float, best: float, worst: float) -> float:
    """
    Score a value against a fixed benchmark band. `best` maps to 100, `worst`
    to 0, linear and clamped between. Works for both directions:
      lower-is-better → best < worst (e.g. emissions 15→100, 250→0)
      higher-is-better → best > worst (e.g. renewable% 75→100, 0→0)
    Absolute and size-independent — no dependence on peers.
    """
    if best == worst:
        return 50.0
    t = (x - worst) / (best - worst)
    return round(max(0.0, min(1.0, t)) * 100, 1)


def build_all(companies, all_values, metric_defs):
    """
    Build per-company core-metric maps, size-normalise, peer-rank, and assemble
    the per-company ESG result dicts. Returns list of company dicts.
    """
    key_units = {m.key: m.unit for m in metric_defs.values() if m.unit}

    # ── Step 1: build {company_name -> {core_key -> entry}} with real>estimate dedup
    # Also keep the agentic pillar keys (the actual pillar scores) and the
    # explainability summary key.
    keep_keys = CORE_KEYS | set(_AGENTIC_PILLAR.values()) | {AGENTIC_SUMMARY_KEY}
    company_raw = {}
    company_summary = {}
    # Whether a company has ever been gap-filled — i.e. has ≥1 estimate row for a
    # core key, even if a real disclosure later outranked it. This (not the
    # metrics_estimated count) is the authoritative "estimation done" marker,
    # so a fully-disclosed company doesn't get re-estimated on every poll.
    has_estimates = {}

    for company in companies:
        by_key = {}
        seen_estimate = False
        for cmv in all_values.get(str(company.id), []):
            m = metric_defs.get(str(cmv.metric_id))
            if not m:
                continue
            key = m.key
            if key not in keep_keys:
                continue
            source = cmv.source or ""
            # esg_summary carries its text in `reasoning`, not `numeric_value`
            # (see explainability_agent.py) -- pull it separately, don't let
            # the numeric_value-is-None filter below drop it.
            if key == AGENTIC_SUMMARY_KEY:
                if cmv.reasoning:
                    company_summary[company.name] = cmv.reasoning
                continue
            if cmv.numeric_value is None:
                continue
            if key in CORE_KEYS and source in _AGENTIC_SOURCES:
                seen_estimate = True
            year = cmv.reporting_year or 0
            rank = _row_rank(source, year)
            existing = by_key.get(key)
            if existing is None or rank > existing["rank"]:
                by_key[key] = {
                    "numeric": cmv.numeric_value,
                    "value": cmv.value,
                    "year": year,
                    "source": source,
                    "rank": rank,
                    "estimated": source in _AGENTIC_SOURCES,
                    "corrected": source in _CORRECTION_SOURCES,
                    "confidence": cmv.confidence,
                    # Ensemble scorer uncertainty output (agentic_ensemble_v1
                    # only; NULL on every other source -- see
                    # db_migrations/005_ensemble_cutover.sql). Carried through
                    # so _pillar() can surface it alongside the pillar score.
                    "low_value": getattr(cmv, "low_value", None),
                    "high_value": getattr(cmv, "high_value", None),
                    "needs_review": getattr(cmv, "needs_review", None),
                    "confidence_label": getattr(cmv, "confidence_label", None),
                    "verdict": getattr(cmv, "verdict", None),
                }
        company_raw[company.name] = by_key
        has_estimates[company.name] = seen_estimate

    # ── Step 2: score each scored core metric against its fixed benchmark band.
    # No peer comparison — each company is judged on its own merit, so real
    # disclosures aren't distorted by optimistic estimates. Absolute quantities
    # (emissions, energy, water, waste) are first intensity-normalised by REVENUE
    # (per $M), making the band size-independent: a startup and an MNC with the
    # same carbon-per-$revenue score identically.
    def _metric_score(cname: str, key: str):
        """0-100 score for a company on one scored core metric, or None."""
        meta = _CORE[key]
        entry = company_raw[cname].get(key)
        if entry is None:
            return None
        if meta["kind"] == "boolean":
            return 100.0 if entry["numeric"] >= 0.5 else 0.0
        val = entry["numeric"]
        if meta["intensity"]:
            denom = company_raw[cname].get(meta["intensity"], {}).get("numeric")
            if not (denom and denom > 0):
                return None  # need the revenue denominator to score an intensity metric
            val = val / denom
        band = meta.get("benchmark")
        if not band:
            return None
        return _band_score(val, band[0], band[1])

    # ── Step 4: assemble per-company results
    results = []
    for company in companies:
        cname = company.name
        by_key = company_raw.get(cname, {})

        def _pillar(cat: str):
            """Return (score, basis, detail) — basis in {'agentic', 'core', 'none'}.
            The agentic pillar score (esg_e/s/g_score -- the ensemble scorer's
            verified, confidence-gated output as of the Phase 6 cutover) is
            authoritative whenever it exists: it carries an evidence trail,
            confidence gating, and critic verification that a raw core-metric
            band average has none of. Core-metric bands are only a fallback
            for companies that predate the agentic pillar score entirely.
            detail is None for 'none'/plain 'core'; for 'agentic' it's a dict
            with confidence_label/verdict always, plus range/needs_review
            when the winning row is agentic_ensemble_v1 with populated
            low/high (every older source leaves those NULL)."""
            entry = by_key.get(_AGENTIC_PILLAR[cat], {})
            ag = entry.get("numeric")
            if ag is not None:
                detail = {
                    "confidence_label": entry.get("confidence_label"),
                    "verdict": entry.get("verdict"),
                }
                if entry.get("low_value") is not None and entry.get("high_value") is not None:
                    detail["range"] = {
                        "low": round(entry["low_value"], 1), "high": round(entry["high_value"], 1),
                    }
                    detail["needs_review"] = bool(entry.get("needs_review"))
                return round(ag, 1), "agentic", detail

            scores = [
                s for k in SCORED_CORE if _CORE[k]["category"] == cat
                for s in (_metric_score(cname, k),) if s is not None
            ]
            if scores:
                return round(sum(scores) / len(scores), 1), "core", None
            return 0.0, "none", None

        e_score, e_basis, e_detail = _pillar("E")
        s_score, s_basis, s_detail = _pillar("S")
        g_score, g_basis, g_detail = _pillar("G")
        total_score = round(
            e_score * _PILLAR_WEIGHTS["E"] + s_score * _PILLAR_WEIGHTS["S"] + g_score * _PILLAR_WEIGHTS["G"], 1
        )

        # Provenance across displayed core metrics
        real_core = [k for k in CORE_KEYS if k in by_key and not by_key[k]["estimated"]]
        est_core = [k for k in CORE_KEYS if k in by_key and by_key[k]["estimated"]]
        if real_core and est_core:
            data_source = "mixed"
        elif real_core:
            data_source = "reported"
        elif est_core:
            data_source = "agentic_estimated"
        else:
            data_source = "no_data"

        def _metric_dict(k):
            e = by_key[k]
            meta = _CORE[k]
            # Format the display value uniformly from numeric+unit+kind, so real
            # and estimated values look consistent (and odd real strings like
            # "18.400005961044574" or a 0-100 value under a Yes/No key get cleaned).
            display = _display_value(meta, float(e["numeric"]))
            d = {
                "raw": e["numeric"],
                "value": display,
                "unit": key_units.get(k, meta["unit"]),
                "score": _metric_score(cname, k),
                "year": e["year"],
                "estimated": e["estimated"],
            }
            if e.get("corrected"):
                d["corrected"] = True  # agent overrode an implausible reported value
            if e["estimated"] and e.get("confidence") is not None:
                d["confidence"] = round(float(e["confidence"]), 2)
            return d

        def _pillar_metrics(cat: str):
            return {k: _metric_dict(k) for k in CORE_METRIC_KEYS if _CORE[k]["category"] == cat and k in by_key}

        env_metrics = _pillar_metrics("E")
        social_metrics = _pillar_metrics("S")
        gov_metrics = _pillar_metrics("G")

        # Year of the displayed core metrics only (ignore hidden agentic pillar rows)
        reporting_year = max(
            (by_key[k]["year"] for k in CORE_KEYS if k in by_key and by_key[k].get("year")),
            default=None,
        )

        results.append({
            "name": cname,
            "country": company.country,
            "esg_scoring": getattr(company, "esg_scoring", "pending"),
            "data_source": data_source,
            "has_estimates": has_estimates.get(cname, False),
            # Completeness marker: revenue is the essential scoring denominator and
            # only ever comes from estimation, so its presence means the company has
            # been gap-filled against the CURRENT core schema.
            "has_revenue": "annual_revenue" in by_key,
            "rating": derive_rating(total_score),
            "esg_scores": {
                "environment": {"score": e_score, "risk": derive_risk(e_score), **(e_detail or {})},
                "social":      {"score": s_score, "risk": derive_risk(s_score), **(s_detail or {})},
                "governance":  {"score": g_score, "risk": derive_risk(g_score), **(g_detail or {})},
                "total":       {"score": total_score, "risk": derive_risk(total_score)},
            },
            "esg_summary": company_summary.get(cname),
            "environmental_metrics": env_metrics or None,
            "social_metrics": social_metrics or None,
            "governance_metrics": gov_metrics or None,
            # real disclosures only — estimates don't count as "disclosed"
            "metrics_disclosed": len(real_core),
            "metrics_estimated": len(est_core),
            "reporting_year": reporting_year,
        })

    results.sort(key=lambda c: c["esg_scores"]["total"]["score"], reverse=True)
    return results


def build_market_esg_json(market_name: str, db=None) -> dict | None:
    """
    Core logic: load companies + metric values for a market from DB,
    compute peer-ranked ESG scores, return the full JSON dict.
    Accepts an optional db session; creates one if not provided.
    """
    close_db = False
    if db is None:
        db = Sessionlocal()
        close_db = True

    try:
        market = db.query(Market).filter(Market.name == market_name).first()
        if not market:
            return None

        companies = (
            db.query(Company)
            .join(MarketCompanyLink, MarketCompanyLink.company_id == Company.id)
            .filter(MarketCompanyLink.market_id == market.id)
            .all()
        )

        if not companies:
            return None

        metric_defs = {str(m.id): m for m in db.query(ESGMetricDefinition).all()}

        all_values = {}
        for company in companies:
            vals = (
                db.query(CompanyMetricValue)
                .filter(CompanyMetricValue.company_id == company.id)
                .all()
            )
            all_values[str(company.id)] = vals

        company_jsons = build_all(companies, all_values, metric_defs)

        avg_score = round(
            sum(c["esg_scores"]["total"]["score"] for c in company_jsons) / len(company_jsons), 1
        ) if company_jsons else 0

        return {
            "market": market_name,
            "industry_avg_esg_score": avg_score,
            "total_companies": len(company_jsons),
            "scoring_method": "fixed_benchmark_revenue_intensity",
            "companies": company_jsons,
        }

    finally:
        if close_db:
            db.close()


def run():
    market_name = sys.argv[1] if len(sys.argv) > 1 else "Textile And Apparel Market"
    output = build_market_esg_json(market_name)
    if not output:
        print(f"Market not found or no companies: {market_name}")
        return

    out_file = "esg_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to {out_file}", flush=True)


if __name__ == "__main__":
    run()
