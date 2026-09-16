"""
B Corp fetcher — looks up a company by name in the locally ingested bcorp_lookup
table and returns ESG metric values mapped to the catalog.

B Corp scores are on a 0-200 scale. We normalise to 0-100 by halving.

Score mapping:
  overall_score              → bcorp_overall_score (stored as-is, normalised)
  impact_area_environment    → bcorp_environment_score
  impact_area_workers        → bcorp_workers_score
  impact_area_community      → bcorp_community_score
  impact_area_customers      → bcorp_customers_score
  impact_area_governance     → bcorp_governance_score
"""

from difflib import SequenceMatcher

from sqlalchemy.orm import Session

from api.v1.models import BCorpLookup


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def fetch_bcorp(company_name: str, db: Session) -> dict:
    """
    Main entry point. Queries bcorp_lookup table for best name match.

    Returns standard fetcher result dict:
        {source, found, bcorp_name, bcorp_country, values: [...], error}
    """
    base = {
        "source": "bcorp",
        "found": False,
        "bcorp_name": None,
        "bcorp_country": None,
        "values": [],
        "error": None,
    }

    try:
        candidates = db.query(BCorpLookup).filter(
            BCorpLookup.company_name.ilike(f"%{company_name[:20]}%")
        ).limit(10).all()

        if not candidates:
            # Fallback: first-word match
            first_word = company_name.split()[0] if company_name.split() else company_name
            candidates = db.query(BCorpLookup).filter(
                BCorpLookup.company_name.ilike(f"%{first_word}%")
            ).limit(10).all()

        if not candidates:
            return base

        best = max(candidates, key=lambda c: _similarity(company_name, c.company_name))
        if _similarity(company_name, best.company_name) < 0.65:
            return base

    except Exception as e:
        base["error"] = f"bcorp lookup failed: {e}"
        return base

    base["found"] = True
    base["bcorp_name"] = best.company_name
    base["bcorp_country"] = best.country

    assessment_year = best.assessment_year or 2023

    def _val(raw) -> dict | None:
        if raw is None:
            return None
        try:
            return {"raw": float(raw), "normalised": round(float(raw) / 2, 2)}
        except (TypeError, ValueError):
            return None

    metric_map = [
        ("bcorp_overall_score",      best.overall_score),
        ("bcorp_environment_score",  best.impact_area_environment),
        ("bcorp_workers_score",      best.impact_area_workers),
        ("bcorp_community_score",    best.impact_area_community),
        ("bcorp_customers_score",    best.impact_area_customers),
        ("bcorp_governance_score",   best.impact_area_governance),
    ]

    values = []
    for metric_key, raw in metric_map:
        parsed = _val(raw)
        if parsed is None:
            continue
        values.append({
            "metric_key": metric_key,
            "value": str(parsed["raw"]),
            "numeric_value": parsed["normalised"],
            "reporting_year": assessment_year,
        })

    base["values"] = values
    return base
