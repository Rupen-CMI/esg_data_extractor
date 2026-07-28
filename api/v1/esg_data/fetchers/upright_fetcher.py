"""
Upright Platform fetcher — looks up a company by name in the locally ingested
upright_lookup table and returns ESG metric values mapped to the catalog.

Upright scores are in cents per dollar of revenue (can be negative = harm).
We store the raw value and use it as numeric_value directly.
"""

from difflib import SequenceMatcher
from sqlalchemy import text
from sqlalchemy.orm import Session


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def fetch_upright(company_name: str, db: Session) -> dict:
    base = {
        "source": "upright",
        "found": False,
        "upright_name": None,
        "upright_country": None,
        "values": [],
        "error": None,
    }

    try:
        # First try full name ILIKE match
        rows = db.execute(
            text("SELECT * FROM upright_lookup WHERE name ILIKE :q LIMIT 10"),
            {"q": f"%{company_name[:30]}%"},
        ).mappings().all()

        if not rows:
            first_word = company_name.split()[0] if company_name.split() else company_name
            rows = db.execute(
                text("SELECT * FROM upright_lookup WHERE name ILIKE :q LIMIT 10"),
                {"q": f"%{first_word}%"},
            ).mappings().all()

        if not rows:
            return base

        best = max(rows, key=lambda r: _similarity(company_name, r["name"]))
        if _similarity(company_name, best["name"]) < 0.60:
            return base

    except Exception as e:
        base["error"] = f"upright lookup failed: {e}"
        return base

    base["found"] = True
    base["upright_name"] = best["name"]
    base["upright_country"] = best["country"]

    REPORTING_YEAR = 2024

    def _v(col) -> dict | None:
        val = best.get(col)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    metric_map = [
        # Aggregate scores
        ("upright_net_impact_ratio",        _v("net_impact_ratio")),
        ("upright_net_impact_percentile",   _v("net_impact_ratio_percentile")),

        # Environment
        ("upright_ghg_negative",            _v("e1_ghg_negative")),
        ("upright_ghg_positive",            _v("e1_ghg_positive")),
        ("upright_non_ghg_negative",        _v("e2_non_ghg_negative")),
        ("upright_scarce_resources",        _v("e3_scarce_resources_negative")),
        ("upright_biodiversity_negative",   _v("e4_biodiversity_negative")),
        ("upright_waste_negative",          _v("e5_waste_negative")),

        # Society
        ("upright_jobs_positive",           _v("s1_jobs_positive")),
        ("upright_taxes_positive",          _v("s2_taxes_positive")),
        ("upright_equality_negative",       _v("s5_equality_negative")),
        ("upright_equality_positive",       _v("s5_equality_positive")),

        # Health
        ("upright_health_negative",         _v("h1_physical_diseases_negative")),
        ("upright_health_positive",         _v("h1_physical_diseases_positive")),

        # SDG key alignments
        ("upright_sdg_13_climate",          _v("sdg_13_aligned")),
        ("upright_sdg_8_decent_work",       _v("sdg_8_aligned")),
        ("upright_sdg_5_gender",            _v("sdg_5_aligned")),
    ]

    values = []
    for metric_key, val in metric_map:
        if val is None:
            continue
        values.append({
            "metric_key": metric_key,
            "value": str(val),
            "numeric_value": val,
            "reporting_year": REPORTING_YEAR,
        })

    base["values"] = values
    return base
