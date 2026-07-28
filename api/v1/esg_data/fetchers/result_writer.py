"""
Persists fetcher results into company_metric_values.
Looks up esg_metric_definitions.id by key, then upserts rows.
"""

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api.v1.models import CompanyMetricValue, ESGMetricDefinition, Company


def _build_metric_key_map(db: Session) -> dict[str, object]:
    """Return {metric_key: metric_uuid} for all known metrics."""
    rows = db.query(ESGMetricDefinition.key, ESGMetricDefinition.id).all()
    return {r.key: r.id for r in rows}


def save_fetch_results(
    db: Session,
    company_id,
    results: list[dict],
    metric_key_map: dict | None = None,
) -> dict:
    """
    Persist a list of FetchResult dicts for one company.

    Each result dict must have:
        source, found, values: [{metric_key, value, numeric_value, reporting_year}]

    Returns summary: {saved: int, skipped_unknown_metric: int, skipped_no_value: int}
    """
    if metric_key_map is None:
        metric_key_map = _build_metric_key_map(db)

    saved = 0
    skipped_unknown = 0
    skipped_no_value = 0

    for result in results:
        if not result.get("found") or not result.get("values"):
            continue
        source = result["source"]

        for entry in result["values"]:
            metric_key = entry.get("metric_key")
            raw_value = entry.get("value")
            numeric_value = entry.get("numeric_value")
            reporting_year = entry.get("reporting_year")

            if not raw_value and numeric_value is None:
                skipped_no_value += 1
                continue

            metric_id = metric_key_map.get(metric_key)
            if not metric_id:
                skipped_unknown += 1
                continue

            stmt = pg_insert(CompanyMetricValue).values(
                company_id=company_id,
                metric_id=metric_id,
                value=raw_value,
                numeric_value=numeric_value,
                reporting_year=reporting_year,
                source=source,
                confidence=1.0,
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_company_metric_year_source",
                set_={
                    "value": stmt.excluded.value,
                    "numeric_value": stmt.excluded.numeric_value,
                },
            )
            db.execute(stmt)
            saved += 1

    if saved > 0:
        company = db.get(Company, company_id)
        if company:
            company.esg_scoring = "reported"

    db.commit()
    return {
        "saved": saved,
        "skipped_unknown_metric": skipped_unknown,
        "skipped_no_value": skipped_no_value,
    }


def update_company_ticker(db: Session, company_id, ticker: str | None) -> None:
    """Set has_public_esg = True and store ticker if we found one."""
    if not ticker:
        return
    company = db.get(Company, company_id)
    if not company:
        return
    company.ticker = ticker
    company.has_public_esg = True
    db.commit()
