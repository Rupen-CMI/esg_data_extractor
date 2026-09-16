import asyncio
import random

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from api.v1.esg_data.fetchers.bcorp_fetcher import fetch_bcorp
from api.v1.esg_data.fetchers.result_writer import (
    save_fetch_results,
    update_company_ticker,
)
from api.v1.esg_data.fetchers.upright_fetcher import fetch_upright
from api.v1.esg_data.fetchers.wikirate_fetcher import fetch_wikirate
from api.v1.esg_data.fetchers.yfinance_fetcher import fetch_yfinance
from api.v1.esg_data.metric_catalog import (
    METRIC_DEFINITIONS,
    SASB_SECTORS,
    SECTOR_METRICS,
)
from api.v1.models import (
    Company,
    ESGMetricDefinition,
    Market,
    MarketCompanyLink,
    MarketMetricLink,
)
from database import db as engine
from zen_client import DEFAULT_MODEL, call_with_prompt

SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

BATCH_SIZE = 5
MAX_CONCURRENT = 3
_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# ---------------------------------------------------------------------------
# Step 1 — Seed metric definitions (idempotent, run once)
# ---------------------------------------------------------------------------

def seed_metric_definitions(db: Session) -> dict:
    """
    Insert all metrics from the catalog into esg_metric_definitions.
    Safe to run multiple times — skips existing rows on key conflict.
    """
    rows = [
        {
            "key": m["key"],
            "name": m["name"],
            "description": m.get("description"),
            "unit": m.get("unit"),
            "category": m["category"],
            "sasb_sector": m.get("sasb_sector"),
            "source_framework": m.get("source_framework"),
            "is_universal": m["is_universal"],
        }
        for m in METRIC_DEFINITIONS
    ]

    stmt = pg_insert(ESGMetricDefinition).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["key"])
    result = db.execute(stmt)
    db.commit()

    inserted = result.rowcount
    return {
        "total_in_catalog": len(rows),
        "inserted": inserted,
        "skipped": len(rows) - inserted,
    }


# ---------------------------------------------------------------------------
# Step 2 — Classify each market into a SASB sector via LLM
# ---------------------------------------------------------------------------

_SECTOR_PROMPT = """Classify the following market name into exactly one SASB sector from this list:

{sectors}

Rules:
- Return ONLY the sector key, nothing else — no explanation, no punctuation
- If the market spans multiple sectors, pick the closest primary one
- If truly unclear, return: general

Examples:
  "Avocado Oil Market" → agriculture_food
  "Dental Implants Market" → healthcare
  "Lithium Ion Battery Market" → extractives
  "Cloud Computing Market" → technology
  "Life Insurance Market" → financials
  "Cryotherapy Market" → healthcare

Market: "{market_name}"
"""


def _classify_sector(market_name: str) -> str:
    sectors_str = ", ".join(SASB_SECTORS)
    prompt = _SECTOR_PROMPT.format(sectors=sectors_str, market_name=market_name)
    result = call_with_prompt(prompt, model=DEFAULT_MODEL, max_tokens=20)
    if not result["ok"]:
        return "general"
    sector = result["raw"].strip().lower().split()[0] if result["raw"].strip() else "general"
    return sector if sector in SASB_SECTORS else "general"


# ---------------------------------------------------------------------------
# Step 3 — Link metrics to a market
# ---------------------------------------------------------------------------

def _get_metric_ids_for_sector(db: Session, sasb_sector: str) -> list:
    """Return UUIDs of all metrics that apply to this sector (universal + sector-specific)."""
    sector_keys = SECTOR_METRICS.get(sasb_sector, [])
    universal_keys = [m["key"] for m in METRIC_DEFINITIONS if m["is_universal"]]
    all_keys = list(set(universal_keys + sector_keys))

    rows = (
        db.query(ESGMetricDefinition.id)
        .filter(ESGMetricDefinition.key.in_(all_keys))
        .all()
    )
    return [r.id for r in rows]


def _link_metrics_to_market(db: Session, market_id, metric_ids: list) -> int:
    if not metric_ids:
        return 0
    rows = [{"market_id": market_id, "metric_id": mid} for mid in metric_ids]
    stmt = pg_insert(MarketMetricLink).values(rows).on_conflict_do_nothing()
    result = db.execute(stmt)
    db.commit()
    return result.rowcount


# ---------------------------------------------------------------------------
# Per-market async task
# ---------------------------------------------------------------------------

async def _process_market(market_id, market_name: str) -> dict:
    async with _semaphore:
        await asyncio.sleep(random.uniform(1, 3))

        loop = asyncio.get_event_loop()
        sector = await loop.run_in_executor(None, lambda: _classify_sector(market_name))

        db: Session = SessionFactory()
        try:
            market = db.get(Market, market_id)
            if not market:
                return {"market": market_name, "sector": None, "linked": 0, "error": "not found"}

            market.sasb_sector = sector
            db.flush()

            metric_ids = _get_metric_ids_for_sector(db, sector)
            linked = _link_metrics_to_market(db, market_id, metric_ids)
            db.commit()

            print(f"[done] {market_name} → {sector} ({linked} metrics linked)")
            return {"market": market_name, "sector": sector, "linked": linked, "error": None}

        except Exception as e:
            db.rollback()
            print(f"[error] {market_name}: {e}")
            return {"market": market_name, "sector": None, "linked": 0, "error": str(e)}
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Orchestrator — called as a FastAPI background task
# ---------------------------------------------------------------------------

async def seed_market_metrics() -> None:
    """
    For every market that has no sasb_sector yet, classify it via LLM and
    populate market_metric_link.  Safe to re-run — already-classified markets
    are skipped.
    """
    db: Session = SessionFactory()
    try:
        unclassified = (
            db.query(Market.id, Market.name)
            .filter(Market.sasb_sector.is_(None))
            .all()
        )
    finally:
        db.close()

    if not unclassified:
        print("All markets already classified.")
        return

    print(f"Classifying {len(unclassified)} markets into SASB sectors...")

    for i in range(0, len(unclassified), BATCH_SIZE):
        batch = unclassified[i: i + BATCH_SIZE]
        tasks = [_process_market(row.id, row.name) for row in batch]
        await asyncio.gather(*tasks)

        if i + BATCH_SIZE < len(unclassified):
            rest = random.randint(5, 12)
            print(f"[rest] {rest}s between batches...")
            await asyncio.sleep(rest)

    print("Market metric seeding complete.")


# ---------------------------------------------------------------------------
# ESG Data Fetch — per-company worker
# ---------------------------------------------------------------------------

_fetch_semaphore = asyncio.Semaphore(3)


def _fetch_company_esg_sync(company_name: str, ticker: str | None) -> list[dict]:
    """Run all fetchers synchronously (called inside run_in_executor)."""
    results = []

    yfin = fetch_yfinance(company_name, existing_ticker=ticker)
    results.append(yfin)

    wiki = fetch_wikirate(company_name)
    results.append(wiki)

    db = SessionFactory()
    try:
        bcorp = fetch_bcorp(company_name, db)
        results.append(bcorp)

        upright = fetch_upright(company_name, db)
        results.append(upright)
    finally:
        db.close()

    return results


async def _fetch_and_save_company(company_id, company_name: str, ticker: str | None,
                                   metric_key_map: dict) -> dict:
    """Fetch ESG data for one company and persist it."""
    async with _fetch_semaphore:
        await asyncio.sleep(random.uniform(0.5, 2.0))
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None,
                lambda: _fetch_company_esg_sync(company_name, ticker)
            )
        except Exception as e:
            print(f"  [error] {company_name}: {e}")
            return {"company": company_name, "saved": 0, "error": str(e)}

        db: Session = SessionFactory()
        try:
            # Update ticker on company row if yfinance found one
            yf_result = next((r for r in results if r["source"] == "yfinance"), None)
            if yf_result and yf_result.get("ticker"):
                update_company_ticker(db, company_id, yf_result["ticker"])

            summary = save_fetch_results(db, company_id, results, metric_key_map)
        finally:
            db.close()

        sources_found = [r["source"] for r in results if r.get("found")]
        print(f"  [done] {company_name} | sources: {sources_found or 'none'} "
              f"| saved: {summary['saved']} values")

        return {
            "company": company_name,
            "sources_found": sources_found,
            **summary,
            "error": None,
        }


# ---------------------------------------------------------------------------
# ESG Data Fetch — market-level orchestrator
# ---------------------------------------------------------------------------

async def fetch_market_esg(market_name: str) -> dict:
    """
    Fetch ESG data for every company linked to a given market and persist
    the results into company_metric_values.

    Safe to re-run — existing values are updated (not duplicated) thanks to
    the upsert in result_writer.

    Returns a summary dict with per-company results.
    """
    db: Session = SessionFactory()
    try:
        market = db.query(Market).filter(Market.name == market_name).first()
        if not market:
            return {"error": f"Market '{market_name}' not found in DB"}

        companies = (
            db.query(Company.id, Company.name, Company.ticker)
            .join(MarketCompanyLink, MarketCompanyLink.company_id == Company.id)
            .filter(MarketCompanyLink.market_id == market.id)
            .all()
        )

        if not companies:
            return {"error": f"No companies linked to market '{market_name}'"}

        # Pre-load metric key → UUID map once (shared across all company tasks)
        metric_key_map = {
            r.key: r.id
            for r in db.query(ESGMetricDefinition.key, ESGMetricDefinition.id).all()
        }

        market_sasb = market.sasb_sector

    finally:
        db.close()

    print(f"\n[fetch_market_esg] '{market_name}' | {len(companies)} companies | "
          f"sector: {market_sasb}")

    company_batch_size = 5
    all_results = []

    for i in range(0, len(companies), company_batch_size):
        batch = companies[i: i + company_batch_size]
        tasks = [
            _fetch_and_save_company(c.id, c.name, c.ticker, metric_key_map)
            for c in batch
        ]
        batch_results = await asyncio.gather(*tasks)
        all_results.extend(batch_results)

        if i + company_batch_size < len(companies):
            await asyncio.sleep(random.randint(3, 8))

    total_saved = sum(r.get("saved", 0) for r in all_results)
    companies_with_data = [r for r in all_results if r.get("sources_found")]

    print(f"\n[fetch_market_esg] Done. {len(companies_with_data)}/{len(companies)} companies "
          f"had data. {total_saved} metric values saved.")

    return {
        "market": market_name,
        "companies_processed": len(companies),
        "companies_with_data": len(companies_with_data),
        "total_values_saved": total_saved,
        "results": all_results,
    }
