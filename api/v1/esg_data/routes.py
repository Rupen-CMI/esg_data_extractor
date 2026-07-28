import asyncio
import concurrent.futures
import logging
import threading
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from uuid import UUID

from database import get_db
from api.v1.models import Company
from .services import seed_metric_definitions, seed_market_metrics, fetch_market_esg
from build_esg_json import build_market_esg_json

log = logging.getLogger(__name__)

router = APIRouter(prefix="/esg", tags=["ESG Data"])


@router.get("/health")
def health():
    return {"status": "running"}


@router.get("/markets")
def list_markets(db: Session = Depends(get_db)):
    """List markets for the UI dropdown, with linked company counts."""
    from sqlalchemy import func
    from api.v1.models import Market, MarketCompanyLink

    rows = (
        db.query(Market.name, Market.status, Market.sasb_sector, func.count(MarketCompanyLink.company_id))
        .outerjoin(MarketCompanyLink, MarketCompanyLink.market_id == Market.id)
        .group_by(Market.id, Market.name, Market.status, Market.sasb_sector)
        .order_by(Market.name)
        .all()
    )
    return [
        {"name": name, "status": status, "sasb_sector": sasb_sector, "company_count": count}
        for name, status, sasb_sector, count in rows
    ]


@router.get("/core-metrics")
def list_core_metrics():
    """Expose CORE_METRICS metadata so the UI can show display names/units."""
    from agentic_estimation.layer_3.metric_estimation_agent import CORE_METRICS

    return [
        {"key": m["key"], "name": m["name"], "unit": m["unit"], "category": m["category"], "direction": m["direction"]}
        for m in CORE_METRICS
    ]


@router.post("/seed-metrics", status_code=201)
def seed_metrics_endpoint(db: Session = Depends(get_db)):
    """
    Seed the esg_metric_definitions table with the full metric catalog.
    Safe to call multiple times — skips existing rows.
    """
    try:
        result = seed_metric_definitions(db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seed-market-metrics", status_code=202)
async def seed_market_metrics_endpoint(background_tasks: BackgroundTasks):
    """
    Background task: classify every unclassified market into a SASB sector
    via LLM, then populate market_metric_link for each one.
    Safe to call multiple times — skips already-classified markets.
    """
    background_tasks.add_task(seed_market_metrics)
    return {
        "status": "started",
        "message": "Market metric seeding running in background. Check server logs for progress.",
    }


class MarketRequest(BaseModel):
    market_name: str


@router.post("/fetch-market", status_code=202)
async def fetch_market_esg_endpoint(body: MarketRequest, background_tasks: BackgroundTasks):
    """
    Background task: fetch ESG data for all companies linked to the given market.
    Safe to re-run — existing values are upserted, not duplicated.
    """
    background_tasks.add_task(fetch_market_esg, body.market_name)
    return {
        "status": "started",
        "market": body.market_name,
        "message": "ESG fetch running in background. Check server logs for progress.",
    }


@router.post("/market-esg")
def get_market_esg(body: MarketRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Return the full ESG compliance JSON for a market.

    Estimation is triggered in the background so every company ends up with the
    full core (star) metric set:
      - esg_scoring='pending'  → full agentic pipeline (pillars + core metrics)
      - reported / already-estimated companies that have never been gap-filled
        (metrics_estimated == 0) → lightweight metrics-only run that estimates
        the core metrics they don't disclose (real disclosures always win).

    Re-poll POST /esg/estimation-status to track progress.
    """
    market_name = body.market_name
    result = build_market_esg_json(market_name, db=db)
    if not result:
        raise HTTPException(status_code=404, detail=f"Market '{market_name}' not found or has no companies")

    # Decide what needs a background run
    to_run: list[tuple[str, str]] = []   # (company_name, mode)
    for c in result["companies"]:
        status = c.get("esg_scoring")
        if status == "pending":
            to_run.append((c["name"], "full"))
        elif status == "failed":
            # run_company_graph is the sole writer of "failed" (an exception
            # or state["error"] during a prior run) -- all writes downstream
            # are upserts, so safe to just retry the full run.
            to_run.append((c["name"], "full"))
        elif status == "processing":
            # _inflight is per-PROCESS (in-memory) -- only skip if THIS
            # server instance actually has it in flight right now. A company
            # stuck at "processing" with no matching _inflight entry means a
            # prior server process died mid-run (crash/restart) and orphaned
            # it -- re-enqueue rather than skip forever. Multi-worker
            # deployments: another worker's in-flight run isn't visible here
            # either, so a re-enqueue in that case duplicates work rather
            # than skips it -- safe, since every downstream write is an
            # upsert (company_metric_values, company_evidence_claims, the
            # esg_scoring status column itself).
            if c["name"] in _inflight:
                continue
            to_run.append((c["name"], "full"))
        elif not c.get("has_revenue"):
            # never gap-filled, OR gap-filled before revenue existed (stale) —
            # either way it lacks the revenue denominator needed to score.
            to_run.append((c["name"], "metrics"))

    if to_run:
        log.info("Triggering estimation for %d companies: %s", len(to_run), to_run)
        background_tasks.add_task(_ensure_estimates, to_run, market_name)

    return result


# Max pipelines running concurrently. Each run makes ~5 DDG calls with a
# 2s minimum gap — 2 concurrent runs keeps DDG pressure well within limits.
# Raise cautiously: 3+ risks collapsing the rate limiter gap under load.
_ESTIMATION_CONCURRENCY = 2

# In-process guard against double-running the same company when the frontend
# re-polls market-esg while a background run is still going.
_inflight: set[str] = set()
_inflight_lock = threading.Lock()


def _ensure_estimates(specs: list[tuple[str, str]], market_name: str) -> None:
    """
    Background task ensuring each company has its estimates.

    IMPORTANT: this is a SYNC function on purpose. The agentic pipeline is
    heavily blocking (web fetches, LLM HTTP calls, sleep-based rate limiters).
    A sync background task runs in Starlette's worker threadpool — NOT on the
    event loop — so it never freezes the server (e.g. the estimation-status
    poll stays instant). An inner ThreadPoolExecutor then runs companies in
    parallel THREADS, giving real concurrency (an asyncio.Semaphore couldn't,
    because blocking work can't overlap on a single event loop). Each company's
    async pipeline is driven by asyncio.run() inside its own worker thread.

    mode 'full' runs the whole pipeline; mode 'metrics' only gap-fills the core
    star metrics. DDG calls still queue through the shared thread-safe
    _DDG_LIMITER, keeping us within safe rate limits across threads.
    """
    import os
    from database import Sessionlocal
    from agentic_estimation.orchestrator import run_metrics_only
    from agentic_estimation.graph import run_company_graph

    # Phase 6 cutover: full-pipeline runs go through the new ensemble scorer
    # (Tier-0 validators + v5 formula + v8 reconcile + Confidence Gate) via
    # graph.py, not the original single-shot scoring_agent path -- the
    # ensemble scorer is proven better on fixed-evidence backtests (see
    # EQUATION_CHANGES.md / PHASE_4_PLAN.md). The critic panel (Phase 4
    # verify) is on by default; ESG_VERIFY=0 is the escape hatch if it needs
    # to be disabled in production without a code change.
    _verify_on = os.getenv("ESG_VERIFY", "1") != "0"

    def _run_one(name: str, mode: str) -> None:
        with _inflight_lock:
            if name in _inflight:
                log.info("[agentic] '%s' already in flight — skipping", name)
                return
            _inflight.add(name)
        try:
            lookup = Sessionlocal()
            try:
                company = lookup.query(Company).filter(Company.name == name).first()
                if not company:
                    log.warning("[agentic] Company '%s' not found in DB — skipping", name)
                    return
                cid = UUID(str(company.id))
                ccountry = company.country or None
            finally:
                lookup.close()

            if mode == "full":
                log.info("[agentic] Full pipeline (ensemble, verify=%s) for '%s'", _verify_on, name)
                result = asyncio.run(run_company_graph(
                    company_name=name, company_id=cid,
                    industry=market_name, country=ccountry,
                    scorer="ensemble", verify=_verify_on,
                ))
                log.info(
                    "[agentic] Done '%s' — E=%.1f S=%.1f G=%.1f (%.1fs), %d metrics",
                    name, result.e_score or 0, result.s_score or 0, result.g_score or 0,
                    result.elapsed_s, len(result.metric_estimates),
                )
            else:
                log.info("[agentic] Metrics-only gap-fill for '%s'", name)
                estimates = asyncio.run(run_metrics_only(
                    company_name=name, company_id=cid,
                    industry=market_name, country=ccountry,
                ))
                log.info("[agentic] Gap-filled '%s' — %d metrics", name, len(estimates))
        except Exception as exc:
            log.error("[agentic] Estimation failed for '%s': %s", name, exc)
        finally:
            with _inflight_lock:
                _inflight.discard(name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=_ESTIMATION_CONCURRENCY) as pool:
        futures = [pool.submit(_run_one, name, mode) for name, mode in specs]
        concurrent.futures.wait(futures)


@router.post("/estimation-status")
def get_estimation_status(body: MarketRequest, db: Session = Depends(get_db)):
    """
    Poll endpoint for the frontend to track estimation progress for a market.

    A company is 'ready' only when its core-metric tables can actually be shown —
    i.e. it has been gap-filled (has ≥1 agentic_metrics_v1 estimate row) and is
    not mid-run. A 'reported' company with only legacy/partial data is NOT ready
    until gap-fill completes, so the frontend keeps polling until the tables fill.

    Frontend flow:
      1. POST /esg/market-esg         — triggers estimation/gap-fill in background
      2. POST /esg/estimation-status  — poll until all_ready is true
    """
    from api.v1.models import Market, MarketCompanyLink, CompanyMetricValue, ESGMetricDefinition

    market_name = body.market_name
    market = db.query(Market).filter(Market.name == market_name).first()
    if not market:
        raise HTTPException(status_code=404, detail=f"Market '{market_name}' not found")

    companies = (
        db.query(Company)
        .join(MarketCompanyLink, MarketCompanyLink.company_id == Company.id)
        .filter(MarketCompanyLink.market_id == market.id)
        .all()
    )
    company_ids = [c.id for c in companies]

    # A company is "complete" once it has a revenue value — the essential scoring
    # denominator, which only comes from a gap-fill run against the current schema.
    revenue_def = db.query(ESGMetricDefinition).filter(ESGMetricDefinition.key == "annual_revenue").first()
    complete_ids = {
        row[0]
        for row in db.query(CompanyMetricValue.company_id)
        .filter(
            CompanyMetricValue.company_id.in_(company_ids),
            CompanyMetricValue.metric_id == revenue_def.id,
        )
        .distinct()
        .all()
    } if (company_ids and revenue_def) else set()

    statuses = []
    for c in companies:
        status = c.esg_scoring or "pending"
        if status == "failed":
            # "failed" is a real terminal status (see graph.run_company_graph)
            # but the frontend doesn't know it yet -- map to "pending" for now
            # (zero frontend change; get_market_esg already re-enqueues
            # "failed" companies the same as "pending" ones). Expose a real
            # `failed` state to the frontend later instead of masking it.
            status = "pending"
        has_estimates = c.id in complete_ids
        in_flight = c.name in _inflight
        ready = has_estimates and status not in ("pending", "processing")
        if ready:
            state = "ready"
        elif status == "processing" or in_flight:
            state = "processing"
        else:
            state = "pending"
        statuses.append({
            "name": c.name,
            "esg_scoring": status,
            "has_estimates": has_estimates,
            "state": state,
            "ready": ready,
        })

    total = len(statuses)
    done = sum(1 for s in statuses if s["ready"])
    in_progress = sum(1 for s in statuses if s["state"] == "processing")
    pending = sum(1 for s in statuses if s["state"] == "pending")

    return {
        "market": market_name,
        "total_companies": total,
        "done": done,
        "processing": in_progress,
        "pending": pending,
        "all_ready": total > 0 and done == total,
        "companies": statuses,
    }
