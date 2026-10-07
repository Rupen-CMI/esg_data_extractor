"""
orchestrator.py — ESG Agentic Pipeline Orchestrator

Wires the full pipeline for a single company:
  1. Signal Agent    — gather web signals
  2. Country Baseline — load country peer-group scores
  3. Scoring Agent   — produce E/S/G scores (0-100)
  4. Evaluator Agent — fact-check scores against signals, correct if needed
  5. Explainability  — write plain-English summary

Entry points:
    run_company(company_name, company_id, industry, country) → PipelineResult
        Async, saves all outputs to DB.

    run_company_dry(company_name, industry, country) → PipelineResult
        Sync, no DB writes. Good for testing.

CLI:
    # Dry run (no DB)
    python -m agentic_estimation.orchestrator dry "Patagonia" --industry "Outdoor Apparel"

    # Full run with DB save (company must exist in companies table)
    python -m agentic_estimation.orchestrator run "Bosch" --industry "Industrial Machinery" --country Germany

    # Pass explicit company_id (UUID) to skip the DB lookup
    python -m agentic_estimation.orchestrator run "Bosch" --id <uuid> --industry "Industrial Machinery"
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header, log_pipeline_start

log = get_logger("orchestrator")


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    company: str
    industry: str
    country: Optional[str]

    # Per-stage outcomes
    signals: dict = field(default_factory=dict)       # source → text
    signals_count: int = 0

    e_score: Optional[float] = None
    s_score: Optional[float] = None
    g_score: Optional[float] = None
    e_reasoning: str = ""
    s_reasoning: str = ""
    g_reasoning: str = ""

    evaluator_verdict: str = ""                        # "pass" or "fix"
    evaluator_note: str = ""

    metric_estimates: dict = field(default_factory=dict)  # core metric key → estimate

    summary: Optional[str] = None

    # Metadata
    elapsed_s: float = 0.0
    saved_to_db: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.e_score is not None


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _set_esg_scoring_status(company_id: UUID, status: str) -> None:
    """Update companies.esg_scoring for a single company. Non-fatal on error."""
    from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
    db_url = os.environ.get("ASYNC_DB_URL", "")
    if not db_url:
        return
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    parsed = urlparse(db_url)
    qs = parse_qs(parsed.query)
    sslmode = qs.pop("ssl", [None])[0]
    clean_qs = urlencode({k: v[0] for k, v in qs.items()})
    if sslmode:
        clean_qs = (clean_qs + "&" if clean_qs else "") + f"sslmode={sslmode}"
    db_url = urlunparse(parsed._replace(query=clean_qs))
    try:
        conn = await asyncpg.connect(db_url)
        await conn.execute(
            "UPDATE companies SET esg_scoring = $1 WHERE id = $2",
            status, str(company_id),
        )
        await conn.close()
    except Exception as exc:
        log.warning("Failed to set esg_scoring='%s' for %s: %s", status, company_id, exc)


# ── Pipeline stages ────────────────────────────────────────────────────────────

def _run_signals(company: str, industry: str, company_id=None, country: Optional[str] = None) -> dict:
    # With a company_id, use the DB-cached path so re-runs don't re-hammer the web.
    if company_id is not None:
        from agentic_estimation.layer_1.signal_agent import get_or_fetch_signals
        return get_or_fetch_signals(company_id, company, industry, country=country)
    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    return fetch_company_signals(company, industry, country=country)


def _run_metadata(company: str, company_id=None) -> dict:
    if company_id is not None:
        from agentic_estimation.layer_1.company_metadata import get_or_fetch_metadata
        return get_or_fetch_metadata(company_id, company)
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    return get_company_metadata(company)


def _run_scoring(company: str, industry: str, country: Optional[str], signals: dict, metadata: dict):
    from agentic_estimation.layer_3.scoring_agent import score_company_sync
    return score_company_sync(company, industry=industry, country=country, signals=signals, metadata=metadata)


def _run_evaluator(score, signals: dict, industry: str):
    from agentic_estimation.layer_4.evaluator_agent import evaluate_company_sync
    return evaluate_company_sync(score, signals, industry=industry)


def _run_explainability(score, industry: str, evidence_by_pillar=None, routing_by_pillar=None):
    from agentic_estimation.layer_4.explainability_agent import explain_company_sync
    return explain_company_sync(score, industry=industry, evidence_by_pillar=evidence_by_pillar,
                                 routing_by_pillar=routing_by_pillar)


def _resolve_baseline(country: Optional[str]):
    """Look up the country ESG baseline via the alias/regional/global-average
    fallback chain (get_country_baseline_with_fallback), or None if country
    itself is empty. Non-fatal. Previously used the bare get_country_baseline
    (exact-or-None), which meant an unresolved alias/ISO3 code or an
    on-dataset-but-baseline-less territory silently fell all the way to a
    flat 50/50/50 neutral in estimate_metrics_sync/estimate_and_save's own
    `if baseline else "50"` branch, even when a real regional or
    global-average baseline was available and should have been used instead."""
    if not country:
        return None
    try:
        from agentic_estimation.layer_1.country_baseline_agent import get_country_baseline_with_fallback
        baseline, _source = get_country_baseline_with_fallback(country)
        return baseline
    except Exception as exc:
        log.warning("Baseline lookup failed for %s: %s", country, exc)
        return None


# ── Dry run (no DB) ────────────────────────────────────────────────────────────

def run_company_dry(
    company_name: str,
    industry: str = "",
    country: Optional[str] = None,
) -> PipelineResult:
    """
    Run the full pipeline without any DB writes.
    Useful for testing a company before it's in the DB.
    """
    t0 = time.monotonic()
    result = PipelineResult(company=company_name, industry=industry, country=country)

    log_pipeline_start(company_name, mode="dry")
    log_header(log, "Orchestrator — DRY RUN",
               company=company_name, industry=industry or "N/A", country=country or "auto-detect")

    # Stage 1: Signals
    log.info("=== Stage 1/5: Signal Agent ===")
    try:
        signals = _run_signals(company_name, industry, country=country)
        result.signals = signals
        result.signals_count = len(signals)
        log.info("Signals gathered: %d sources", result.signals_count)
    except Exception as exc:
        result.error = f"Signal agent failed: {exc}"
        log.error(result.error)
        return result

    # Stage 1b: Metadata (non-fatal — scoring falls back gracefully if this fails)
    log.info("=== Stage 1b: Company Metadata ===")
    metadata = {}
    try:
        metadata = _run_metadata(company_name)
        log.info("Metadata source: %s", metadata.get("source") or "none")
    except Exception as exc:
        log.warning("Metadata lookup failed (%s) — scoring will proceed without it", exc)

    # Stage 2: Scoring
    log.info("=== Stage 2/5: Scoring Agent ===")
    try:
        score = _run_scoring(company_name, industry, country, signals, metadata)
        if score is None:
            result.error = "Scoring agent returned None"
            log.error(result.error)
            return result
        result.country = score.country
        log.info("Scores: E=%.1f S=%.1f G=%.1f (country=%s)", score.e_score, score.s_score, score.g_score, score.country)
    except Exception as exc:
        result.error = f"Scoring agent failed: {exc}"
        log.error(result.error)
        return result

    # Stage 3: Evaluator
    log.info("=== Stage 3/5: Evaluator Agent ===")
    try:
        eval_result = _run_evaluator(score, signals, industry)
        if eval_result is None:
            log.warning("Evaluator returned None — keeping original scores")
            final_score = score
            result.evaluator_verdict = "skipped"
            result.evaluator_note = "Evaluator LLM call failed; original scores kept"
        else:
            final_score = eval_result.final_score
            result.evaluator_verdict = eval_result.verdict
            result.evaluator_note = eval_result.evaluator_note
            log.info("Evaluator verdict: %s", eval_result.verdict.upper())
    except Exception as exc:
        log.warning("Evaluator failed (%s) — keeping original scores", exc)
        final_score = score
        result.evaluator_verdict = "skipped"
        result.evaluator_note = str(exc)

    result.e_score = final_score.e_score
    result.s_score = final_score.s_score
    result.g_score = final_score.g_score
    result.e_reasoning = final_score.e_reasoning
    result.s_reasoning = final_score.s_reasoning
    result.g_reasoning = final_score.g_reasoning

    # Stage 3b: Core Metric Estimation (dry — estimate only, no DB write)
    log.info("=== Stage 3b: Core Metric Estimation ===")
    try:
        from agentic_estimation.layer_3.metric_estimation_agent import estimate_metrics_sync
        baseline = _resolve_baseline(result.country)
        result.metric_estimates = estimate_metrics_sync(
            company_name, industry, result.country, signals, metadata, baseline,
        )
        log.info("Estimated %d core metrics", len(result.metric_estimates))
    except Exception as exc:
        log.warning("Metric estimation failed (%s) — skipped", exc)

    # Stage 4: Explainability
    log.info("=== Stage 4/5: Explainability Agent ===")
    try:
        expl = _run_explainability(final_score, industry)
        if expl is None:
            log.warning("Explainability returned None — summary skipped")
        else:
            result.summary = expl.summary
    except Exception as exc:
        log.warning("Explainability failed (%s) — summary skipped", exc)

    result.elapsed_s = time.monotonic() - t0
    log.info("Pipeline complete in %.1fs", result.elapsed_s)
    return result


# ── Full async run with DB writes ──────────────────────────────────────────────

async def run_company(
    company_name: str,
    company_id: UUID,
    industry: str = "",
    country: Optional[str] = None,
) -> PipelineResult:
    """
    Run the full pipeline (original single-shot LLM scorer) and save all
    outputs to the DB. company_id must match a row in the companies table.

    LEGACY as of the Phase 6 cutover (2026-07-21): production
    (api/v1/esg_data/routes.py) now calls
    agentic_estimation.graph.run_company_graph(..., scorer="ensemble")
    instead of this function -- the ensemble scorer (Tier-0 validators + v5
    formula + v8 reconcile + Confidence Gate + Phase-4 critics) is proven
    better on fixed-evidence backtests. This function is retained only for
    calibration_harness.py's --scorer llm baseline comparisons.
    """
    t0 = time.monotonic()
    result = PipelineResult(company=company_name, industry=industry, country=country)

    log_pipeline_start(company_name, mode="full")
    log_header(log, "Orchestrator — FULL RUN",
               company=company_name, industry=industry or "N/A",
               country=country or "auto-detect", company_id=str(company_id))

    await _set_esg_scoring_status(company_id, "processing")

    # Stage 1: Signals (DB-cached)
    log.info("=== Stage 1/5: Signal Agent ===")
    try:
        signals = _run_signals(company_name, industry, company_id=company_id, country=country)
        result.signals = signals
        result.signals_count = len(signals)
        log.info("Signals gathered: %d sources", result.signals_count)
    except Exception as exc:
        result.error = f"Signal agent failed: {exc}"
        log.error(result.error)
        await _set_esg_scoring_status(company_id, "failed")
        return result

    # Stage 1b: Metadata (non-fatal)
    log.info("=== Stage 1b: Company Metadata ===")
    metadata = {}
    try:
        metadata = _run_metadata(company_name, company_id=company_id)
        log.info("Metadata source: %s", metadata.get("source") or "none")
    except Exception as exc:
        log.warning("Metadata lookup failed (%s) — scoring will proceed without it", exc)

    # Stage 2: Scoring (already saves to DB internally)
    log.info("=== Stage 2/5: Scoring Agent ===")
    try:
        from agentic_estimation.layer_3.scoring_agent import score_company
        score = await score_company(
            company_name=company_name,
            company_id=company_id,
            industry=industry,
            country=country,
            signals=signals,
            metadata=metadata,
        )
        if score is None:
            result.error = "Scoring agent returned None"
            log.error(result.error)
            await _set_esg_scoring_status(company_id, "failed")
            return result
        result.country = score.country
        log.info("Scores saved: E=%.1f S=%.1f G=%.1f", score.e_score, score.s_score, score.g_score)
    except Exception as exc:
        result.error = f"Scoring agent failed: {exc}"
        log.error(result.error)
        await _set_esg_scoring_status(company_id, "failed")
        return result

    # Stage 3: Evaluator (saves corrected scores if verdict=fix)
    log.info("=== Stage 3/5: Evaluator Agent ===")
    try:
        from agentic_estimation.layer_4.evaluator_agent import evaluate_company
        eval_result = await evaluate_company(
            score=score,
            company_id=company_id,
            signals=signals,
            industry=industry,
        )
        if eval_result is None:
            log.warning("Evaluator returned None — keeping original scores")
            final_score = score
            result.evaluator_verdict = "skipped"
            result.evaluator_note = "Evaluator LLM call failed; original scores kept"
        else:
            final_score = eval_result.final_score
            result.evaluator_verdict = eval_result.verdict
            result.evaluator_note = eval_result.evaluator_note
            log.info("Evaluator verdict: %s", eval_result.verdict.upper())
    except Exception as exc:
        log.warning("Evaluator failed (%s) — keeping original scores", exc)
        final_score = score
        result.evaluator_verdict = "skipped"
        result.evaluator_note = str(exc)

    result.e_score = final_score.e_score
    result.s_score = final_score.s_score
    result.g_score = final_score.g_score
    result.e_reasoning = final_score.e_reasoning
    result.s_reasoning = final_score.s_reasoning
    result.g_reasoning = final_score.g_reasoning

    # Stage 3b: Core Metric Estimation (saves estimates to DB, source=agentic_metrics_v1)
    log.info("=== Stage 3b: Core Metric Estimation ===")
    try:
        from agentic_estimation.layer_3.metric_estimation_agent import estimate_and_save
        baseline = _resolve_baseline(result.country)
        result.metric_estimates = await estimate_and_save(
            company_name, company_id, industry, result.country, signals, metadata, baseline,
        )
        log.info("Saved %d estimated core metrics", len(result.metric_estimates))
    except Exception as exc:
        log.warning("Metric estimation failed (%s) — skipped", exc)

    # Stage 4: Explainability (saves summary to DB)
    log.info("=== Stage 4/5: Explainability Agent ===")
    try:
        from agentic_estimation.layer_4.explainability_agent import explain_company
        expl = await explain_company(
            score=final_score,
            company_id=company_id,
            industry=industry,
        )
        if expl is None:
            log.warning("Explainability returned None — summary skipped")
        else:
            result.summary = expl.summary
            log.info("Summary saved (%d chars)", len(expl.summary))
    except Exception as exc:
        log.warning("Explainability failed (%s) — summary skipped", exc)

    await _set_esg_scoring_status(company_id, "estimated")

    result.elapsed_s = time.monotonic() - t0
    result.saved_to_db = True
    log.info("Pipeline complete in %.1fs — all outputs saved to DB", result.elapsed_s)
    return result


# ── Metrics-only run (gap-fill for companies that already have pillar data) ────

async def run_metrics_only(
    company_name: str,
    company_id: UUID,
    industry: str = "",
    country: Optional[str] = None,
) -> dict:
    """
    Estimate and save ONLY the core star metrics for a company — no pillar
    scoring, evaluator, explainability, or status change. Used to gap-fill
    reported/already-estimated companies so every company has a uniform core
    metric set. Real disclosed values still win in build_esg_json.

    Returns the estimate dict (empty on failure).
    """
    t0 = time.monotonic()
    log_header(log, "Metrics-Only Run",
               company=company_name, industry=industry or "N/A",
               country=country or "auto-detect", company_id=str(company_id))

    # Signals — DB-cached (reuses prior gather; avoids re-hammering the web)
    try:
        signals = _run_signals(company_name, industry, company_id=company_id, country=country)
        log.info("Signals gathered: %d sources", len(signals))
    except Exception as exc:
        log.error("Metrics-only: signal gathering failed: %s", exc)
        signals = {}

    # Metadata (cached in DB by company_id where possible)
    metadata = {}
    try:
        metadata = _run_metadata(company_name, company_id=company_id)
    except Exception as exc:
        log.warning("Metrics-only: metadata lookup failed (%s)", exc)

    resolved_country = country or metadata.get("country")
    baseline = _resolve_baseline(resolved_country)

    try:
        from agentic_estimation.layer_3.metric_estimation_agent import estimate_and_save
        estimates = await estimate_and_save(
            company_name, company_id, industry, resolved_country, signals, metadata, baseline,
        )
    except Exception as exc:
        log.error("Metrics-only: estimation failed: %s", exc)
        estimates = {}

    log.info("Metrics-only complete in %.1fs — %d metrics", time.monotonic() - t0, len(estimates))
    return estimates


# ── Programmatic helper for API use ───────────────────────────────────────────

async def run_for_company_name(
    company_name: str,
    industry: str = "",
    country: Optional[str] = None,
) -> PipelineResult:
    """
    Convenience wrapper for the API: looks up company_id from the DB by name,
    then runs the full pipeline. Raises RuntimeError if company not found.

    Usage in FastAPI:
        from agentic_estimation.orchestrator import run_for_company_name
        result = await run_for_company_name("Bosch", industry="Industrial Machinery")
    """
    from agentic_estimation.shared.db_company_lookup import resolve_company_id_standalone

    match = await resolve_company_id_standalone(company_name)
    if match is None:
        raise RuntimeError(f"Company '{company_name}' not found in DB")
    company_id, actual_name = match

    return await run_company(actual_name, company_id, industry=industry, country=country)


# ── CLI output ─────────────────────────────────────────────────────────────────

def _print_result(result: PipelineResult) -> None:
    width = 54
    sep = "=" * width
    print(f"\n{sep}")
    print(f"  ESG Pipeline Result — {result.company}")
    print(sep)
    print(f"  Industry  : {result.industry or 'N/A'}")
    print(f"  Country   : {result.country or 'Unknown'}")
    print(f"  Signals   : {result.signals_count} sources")
    print(f"  Elapsed   : {result.elapsed_s:.1f}s")
    print(f"  DB saved  : {'yes' if result.saved_to_db else 'no (dry run)'}")
    print(sep)

    if result.error:
        print(f"  ERROR: {result.error}")
        print(sep)
        return

    print(f"  E score   : {result.e_score:.1f}/100")
    print(f"            {result.e_reasoning}")
    print(f"  S score   : {result.s_score:.1f}/100")
    print(f"            {result.s_reasoning}")
    print(f"  G score   : {result.g_score:.1f}/100")
    print(f"            {result.g_reasoning}")
    print(sep)
    print(f"  Evaluator : {result.evaluator_verdict.upper()}")
    print(f"            {result.evaluator_note}")
    print(sep)
    if result.summary:
        print(f"\n  Summary:\n")
        # Word-wrap at ~70 chars
        words = result.summary.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 70:
                lines.append("  " + " ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append("  " + " ".join(line))
        print("\n".join(lines))
    else:
        print("  Summary   : (not generated)")
    print(f"\n{sep}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m agentic_estimation.orchestrator dry <company> [--industry ...] [--country ...]")
        print("  python -m agentic_estimation.orchestrator run <company> [--industry ...] [--country ...] [--id <uuid>]")
        sys.exit(1)

    mode = sys.argv[1]
    company = sys.argv[2]
    industry, country, company_id_str = "", None, None

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--industry" and i + 1 < len(args):
            industry = args[i + 1]; i += 2
        elif args[i] == "--country" and i + 1 < len(args):
            country = args[i + 1]; i += 2
        elif args[i] == "--id" and i + 1 < len(args):
            company_id_str = args[i + 1]; i += 2
        else:
            i += 1

    if mode == "dry":
        result = run_company_dry(company, industry=industry, country=country)
        _print_result(result)
        sys.exit(0 if result.ok else 1)

    elif mode == "run":
        async def _run():
            if company_id_str:
                cid = UUID(company_id_str)
                result = await run_company(company, cid, industry=industry, country=country)
            else:
                result = await run_for_company_name(company, industry=industry, country=country)
            _print_result(result)
            sys.exit(0 if result.ok else 1)

        asyncio.run(_run())

    else:
        print(f"Unknown mode '{mode}'. Use 'dry' or 'run'.")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
