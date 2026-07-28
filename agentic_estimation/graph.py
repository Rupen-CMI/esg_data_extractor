"""
graph.py — LangGraph orchestration for the ESG pipeline. THE PRODUCTION
ORCHESTRATOR as of the Phase 6 cutover (2026-07-21) — api/v1/esg_data/routes.py
calls run_company_graph(..., scorer="ensemble") for every full-pipeline
estimation; orchestrator.py's original scoring_agent-based run_company is no
longer invoked in production (kept only for calibration_harness.py's
--scorer llm A/B baseline).

Three scorer paths coexist in one graph, routed by the `scorer` state field:
  'llm'      — the original single-shot scorer (scoring -> evaluator ->
               metrics -> explainability). Legacy; calibration baseline only.
  'formula'  — deterministic formula only, no ensemble/verification. Used to
               isolate the formula's own skill in backtests.
  'ensemble' — THE PRODUCTION PATH: extract_claims (+ Tier-0 validators) ->
               formula_score (v5 saturation) -> holistic_llm -> reconcile
               (v8) -> verify_estimate (Confidence Gate + gated Phase-4
               critics) -> persist_ensemble_scores -> metrics_persist ->
               explainability_persist -> mark_estimated.

Every stage function is a thin wrapper around either orchestrator.py's
legacy _run_* helpers (llm path) or the layer_2/layer_3/layer_4 modules
directly (formula/ensemble paths) — reused, not reimplemented.

Dry vs. full run: ONE graph. A `dry_run` flag in the input state routes
around all DB-persisting nodes via conditional edges (dry ensemble runs never
reach persist_ensemble_scores/metrics_persist/explainability_persist —
calibration_harness.py depends on this staying true).

CLI:
    python -m agentic_estimation.graph dry "Patagonia" --industry "Outdoor Apparel" --scorer ensemble --verify
    python -m agentic_estimation.graph run "Bosch" --industry "Industrial Machinery" --country Germany --scorer ensemble --verify
"""

import asyncio
import sys
import time
from typing import Optional, TypedDict
from uuid import UUID

from langgraph.graph import StateGraph, END

from agentic_estimation.shared.pipeline_logger import get_logger, log_header, log_pipeline_start
from agentic_estimation.orchestrator import (
    PipelineResult,
    _run_signals,
    _run_metadata,
    _run_scoring,
    _run_evaluator,
    _run_explainability,
    _resolve_baseline,
    _set_esg_scoring_status,
)

log = get_logger("graph")


# ── Graph state ──────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # Input
    company_name: str
    company_id: Optional[UUID]
    industry: str
    country: Optional[str]
    dry_run: bool
    scorer: str               # 'llm' (default) | 'formula' | 'ensemble' -- see PHASE_2_PLAN.md Step 6, Phase 3 Step 4

    # Threaded through stages
    signals: dict
    metadata: dict
    score: object            # ESGScore (scorer='llm')
    final_score: object      # ESGScore (post-evaluator, scorer='llm')
    evaluator_verdict: str
    evaluator_note: str
    metric_estimates: dict
    summary: Optional[str]

    # Phase 2 formula path (scorer='formula')
    claims: list              # list[ExtractedClaim]
    formula_scores: dict      # {'E'|'S'|'G': PillarFormulaScore}

    # Phase 3 ensemble path (scorer='ensemble') -- reuses claims/formula_scores
    # above (same extract_claims -> formula_score nodes), adds a holistic LLM
    # vote and the reconciled merge on top.
    holistic_score: object    # ESGScore | None -- from holistic_estimator.holistic_vote()
    reconciled: dict          # {'E'|'S'|'G': ReconciledScore} -- from reconcile.reconcile_all()

    # Phase 4 verification (scorer='ensemble', opt-in via `verify` flag) --
    # dict[pillar, VerifiedScore] from estimate_verifier.verify_reconciled().
    # Absent/None when verification didn't run (verify=False or an error) --
    # _populate_result_from_ensemble falls back to the unverified reconciled
    # rendering in that case.
    verify: bool              # opt-in flag, mirrors calibration_harness.py's _VERIFY
    verified: dict            # {'E'|'S'|'G': VerifiedScore}

    # Control
    error: Optional[str]


# ── Nodes (each thinly wraps an existing orchestrator._run_* helper) ─────────

def node_signals(state: PipelineState) -> dict:
    log.info("=== Node: signals ===")
    try:
        # node_metadata now runs BEFORE signals (metadata -> signals edge,
        # fixed per DEFECT_FIX_PLAN.md 1.4 / PHASE_5_PLAN.md 0.3) and writes
        # state['country'] via _resolve_state_country -- this is the RESOLVED
        # country (input country, or metadata['country'] fallback, canonicalized
        # where possible), not just the raw input. A country only discoverable
        # from metadata (caller passed none) now reaches the localized_esg
        # signal source on the very first gather instead of never (no cache
        # TTL means missing it once was previously permanent per company).
        signals = _run_signals(state["company_name"], state["industry"],
                                company_id=state.get("company_id"), country=state.get("country"))
        log.info("Signals gathered: %d sources", len(signals))
        return {"signals": signals}
    except Exception as exc:
        return {"error": f"Signal agent failed: {exc}"}


def _resolve_state_country(input_country: Optional[str], metadata: dict) -> Optional[str]:
    """Single country-resolution point for every scorer route (node_metadata
    runs before signals and all scorer paths, so this result is available to
    every downstream node including node_signals). Previously ONLY the legacy llm nodes ever wrote
    state['country'] (from score.country, resolved internally by
    scoring_agent.py) -- the ensemble/formula paths just read back whatever
    was passed into run_company_graph unchanged, which is often "" or None
    (see e.g. api/v1/esg_data/routes.py's company.country or "" bug, fixed
    alongside this). That meant every downstream ensemble consumer (formula
    baseline, peer anchor, holistic vote, Tier-0 country-ceiling check) could
    silently run with NO country context even when metadata resolution
    would have found one.

    "" -> None (empty string is not a country). Falls back to
    metadata['country'] when input_country is empty. Canonicalizes via
    resolve_country_name() where possible; keeps the raw string when it
    doesn't resolve -- get_country_baseline_with_fallback has its own
    alias/regional/global chain downstream, so an unresolved-here string
    still gets a real fallback baseline rather than being dropped."""
    country = input_country or None
    if not country:
        meta_country = metadata.get("country")
        country = meta_country or None
    if not country:
        return None
    from agentic_estimation.layer_1.country_baseline_agent import resolve_country_name
    resolved = resolve_country_name(country)
    return resolved or country


def node_metadata(state: PipelineState) -> dict:
    log.info("=== Node: metadata ===")
    try:
        metadata = _run_metadata(state["company_name"], company_id=state.get("company_id"))
        log.info("Metadata source: %s", metadata.get("source") or "none")
    except Exception as exc:
        log.warning("Metadata lookup failed (%s) — scoring proceeds without it", exc)
        metadata = {}
    country = _resolve_state_country(state.get("country"), metadata)
    log.info("Resolved country: %s", country or "(none)")
    return {"metadata": metadata, "country": country}


# ── Phase 2 formula-scorer nodes (scorer='formula') ───────────────────────────

def _gather_all_signals(state: PipelineState) -> dict:
    """Merge signals from all three Layer-1 collectors. node_signals (already
    run before this) only covers signal_agent's sources -- the pillar
    extractors also need governance/facility collector text, which the old
    LLM scorer path never required."""
    from agentic_estimation.layer_1.governance_collector import get_or_fetch_governance_signals, fetch_governance_signals
    from agentic_estimation.layer_1.facility_extractor import get_or_fetch_facility_signals, fetch_facility_signals

    company_id = state.get("company_id")
    company_name = state["company_name"]
    industry = state["industry"]

    merged = dict(state.get("signals", {}))
    if company_id is not None:
        merged.update(get_or_fetch_governance_signals(company_id, company_name))
        merged.update(get_or_fetch_facility_signals(company_id, company_name, industry))
    else:
        merged.update(fetch_governance_signals(company_name))
        merged.update(fetch_facility_signals(company_name, industry))
    return merged


def _node_extract_claims(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: extract_claims (scorer=formula) ===")
    from agentic_estimation.layer_2.pillar_extractors import extract_all_claims
    from agentic_estimation.layer_2.climate_trace_anchor import ct_anchor_claims

    try:
        all_signals = _gather_all_signals(state)
        claims = extract_all_claims(state["company_name"], all_signals, state.get("metadata", {}))
        claims += ct_anchor_claims(
            state["company_name"], company_id=state.get("company_id"), country=state.get("country"),
        )

        from agentic_estimation.layer_2.claim_validators import validate_claims
        n_before = len(claims)
        claims, flags = validate_claims(claims, signals=all_signals, country=state.get("country"))
        if flags:
            n_dropped = n_before - len(claims)
            n_capped = sum(1 for f in flags if f.action == "capped")
            log.info("Tier-0 validators: %d claim(s) dropped, %d capped", n_dropped, n_capped)

        log.info("extracted %d claims (%d signals)", len(claims), len(all_signals))
        return {"signals": all_signals, "claims": claims}
    except Exception as exc:
        return {"error": f"Claim extraction failed: {exc}"}


def node_extract_claims_dry(state: PipelineState) -> dict:
    return _node_extract_claims(state)


async def node_extract_claims_persist(state: PipelineState) -> dict:
    """Same extraction as the dry path, plus persisting claims to
    company_evidence_claims (source_signal_id resolution happens inside
    persist_claims itself)."""
    result = _node_extract_claims(state)
    if result.get("error") or "claims" not in result:
        return result

    try:
        from agentic_estimation.layer_2.pillar_extractors import persist_claims
        n = await persist_claims(state["company_id"], result["claims"])
        log.info("persisted %d claims", n)
    except Exception as exc:
        log.warning("claim persistence failed (%s) -- scoring proceeds on in-memory claims", exc)
    return result


def node_formula_score(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: formula_score ===")
    from agentic_estimation.layer_3.formula_estimator import compute_formula_scores

    try:
        scores = compute_formula_scores(
            state.get("claims", []), state.get("country"), state.get("metadata", {}),
            company_name=state.get("company_name"), sector=state.get("industry"),
            signals=state.get("signals", {}),
        )
        log.info("formula scores: E=%.1f S=%.1f G=%.1f",
                  scores["E"].score, scores["S"].score, scores["G"].score)
        return {"formula_scores": scores}
    except Exception as exc:
        return {"error": f"Formula estimator failed: {exc}"}


# ── Phase 3 ensemble-scorer nodes (scorer='ensemble') ─────────────────────────
# Sequential, not LangGraph fan-out -- per the Phase 3 plan, fan-out adds
# state-merge complexity for zero benefit at just one extra estimator. Each
# node degrades to a missing vote on exception (holistic_vote/reconcile_all
# already do this internally), never errors the whole run.

def node_holistic_llm(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: holistic_llm (scorer=ensemble) ===")
    from agentic_estimation.layer_3.holistic_estimator import holistic_vote

    holistic = holistic_vote(
        state["company_name"], state["industry"], state.get("country"),
        state.get("signals", {}), state.get("metadata", {}),
    )
    if holistic is None:
        log.warning("holistic vote missing -- reconcile will fall back to formula-only")
    else:
        log.info("holistic vote: E=%.1f S=%.1f G=%.1f", holistic.e_score, holistic.s_score, holistic.g_score)
    return {"holistic_score": holistic}


def node_reconcile(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: reconcile (scorer=ensemble) ===")
    from agentic_estimation.layer_3.reconcile import reconcile_all

    try:
        reconciled = reconcile_all(state.get("formula_scores", {}), state.get("holistic_score"))
        log.info(
            "reconciled: E=%.1f(%s) S=%.1f(%s) G=%.1f(%s)",
            reconciled["E"].score, reconciled["E"].confidence,
            reconciled["S"].score, reconciled["S"].confidence,
            reconciled["G"].score, reconciled["G"].confidence,
        )
        return {"reconciled": reconciled}
    except Exception as exc:
        return {"error": f"Reconcile failed: {exc}"}


def node_verify_estimate(state: PipelineState) -> dict:
    """Phase 4, opt-in via state['verify']=True. Single node shared by dry
    and persist runs (verification itself writes nothing to the DB -- same
    boundary the Confidence Gate already respects). Composes the built
    qc_assess()/gate() with the gated critic panel + bounded retry -- see
    estimate_verifier.py and PHASE_4_PLAN.md. A verifier exception degrades
    to no verified output (never errors the whole run); downstream rendering
    falls back to the unverified reconciled score."""
    if state.get("error"):
        return {}
    if not state.get("verify"):
        return {}
    log.info("=== Node: verify_estimate (scorer=ensemble, Phase 4) ===")
    from agentic_estimation.layer_4.estimate_verifier import verify_reconciled

    try:
        verified = verify_reconciled(
            state["company_name"], state.get("reconciled", {}), state.get("formula_scores", {}),
            state.get("holistic_score"), state.get("claims", []), state.get("signals", {}),
            state.get("metadata", {}), state.get("country"),
        )
        log.info("verified: E=%s S=%s G=%s",
                  verified["E"].verdict, verified["S"].verdict, verified["G"].verdict)
        return {"verified": verified}
    except Exception as exc:
        log.warning("verify_estimate failed (%s) -- falling back to unverified gate output", exc)
        return {}


def node_scoring(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: scoring ===")
    try:
        score = _run_scoring(
            state["company_name"], state["industry"], state.get("country"),
            state.get("signals", {}), state.get("metadata", {}),
        )
        if score is None:
            return {"error": "Scoring agent returned None"}
        log.info("Scores: E=%.1f S=%.1f G=%.1f (country=%s)", score.e_score, score.s_score, score.g_score, score.country)
        return {"score": score, "country": score.country}
    except Exception as exc:
        return {"error": f"Scoring agent failed: {exc}"}


def node_evaluator(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: evaluator ===")
    score = state["score"]
    try:
        eval_result = _run_evaluator(score, state.get("signals", {}), state["industry"])
        if eval_result is None:
            log.warning("Evaluator returned None — keeping original scores")
            return {"final_score": score, "evaluator_verdict": "skipped",
                    "evaluator_note": "Evaluator LLM call failed; original scores kept"}
        log.info("Evaluator verdict: %s", eval_result.verdict.upper())
        return {"final_score": eval_result.final_score, "evaluator_verdict": eval_result.verdict,
                "evaluator_note": eval_result.evaluator_note}
    except Exception as exc:
        log.warning("Evaluator failed (%s) — keeping original scores", exc)
        return {"final_score": score, "evaluator_verdict": "skipped", "evaluator_note": str(exc)}


def node_metrics_dry(state: PipelineState) -> dict:
    """Dry-run metric estimation: estimate only, no DB write."""
    if state.get("error"):
        return {}
    log.info("=== Node: metric estimation (dry) ===")
    try:
        from agentic_estimation.layer_3.metric_estimation_agent import estimate_metrics_sync
        baseline = _resolve_baseline(state.get("country"))
        estimates = estimate_metrics_sync(
            state["company_name"], state["industry"], state.get("country"),
            state.get("signals", {}), state.get("metadata", {}), baseline,
        )
        log.info("Estimated %d core metrics", len(estimates))
        return {"metric_estimates": estimates}
    except Exception as exc:
        log.warning("Metric estimation failed (%s) — skipped", exc)
        return {"metric_estimates": {}}


def node_explainability(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: explainability ===")
    try:
        expl = _run_explainability(state["final_score"], state["industry"])
        if expl is None:
            log.warning("Explainability returned None — summary skipped")
            return {"summary": None}
        return {"summary": expl.summary}
    except Exception as exc:
        log.warning("Explainability failed (%s) — summary skipped", exc)
        return {"summary": None}


# ── DB-persisting nodes (full-run path only) ──────────────────────────────────

async def node_scoring_persist(state: PipelineState) -> dict:
    """Full-run scoring: score_company_sync + DB save, mirrors orchestrator.run_company Stage 2."""
    if state.get("error"):
        return {}
    log.info("=== Node: scoring (persist) ===")

    try:
        from agentic_estimation.layer_3.scoring_agent import score_company
        score = await score_company(
            company_name=state["company_name"], company_id=state["company_id"],
            industry=state["industry"], country=state.get("country"),
            signals=state.get("signals", {}), metadata=state.get("metadata", {}),
        )
        if score is None:
            return {"error": "Scoring agent returned None"}
        log.info("Scores saved: E=%.1f S=%.1f G=%.1f", score.e_score, score.s_score, score.g_score)
        return {"score": score, "country": score.country}
    except Exception as exc:
        return {"error": f"Scoring agent failed: {exc}"}


async def node_evaluator_persist(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: evaluator (persist) ===")
    score = state["score"]

    try:
        from agentic_estimation.layer_4.evaluator_agent import evaluate_company
        eval_result = await evaluate_company(
            score=score, company_id=state["company_id"],
            signals=state.get("signals", {}), industry=state["industry"],
        )
        if eval_result is None:
            log.warning("Evaluator returned None — keeping original scores")
            return {"final_score": score, "evaluator_verdict": "skipped",
                    "evaluator_note": "Evaluator LLM call failed; original scores kept"}
        log.info("Evaluator verdict: %s", eval_result.verdict.upper())
        return {"final_score": eval_result.final_score, "evaluator_verdict": eval_result.verdict,
                "evaluator_note": eval_result.evaluator_note}
    except Exception as exc:
        log.warning("Evaluator failed (%s) — keeping original scores", exc)
        return {"final_score": score, "evaluator_verdict": "skipped", "evaluator_note": str(exc)}


async def node_metrics_persist(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: metric estimation (persist) ===")

    try:
        from agentic_estimation.layer_3.metric_estimation_agent import estimate_and_save
        baseline = _resolve_baseline(state.get("country"))
        estimates = await estimate_and_save(
            state["company_name"], state["company_id"], state["industry"],
            state.get("country"), state.get("signals", {}), state.get("metadata", {}), baseline,
        )
        log.info("Saved %d estimated core metrics", len(estimates))
        return {"metric_estimates": estimates}
    except Exception as exc:
        log.warning("Metric estimation failed (%s) — skipped", exc)
        return {"metric_estimates": {}}


async def node_explainability_persist(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    log.info("=== Node: explainability (persist) ===")

    try:
        from agentic_estimation.layer_4.explainability_agent import explain_company
        expl = await explain_company(
            score=state["final_score"], company_id=state["company_id"], industry=state["industry"],
        )
        if expl is None:
            log.warning("Explainability returned None — summary skipped")
            return {"summary": None}
        log.info("Summary saved (%d chars)", len(expl.summary))
        return {"summary": expl.summary}
    except Exception as exc:
        log.warning("Explainability failed (%s) — summary skipped", exc)
        return {"summary": None}


def _ensemble_render(rs, gated_out, vscore) -> str:
    """Shared rendering used both for PipelineResult.e/s/g_reasoning
    (_populate_result_from_ensemble) and the persisted reasoning text
    (node_persist_ensemble_scores) -- kept in one place so the DB value
    always matches what a caller would see in the dry-run result."""
    vote_parts = [f"{v.estimator} {v.score:.1f} (w={rs.weights_used.get(v.estimator, 0):.2f})" for v in rs.votes]
    spread_str = f"{rs.spread:.1f}" if rs.spread is not None else "n/a"
    base = (f"{' | '.join(vote_parts)} -> {rs.score:.1f} in [{rs.low:.1f}, {rs.high:.1f}] "
            f"(spread={spread_str}, confidence={rs.confidence})")
    if vscore is not None:
        base += f" -- VERIFIED: {vscore.verdict}"
        if vscore.mode == "range":
            base += f", RANGE [{vscore.low:.1f}, {vscore.high:.1f}], needs review ({vscore.reason})"
        if vscore.objections:
            base += f" | objections: {'; '.join(vscore.objections)}"
    elif gated_out is not None and gated_out.mode == "range":
        base += f" -- RANGE [{gated_out.low:.1f}, {gated_out.high:.1f}], needs review ({gated_out.reason})"
    return base


async def node_persist_ensemble_scores(state: PipelineState) -> dict:
    """Full-run only (see the ensemble edge wiring below -- dry runs never
    reach this node): writes reconcile.py's (+ estimate_verifier.py's, when
    verify ran) output to company_metric_values via ensemble_persistence.py.
    This is the write the ensemble path has been missing since Phase 3 --
    node_mark_estimated only ever flipped a status flag; nothing before
    this node ever persisted a score. Also builds a final_score object
    (ESGScore-shaped) so node_explainability_persist -- built for the old
    scoring_agent.ESGScore contract -- can run unmodified on the ensemble
    path's output."""
    if state.get("error"):
        return {}
    log.info("=== Node: persist_ensemble_scores ===")
    reconciled = state.get("reconciled")
    if not reconciled:
        return {"error": "No reconciled scores to persist"}

    from agentic_estimation.layer_3.confidence_gate import qc_assess, gate
    formula_scores = state.get("formula_scores", {})
    qc = qc_assess(formula_scores) if formula_scores else {}
    gated = gate(reconciled, qc) if qc else {}
    verified = state.get("verified") or {}

    reasonings = {
        p: _ensemble_render(reconciled[p], gated.get(p), verified.get(p)) for p in ("E", "S", "G")
    }

    try:
        from agentic_estimation.layer_3.ensemble_persistence import persist_ensemble_scores
        await persist_ensemble_scores(state["company_id"], reconciled, verified or None, reasonings, gated)
    except Exception as exc:
        return {"error": f"Ensemble score persistence failed: {exc}"}

    from agentic_estimation.layer_3.scoring_agent import ESGScore
    final_score = ESGScore(
        company=state["company_name"],
        e_score=reconciled["E"].score, s_score=reconciled["S"].score, g_score=reconciled["G"].score,
        e_reasoning=reasonings["E"], s_reasoning=reasonings["S"], g_reasoning=reasonings["G"],
        country=state.get("country"), signals_used=len(state.get("signals", {})),
    )
    return {"final_score": final_score}


async def node_mark_processing(state: PipelineState) -> dict:
    await _set_esg_scoring_status(state["company_id"], "processing")
    return {}


async def node_mark_estimated(state: PipelineState) -> dict:
    if state.get("error"):
        return {}
    await _set_esg_scoring_status(state["company_id"], "estimated")
    return {}


# ── Conditional routing ───────────────────────────────────────────────────────

def _route_after_signals(state: PipelineState) -> str:
    """Routes on (dry_run, scorer) -- called after the signals node (metadata
    now runs BEFORE signals; see PHASE_5_PLAN.md 0.3). The old LLM path
    (scoring_dry/persist) remains for calibration_harness.py's --scorer llm
    baseline only -- it is no longer the production path (see module
    docstring, Phase 6 cutover). ensemble reuses the exact same
    extract_claims/formula_score nodes as formula -- they diverge only AFTER
    formula_score (see _route_after_formula_score)."""
    is_full = not state.get("dry_run", True)
    if state.get("scorer") in ("formula", "ensemble"):
        return "extract_claims_persist" if is_full else "extract_claims_dry"
    return "scoring_persist" if is_full else "scoring_dry"


def _route_after_formula_score(state: PipelineState) -> str:
    """formula stops here (-> mark_estimated/END); ensemble continues on to
    the holistic LLM vote + reconcile before ending."""
    if state.get("scorer") == "ensemble":
        return "holistic_llm"
    return "mark_estimated" if not state.get("dry_run", True) else "end"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("mark_processing", node_mark_processing)
    g.add_node("signals", node_signals)
    g.add_node("metadata", node_metadata)
    g.add_node("scoring_dry", node_scoring)
    g.add_node("scoring_persist", node_scoring_persist)
    g.add_node("evaluator_dry", node_evaluator)
    g.add_node("evaluator_persist", node_evaluator_persist)
    g.add_node("metrics_dry", node_metrics_dry)
    g.add_node("metrics_persist", node_metrics_persist)
    g.add_node("explainability_dry", node_explainability)
    g.add_node("explainability_persist", node_explainability_persist)
    g.add_node("mark_estimated", node_mark_estimated)

    # Phase 2 formula path -- retained for calibration_harness.py's
    # --scorer formula isolation runs (see PHASE_2_PLAN.md Step 6); the
    # ensemble path below (which shares these two nodes) is production.
    g.add_node("extract_claims_dry", node_extract_claims_dry)
    g.add_node("extract_claims_persist", node_extract_claims_persist)
    g.add_node("formula_score", node_formula_score)

    # Phase 3 ensemble path -- shares extract_claims_*/formula_score with the
    # formula path above (same claims + formula scores are the ensemble's
    # first vote), diverges after formula_score to add a holistic LLM vote
    # and reconcile the two.
    g.add_node("holistic_llm", node_holistic_llm)
    g.add_node("reconcile", node_reconcile)
    g.add_node("verify_estimate", node_verify_estimate)
    g.add_node("persist_ensemble_scores", node_persist_ensemble_scores)

    # metadata runs BEFORE signals (PHASE_5_PLAN.md 0.3 / DEFECT_FIX_PLAN.md 1.4):
    # node_metadata is the single country-resolution point (_resolve_state_country),
    # and node_signals needs the RESOLVED country -- not just the raw input -- to
    # reach the localized_esg signal source on the very first gather for a company
    # whose country is only discoverable from metadata (no cache TTL means missing
    # this once was previously permanent).
    def _entry(state: PipelineState) -> str:
        return "mark_processing" if not state.get("dry_run", True) else "metadata"

    g.set_conditional_entry_point(_entry, {"mark_processing": "mark_processing", "metadata": "metadata"})
    g.add_edge("mark_processing", "metadata")
    g.add_edge("metadata", "signals")

    g.add_conditional_edges(
        "signals", _route_after_signals,
        {
            "scoring_dry": "scoring_dry", "scoring_persist": "scoring_persist",
            "extract_claims_dry": "extract_claims_dry", "extract_claims_persist": "extract_claims_persist",
        },
    )

    g.add_edge("holistic_llm", "reconcile")
    g.add_edge("reconcile", "verify_estimate")
    # Full runs continue on to persist scores + metrics + explainability
    # (mirroring the LLM path's evaluator->metrics->explainability chain);
    # dry runs never persist anything, same as every other dry branch.
    g.add_conditional_edges(
        "verify_estimate",
        lambda state: "persist_ensemble_scores" if not state.get("dry_run", True) else "end",
        {"persist_ensemble_scores": "persist_ensemble_scores", "end": END},
    )
    g.add_edge("persist_ensemble_scores", "metrics_persist")

    g.add_edge("scoring_dry", "evaluator_dry")
    g.add_edge("evaluator_dry", "metrics_dry")
    g.add_edge("metrics_dry", "explainability_dry")
    g.add_edge("explainability_dry", END)

    g.add_edge("scoring_persist", "evaluator_persist")
    g.add_edge("evaluator_persist", "metrics_persist")
    g.add_edge("metrics_persist", "explainability_persist")
    g.add_edge("explainability_persist", "mark_estimated")
    g.add_edge("mark_estimated", END)

    # Formula path: extract_claims -> formula_score -> END (dry) or
    # -> mark_estimated -> END (full). No evaluator/metrics/explainability --
    # Phase 2 isolates the formula's own skill (same "--with-evaluator ignored"
    # rule as calibration_harness.py's --scorer formula).
    g.add_edge("extract_claims_dry", "formula_score")
    g.add_edge("extract_claims_persist", "formula_score")
    g.add_conditional_edges(
        "formula_score", _route_after_formula_score,
        {"holistic_llm": "holistic_llm", "mark_estimated": "mark_estimated", "end": END},
    )

    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ── Public entry points (drop-in for orchestrator.run_company_dry / run_company) ──

def _populate_result_common(result: PipelineResult, final_state: dict) -> None:
    """Fields both scorer paths fill identically."""
    result.signals = final_state.get("signals", {})
    result.signals_count = len(result.signals)
    result.error = final_state.get("error")
    result.summary = final_state.get("summary")
    result.metric_estimates = final_state.get("metric_estimates", {})


def _populate_result_from_llm(result: PipelineResult, final_state: dict) -> None:
    final_score = final_state.get("final_score") or final_state.get("score")
    if final_score is not None:
        result.e_score, result.s_score, result.g_score = final_score.e_score, final_score.s_score, final_score.g_score
        result.e_reasoning, result.s_reasoning, result.g_reasoning = (
            final_score.e_reasoning, final_score.s_reasoning, final_score.g_reasoning,
        )
    result.evaluator_verdict = final_state.get("evaluator_verdict", "")
    result.evaluator_note = final_state.get("evaluator_note", "")


def _populate_result_from_formula(result: PipelineResult, final_state: dict) -> None:
    """Maps formula_scores (PillarFormulaScore per pillar, with a full
    contributions audit trail) onto PipelineResult's e/s/g_score fields --
    the reasoning fields carry a compact rendering of each contribution so
    the same _print_result() works for both scorer paths."""
    scores = final_state.get("formula_scores")
    if not scores:
        return
    result.e_score, result.s_score, result.g_score = scores["E"].score, scores["S"].score, scores["G"].score

    def _render(pfs) -> str:
        if not pfs.contributions:
            return f"No evidence -- score is the country baseline ({pfs.baseline:.1f}, source={pfs.baseline_source})."
        parts = [f"{c.factor} ({c.points:+.1f})" for c in
                 sorted(pfs.contributions, key=lambda x: -abs(x.points))]
        return f"baseline {pfs.baseline:.1f} ({pfs.baseline_source}) + " + ", ".join(parts)

    result.e_reasoning, result.s_reasoning, result.g_reasoning = (
        _render(scores["E"]), _render(scores["S"]), _render(scores["G"]),
    )
    result.evaluator_verdict = "n/a (formula scorer)"


def _populate_result_from_ensemble(result: PipelineResult, final_state: dict) -> None:
    """Maps reconciled (ReconciledScore per pillar) onto PipelineResult --
    reasoning renders the vote breakdown (formula weight/confidence, holistic
    weight/confidence, spread) so the merge is auditable the same way a
    formula-only run's contribution list is."""
    reconciled = final_state.get("reconciled")
    if not reconciled:
        # holistic_llm/reconcile never ran (e.g. formula_score itself errored)
        # -- fall back to whatever formula_scores exist, same as scorer='formula'.
        _populate_result_from_formula(result, final_state)
        return
    result.e_score, result.s_score, result.g_score = (
        reconciled["E"].score, reconciled["S"].score, reconciled["G"].score,
    )

    # Confidence Gate (confidence_gate.py): point score vs range + needs_review.
    # Numeric score fields above are unchanged either way -- PipelineResult
    # (the in-memory dataclass) still has no band fields; the gate's verdict
    # is rendered into the reasoning string instead. Ranges/verdicts ARE now
    # persisted to the DB (company_metric_values.low_value/high_value/verdict
    # -- see ensemble_persistence.py, node_persist_ensemble_scores), so this
    # is only a display-object limitation, not a data-loss one.
    gated = {}
    formula_scores = final_state.get("formula_scores")
    if formula_scores:
        try:
            from agentic_estimation.layer_3.confidence_gate import qc_assess, gate
            qc = qc_assess(formula_scores)
            gated = gate(reconciled, qc)
        except Exception as exc:
            log.warning("confidence gate failed (%s) -- reasoning falls back to ungated rendering", exc)

    # Phase 4 verify (estimate_verifier.py), when the verify_estimate node
    # ran (state['verify']=True and it didn't error). VerifiedScore's
    # mode/low/high/reason SUPERSEDE the plain gate output above for
    # rendering purposes when present -- e.g. a successful critic retry can
    # change the routing (range -> point) relative to the pre-verification
    # gate snapshot. Falls back to `gated` when verification didn't run.
    verified = final_state.get("verified") or {}

    result.e_reasoning, result.s_reasoning, result.g_reasoning = (
        _ensemble_render(reconciled["E"], gated.get("E"), verified.get("E")),
        _ensemble_render(reconciled["S"], gated.get("S"), verified.get("S")),
        _ensemble_render(reconciled["G"], gated.get("G"), verified.get("G")),
    )
    result.evaluator_verdict = "n/a (ensemble scorer)"


def run_company_dry_graph(
    company_name: str, industry: str = "", country: Optional[str] = None, scorer: str = "llm",
    verify: bool = False,
) -> PipelineResult:
    """Same contract as orchestrator.run_company_dry, routed through the LangGraph scaffold.
    scorer='formula' routes through the Phase 2 evidence-claims + deterministic formula path
    instead of the original single-shot LLM scorer -- see PHASE_2_PLAN.md Step 6.
    verify=True (ensemble scorer only) additionally runs Phase 4's gated
    critic panel + bounded retry via the verify_estimate node -- see
    estimate_verifier.py and PHASE_4_PLAN.md. Adds real LLM cost on pillars
    landing medium-confidence + QC-ok; ignored for scorer != 'ensemble'.

    INVARIANT (DEFECT_FIX_PLAN.md 1.3): this calls the SYNC graph.invoke(),
    which cannot await a coroutine node. This is only safe because every
    node that is `async def` (mark_processing, mark_estimated,
    extract_claims_persist, scoring_persist, evaluator_persist,
    metrics_persist, explainability_persist, persist_ensemble_scores) is
    reachable ONLY on the full-run path -- both _entry and
    _route_after_signals route strictly on `not state.get("dry_run", True)`,
    and this function always passes dry_run=True. If a future change ever
    lets a dry-run state reach one of those nodes, graph.invoke() would need
    to run it as a bare (un-awaited) coroutine -- a real, structural bug, not
    a nested-asyncio.run() one. Verify routing before adding an async node
    reachable from here."""
    t0 = time.monotonic()
    log_pipeline_start(company_name, mode="dry-graph")
    log_header(log, "Graph Orchestrator — DRY RUN",
               company=company_name, industry=industry or "N/A", country=country or "auto-detect",
               scorer=scorer, verify=verify)

    graph = get_graph()
    final_state = graph.invoke({
        "company_name": company_name, "industry": industry, "country": country,
        "dry_run": True, "scorer": scorer, "verify": verify and scorer == "ensemble",
    })

    result = PipelineResult(company=company_name, industry=industry, country=final_state.get("country") or country)
    _populate_result_common(result, final_state)
    if scorer == "ensemble":
        _populate_result_from_ensemble(result, final_state)
    elif scorer == "formula":
        _populate_result_from_formula(result, final_state)
    else:
        _populate_result_from_llm(result, final_state)
    result.elapsed_s = time.monotonic() - t0
    log.info("Graph pipeline complete in %.1fs", result.elapsed_s)
    return result


async def run_company_graph(
    company_name: str, company_id: UUID, industry: str = "", country: Optional[str] = None, scorer: str = "llm",
    verify: bool = False,
) -> PipelineResult:
    """Same contract as orchestrator.run_company, routed through the LangGraph scaffold.
    scorer='formula' persists claims to company_evidence_claims and computes scores via
    the deterministic formula instead of the original single-shot LLM scorer.
    verify=True: see run_company_dry_graph's docstring -- same Phase 4 opt-in."""
    t0 = time.monotonic()
    log_pipeline_start(company_name, mode="full-graph")
    log_header(log, "Graph Orchestrator — FULL RUN",
               company=company_name, industry=industry or "N/A",
               country=country or "auto-detect", company_id=str(company_id), scorer=scorer, verify=verify)

    graph = get_graph()
    try:
        final_state = await graph.ainvoke({
            "company_name": company_name, "company_id": company_id,
            "industry": industry, "country": country, "dry_run": False, "scorer": scorer,
            "verify": verify and scorer == "ensemble",
        })
    except Exception:
        # node_mark_estimated is the ONLY writer of "estimated", and it
        # no-ops on state["error"] (correctly -- a half-scored company
        # shouldn't look done). But nothing else ever wrote a terminal
        # status on the error path, so a raised exception left the company
        # at "processing" (set by node_mark_processing) FOREVER -- and
        # get_market_esg's trigger loop unconditionally skips any company
        # already "processing", so this was a permanent orphan, not a
        # retryable failure. run_company_graph is the sole writer of
        # "failed" (mirrors node_mark_estimated being the sole writer of
        # "estimated") so routes.py can safely re-enqueue on "failed".
        await _set_esg_scoring_status(company_id, "failed")
        raise

    if final_state.get("error"):
        await _set_esg_scoring_status(company_id, "failed")

    result = PipelineResult(company=company_name, industry=industry, country=final_state.get("country") or country)
    _populate_result_common(result, final_state)
    if scorer == "ensemble":
        _populate_result_from_ensemble(result, final_state)
    elif scorer == "formula":
        _populate_result_from_formula(result, final_state)
    else:
        _populate_result_from_llm(result, final_state)
    result.elapsed_s = time.monotonic() - t0
    result.saved_to_db = result.error is None
    log.info("Graph pipeline complete in %.1fs — saved_to_db=%s", result.elapsed_s, result.saved_to_db)
    return result


# ── CLI (mirrors orchestrator.py's _cli) ──────────────────────────────────────

def _print_result(result: PipelineResult) -> None:
    from agentic_estimation.orchestrator import _print_result as _orig_print
    _orig_print(result)


def _cli() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m agentic_estimation.graph dry <company> [--industry ...] [--country ...] [--scorer llm|formula|ensemble] [--verify]")
        print("  python -m agentic_estimation.graph run <company> [--industry ...] [--country ...] [--id <uuid>] [--scorer llm|formula|ensemble] [--verify]")
        sys.exit(1)

    mode = sys.argv[1]
    company = sys.argv[2]
    industry, country, company_id_str, scorer, verify = "", None, None, "llm", False

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--industry" and i + 1 < len(args):
            industry = args[i + 1]; i += 2
        elif args[i] == "--country" and i + 1 < len(args):
            country = args[i + 1]; i += 2
        elif args[i] == "--id" and i + 1 < len(args):
            company_id_str = args[i + 1]; i += 2
        elif args[i] == "--scorer" and i + 1 < len(args):
            scorer = args[i + 1]; i += 2
        elif args[i] == "--verify":
            verify = True; i += 1
        else:
            i += 1

    if verify and scorer != "ensemble":
        print("ERROR: --verify requires --scorer ensemble")
        sys.exit(1)

    if mode == "dry":
        result = run_company_dry_graph(company, industry=industry, country=country, scorer=scorer, verify=verify)
        _print_result(result)
        sys.exit(0 if result.ok else 1)
    elif mode == "run":
        if not company_id_str:
            print("ERROR: --id <uuid> required for 'run' mode in this scaffold")
            sys.exit(1)
        result = asyncio.run(run_company_graph(company, UUID(company_id_str), industry=industry, country=country,
                                                scorer=scorer, verify=verify))
        _print_result(result)
        sys.exit(0 if result.ok else 1)
    else:
        print(f"Unknown mode '{mode}'. Use 'dry' or 'run'.")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
