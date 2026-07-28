"""
ensemble_persistence.py — Phase 6 cutover: persist the ensemble scorer's
output (reconcile.py + confidence_gate.py + estimate_verifier.py) to
company_metric_values, the same table scoring_agent.py's _upsert_score
writes to for the legacy path.

WHY THIS EXISTS: the ensemble path (extract_claims -> formula_score ->
holistic_llm -> reconcile -> verify_estimate) computed a full, better-tested
score for every company, but until this module existed nothing wrote it
anywhere -- node_mark_estimated only flipped a status flag. This is the
missing write, using source 'agentic_ensemble_v1' so it coexists with
(and, via build_esg_json.py's updated _SOURCE_PRIORITY, outranks) rows from
'agentic_scoring_v1'/'agentic_evaluator_v1'.

Unlike the point-only legacy write, this also persists the Confidence
Gate's range/needs_review and (when Phase 4 verification ran) the critic
panel's verdict -- see db_migrations/005_ensemble_cutover.sql for the new
nullable columns this requires. needs_review reflects the gate's verdict
unconditionally (the gate always runs); only `verdict` requires --verify
and is NULL without it, since there's no gate-level equivalent for a
critic's pass/refute call.
"""

import os
from typing import Optional
from uuid import UUID

import asyncpg

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("ensemble_persistence")

_METRIC_KEYS = {"E": "esg_e_score", "S": "esg_s_score", "G": "esg_g_score"}


async def _get_metric_ids(conn: asyncpg.Connection) -> dict[str, UUID]:
    keys = list(_METRIC_KEYS.values())
    rows = await conn.fetch(
        "SELECT id, key FROM esg_metric_definitions WHERE key = ANY($1::text[])", keys,
    )
    mapping = {row["key"]: UUID(str(row["id"])) for row in rows}
    missing = set(keys) - set(mapping.keys())
    if missing:
        raise RuntimeError(
            f"Missing metric definitions in DB: {missing}. Run db_migrations/001_agentic_pipeline.sql."
        )
    return mapping


# reconcile.py's ReconciledScore has no single numeric confidence -- only the
# 3-way label (see reconcile.py's own docstring on why: per-vote
# self_confidence values aren't reducible to one number without inventing a
# weighting scheme). This is a fixed, documented, APPROXIMATE mapping so the
# numeric `confidence` column (populated by the legacy metric_estimation_agent
# path, company_metric_values.confidence, nullable) isn't silently left NULL
# on every ensemble-scored row -- not a measured probability, just a
# consistent ordinal stand-in matching the label it's derived from.
_CONFIDENCE_LABEL_TO_NUMERIC = {"high": 0.9, "medium": 0.6, "low": 0.3}


async def _upsert_ensemble_score(
    conn: asyncpg.Connection,
    company_id: UUID,
    metric_id: UUID,
    score: float,
    low: float,
    high: float,
    confidence_label: str,
    reasoning: str,
    verdict: Optional[str],
    needs_review: Optional[bool],
    source: str = "agentic_ensemble_v1",
) -> None:
    display_value = (f"{low:.1f}-{high:.1f}/100" if needs_review else f"{score:.1f}/100")
    confidence_numeric = _CONFIDENCE_LABEL_TO_NUMERIC.get(confidence_label)
    await conn.execute(
        """
        INSERT INTO company_metric_values
            (company_id, metric_id, numeric_value, value, confidence, reasoning, source, reporting_year,
             low_value, high_value, confidence_label, verdict, needs_review)
        VALUES ($1, $2, $3, $4, $5, $6, $7, EXTRACT(YEAR FROM now())::int, $8, $9, $10, $11, $12)
        ON CONFLICT (company_id, metric_id, reporting_year, source)
        DO UPDATE SET
            numeric_value    = EXCLUDED.numeric_value,
            value            = EXCLUDED.value,
            confidence       = EXCLUDED.confidence,
            reasoning        = EXCLUDED.reasoning,
            low_value        = EXCLUDED.low_value,
            high_value       = EXCLUDED.high_value,
            confidence_label = EXCLUDED.confidence_label,
            verdict          = EXCLUDED.verdict,
            needs_review     = EXCLUDED.needs_review
        """,
        str(company_id), str(metric_id), score, display_value, confidence_numeric, reasoning, source,
        low, high, confidence_label, verdict, needs_review,
    )


async def persist_ensemble_scores(
    company_id: UUID,
    reconciled: dict,
    verified: Optional[dict],
    reasonings: dict,
    gated: Optional[dict] = None,
) -> None:
    """reconciled: dict[pillar, ReconciledScore]. verified: dict[pillar,
    VerifiedScore] or None (when Phase 4 verification wasn't run). gated:
    dict[pillar, GatedScore] or None -- confidence_gate.gate()'s output,
    which runs unconditionally regardless of --verify. needs_review comes
    from verified when present, else from gated (NOT hardcoded False/None --
    the gate alone already knows whether a score needs review; verify only
    adds critic verdicts on top). verdict persists as NULL only when verify
    genuinely didn't run, since there's no gate-level equivalent for it.
    reasonings: dict[pillar, str] -- the rendered vote/verdict text (reuse
    graph._populate_result_from_ensemble's _render output so the persisted
    reasoning matches what PipelineResult shows)."""
    db_url = os.environ.get("ASYNC_DB_URL", "")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set in environment")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    try:
        metric_ids = await _get_metric_ids(conn)
        for pillar in ("E", "S", "G"):
            rs = reconciled[pillar]
            vs = verified.get(pillar) if verified else None

            if vs is not None:
                score, low, high = vs.score, vs.low, vs.high
                needs_review = vs.needs_review
                verdict = vs.verdict
            else:
                score, low, high = rs.score, rs.low, rs.high
                gs = gated.get(pillar) if gated else None
                needs_review = gs.needs_review if gs is not None else None
                verdict = None

            metric_key = _METRIC_KEYS[pillar]
            await _upsert_ensemble_score(
                conn, company_id, metric_ids[metric_key], score, low, high,
                rs.confidence, reasonings.get(pillar, ""), verdict, needs_review,
            )
        log.info(
            "[%s] saved ensemble scores E=%.1f S=%.1f G=%.1f",
            company_id, reconciled["E"].score, reconciled["S"].score, reconciled["G"].score,
        )
    finally:
        await conn.close()
