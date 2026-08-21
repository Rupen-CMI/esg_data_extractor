"""
estimate_verifier.py — Phase 4 orchestration: composes the already-built
QC gate (confidence_gate.py) with the new gated critic panel
(critic_panel.py) and a bounded retry loop. See PHASE_4_PLAN.md sections
2-4 for the full design, including the five gaps patched into the plan
before this was built (retryability guard, selective claim replacement,
fail-closed re-extraction, the documented agreement-by-corruption risk,
and the rich-company vs thin-population test split).

This module does NOT replace confidence_gate.py -- qc_assess() and gate()
are called here exactly as the harness/graph already call them. The only
new decision this module adds is WHEN to additionally run the critic
panel, and what to do with its verdict.

No DB writes. No LLM calls when a pillar is already routed to skip.
"""

from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.layer_3.confidence_gate import qc_assess, gate

log = get_logger("estimate_verifier")

_MAX_RETRIES = 1
_MAX_PANEL_ROUNDS = 2

# Methods whose winning claim cannot be changed by re-extraction -- a
# converged flagged_factor on one of these can't trigger a retry (see
# PHASE_4_PLAN.md section 2, decision 4 / gap 1).
_NON_RETRYABLE_METHODS = {"peer_anchor", "dataset_lookup", "peer_ratio_fallback", "coarse_bucket"}


@dataclass
class VerifiedScore:
    pillar: str
    mode: str                # 'point' | 'range'
    score: float
    low: float
    high: float
    confidence: str
    needs_review: bool
    verdict: str              # 'skipped' | 'passed' | 'passed_after_retry' | 'refuted'
    retried: bool = False
    objections: list = field(default_factory=list)
    reason: str = ""
    critic_calls: int = 0     # 3 per panel round actually run (0 if skipped)


def _factor_method(formula_score, factor_key: Optional[str]) -> Optional[str]:
    if not factor_key:
        return None
    for c in (formula_score.contributions or []):
        if c.factor == factor_key:
            return c.method
    return None


def _wrap_gated(pillar: str, gated_out, verdict: str, critic_calls: int = 0) -> VerifiedScore:
    return VerifiedScore(
        pillar=pillar, mode=gated_out.mode, score=gated_out.score, low=gated_out.low,
        high=gated_out.high, confidence=gated_out.confidence, needs_review=gated_out.needs_review,
        verdict=verdict, reason=gated_out.reason, critic_calls=critic_calls,
    )


def _retry_pillar(
    pillar: str, company: str, signals: dict, metadata: Optional[dict], country: Optional[str],
    claims: list, flagged_factor: str, objections: list, model: Optional[str] = None,
) -> Optional[list]:
    """Re-extract ONLY this pillar's `method == 'extracted'` claims, with the
    critics' objection injected. Returns the NEW full claim list for this
    pillar (re-extracted claims + preserved non-extracted claims), or None
    on re-extraction failure (caller must treat None as a failed retry --
    gap 3, fail-closed).

    Preserves dataset_lookup/peer_ratio/coarse_bucket claims for this
    pillar untouched -- re-extraction only ever produces 'extracted'
    claims, so blindly replacing "this pillar's claims" would silently
    delete real Climate TRACE / ratio-estimator evidence that re-extraction
    has no way to regenerate (gap 2)."""
    from agentic_estimation.layer_2.pillar_extractors import extract_pillar_claims
    from agentic_estimation.layer_2.claim_validators import validate_claims

    pillar_claims = [c for c in claims if c.pillar == pillar]
    preserved = [c for c in pillar_claims if c.method != "extracted"]

    objection_text = "; ".join(objections) if objections else "(no detail provided)"
    try:
        new_extracted = extract_pillar_claims(
            pillar, company, signals, metadata,
            objection={"flagged_factor": flagged_factor, "objection_text": objection_text},
            model=model,
        )
    except Exception as exc:
        log.warning("[%s/%s] retry re-extraction raised: %s -- treating as failed retry",
                    company, pillar, exc)
        return None

    if new_extracted is None:
        log.warning("[%s/%s] retry re-extraction returned None -- treating as failed retry",
                    company, pillar)
        return None

    # Empty re-extraction is not automatically a failure (the objection may
    # legitimately be "drop the claim, no real support exists" and the
    # extractor correctly emits nothing) -- but it IS a failure if the
    # underlying LLM call itself errored, which extract_pillar_claims
    # already logs and returns [] for. We can't distinguish "correctly
    # empty" from "LLM failed" at this layer without deeper plumbing, so we
    # accept an empty result here (preserved claims + zero extracted is a
    # valid retry outcome: the reviewer's objection was upheld and the
    # claim was dropped). A rescore on preserved-only claims is honest,
    # not gutted -- it's exactly what "drop it" should produce.
    new_claims_for_pillar, flags = validate_claims(new_extracted, signals=signals, country=country)

    other_pillar_claims = [c for c in claims if c.pillar != pillar]
    return other_pillar_claims + preserved + new_claims_for_pillar


def verify_reconciled(
    company: str,
    reconciled: dict,
    formula_scores: dict,
    holistic,
    claims: list,
    signals: dict,
    metadata: Optional[dict],
    country: Optional[str],
    model: Optional[str] = None,
) -> dict:
    """Returns dict[pillar, VerifiedScore]. Composes qc_assess()/gate() (the
    already-built Confidence Gate) with the gated critic panel. Never
    raises -- a critic-panel or retry failure degrades to the ORIGINAL
    gate output for that pillar, never a crash.

    model: optional override forwarded to run_critic_panel. None (default)
    preserves today's exact behavior."""
    from agentic_estimation.layer_4.critic_panel import run_critic_panel
    from agentic_estimation.layer_3.reconcile import reconcile_all

    qc = qc_assess(formula_scores)
    gated = gate(reconciled, qc)

    results: dict[str, VerifiedScore] = {}

    for pillar, gated_out in gated.items():
        rs = reconciled[pillar]
        fs = formula_scores[pillar]

        # Skip cases: already-range (thin QC or low confidence) never
        # benefits from critics (a critic cannot upgrade a range -- see
        # PHASE_4_PLAN.md section 2, decision 1). High-confidence + QC-ok
        # also skips (original locked decision, unchanged).
        if gated_out.mode == "range":
            results[pillar] = _wrap_gated(pillar, gated_out, verdict="skipped")
            continue
        if rs.confidence == "high":
            results[pillar] = _wrap_gated(pillar, gated_out, verdict="skipped")
            continue
        if rs.confidence != "medium":
            # Defensive: point mode with a confidence label that's neither
            # 'high' nor 'medium' shouldn't happen (gate() only emits point
            # when confidence != 'low' and qc == 'ok'), but fail safe to
            # skip rather than assume medium-zone behavior.
            results[pillar] = _wrap_gated(pillar, gated_out, verdict="skipped")
            continue

        # medium confidence + QC ok -> run the panel.
        try:
            panel = run_critic_panel(pillar, company, rs, fs, holistic, claims, signals, metadata,
                                      model=model)
        except Exception as exc:
            log.warning("[%s/%s] critic panel raised: %s -- falling back to original gate output",
                        company, pillar, exc)
            results[pillar] = _wrap_gated(pillar, gated_out, verdict="skipped")
            continue

        critic_calls = 3  # one call per lens, this round

        if not panel.refuted:
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="point", score=gated_out.score, low=gated_out.low,
                high=gated_out.high, confidence=rs.confidence, needs_review=False,
                verdict="passed", reason="critic panel: majority pass", critic_calls=critic_calls,
            )
            continue

        # Refuted. Retryable only if a factor converged AND that factor's
        # winning claim is actually re-extractable (gap 1).
        flagged = panel.flagged_factor
        flagged_method = _factor_method(fs, flagged) if flagged else None
        retryable = flagged is not None and flagged_method not in _NON_RETRYABLE_METHODS

        if not retryable:
            reason = (f"refuted, no retryable convergence (flagged={flagged!r}, "
                      f"method={flagged_method!r})")
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="range", score=rs.score, low=rs.low, high=rs.high,
                confidence="low", needs_review=True, verdict="refuted",
                objections=panel.objections, reason=reason, critic_calls=critic_calls,
            )
            continue

        # Bounded retry (max 1).
        new_claims = _retry_pillar(pillar, company, signals, metadata, country, claims,
                                    flagged, panel.objections, model=model)
        if new_claims is None:
            # Gap 3: re-extraction failed -> fail closed, do NOT rescore on
            # a gutted/unknown claim set.
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="range", score=rs.score, low=rs.low, high=rs.high,
                confidence="low", needs_review=True, verdict="refuted", retried=True,
                objections=panel.objections, reason="retry re-extraction failed",
                critic_calls=critic_calls,
            )
            continue

        from agentic_estimation.layer_3.formula_estimator import compute_formula_scores
        retried_formula_scores = compute_formula_scores(
            new_claims, country, metadata, company_name=company, sector=metadata.get("industry") if metadata else None,
            signals=signals, truth_source="upright",
        )
        retried_reconciled_all = reconcile_all(retried_formula_scores, holistic)
        retried_rs = retried_reconciled_all[pillar]
        retried_fs = retried_formula_scores[pillar]

        retried_qc = qc_assess({pillar: retried_fs})
        retried_gated = gate({pillar: retried_rs}, retried_qc)[pillar]

        if retried_gated.mode == "range" or retried_rs.confidence != "medium":
            # Re-reconciling on corrected evidence itself resolved the
            # ambiguity (either genuinely thin now, or confidently high) --
            # accept that routing without spending a second panel round.
            verdict = "passed_after_retry" if retried_gated.mode == "point" else "refuted"
            results[pillar] = VerifiedScore(
                pillar=pillar, mode=retried_gated.mode, score=retried_gated.score,
                low=retried_gated.low, high=retried_gated.high,
                confidence=retried_rs.confidence if retried_gated.mode == "point" else "low",
                needs_review=(retried_gated.mode == "range"), verdict=verdict, retried=True,
                objections=panel.objections, reason="retry changed evidence; re-routed by gate",
                critic_calls=critic_calls,
            )
            continue

        try:
            panel2 = run_critic_panel(pillar, company, retried_rs, retried_fs, holistic,
                                       new_claims, signals, metadata, model=model)
        except Exception as exc:
            log.warning("[%s/%s] round-2 critic panel raised: %s -- treating as refuted",
                        company, pillar, exc)
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="range", score=retried_rs.score, low=retried_rs.low,
                high=retried_rs.high, confidence="low", needs_review=True, verdict="refuted",
                retried=True, objections=panel.objections, reason=f"round-2 panel error: {exc}",
                critic_calls=critic_calls,
            )
            continue

        critic_calls += 3

        if not panel2.refuted:
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="point", score=retried_gated.score, low=retried_gated.low,
                high=retried_gated.high, confidence=retried_rs.confidence, needs_review=False,
                verdict="passed_after_retry", retried=True, objections=panel.objections,
                reason="round-2 panel: majority pass", critic_calls=critic_calls,
            )
        else:
            # Second refute is final -- no second retry (locked decision 3).
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="range", score=retried_rs.score, low=retried_rs.low,
                high=retried_rs.high, confidence="low", needs_review=True, verdict="refuted",
                retried=True, objections=panel.objections + panel2.objections,
                reason="refuted again after retry -- final", critic_calls=critic_calls,
            )

    return results
