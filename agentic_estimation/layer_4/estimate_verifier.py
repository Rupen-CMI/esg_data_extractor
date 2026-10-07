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
    verdict: str              # 'skipped' | 'passed' | 'passed_after_retry' | 'refuted' | 'inconclusive'
    retried: bool = False
    objections: list = field(default_factory=list)
    reason: str = ""
    critic_calls: int = 0     # up to 3 per panel round actually run (0 if
                              # skipped, 2 when the first two lenses already
                              # agreed and the 3rd was skipped -- see
                              # critic_panel.run_critic_panel's early exit)


def _factor_method(formula_score, factor_key: Optional[str]) -> Optional[str]:
    if not factor_key:
        return None
    for c in (formula_score.contributions or []):
        if c.factor == factor_key:
            return c.method
    return None


def _text_actionable_objections(panel) -> list:
    """Only evidence_support ever sees cited signal text (critic_panel.py's
    _critic_b_prompt/_critic_c_prompt never show it any) -- its objections
    are things a re-extraction can actually act on ("the text doesn't say
    this"). peer_plausibility/internal_consistency object to the
    *reconciled score's arithmetic* or a cross-estimator contradiction, not
    to the cited text, so feeding their objections into a "re-read the
    signal" prompt gives the extractor a complaint it structurally cannot
    address. Filters panel.verdicts (which carries per-critic attribution)
    rather than panel.objections (already-flattened, unattributed strings)."""
    return [v.objection for v in panel.verdicts
            if v.critic == "evidence_support" and v.verdict == "refute"]


def _flagged_claim_reasoning(claims: list, pillar: str, factor_key: Optional[str]) -> Optional[str]:
    """Returns the raw reasoning/evidence text of this pillar's ExtractedClaim(s)
    for factor_key (ExtractedClaim.reasoning -- NOT Contribution.claim_reasoning,
    a different dataclass), joined and sorted so re-extraction re-emitting the
    same claim(s) in a different order still compares equal. None if no claim
    exists for that factor (nothing to compare -- caller treats that as "not
    a no-op", since a factor going from present to absent IS a real change)."""
    if not factor_key:
        return None
    texts = sorted(
        c.reasoning for c in claims
        if c.pillar == pillar and c.factor == factor_key and getattr(c, "reasoning", None)
    )
    return "|".join(texts) if texts else None


def _wrap_gated(pillar: str, gated_out, verdict: str, critic_calls: int = 0) -> VerifiedScore:
    return VerifiedScore(
        pillar=pillar, mode=gated_out.mode, score=gated_out.score, low=gated_out.low,
        high=gated_out.high, confidence=gated_out.confidence, needs_review=gated_out.needs_review,
        verdict=verdict, reason=gated_out.reason, critic_calls=critic_calls,
    )


def _retry_pillar(
    pillar: str, company: str, signals: dict, metadata: Optional[dict], country: Optional[str],
    claims: list, flagged_factor: str, objections: list, model: Optional[str] = None,
    corrected_excerpt: Optional[str] = None,
) -> Optional[list]:
    """Re-extract ONLY this pillar's `method == 'extracted'` claims, with the
    critics' objection injected. Returns the NEW full claim list for this
    pillar (re-extracted claims + preserved non-extracted claims), or None
    on re-extraction failure (caller must treat None as a failed retry --
    gap 3, fail-closed).

    objections: pass ONLY text-actionable objections here (i.e. from
    evidence_support, the one lens shown cited signal text -- see
    _text_actionable_objections below). peer_plausibility/internal_consistency
    object to the *reconciled score's arithmetic*, not to the cited text;
    re-extraction has no way to act on "the swing is implausible given thin
    evidence" and dumping that into the same "re-read the text" framing the
    objection template uses just confuses the re-extraction LLM (found
    2026-09-21).

    corrected_excerpt: optional verbatim passage the evidence_support critic
    quoted from the FULL cited signal (see critic_panel.py's CriticPanelResult
    -- it now sees untruncated text specifically so it can quote a real
    passage instead of guessing blind). When present, forwarded into the
    extraction objection so the re-extraction LLM is pointed at a specific
    passage to verify against the original, rather than re-searching the
    same signal text from scratch a second time.

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
            objection={"flagged_factor": flagged_factor, "objection_text": objection_text,
                       "corrected_excerpt": corrected_excerpt},
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
    routing: Optional[dict] = None,
) -> dict:
    """Returns dict[pillar, VerifiedScore]. Composes qc_assess()/gate() (the
    already-built Confidence Gate) with the gated critic panel. Never
    raises -- a critic-panel or retry failure degrades to the ORIGINAL
    gate output for that pillar, never a crash.

    model: optional override forwarded to run_critic_panel. None (default)
    preserves today's exact behavior.

    routing: optional dict[pillar, {"route", "rung"}] from the evidence
    router (EVIDENCE_ROUTE_PLAN.md sec1.3/sec2), forwarded to gate() so its
    GatedOutput carries the real route/basis/rung instead of the rich/
    evidence/None defaults. None (default) preserves today's exact output."""
    from agentic_estimation.layer_4.critic_panel import run_critic_panel
    from agentic_estimation.layer_3.reconcile import reconcile_all

    qc = qc_assess(formula_scores)
    gated = gate(reconciled, qc, routing=routing)

    results: dict[str, VerifiedScore] = {}

    for pillar, gated_out in gated.items():
        rs = reconciled[pillar]
        fs = formula_scores[pillar]

        # Skip cases: already-range (thin QC or low confidence) never
        # benefits from critics (a critic cannot upgrade a range -- see
        # PHASE_4_PLAN.md section 2, decision 1). High-confidence + QC-ok
        # also skips (original locked decision, unchanged). basis=='prior'
        # (EVIDENCE_ROUTE_PLAN.md sec4.1) is an ADDITIONAL, redundant-by-
        # construction guard for the thin/ladder route -- a thin pillar's
        # ReconciledScore already has n_votes<=1 (holistic suppressed per
        # node_reconcile), so mode=='range' already fires for it via the
        # low_confidence branch; this check exists so the skip stays
        # correct even if `mode` is ever flipped to always emit 'range'
        # (sec4.1 lists this exact re-key as required before that flip --
        # not done yet, this is the re-keyed condition ready for when it is).
        if gated_out.mode == "range" or gated_out.basis == "prior":
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
            # A genuine crash inside run_critic_panel itself (a real code
            # bug -- LLM/network failures already abstain gracefully inside
            # _run_one_critic and never raise out of here). This must NOT
            # collapse to verdict="skipped": "skipped" means "never routed
            # to the panel at all" (high confidence / thin / basis=='prior'
            # above), which downstream readers (graph.py's
            # e/s/g_needs_review, explainability_agent.py's review flag)
            # take as "no review needed". A pillar that WAS sent to the
            # panel and then crashed is exactly the opposite -- unverified,
            # not verified-and-fine. Mirrors the inconclusive branch below:
            # same "attempted but couldn't reach a verdict" semantics, just
            # a code exception instead of <2 LLM responders. Found 2026-09-21.
            log.warning("[%s/%s] critic panel raised: %s -- falling back to original gate output, "
                        "flagged for review", company, pillar, exc)
            results[pillar] = VerifiedScore(
                pillar=pillar, mode=gated_out.mode, score=gated_out.score, low=gated_out.low,
                high=gated_out.high, confidence=gated_out.confidence, needs_review=True,
                verdict="inconclusive", reason=f"critic panel raised an exception: {exc}",
                critic_calls=0,
            )
            continue

        # len(panel.verdicts), not a hardcoded 3 -- run_critic_panel can
        # short-circuit to 2 calls when the first two already agree (see
        # its own docstring), so this must reflect what actually ran.
        critic_calls = len(panel.verdicts)

        if panel.inconclusive:
            # <2 critics responded -- an infra failure (LLM calls erroring
            # out), not a real disagreement. Fail-open keeps the original
            # point score (no actual evidence it's wrong), but this must
            # NOT read the same as a genuine majority pass -- flagged via
            # both a distinct verdict string and needs_review=True, so a
            # downstream consumer can tell "nobody actually checked" apart
            # from "3 critics reviewed it and agreed." Added 2026-09-19.
            log.warning("[%s/%s] critic panel inconclusive (too few responders) -- "
                        "keeping original score but flagging for review", company, pillar)
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="point", score=gated_out.score, low=gated_out.low,
                high=gated_out.high, confidence=rs.confidence, needs_review=True,
                verdict="inconclusive", reason="critic panel could not obtain enough responses "
                "to reach a verdict (fewer than 2 of 3 critics responded)", critic_calls=critic_calls,
            )
            continue

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

        # Bounded retry (max 1). Only evidence_support's objections are
        # text-actionable -- see _text_actionable_objections. The full,
        # unfiltered panel.objections still goes into every VerifiedScore's
        # own `objections` field elsewhere in this function (that's an
        # audit trail for a human/downstream reader, not extractor input).
        actionable_objections = _text_actionable_objections(panel)
        new_claims = _retry_pillar(pillar, company, signals, metadata, country, claims,
                                    flagged, actionable_objections, model=model,
                                    corrected_excerpt=panel.corrected_excerpt)
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

        # Gap 4: the retry may have re-emitted the SAME claim for the
        # flagged factor, ignoring the objection entirely (an LLM no-op --
        # not a call failure, so it isn't caught by the None check above).
        # Without this check a no-op retry sails into round 2 looking
        # "fixed" on paper, relying entirely on the critics noticing from
        # claim_reasoning text alone. Compare the flagged factor's winning
        # claim reasoning before vs after -- if it's byte-identical, the
        # objection plainly wasn't acted on. Found 2026-09-21.
        old_reasoning = _flagged_claim_reasoning(claims, pillar, flagged)
        new_reasoning = _flagged_claim_reasoning(new_claims, pillar, flagged)
        if old_reasoning is not None and old_reasoning == new_reasoning:
            log.warning("[%s/%s] retry re-extraction re-emitted an identical claim for flagged "
                        "factor %r -- objection was not acted on, treating as refuted",
                        company, pillar, flagged)
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="range", score=rs.score, low=rs.low, high=rs.high,
                confidence="low", needs_review=True, verdict="refuted", retried=True,
                objections=panel.objections, reason="retry re-emitted an identical claim for the "
                "flagged factor -- objection not acted on", critic_calls=critic_calls,
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
                # reuse gate()'s own needs_review verbatim, same convention
                # as _wrap_gated -- not re-derived from mode here, so this
                # can't silently drift from gate()'s real logic if it ever
                # grows a needs_review trigger beyond mode=='range' (found
                # 2026-09-21).
                needs_review=retried_gated.needs_review, verdict=verdict, retried=True,
                objections=panel.objections, reason="retry changed evidence; re-routed by gate",
                critic_calls=critic_calls,
            )
            continue

        try:
            panel2 = run_critic_panel(pillar, company, retried_rs, retried_fs, holistic,
                                       new_claims, signals, metadata, model=model,
                                       prior_round={"objections": panel.objections,
                                                    "corrected_excerpt": panel.corrected_excerpt})
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

        critic_calls += len(panel2.verdicts)  # same short-circuit accounting as round 1

        if panel2.inconclusive:
            # Same distinction as round 1 -- see that branch's comment.
            log.warning("[%s/%s] round-2 critic panel inconclusive (too few responders) -- "
                        "keeping retried score but flagging for review", company, pillar)
            results[pillar] = VerifiedScore(
                pillar=pillar, mode="point", score=retried_gated.score, low=retried_gated.low,
                high=retried_gated.high, confidence=retried_rs.confidence, needs_review=True,
                verdict="inconclusive", retried=True, objections=panel.objections,
                reason="round-2 critic panel could not obtain enough responses to reach a verdict",
                critic_calls=critic_calls,
            )
        elif not panel2.refuted:
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
