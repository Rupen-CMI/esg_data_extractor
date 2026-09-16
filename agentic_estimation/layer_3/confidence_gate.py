"""
confidence_gate.py — QC evidence-sufficiency + Confidence Gate
(UPDATED_AGENTIC_WORKFLOW.md's "Coverage & Evidence Mass Sufficient?" and
"Confidence Gate" diamonds). Pure code, no LLM, no DB.

WHY THIS EXISTS: reconcile.py already computes a per-pillar `confidence`
label ('high'/'medium'/'low') and a `[low, high]` range for every company,
every run -- but nothing in the live pipeline consumes them. A company with
zero real evidence (confirmed live this session: the seed=314 backtest
sample is dominated by companies with no findable web footprint) still gets
a confident-looking point score shipped, identical in shape to a company
with five corroborated claims. This module is the missing consumer: it
decides, per pillar, whether the output should be a point score (we have
real grounds for one) or an honest range + needs_review flag (we don't).

Evidence Recovery (the workflow doc's other consumer of the QC verdict) is
explicitly DEFERRED (see UPDATED_AGENTIC_WORKFLOW.md's Build Status section
and this session's evidence-recovery probe) -- so QC here is OBSERVATIONAL
ONLY. It never blocks or loops; it feeds the gate and the calibration
report. This matches the workflow doc's own "no (budget spent) -> proceed
anyway" branch, just with the recovery step skipped entirely for now.
"""

from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("confidence_gate")

_QC_VERDICT_THIN = "thin"
_QC_VERDICT_OK = "ok"


@dataclass
class QCVerdict:
    pillar: str
    verdict: str          # 'ok' | 'thin'
    evidence_mass: Optional[float]   # None when no breakdown (linear path)
    gate_fired: Optional[bool]
    n_claim_contribs: Optional[int]
    reason: str


@dataclass
class GatedOutput:
    pillar: str
    mode: str              # 'point' | 'range' -- see EVIDENCE_ROUTE_PLAN.md sec4/4.1:
                            # under the (not-yet-flipped) range-always output contract this
                            # field becomes a constant and carries no signal; `basis` is its
                            # replacement. Kept computing exactly as before for now -- flipping
                            # it to always 'range' is a separate step (re-keys 3 call sites
                            # that branch on it, most importantly estimate_verifier.py's critic
                            # skip, which would silently skip every pillar if flipped without
                            # also re-keying that condition to basis=="prior" first).
    score: float
    low: float
    high: float
    confidence: str        # passthrough from ReconciledScore
    needs_review: bool
    reason: str
    # New fields (EVIDENCE_ROUTE_PLAN.md sec4) -- additive, always populated,
    # safe for existing callers that only read the fields above.
    route: str = "rich"                  # 'rich' | 'thin' -- which path produced this pillar
    basis: str = "evidence"              # 'evidence' | 'prior' -- what the score actually rests on
    rung: Optional[str] = None           # ladder rung name, thin route only; None on rich


def qc_assess(formula_scores: dict) -> dict[str, QCVerdict]:
    """formula_scores: dict[pillar, PillarFormulaScore] (from
    compute_formula_scores()). Reads PillarFormulaScore.breakdown when present
    (use_saturation=True path); falls back to counting non-peer-anchor
    contributions when breakdown is None (linear path, or saturation
    unavailable for any reason) -- never raises, never blocks."""
    out: dict[str, QCVerdict] = {}
    for pillar, pfs in formula_scores.items():
        breakdown = getattr(pfs, "breakdown", None)
        if breakdown is not None:
            thin = bool(breakdown.gate_fired) or breakdown.n_claim_contribs == 0
            reason = (f"evidence_mass={breakdown.evidence_mass:.2f}, "
                      f"gate_fired={breakdown.gate_fired}, "
                      f"n_claim_contribs={breakdown.n_claim_contribs}")
            out[pillar] = QCVerdict(
                pillar=pillar, verdict=_QC_VERDICT_THIN if thin else _QC_VERDICT_OK,
                evidence_mass=breakdown.evidence_mass, gate_fired=breakdown.gate_fired,
                n_claim_contribs=breakdown.n_claim_contribs, reason=reason,
            )
            continue

        # Linear/no-breakdown fallback: count real (non-peer-anchor) contributions.
        n_real = sum(1 for c in (pfs.contributions or []) if c.method != "peer_anchor")
        thin = n_real == 0
        out[pillar] = QCVerdict(
            pillar=pillar, verdict=_QC_VERDICT_THIN if thin else _QC_VERDICT_OK,
            evidence_mass=None, gate_fired=None, n_claim_contribs=n_real,
            reason=f"no saturation breakdown available -- counted {n_real} non-peer-anchor contributions",
        )
    return out


def gate(reconciled: dict, qc: dict[str, "QCVerdict"],
         routing: Optional[dict[str, dict]] = None) -> dict[str, GatedOutput]:
    """reconciled: dict[pillar, ReconciledScore] (from reconcile_all()).
    qc: dict[pillar, QCVerdict] (from qc_assess() above).
    routing: optional dict[pillar, {"route": "rich"|"thin", "rung": str|None}]
        from the evidence router (EVIDENCE_ROUTE_PLAN.md sec1.3/sec2) --
        None (default) preserves today's exact output (route='rich',
        basis='evidence', rung=None on every pillar, matching what every
        existing caller already gets since GatedOutput's new fields default
        to those values). Passing routing populates `route`/`basis`/`rung`
        from the actual router decision instead of the defaults.

    Deterministic rule: emit a range (mode='range', needs_review=True) when
    EITHER reconcile's own confidence label is 'low' OR the QC verdict for
    that pillar is 'thin' -- either signal alone is enough to distrust a
    point score. Otherwise emit the point score. The range values are
    reconcile's own low/high (already computed, never re-derived here); the
    displayed score is always the reconciled score/midpoint, in both modes,
    so a 'range' output isn't a different number, just a different framing
    of the same estimate plus an honest confidence-attached range."""
    out: dict[str, GatedOutput] = {}
    for pillar, rs in reconciled.items():
        qc_verdict = qc.get(pillar)
        qc_thin = qc_verdict is not None and qc_verdict.verdict == _QC_VERDICT_THIN
        low_confidence = rs.confidence == "low"

        route_info = (routing or {}).get(pillar, {})
        route = route_info.get("route", "rich")
        rung = route_info.get("rung")
        basis = "prior" if route == "thin" else "evidence"

        if low_confidence or qc_thin:
            reasons = []
            if low_confidence:
                reasons.append(f"reconcile confidence='low' (spread={rs.spread})")
            if qc_thin:
                reasons.append(f"QC verdict='thin' ({qc_verdict.reason})" if qc_verdict else "QC verdict='thin'")
            out[pillar] = GatedOutput(
                pillar=pillar, mode="range", score=rs.score, low=rs.low, high=rs.high,
                confidence=rs.confidence, needs_review=True, reason="; ".join(reasons),
                route=route, basis=basis, rung=rung,
            )
        else:
            out[pillar] = GatedOutput(
                pillar=pillar, mode="point", score=rs.score, low=rs.low, high=rs.high,
                confidence=rs.confidence, needs_review=False,
                reason=f"reconcile confidence='{rs.confidence}', QC verdict='ok'",
                route=route, basis=basis, rung=rung,
            )
    return out
