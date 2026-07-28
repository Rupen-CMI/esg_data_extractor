"""
QC evidence-sufficiency assessment + Confidence Gate
(agentic_estimation/layer_3/confidence_gate.py). Ported from this session's
scratchpad verification script -- 10 checks.
"""
from agentic_estimation.layer_3.confidence_gate import qc_assess, gate, QCVerdict
from agentic_estimation.layer_3.reconcile import ReconciledScore
from agentic_estimation.layer_3.formula_estimator import PillarFormulaScore, Contribution
from agentic_estimation.layer_3.saturation_score import SaturationBreakdown


def _make_breakdown(gate_fired, n_claim_contribs, evidence_mass=0.0):
    return SaturationBreakdown(
        pillar="E", baseline=50.0, score=50.0, delta=0.0, evidence_mass=evidence_mass,
        gate_fired=gate_fired, coverage=0.0, coverage_multiplier=0.6, a_used=40.0, k_used=1.0,
        n_claim_contribs=n_claim_contribs, used_peer_anchor=False,
    )


def _make_reconciled(score, low, high, spread, confidence, n_votes=2):
    return ReconciledScore(pillar="E", score=score, low=low, high=high, spread=spread,
                            votes=[], weights_used={}, n_votes=n_votes, confidence=confidence)


# ── qc_assess ─────────────────────────────────────────────────────────────

def test_qc_gate_fired_is_thin():
    pfs = PillarFormulaScore(pillar="E", baseline=50.0, baseline_source="exact", score=50.0,
                              contributions=[], breakdown=_make_breakdown(gate_fired=True, n_claim_contribs=0))
    qc = qc_assess({"E": pfs})
    assert qc["E"].verdict == "thin"


def test_qc_zero_claim_contribs_is_thin_even_if_gate_not_fired():
    pfs = PillarFormulaScore(pillar="E", baseline=50.0, baseline_source="exact", score=50.0,
                              contributions=[], breakdown=_make_breakdown(gate_fired=False, n_claim_contribs=0))
    qc = qc_assess({"E": pfs})
    assert qc["E"].verdict == "thin"


def test_qc_real_evidence_is_ok():
    pfs = PillarFormulaScore(pillar="E", baseline=50.0, baseline_source="exact", score=65.0,
                              contributions=[],
                              breakdown=_make_breakdown(gate_fired=False, n_claim_contribs=3, evidence_mass=5.0))
    qc = qc_assess({"E": pfs})
    assert qc["E"].verdict == "ok"


def test_qc_linear_fallback_with_real_contribution_is_ok():
    """No saturation breakdown (linear path) -> counts non-peer-anchor contributions."""
    c_real = Contribution(factor="net_zero_pledge", weight=5, confidence=0.8, delta=0.5, points=2.0,
                           claim_reasoning="x", method="extracted")
    c_peer = Contribution(factor="_peer_anchor", weight=10, confidence=0.5, delta=0.3, points=1.5,
                           claim_reasoning="peer", method="peer_anchor")
    pfs = PillarFormulaScore(pillar="S", baseline=50.0, baseline_source="exact", score=55.0,
                              contributions=[c_real, c_peer], breakdown=None)
    qc = qc_assess({"S": pfs})
    assert qc["S"].verdict == "ok"


def test_qc_linear_fallback_with_only_peer_anchor_is_thin():
    c_peer = Contribution(factor="_peer_anchor", weight=10, confidence=0.5, delta=0.3, points=1.5,
                           claim_reasoning="peer", method="peer_anchor")
    pfs = PillarFormulaScore(pillar="G", baseline=50.0, baseline_source="exact", score=51.5,
                              contributions=[c_peer], breakdown=None)
    qc = qc_assess({"G": pfs})
    assert qc["G"].verdict == "thin"


# ── gate ──────────────────────────────────────────────────────────────────

def test_gate_low_confidence_routes_to_range():
    rs = _make_reconciled(60.0, 40.0, 80.0, 40.0, "low")
    qc_ok = {"E": QCVerdict(pillar="E", verdict="ok", evidence_mass=5.0, gate_fired=False, n_claim_contribs=3, reason="")}
    g = gate({"E": rs}, qc_ok)
    assert g["E"].mode == "range"
    assert g["E"].needs_review


def test_gate_high_confidence_and_qc_ok_routes_to_point():
    rs = _make_reconciled(70.0, 65.0, 75.0, 5.0, "high")
    qc_ok = {"E": QCVerdict(pillar="E", verdict="ok", evidence_mass=5.0, gate_fired=False, n_claim_contribs=3, reason="")}
    g = gate({"E": rs}, qc_ok)
    assert g["E"].mode == "point"
    assert not g["E"].needs_review


def test_gate_qc_thin_alone_forces_range_even_at_medium_confidence():
    rs = _make_reconciled(55.0, 50.0, 60.0, 10.0, "medium")
    qc_thin = {"E": QCVerdict(pillar="E", verdict="thin", evidence_mass=0.0, gate_fired=True, n_claim_contribs=0, reason="")}
    g = gate({"E": rs}, qc_thin)
    assert g["E"].mode == "range"
    assert g["E"].needs_review


def test_gate_bare_baseline_no_votes_routes_to_range():
    """reconcile_pillar's own no-votes fallback (score=50, low=20, high=80,
    confidence='low') must still be correctly turned into a range by the gate."""
    rs_bare = ReconciledScore(pillar="E", score=50.0, low=20.0, high=80.0, spread=None,
                               votes=[], weights_used={}, n_votes=0, confidence="low")
    qc_thin = {"E": QCVerdict(pillar="E", verdict="thin", evidence_mass=0.0, gate_fired=True, n_claim_contribs=0, reason="")}
    g = gate({"E": rs_bare}, qc_thin)
    assert g["E"].mode == "range"
    assert g["E"].low == 20.0
    assert g["E"].high == 80.0


def test_gate_preserves_exact_score_low_high_regardless_of_mode():
    rs = _make_reconciled(72.3, 68.0, 76.0, 8.0, "high")
    qc_ok = {"E": QCVerdict(pillar="E", verdict="ok", evidence_mass=5.0, gate_fired=False, n_claim_contribs=3, reason="")}
    g = gate({"E": rs}, qc_ok)
    assert g["E"].score == 72.3
    assert g["E"].low == 68.0
    assert g["E"].high == 76.0
