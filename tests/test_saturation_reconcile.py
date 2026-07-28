"""
v5 saturation formula (agentic_estimation/layer_3/saturation_score.py) and
v8 reconcile (agentic_estimation/layer_3/reconcile.py) synthetic arithmetic
tests -- hand-checked against the documented math in each module's
docstring. These were run informally during this session's build; this file
makes them a permanent regression suite.
"""
import math

import pytest

from agentic_estimation.layer_3.saturation_score import (
    saturate_pillar, PillarSatParams, _BETA, _EVIDENCE_THRESHOLD,
)
from agentic_estimation.layer_3.formula_estimator import Contribution
from agentic_estimation.layer_3.reconcile import reconcile_pillar, _confidence_label
from agentic_estimation.layer_3.formula_estimator import PillarFormulaScore


# ── saturate_pillar ──────────────────────────────────────────────────────────

def test_saturation_no_evidence_is_exactly_baseline():
    """Empty contributions -> delta=0 -> tanh(0)=0 -> score == baseline exactly."""
    breakdown = saturate_pillar(pillar="E", baseline=50.0, contributions=[], registry_weight_sum=100.0)
    assert breakdown.score == 50.0
    assert breakdown.delta == 0.0
    assert breakdown.evidence_mass == 0.0
    assert breakdown.gate_fired is True  # 0 < threshold


def test_saturation_evidence_gate_fires_below_threshold():
    """A single weak claim (w*c' well below 2.5) gates out -- claims ignored,
    score stays at baseline (no peer anchor to fall back on either)."""
    weak = Contribution(factor="cdp_disclosure", weight=4, confidence=0.3, delta=1.0,
                         points=1.2, claim_reasoning="x", method="extracted")
    # mass = 4 * 0.3 * 0.9 (extracted trust) = 1.08 < 2.5
    breakdown = saturate_pillar(pillar="E", baseline=50.0, contributions=[weak], registry_weight_sum=100.0)
    assert breakdown.gate_fired is True
    assert breakdown.n_claim_contribs == 0
    assert breakdown.score == 50.0


def test_saturation_strong_single_claim_passes_gate_and_moves_score():
    """A single strong claim (w*c' well above 2.5) passes the gate and
    produces a real positive swing."""
    strong = Contribution(factor="net_zero_pledge", weight=5, confidence=0.9, delta=1.0,
                           points=4.5, claim_reasoning="x", method="extracted")
    # mass = 5 * 0.9 * 0.9 (extracted trust) = 4.05 >= 2.5 -> gate passes
    breakdown = saturate_pillar(pillar="E", baseline=50.0, contributions=[strong], registry_weight_sum=10.0)
    assert breakdown.gate_fired is False
    assert breakdown.n_claim_contribs == 1
    assert breakdown.delta > 0
    assert breakdown.score > 50.0


def test_saturation_all_positive_evidence_approaches_baseline_plus_A():
    """delta -> 1 (all-positive, single claim) -> score ~= baseline +
    A*tanh(k)*CovMult. Hand-checked: A=40, k=1.0 (E defaults),
    tanh(1)=0.7615941..., full coverage (weight equals registry_weight_sum)
    -> CovMult = 1.0."""
    strong = Contribution(factor="net_zero_pledge", weight=5, confidence=1.0, delta=1.0,
                           points=5.0, claim_reasoning="x", method="dataset_lookup")  # trust=1.0, no discount
    breakdown = saturate_pillar(pillar="E", baseline=50.0, contributions=[strong], registry_weight_sum=5.0)
    assert breakdown.delta == pytest.approx(1.0)
    assert breakdown.coverage == pytest.approx(1.0)
    assert breakdown.coverage_multiplier == pytest.approx(1.0)
    expected = 50.0 + 40.0 * math.tanh(1.0) * 1.0
    assert breakdown.score == pytest.approx(expected, abs=0.01)


def test_saturation_gate_fired_falls_back_to_peer_anchor_only():
    """Claims gate out (weak), but a peer_anchor contribution is exempt from
    the gate -- the score should move based on the peer anchor alone, not
    stay pinned to baseline."""
    weak_claim = Contribution(factor="cdp_disclosure", weight=4, confidence=0.2, delta=1.0,
                               points=0.8, claim_reasoning="x", method="extracted")
    peer = Contribution(factor="_peer_anchor", weight=10, confidence=0.5, delta=0.6,
                         points=3.0, claim_reasoning="peer stat", method="peer_anchor")
    breakdown = saturate_pillar(pillar="E", baseline=50.0, contributions=[weak_claim, peer],
                                 registry_weight_sum=100.0)
    assert breakdown.gate_fired is True
    assert breakdown.used_peer_anchor is True
    assert breakdown.n_claim_contribs == 0  # claim excluded post-gate
    assert breakdown.score > 50.0  # peer anchor alone still moves it


def test_saturation_negative_delta_uses_a_neg():
    """A negative-delta claim should use a_neg, not a_pos -- verified by
    passing asymmetric params and checking the score matches a_neg's math."""
    params = PillarSatParams(a_pos=40.0, a_neg=20.0, k=1.0)
    bad_claim = Contribution(factor="environmental_controversy", weight=10, confidence=1.0, delta=-1.0,
                              points=-10.0, claim_reasoning="x", method="dataset_lookup")
    breakdown = saturate_pillar(pillar="E", baseline=50.0, contributions=[bad_claim],
                                 registry_weight_sum=10.0, params=params)
    assert breakdown.delta == pytest.approx(-1.0)
    assert breakdown.a_used == 20.0  # a_neg, not a_pos
    expected = 50.0 + 20.0 * math.tanh(-1.0) * 1.0
    assert breakdown.score == pytest.approx(expected, abs=0.01)


def test_saturation_score_always_clamped_0_100():
    """Even with an extreme baseline near the ceiling, score never exceeds 100."""
    strong = Contribution(factor="net_zero_pledge", weight=20, confidence=1.0, delta=1.0,
                           points=20.0, claim_reasoning="x", method="dataset_lookup")
    breakdown = saturate_pillar(pillar="E", baseline=95.0, contributions=[strong], registry_weight_sum=20.0)
    assert 0.0 <= breakdown.score <= 100.0


# ── reconcile_pillar ─────────────────────────────────────────────────────────

def _fps(contributions, baseline=50.0, score=60.0):
    return PillarFormulaScore(pillar="E", baseline=baseline, baseline_source="exact",
                               score=score, contributions=contributions)


def test_reconcile_both_votes_missing_returns_honest_bare_baseline():
    result = reconcile_pillar("E", None, None)
    assert result.score == 50.0
    assert result.low == 20.0
    assert result.high == 80.0
    assert result.confidence == "low"
    assert result.n_votes == 0


def test_reconcile_formula_only_bare_baseline_confidence_is_point_four():
    """0 contributions -> M_trust=0 -> c_f = min(1.0, 0.4 + 0.04*0) = 0.4 exactly."""
    fps = _fps([], score=50.0)
    result = reconcile_pillar("E", fps, None)
    assert result.n_votes == 1
    assert result.spread is None
    assert result.confidence == "low"  # n==1 always -> low
    # single vote -> score passes through as formula's own score
    assert result.score == pytest.approx(50.0)
    # band is score +/- 15
    assert result.low == pytest.approx(35.0)
    assert result.high == pytest.approx(65.0)


def test_reconcile_weighted_merge_matches_hand_computation():
    """Formula + holistic, both present -- verify the exact weighted-mean
    arithmetic against E's pillar weights (0.7/0.3) and the documented
    confidence formulas."""
    c = Contribution(factor="net_zero_pledge", weight=5, confidence=0.8, delta=1.0,
                      points=4.0, claim_reasoning="x", method="extracted")
    # M_trust = 5 * 0.8 * 0.9 (extracted trust) = 3.6 -> c_f = min(1, 0.4+0.04*3.6) = 0.544
    fps = _fps([c], score=70.0)
    result = reconcile_pillar("E", fps, holistic_score=50.0)

    c_f = min(1.0, 0.4 + 0.04 * (5 * 0.8 * 0.9))
    c_h = 0.5
    eff_f = 0.7 * c_f
    eff_h = 0.3 * c_h
    total = eff_f + eff_h
    w_f, w_h = eff_f / total, eff_h / total
    expected_score = w_f * 70.0 + w_h * 50.0

    assert result.n_votes == 2
    assert result.weights_used["formula"] == pytest.approx(w_f)
    assert result.weights_used["holistic"] == pytest.approx(w_h)
    assert result.score == pytest.approx(expected_score)
    assert result.spread == pytest.approx(20.0)  # |70-50|
    assert result.low == pytest.approx(48.0)   # min(scores)-2
    assert result.high == pytest.approx(72.0)  # max(scores)+2


def test_reconcile_missing_holistic_falls_back_to_formula_alone():
    fps = _fps([], score=65.0)
    result = reconcile_pillar("E", fps, holistic_score=None)
    assert result.n_votes == 1
    assert result.weights_used == {"formula": 1.0}
    assert result.score == pytest.approx(65.0)


def test_reconcile_score_clamped_0_100():
    fps = _fps([], score=150.0)  # shouldn't happen upstream, but reconcile must still clamp
    result = reconcile_pillar("E", fps, holistic_score=None)
    assert result.score <= 100.0


def test_confidence_label_high_only_when_two_votes_and_tight_spread():
    assert _confidence_label(2, 5.0) == "high"
    assert _confidence_label(2, 10.0) == "high"
    assert _confidence_label(2, 10.01) == "medium"


def test_confidence_label_low_when_single_vote_or_wide_spread():
    assert _confidence_label(1, None) == "low"
    assert _confidence_label(2, 26.0) == "low"


def test_confidence_label_medium_otherwise():
    assert _confidence_label(2, 15.0) == "medium"
