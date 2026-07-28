"""
Phase 4 verifier orchestration (agentic_estimation/layer_4/estimate_verifier.py).
Ported from this session's scratchpad verification script -- 10 checks,
covering the full skip/pass/refute/retry state machine and the 5 gaps
patched into PHASE_4_PLAN.md before this was built: the retryability guard
(peer_anchor/dataset_lookup factors can't be fixed by re-extraction), fail-
closed re-extraction, and the two round-2-panel branches (gate resolves the
ambiguity vs. the panel genuinely runs again).

Tests 6/6b hit compute_formula_scores/reconcile_all with REAL claims, which
make a live DB call (peer_anchor lookup) -- these are the slow tests in this
suite (~1-2 min combined, needs ASYNC_DB_URL reachable). Marked `slow` so a
quick local run can skip them with `-m "not slow"`; CI should still run them.
"""
import pytest

from agentic_estimation.layer_3.formula_estimator import PillarFormulaScore, Contribution
from agentic_estimation.layer_3.saturation_score import SaturationBreakdown
from agentic_estimation.layer_3.reconcile import ReconciledScore
from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.layer_4 import critic_panel as real_cp
from agentic_estimation.layer_4 import estimate_verifier as ev
from agentic_estimation.layer_4.critic_panel import CriticPanelResult


def _make_breakdown(gate_fired=False, n_claim_contribs=3, evidence_mass=5.0):
    return SaturationBreakdown(
        pillar="E", baseline=50.0, score=60.0, delta=0.3, evidence_mass=evidence_mass,
        gate_fired=gate_fired, coverage=0.5, coverage_multiplier=0.8, a_used=40.0, k_used=1.0,
        n_claim_contribs=n_claim_contribs, used_peer_anchor=False,
    )


def _make_pfs(pillar, contributions, gate_fired=False, n_claim_contribs=3):
    return PillarFormulaScore(
        pillar=pillar, baseline=50.0, baseline_source="exact", score=60.0,
        contributions=contributions,
        breakdown=_make_breakdown(gate_fired=gate_fired, n_claim_contribs=n_claim_contribs),
    )


def _make_reconciled(pillar, score=60.0, low=55.0, high=65.0, spread=10.0, confidence="medium", n_votes=2):
    return ReconciledScore(pillar=pillar, score=score, low=low, high=high, spread=spread,
                            votes=[], weights_used={}, n_votes=n_votes, confidence=confidence)


def _run_verify_with_panel(monkeypatch, scripted_panel_results, formula_scores, reconciled,
                            claims=None, retry_claims_result="__no_retry_expected__", holistic=None):
    """Monkeypatch run_critic_panel (imported INSIDE verify_reconciled via a
    late `from agentic_estimation.layer_4.critic_panel import
    run_critic_panel` -- patch the source module's attribute so the late
    import picks it up) to return successive scripted CriticPanelResults,
    and optionally mock the retry's re-extraction via _retry_pillar."""
    call_count = [0]

    def fake_run_critic_panel(pillar, company, rs, fs, holistic_, claims_, signals, metadata):
        idx = call_count[0]
        call_count[0] += 1
        return scripted_panel_results[idx]

    monkeypatch.setattr(real_cp, "run_critic_panel", fake_run_critic_panel)
    if retry_claims_result != "__no_retry_expected__":
        monkeypatch.setattr(ev, "_retry_pillar", lambda *a, **k: retry_claims_result)

    return ev.verify_reconciled(
        "Test Co", reconciled, formula_scores, holistic, claims or [], {}, {"industry": "Testing"}, "United States",
    )


_C_REAL = Contribution(factor="net_zero_pledge", weight=5, confidence=0.8, delta=0.5, points=2.0,
                        claim_reasoning="x", method="extracted")


def test_thin_qc_skips_to_range_zero_critic_calls(monkeypatch):
    fs_thin = {"E": _make_pfs("E", [], gate_fired=True, n_claim_contribs=0)}
    rs_thin = {"E": _make_reconciled("E", confidence="low", spread=40.0)}
    result = _run_verify_with_panel(monkeypatch, [], fs_thin, rs_thin)
    assert result["E"].verdict == "skipped"
    assert result["E"].mode == "range"
    assert result["E"].critic_calls == 0


def test_high_confidence_and_qc_ok_skips_to_point_zero_critic_calls(monkeypatch):
    fs_high = {"E": _make_pfs("E", [_C_REAL], gate_fired=False, n_claim_contribs=1)}
    rs_high = {"E": _make_reconciled("E", confidence="high", spread=5.0)}
    result = _run_verify_with_panel(monkeypatch, [], fs_high, rs_high)
    assert result["E"].verdict == "skipped"
    assert result["E"].mode == "point"
    assert result["E"].critic_calls == 0


def test_medium_confidence_panel_passes(monkeypatch):
    fs_med = {"E": _make_pfs("E", [_C_REAL], gate_fired=False, n_claim_contribs=1)}
    rs_med = {"E": _make_reconciled("E", confidence="medium", spread=10.0)}
    pass_panel = CriticPanelResult(pillar="E", verdicts=[], refuted=False, flagged_factor=None, objections=[])
    result = _run_verify_with_panel(monkeypatch, [pass_panel], fs_med, rs_med)
    assert result["E"].verdict == "passed"
    assert result["E"].mode == "point"
    assert result["E"].critic_calls == 3


def test_refuted_without_convergence_routes_to_range_no_retry(monkeypatch):
    fs_med = {"E": _make_pfs("E", [_C_REAL], gate_fired=False, n_claim_contribs=1)}
    rs_med = {"E": _make_reconciled("E", confidence="medium", spread=10.0)}
    refute_no_conv = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor=None,
                                        objections=["implausible"])
    result = _run_verify_with_panel(monkeypatch, [refute_no_conv], fs_med, rs_med)
    assert result["E"].verdict == "refuted"
    assert result["E"].mode == "range"
    assert result["E"].retried is False


def test_refuted_converged_on_non_retryable_factor_no_retry(monkeypatch):
    """peer_anchor/dataset_lookup contributions can't be fixed by
    re-extraction -- a converged flag on one routes straight to range."""
    c_peer = Contribution(factor="_peer_anchor", weight=10, confidence=0.5, delta=0.3, points=1.5,
                           claim_reasoning="peer", method="peer_anchor")
    fs_peer = {"E": _make_pfs("E", [c_peer], gate_fired=False, n_claim_contribs=1)}
    rs_med = {"E": _make_reconciled("E", confidence="medium", spread=10.0)}
    refute_peer = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor="_peer_anchor",
                                     objections=["implausible peer stat"])
    result = _run_verify_with_panel(monkeypatch, [refute_peer], fs_peer, rs_med)
    assert result["E"].verdict == "refuted"
    assert result["E"].retried is False


class _FakeHolistic:
    e_score = 55.0
    s_score = 50.0
    g_score = 50.0
    e_reasoning = "some environmental activity noted"
    s_reasoning = "x"
    g_reasoning = "x"


@pytest.fixture(scope="module")
def strong_evidence_fixture():
    """Real compute_formula_scores/reconcile_all output on 3 real claims --
    strong enough evidence_mass that BOTH the initial AND retried reconcile
    land medium+QC-ok (not thin), so a round-2 panel call is actually
    reachable. Hits the DB once (peer_anchor lookup) -- shared across the
    3 tests that need it via module scope."""
    from agentic_estimation.layer_3.formula_estimator import compute_formula_scores
    from agentic_estimation.layer_3.reconcile import reconcile_all

    claims = [
        ExtractedClaim(factor="net_zero_pledge", pillar="E", polarity=1, strength=0.8, confidence=0.8,
                       value=None, source_tag="net_zero", reasoning="pledge found", method="extracted"),
        ExtractedClaim(factor="sbti_commitment", pillar="E", polarity=1, strength=0.7, confidence=0.7,
                       value=None, source_tag="sbti", reasoning="sbti found", method="extracted"),
        ExtractedClaim(factor="cdp_disclosure", pillar="E", polarity=1, strength=0.6, confidence=0.6,
                       value=None, source_tag="cdp", reasoning="cdp found", method="extracted"),
    ]
    fs = compute_formula_scores(claims, "United States", {"industry": "Testing"},
                                 company_name="Test Co", sector="Testing", signals={})
    rc = reconcile_all(fs, _FakeHolistic())
    assert rc["E"].confidence == "medium", "fixture must land medium for these tests to be meaningful"
    return claims, fs, rc


@pytest.mark.slow
def test_round2_panel_invoked_and_passes(monkeypatch, strong_evidence_fixture):
    claims, fs, rc = strong_evidence_fixture
    refute_conv = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor="net_zero_pledge",
                                     objections=["text doesn't support this"])
    pass_round2 = CriticPanelResult(pillar="E", verdicts=[], refuted=False, flagged_factor=None, objections=[])

    result = _run_verify_with_panel(
        monkeypatch, [refute_conv, pass_round2], {"E": fs["E"]}, {"E": rc["E"]},
        claims=claims, retry_claims_result=claims, holistic=_FakeHolistic(),
    )
    assert result["E"].verdict == "passed_after_retry"
    assert result["E"].mode == "point"
    assert result["E"].retried is True
    assert result["E"].critic_calls == 6


@pytest.mark.slow
def test_round2_panel_refutes_again_is_final_no_third_retry(monkeypatch, strong_evidence_fixture):
    claims, fs, rc = strong_evidence_fixture
    refute_conv = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor="net_zero_pledge",
                                     objections=["text doesn't support this"])
    refute_round2 = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor="net_zero_pledge",
                                       objections=["still doesn't hold up"])

    result = _run_verify_with_panel(
        monkeypatch, [refute_conv, refute_round2], {"E": fs["E"]}, {"E": rc["E"]},
        claims=claims, retry_claims_result=claims, holistic=_FakeHolistic(),
    )
    assert result["E"].verdict == "refuted"
    assert result["E"].mode == "range"
    assert result["E"].retried is True
    assert result["E"].critic_calls == 6
    assert len(result["E"].objections) == 2  # accumulated across both rounds


def test_weak_retry_reroutes_via_gate_alone_no_second_panel_call(monkeypatch):
    """A single-claim retry has too little evidence_mass to clear QC again --
    the re-gate resolves the ambiguity (to point or range) WITHOUT a second
    panel call. Distinct code path from the strong-evidence round-2 tests
    above."""
    fs_med = {"E": _make_pfs("E", [_C_REAL], gate_fired=False, n_claim_contribs=1)}
    rs_med = {"E": _make_reconciled("E", confidence="medium", spread=10.0)}
    refute_conv = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor="net_zero_pledge",
                                     objections=["text doesn't support this"])
    claim1 = ExtractedClaim(factor="net_zero_pledge", pillar="E", polarity=1, strength=0.8, confidence=0.8,
                             value=None, source_tag="net_zero", reasoning="pledge found", method="extracted")

    result = _run_verify_with_panel(monkeypatch, [refute_conv], fs_med, rs_med,
                                     claims=[claim1], retry_claims_result=[claim1])
    assert result["E"].retried is True
    assert result["E"].critic_calls == 3  # only round 1 -- no second panel call
    assert result["E"].verdict in ("passed_after_retry", "refuted")


def test_retry_reextraction_failure_fails_closed(monkeypatch):
    """_retry_pillar returning None (re-extraction failed) must NOT rescore
    a gutted pillar as if corrected -- fails closed to refuted/range."""
    fs_med = {"E": _make_pfs("E", [_C_REAL], gate_fired=False, n_claim_contribs=1)}
    rs_med = {"E": _make_reconciled("E", confidence="medium", spread=10.0)}
    refute_conv = CriticPanelResult(pillar="E", verdicts=[], refuted=True, flagged_factor="net_zero_pledge",
                                     objections=["text doesn't support this"])
    claim1 = ExtractedClaim(factor="net_zero_pledge", pillar="E", polarity=1, strength=0.8, confidence=0.8,
                             value=None, source_tag="net_zero", reasoning="pledge found", method="extracted")

    result = _run_verify_with_panel(monkeypatch, [refute_conv], fs_med, rs_med,
                                     claims=[claim1], retry_claims_result=None)
    assert result["E"].verdict == "refuted"
    assert result["E"].mode == "range"
    assert result["E"].needs_review is True
    assert result["E"].retried is True
