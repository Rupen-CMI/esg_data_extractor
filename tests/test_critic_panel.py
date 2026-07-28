"""
Phase 4 critic panel (agentic_estimation/layer_4/critic_panel.py). Ported
from this session's scratchpad verification script -- 14 checks, covering
response parsing (fail-closed), majority/convergence logic, and the
source_tag-recovery fix (Contribution has no source_tag field -- Critic A
must re-derive the winning claim's tag the same way formula_estimator's
_pick_best_claim does).

No real LLM calls anywhere in this file -- _run_one_critic is monkeypatched
to return scripted CriticVerdicts.
"""
import pytest

from agentic_estimation.layer_4 import critic_panel as cp
from agentic_estimation.layer_4.critic_panel import CriticVerdict, _parse_critic_response


# ── _parse_critic_response ──────────────────────────────────────────────────

def test_parse_pass_verdict():
    v = _parse_critic_response('{"verdict": "pass", "flagged_factor": null, "objection": "looks fine"}',
                                "test_critic", {"factor_a", "factor_b"})
    assert v.verdict == "pass"
    assert v.flagged_factor is None


def test_parse_refute_with_valid_flagged_factor_kept():
    v = _parse_critic_response('{"verdict": "refute", "flagged_factor": "factor_a", "objection": "bad claim"}',
                                "test_critic", {"factor_a", "factor_b"})
    assert v.verdict == "refute"
    assert v.flagged_factor == "factor_a"


def test_parse_flagged_factor_not_in_valid_set_is_nulled():
    """A critic can't flag a factor that isn't in this pillar's own audit trail."""
    v = _parse_critic_response('{"verdict": "refute", "flagged_factor": "nonexistent_factor", "objection": "x"}',
                                "test_critic", {"factor_a", "factor_b"})
    assert v.verdict == "refute"
    assert v.flagged_factor is None


def test_parse_unparseable_text_abstains():
    v = _parse_critic_response("not json at all", "test_critic", {"factor_a"})
    assert v.verdict == "abstain"


def test_parse_invalid_verdict_enum_abstains():
    v = _parse_critic_response('{"verdict": "maybe", "flagged_factor": null, "objection": "x"}',
                                "test_critic", {"factor_a"})
    assert v.verdict == "abstain"


def test_parse_empty_string_abstains():
    v = _parse_critic_response("", "test_critic", {"factor_a"})
    assert v.verdict == "abstain"


# ── majority + convergence logic (via run_critic_panel) ─────────────────────

class _FakeContribution:
    def __init__(self, factor, method="extracted", confidence=0.6, claim_reasoning="x", weight=5, points=1.0, delta=0.2):
        self.factor = factor
        self.method = method
        self.confidence = confidence
        self.claim_reasoning = claim_reasoning
        self.weight = weight
        self.points = points
        self.delta = delta


class _FakeFormulaScore:
    def __init__(self, contributions, baseline=50.0, baseline_source="exact", peer_anchor=None, breakdown=None):
        self.contributions = contributions
        self.baseline = baseline
        self.baseline_source = baseline_source
        self.peer_anchor = peer_anchor
        self.breakdown = breakdown


class _FakeReconciled:
    def __init__(self, score=60.0, spread=15.0, confidence="medium"):
        self.score = score
        self.spread = spread
        self.confidence = confidence


def _run_with_scripted_verdicts(monkeypatch, scripted, formula_score=None, claims=None):
    """scripted: dict[critic_name] -> CriticVerdict"""
    def fake_run_one(critic_name, prompt, valid_factors):
        return scripted[critic_name]
    monkeypatch.setattr(cp, "_run_one_critic", fake_run_one)
    fs = formula_score or _FakeFormulaScore([_FakeContribution("factor_a")])
    return cp.run_critic_panel("E", "Test Co", _FakeReconciled(), fs, None, claims or [], {})


def test_three_pass_not_refuted(monkeypatch):
    scripted = {
        "evidence_support": CriticVerdict("evidence_support", "pass", None, ""),
        "peer_plausibility": CriticVerdict("peer_plausibility", "pass", None, ""),
        "internal_consistency": CriticVerdict("internal_consistency", "pass", None, ""),
    }
    result = _run_with_scripted_verdicts(monkeypatch, scripted)
    assert result.refuted is False


def test_two_of_three_refute_same_factor_converges(monkeypatch):
    scripted = {
        "evidence_support": CriticVerdict("evidence_support", "refute", "factor_a", "bad"),
        "peer_plausibility": CriticVerdict("peer_plausibility", "refute", "factor_a", "implausible"),
        "internal_consistency": CriticVerdict("internal_consistency", "pass", None, ""),
    }
    result = _run_with_scripted_verdicts(monkeypatch, scripted)
    assert result.refuted is True
    assert result.flagged_factor == "factor_a"
    assert len(result.objections) == 2  # collected from refuters only


def test_two_of_three_refute_different_factors_no_convergence(monkeypatch):
    scripted = {
        "evidence_support": CriticVerdict("evidence_support", "refute", "factor_a", "bad"),
        "peer_plausibility": CriticVerdict("peer_plausibility", "refute", "factor_b", "implausible"),
        "internal_consistency": CriticVerdict("internal_consistency", "pass", None, ""),
    }
    result = _run_with_scripted_verdicts(monkeypatch, scripted)
    assert result.refuted is True
    assert result.flagged_factor is None


def test_one_of_three_refute_not_refuted(monkeypatch):
    """Majority needs >=2 refutes."""
    scripted = {
        "evidence_support": CriticVerdict("evidence_support", "refute", "factor_a", "bad"),
        "peer_plausibility": CriticVerdict("peer_plausibility", "pass", None, ""),
        "internal_consistency": CriticVerdict("internal_consistency", "pass", None, ""),
    }
    result = _run_with_scripted_verdicts(monkeypatch, scripted)
    assert result.refuted is False


def test_fewer_than_two_responders_fails_open(monkeypatch):
    scripted = {
        "evidence_support": CriticVerdict("evidence_support", "abstain", None, ""),
        "peer_plausibility": CriticVerdict("peer_plausibility", "abstain", None, ""),
        "internal_consistency": CriticVerdict("internal_consistency", "refute", "factor_a", "bad"),
    }
    result = _run_with_scripted_verdicts(monkeypatch, scripted)
    assert result.refuted is False


def test_all_abstain_fails_open(monkeypatch):
    scripted = {
        "evidence_support": CriticVerdict("evidence_support", "abstain", None, ""),
        "peer_plausibility": CriticVerdict("peer_plausibility", "abstain", None, ""),
        "internal_consistency": CriticVerdict("internal_consistency", "abstain", None, ""),
    }
    result = _run_with_scripted_verdicts(monkeypatch, scripted)
    assert result.refuted is False


# ── _winning_claim_source_tags ───────────────────────────────────────────────

class _FakeClaim:
    def __init__(self, factor, source_tag, confidence, method="extracted"):
        self.factor = factor
        self.source_tag = source_tag
        self.confidence = confidence
        self.method = method


def test_winning_claim_source_tag_picks_highest_confidence():
    """Contribution has no source_tag field -- Critic A must re-derive the
    winning claim's tag via the same selection rule formula_estimator's
    _pick_best_claim uses. Regression test for a real bug found during
    this session's build."""
    claims = [
        _FakeClaim("factor_a", "source_low", 0.4),
        _FakeClaim("factor_a", "source_high", 0.9),
    ]
    fs = _FakeFormulaScore([_FakeContribution("factor_a")])
    tags = cp._winning_claim_source_tags(fs, claims)
    assert tags["factor_a"] == "source_high"
