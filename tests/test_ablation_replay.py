"""
ablation_replay.py tests: synthetic v2-dump fixture (2 companies) exercising
each variant transformation directly, plus the paired-bootstrap stats on
constructed data with a known answer. Style follows test_golden_rescore.py
(dump-shaped fixtures, no live LLM/DB calls -- everything here is pure
in-memory dicts fed straight into the replay engine).
"""
import pytest

from calibration import ablation_replay as ar


# ── Synthetic v2 dump fixture ──────────────────────────────────────────────

def _company(name, truth_e, e_baseline, claim_delta, peer_pctile, holistic_e):
    return {
        "name": name,
        "truth": {"E": truth_e, "S": None, "G": None},
        "country": "Germany",
        "metadata": {},
        "signals": {"wikipedia": "Some company text mentioning a net zero pledge."},
        "raw_claims": [
            {"factor": "net_zero_pledge", "pillar": "E", "polarity": 1, "strength": 1.0,
             "confidence": 0.9, "value": None, "source_tag": "wikipedia",
             "reasoning": "stated net-zero pledge", "method": "extracted"},
        ],
        "kept_claims": [
            {"factor": "net_zero_pledge", "pillar": "E", "polarity": 1, "strength": 1.0,
             "confidence": 0.9, "value": None, "source_tag": "wikipedia",
             "reasoning": "stated net-zero pledge", "method": "extracted"},
        ],
        "flags": [],
        "holistic": {"E": holistic_e, "S": None, "G": None},
        "pillars": {
            "E": {
                "baseline": e_baseline,
                "baseline_source": "exact",
                "registry_weight_sum": 100.0,
                "contributions": [
                    {"factor": "net_zero_pledge", "weight": 5.0, "confidence": 0.9,
                     "delta": claim_delta, "method": "extracted"},
                    {"factor": "_peer_anchor", "weight": 10.0, "confidence": 0.5,
                     "delta": 2 * (peer_pctile / 100.0) - 1, "method": "peer_anchor"},
                ],
                "peer_anchor": {"pillar": "E", "percentile": peer_pctile, "confidence": 0.5,
                                "n_peers": 12, "tier": "sector_country", "basis": "test"},
            }
        },
    }


@pytest.fixture
def dump_v2():
    return [
        _company("Alpha Corp", truth_e=80.0, e_baseline=50.0, claim_delta=1.0, peer_pctile=90.0, holistic_e=85.0),
        _company("Beta Inc", truth_e=20.0, e_baseline=50.0, claim_delta=-1.0, peer_pctile=10.0, holistic_e=15.0),
    ]


@pytest.fixture
def dump_v1(dump_v2):
    """Same pillar/contribution shape but no raw_claims/signals -- the
    existing tune_saturation.py dump format."""
    out = []
    for rec in dump_v2:
        r = {k: v for k, v in rec.items() if k not in ("raw_claims", "kept_claims", "flags", "signals", "country", "metadata")}
        out.append(r)
    return out


# ── Contributions-level variant transforms ─────────────────────────────────

def test_no_peer_anchor_removes_exactly_the_peer_contribution(dump_v2):
    rec = dump_v2[0]
    contribs = ar._CONTRIB_LEVEL_VARIANTS["no_peer_anchor"](rec, "E")
    methods = [c["method"] for c in contribs]
    assert "peer_anchor" not in methods
    assert "extracted" in methods
    assert len(contribs) == len(rec["pillars"]["E"]["contributions"]) - 1


def test_baseline_only_drops_all_contributions(dump_v2):
    rec = dump_v2[0]
    contribs = ar._CONTRIB_LEVEL_VARIANTS["baseline_only"](rec, "E")
    assert contribs == []


def test_base_keeps_all_contributions(dump_v2):
    rec = dump_v2[0]
    contribs = ar._CONTRIB_LEVEL_VARIANTS["base"](rec, "E")
    assert len(contribs) == 2


# ── _score_variant: end-to-end per-variant scoring on the fixture ─────────

def test_holistic_only_returns_the_dumped_holistic_value(dump_v2):
    rec = dump_v2[0]
    score = ar._score_variant(rec, "E", "holistic_only")
    assert score == 85.0


def test_no_holistic_scores_without_error_and_in_range(dump_v2):
    rec = dump_v2[0]
    score = ar._score_variant(rec, "E", "no_holistic")
    assert score is not None
    assert 0.0 <= score <= 100.0


def test_formula_only_ignores_holistic_vote(dump_v2):
    """formula_only should be identical whether we mutate the dumped holistic
    value or not -- it never reads it."""
    rec = dump_v2[0]
    score_a = ar._score_variant(rec, "E", "formula_only")
    rec2 = dict(rec)
    rec2["holistic"] = {"E": 1.0, "S": None, "G": None}
    score_b = ar._score_variant(rec2, "E", "formula_only")
    assert score_a == score_b


def test_blend_50_50_differs_from_base_when_formula_and_holistic_disagree():
    """base uses 0.7/0.3 (formula-heavy); blend_50_50 shifts weight toward
    holistic. With a strong claim (formula pulls high) but a deliberately
    lower holistic vote, the two blends must produce different scores."""
    rec = _company("Gamma Ltd", truth_e=70.0, e_baseline=50.0, claim_delta=1.0,
                    peer_pctile=90.0, holistic_e=10.0)
    base_score = ar._score_variant(rec, "E", "base")
    blend_score = ar._score_variant(rec, "E", "blend_50_50")
    assert base_score != blend_score
    # more holistic weight in the blend should pull the score DOWN toward the low holistic vote
    assert blend_score < base_score


def test_no_tier0_and_no_freshness_replayable_on_v2_dump(dump_v2):
    rec = dump_v2[0]
    assert ar._score_variant(rec, "E", "no_tier0") is not None
    assert ar._score_variant(rec, "E", "no_freshness") is not None


def test_claims_level_variants_not_replayable_on_v1_dump(dump_v1):
    rec = dump_v1[0]
    assert ar._score_variant(rec, "E", "no_tier0") is None
    assert ar._score_variant(rec, "E", "no_freshness") is None


def test_v1_dump_still_supports_contribution_level_variants(dump_v1):
    rec = dump_v1[0]
    assert ar._score_variant(rec, "E", "base") is not None
    assert ar._score_variant(rec, "E", "no_peer_anchor") is not None


# ── Peer-first baseline variants ──────────────────────────────────────────

def test_peer_baseline_uses_percentile_not_country_baseline(dump_v2):
    """Alpha's peer percentile (90) is far above its country baseline (50) --
    a peer-first baseline must land the formula higher than base's."""
    rec = dump_v2[0]
    peer_first = ar._score_variant(rec, "E", "peer_baseline")
    base = ar._score_variant(rec, "E", "base")
    assert peer_first is not None
    assert peer_first > base


def test_peer_baseline_blend_sits_between_pure_forms(dump_v2):
    """Confidence 0.5 -> blend baseline = 0.5*90 + 0.5*50 = 70, strictly
    between the country baseline (50) and the pure peer percentile (90)."""
    rec = dump_v2[0]
    pure = ar._score_variant(rec, "E", "peer_baseline")
    blend = ar._score_variant(rec, "E", "peer_baseline_blend")
    base = ar._score_variant(rec, "E", "base")
    assert base < blend < pure


def test_peer_baseline_abstain_falls_back_to_country_baseline(dump_v2):
    import copy
    rec = copy.deepcopy(dump_v2[0])
    rec["pillars"]["E"]["peer_anchor"] = {"pillar": "E", "percentile": None, "confidence": 0.0,
                                           "n_peers": 0, "tier": "abstain", "basis": "test"}
    # abstain also means the dumped contributions carry no peer line
    rec["pillars"]["E"]["contributions"] = [
        c for c in rec["pillars"]["E"]["contributions"] if c["method"] != "peer_anchor"]
    peer_first = ar._score_variant(rec, "E", "peer_baseline")
    base = ar._score_variant(rec, "E", "base")
    assert peer_first == base


def test_peer_baseline_not_replayable_on_v1_dump(dump_v1):
    import copy
    rec = copy.deepcopy(dump_v1[0])
    del rec["pillars"]["E"]["peer_anchor"]
    assert ar._score_variant(rec, "E", "peer_baseline") is None
    assert ar._score_variant(rec, "E", "peer_baseline_blend") is None


# ── Paired stats: bootstrap CI + verdicts on constructed data ─────────────

def test_bootstrap_ci_all_negative_deltas_is_better():
    """Sign-flip (Rademacher) bootstrap tests the null 'this company's delta
    sign is arbitrary' -- a fixture needs actual per-item variance for that
    null to be rejectable (identical repeated magnitudes make every resampled
    mean span +/-|mean| regardless of consistency). A large, noisy-but-always-
    negative sample should still exclude zero -> BETTER."""
    import random
    rng = random.Random(7)
    deltas = [-5.0 + rng.uniform(-1.0, 1.0) for _ in range(200)]
    assert all(d < 0 for d in deltas)
    lo, hi, mean = ar._bootstrap_ci(deltas, seed=1)
    assert mean < 0
    assert ar._verdict(lo, hi) == "BETTER"


def test_bootstrap_ci_all_positive_deltas_is_worse():
    import random
    rng = random.Random(7)
    deltas = [5.0 + rng.uniform(-1.0, 1.0) for _ in range(200)]
    assert all(d > 0 for d in deltas)
    lo, hi, mean = ar._bootstrap_ci(deltas, seed=1)
    assert mean > 0
    assert ar._verdict(lo, hi) == "WORSE"


def test_bootstrap_ci_noisy_zero_mean_is_undecidable():
    """Deltas scattered around zero with no consistent sign -- CI should span
    zero -> UNDECIDABLE."""
    deltas = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5] * 4
    lo, hi, mean = ar._bootstrap_ci(deltas, seed=1)
    assert ar._verdict(lo, hi) == "UNDECIDABLE"


def test_bootstrap_ci_empty_deltas_is_undecidable():
    lo, hi, mean = ar._bootstrap_ci([], seed=1)
    assert lo is None and hi is None and mean is None
    assert ar._verdict(lo, hi) == "UNDECIDABLE"


# ── percentile ranks / MAE / spearman sanity ──────────────────────────────

def test_percentile_ranks_monotonic():
    ranks = ar._percentile_ranks([10.0, 30.0, 20.0])
    # index 0 (10.0) is lowest -> lowest rank, index 1 (30.0) highest -> highest rank
    assert ranks[0] < ranks[2] < ranks[1]


def test_percentile_mae_perfect_agreement_is_zero():
    preds = [10.0, 20.0, 30.0, 40.0]
    truths = [1.0, 2.0, 3.0, 4.0]
    assert ar._percentile_mae(preds, truths) == 0.0


def test_spearman_perfect_negative_correlation():
    preds = [1.0, 2.0, 3.0, 4.0]
    truths = [4.0, 3.0, 2.0, 1.0]
    rho = ar._spearman(preds, truths)
    assert abs(rho - (-1.0)) < 1e-9


# ── run(): end-to-end over the fixture, base variant present + verdicts shaped right ─

def test_run_end_to_end_on_v2_dump(tmp_path, dump_v2):
    dump_path = tmp_path / "dump.json"
    import json
    dump_path.write_text(json.dumps(dump_v2), encoding="utf-8")

    report = ar.run(dump_path, ["base", "no_peer_anchor", "no_tier0"], out_path=None)
    assert report["n_companies"] == 2
    e_report = report["pillars"]["E"]
    assert e_report["base"]["verdict"] == "BASE"
    assert "no_peer_anchor" in e_report
    assert "no_tier0" in e_report


def test_run_skips_claims_level_variants_on_v1_dump(tmp_path, dump_v1):
    dump_path = tmp_path / "dump_v1.json"
    import json
    dump_path.write_text(json.dumps(dump_v1), encoding="utf-8")

    report = ar.run(dump_path, ["base", "no_tier0"], out_path=None)
    assert "no_tier0" not in report["pillars"]["E"]
