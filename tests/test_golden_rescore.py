"""
Golden dump->rescore regression test. Loads a dump of real captured evidence
(agentic_estimation.layer_3.formula_estimator.PillarFormulaScore
contributions + the holistic vote, produced once by
`python -m calibration.tune_saturation dump`) and re-scores it OFFLINE with
the CURRENT library code (saturate_pillar + reconcile_pillar at the
pipeline's default (A,k) params). Any score drift on IDENTICAL input evidence
means a formula/reconcile change altered behavior -- on purpose (update the
recorded expected value in the same commit) or by accident (this test catches
it either way).

This is the standing golden-data mechanism this project uses instead of a
separate fixture format -- the dump IS the fixture, and re-scoring it IS the
golden check (see EVALUATION_STRATEGIES.md Action Queue #2).

SKIP-IF-ABSENT: dump files are generated offline (~20 min for n=30, real
signal gathering + LLM calls) and are not checked in as small fixtures --
this test skips cleanly if the dump file isn't present, so `pytest tests/`
stays green in an environment that hasn't run the dump step, while still
running for real wherever the dump exists (e.g. this repo, post-refresh).
"""
import json
from pathlib import Path

import pytest

_DUMP_PATH = Path(__file__).resolve().parent.parent / "calibration" / "sat_ens_seed42_t0.json"

pytestmark = pytest.mark.skipif(
    not _DUMP_PATH.exists(),
    reason=f"golden dump not found at {_DUMP_PATH} -- run "
           f"'python -m calibration.tune_saturation dump --seed 42 --n 30 --out {_DUMP_PATH.name}' first",
)


@pytest.fixture(scope="module")
def dump_records():
    return json.loads(_DUMP_PATH.read_text(encoding="utf-8"))


def _rescore(rec: dict, pillar: str) -> float:
    """Reproduces calibration.tune_saturation._reconciled_pillar_score at
    the pipeline's DEFAULT (A,k) -- 40.0, 1.0 -- exactly what production uses
    unless a --sat-ak override is passed."""
    from types import SimpleNamespace
    from agentic_estimation.layer_3.saturation_score import saturate_pillar, PillarSatParams
    from agentic_estimation.layer_3.formula_estimator import PillarFormulaScore, Contribution
    from agentic_estimation.layer_3.reconcile import reconcile_pillar

    pr = rec["pillars"][pillar]
    contribs_ns = [SimpleNamespace(**c, points=0.0, claim_reasoning="") for c in pr["contributions"]]
    bd = saturate_pillar(
        pillar=pillar, baseline=pr["baseline"], contributions=contribs_ns,
        registry_weight_sum=pr["registry_weight_sum"], params=PillarSatParams(a_pos=40.0, a_neg=40.0, k=1.0),
    )
    contribs = [Contribution(factor=c["factor"], weight=c["weight"], confidence=c["confidence"],
                              delta=c["delta"], points=0.0, claim_reasoning="", method=c["method"])
                for c in pr["contributions"]]
    pfs = PillarFormulaScore(pillar=pillar, baseline=pr["baseline"], baseline_source="exact",
                              score=bd.score, contributions=contribs)
    holistic_val = (rec.get("holistic") or {}).get(pillar)
    return reconcile_pillar(pillar, pfs, holistic_val).score


def test_dump_has_records(dump_records):
    assert len(dump_records) > 0


def test_rescore_is_reproducible_within_one_run(dump_records):
    """The core golden property: rescoring the SAME dump twice with the SAME
    code must produce IDENTICAL scores -- saturate_pillar/reconcile_pillar
    are pure functions with no hidden state or randomness."""
    for rec in dump_records:
        for pillar in ("E", "S", "G"):
            if pillar not in rec["pillars"]:
                continue
            first = _rescore(rec, pillar)
            second = _rescore(rec, pillar)
            assert first == second, f"{rec['name']}/{pillar}: rescore is non-deterministic ({first} != {second})"


def test_rescore_produces_scores_in_valid_range(dump_records):
    """Sanity bound -- every rescored pillar must land in [0, 100]."""
    for rec in dump_records:
        for pillar in ("E", "S", "G"):
            if pillar not in rec["pillars"]:
                continue
            score = _rescore(rec, pillar)
            assert 0.0 <= score <= 100.0, f"{rec['name']}/{pillar}: score {score} out of [0,100]"


def test_rescore_matches_recorded_expected_values(dump_records, tmp_path):
    """The actual regression check: compare against a checked-in snapshot of
    expected scores. On first run against a freshly refreshed dump, this
    WRITES the snapshot (calibration/sat_ens_seed42_t0.expected.json) rather
    than failing -- subsequent runs compare against it. A deliberate formula
    change should regenerate the snapshot in the same commit (delete it and
    rerun once)."""
    snapshot_path = _DUMP_PATH.with_suffix(".expected.json")
    expected = {}
    if snapshot_path.exists():
        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))

    actual = {}
    mismatches = []
    for rec in dump_records:
        actual[rec["name"]] = {}
        for pillar in ("E", "S", "G"):
            if pillar not in rec["pillars"]:
                continue
            score = _rescore(rec, pillar)
            rounded = round(score, 4)
            actual[rec["name"]][pillar] = rounded
            if expected and rec["name"] in expected and pillar in expected[rec["name"]]:
                exp = expected[rec["name"]][pillar]
                if abs(rounded - exp) > 1e-6:
                    mismatches.append(f"{rec['name']}/{pillar}: expected {exp}, got {rounded}")

    if not expected:
        snapshot_path.write_text(json.dumps(actual, indent=2), encoding="utf-8")
        pytest.skip(f"no snapshot existed -- wrote {snapshot_path.name}; rerun to verify against it")

    assert not mismatches, "score drift vs recorded snapshot:\n" + "\n".join(mismatches)
