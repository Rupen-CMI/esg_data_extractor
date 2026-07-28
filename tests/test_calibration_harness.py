"""
compute_report's confidence-label calibration block (calibration_harness.py).
Answers the actual question a Spearman/pct_MAE report can't: does reconcile.py's
'high'/'medium'/'low' confidence LABEL track real |pred-truth| error, or is it
decorative? Synthetic DataFrames here stand in for a live backtest run so this
logic is verified without hitting the network/DB.
"""
import pandas as pd

from agentic_estimation.calibration_harness import compute_report


def _base_row(name, pred_e, truth_e, confidence_e):
    return {
        "name": name, "ok": True, "signals_count": 5,
        "pred_e": pred_e, "pred_s": None, "pred_g": None,
        "truth_e": truth_e, "truth_s": None, "truth_g": None, "truth_total": None,
        "confidence_e": confidence_e, "confidence_s": None, "confidence_g": None,
        "spread_e": None, "spread_s": None, "spread_g": None,
        "gate_e": None, "gate_s": None, "gate_g": None,
        "verdict_e": None, "verdict_s": None, "verdict_g": None,
    }


def test_labels_that_track_error_are_reported_monotonic():
    """'high' confidence rows are near-perfect, 'medium' somewhat off, 'low'
    way off -- the labels are doing their job, and the report should say so."""
    rows = [
        _base_row("h1", 50.0, 51.0, "high"), _base_row("h2", 60.0, 59.0, "high"),
        _base_row("h3", 40.0, 41.0, "high"),
        _base_row("m1", 50.0, 60.0, "medium"), _base_row("m2", 60.0, 68.0, "medium"),
        _base_row("m3", 40.0, 50.0, "medium"),
        _base_row("l1", 50.0, 80.0, "low"), _base_row("l2", 60.0, 20.0, "low"),
        _base_row("l3", 40.0, 90.0, "low"),
    ]
    report = compute_report(pd.DataFrame(rows))
    e = report["confidence_calibration"]["E"]
    assert e["buckets"]["high"]["n"] == 3
    assert e["buckets"]["medium"]["n"] == 3
    assert e["buckets"]["low"]["n"] == 3
    assert e["buckets"]["high"]["mean_abs_error"] < e["buckets"]["medium"]["mean_abs_error"]
    assert e["buckets"]["medium"]["mean_abs_error"] < e["buckets"]["low"]["mean_abs_error"]
    assert e["monotonic_high_to_low"] is True


def test_decorative_labels_are_flagged_non_monotonic():
    """'high' confidence rows are actually WORSE than 'low' -- the label isn't
    tracking real error at all. The report must say NO, not paper over it."""
    rows = [
        _base_row("h1", 50.0, 90.0, "high"), _base_row("h2", 60.0, 10.0, "high"),
        _base_row("h3", 40.0, 95.0, "high"),
        _base_row("l1", 50.0, 51.0, "low"), _base_row("l2", 60.0, 59.0, "low"),
        _base_row("l3", 40.0, 41.0, "low"),
    ]
    report = compute_report(pd.DataFrame(rows))
    e = report["confidence_calibration"]["E"]
    assert e["buckets"]["high"]["mean_abs_error"] > e["buckets"]["low"]["mean_abs_error"]
    assert e["monotonic_high_to_low"] is False


def test_small_bucket_reports_raw_errors_not_hidden():
    """n<3 in a bucket can't support a stable mean -- but it must still show
    up (raw per-row errors), not silently vanish as if untested."""
    rows = [
        _base_row("h1", 50.0, 51.0, "high"),
        _base_row("m1", 50.0, 60.0, "medium"), _base_row("m2", 60.0, 68.0, "medium"),
        _base_row("m3", 40.0, 50.0, "medium"),
    ]
    report = compute_report(pd.DataFrame(rows))
    e = report["confidence_calibration"]["E"]
    assert e["buckets"]["high"]["n"] == 1
    assert "mean_abs_error" not in e["buckets"]["high"]
    assert e["buckets"]["high"]["abs_errors"] == [1.0]
    assert e["monotonic_high_to_low"] is None  # only one bucket populated -- can't judge


def test_no_confidence_data_omits_the_whole_block():
    """Non-ensemble scorer runs (llm/formula) never populate confidence_e/s/g
    -- the block must not appear at all, not show empty/misleading buckets."""
    rows = [{
        "name": "x", "ok": True, "signals_count": 1,
        "pred_e": 50.0, "pred_s": None, "pred_g": None,
        "truth_e": 55.0, "truth_s": None, "truth_g": None, "truth_total": None,
        "confidence_e": None, "confidence_s": None, "confidence_g": None,
        "spread_e": None, "spread_s": None, "spread_g": None,
        "gate_e": None, "gate_s": None, "gate_g": None,
        "verdict_e": None, "verdict_s": None, "verdict_g": None,
    }]
    report = compute_report(pd.DataFrame(rows))
    assert "confidence_calibration" not in report
