"""
export_demo_data.py — one-shot, read-only export of already-calculated ESG
data into a static JS file for the demo UI (demo_ui/index.html).

Reuses build_esg_json.py's build_market_esg_json() for the core per-company
scores/metrics assembly (no reimplementation). Adds what that function omits:
per-pillar reasoning text and the plain-English summary, plus the total
company count in the DB.

No DB writes. Run from repo root:
    venv/Scripts/python.exe demo_ui/export_demo_data.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Sessionlocal
from api.v1.models import Market, Company
from build_esg_json import build_market_esg_json

# Markets to include -- picked live (2026-07-16) as the ones with real
# calculated depth: pillar scores + reasoning + summaries, or real reported
# disclosures. Not auto-discovered -- a fixed, known-good list keeps the demo
# deterministic and avoids surfacing a mostly-empty market by accident.
# Cyber-Physical Systems Security (global) deliberately excluded per user
# request -- 141 companies is too large for the demo.
DEMO_MARKETS = [
    "Plant-Based Cheese Market",
    "Automotive Infotainment Market",
]



def _describe_pillar(label: str, score_obj: dict, metrics: dict | None) -> str:
    """One plain sentence built ONLY from this pillar's own already-computed
    score/risk (esg_scores) and its own metrics dict -- no LLM text, no
    cross-referencing a different scoring path. Fixes a real bug found live:
    the old export attached the LLM evaluator's pillar reasoning text
    ("score set to 0, no evidence") to a DIFFERENT, unrelated score actually
    being displayed (Nush Foods showed E=90.9 with reasoning explaining a
    0.0) -- two disconnected scoring paths in the DB disagreed, and the
    export paired the score from one with the narrative from the other.
    This generator only ever describes the number sitting right next to it."""
    score = score_obj["score"]
    risk = score_obj["risk"]
    metrics = metrics or {}
    n_metrics = len(metrics)
    disclosed = sum(1 for m in metrics.values() if not m.get("estimated"))
    if n_metrics == 0:
        return f"{label}: {score:.0f}/100 ({risk} risk) -- no supporting metrics on file."
    basis = f"{disclosed} disclosed" if disclosed == n_metrics else (
        "all estimated" if disclosed == 0 else f"{disclosed} disclosed, {n_metrics - disclosed} estimated"
    )
    return f"{label}: {score:.0f}/100 ({risk} risk), based on {n_metrics} tracked metric(s) ({basis})."


def _describe_company(c: dict) -> str:
    """One short paragraph summarizing a company's ESG record, built purely
    from fields already present in its own exported dict (scores, rating,
    metric counts) -- no new data, no LLM call."""
    s = c["esg_scores"]
    return (
        f"{c['name']} scores {s['total']['score']:.0f}/100 overall ({s['total']['risk']} risk, rated '{c['rating']}'), "
        f"with Environment {s['environment']['score']:.0f}, Social {s['social']['score']:.0f}, "
        f"and Governance {s['governance']['score']:.0f}."
    )


def build_market_payload(market_name: str, db) -> dict | None:
    base = build_market_esg_json(market_name, db=db)
    if base is None:
        return None

    market = db.query(Market).filter(Market.name == market_name).first()

    for c in base["companies"]:
        # reasoning/summary generated purely from this company's own already-
        # computed score/metrics fields -- see _describe_pillar/
        # _describe_company docstrings for why the old LLM-text fields were
        # removed (they described a DIFFERENT score than the one shown).
        c["reasoning"] = {
            "environment": _describe_pillar("Environment", c["esg_scores"]["environment"], c.get("environmental_metrics")),
            "social": _describe_pillar("Social", c["esg_scores"]["social"], c.get("social_metrics")),
            "governance": _describe_pillar("Governance", c["esg_scores"]["governance"], c.get("governance_metrics")),
        }
        c["summary"] = _describe_company(c)

    # Drop companies with no computed ESG data at all ("Not yet processed" --
    # data_source == "no_data") -- per explicit user request, the demo only
    # shows companies with real results. Recompute market-level totals so
    # they stay consistent with what's actually shown.
    kept = [c for c in base["companies"] if c["data_source"] != "no_data"]
    dropped = [c["name"] for c in base["companies"] if c["data_source"] == "no_data"]
    if dropped:
        print(f"    (dropped {len(dropped)} unprocessed company(ies): {', '.join(dropped)})")
    base["companies"] = kept
    base["total_companies"] = len(kept)
    base["industry_avg_esg_score"] = (
        round(sum(c["esg_scores"]["total"]["score"] for c in kept) / len(kept), 1) if kept else 0
    )

    base["status"] = market.status if market else None
    return base


def main():
    db = Sessionlocal()
    try:
        total_companies_in_db = db.query(Company).count()

        markets_out = []
        for market_name in DEMO_MARKETS:
            payload = build_market_payload(market_name, db)
            if payload is None:
                print(f"  SKIP (not found or empty): {market_name}")
                continue
            markets_out.append(payload)
            print(f"  OK  {market_name}: {payload['total_companies']} companies")

        result = {
            "total_companies_in_db": total_companies_in_db,
            "markets": markets_out,
        }

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data.js")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("window.DEMO_DATA = ")
            json.dump(result, f, indent=2, default=str)
            f.write(";\n")

        print(f"\nWrote {out_path}")
        print(f"Total companies in DB: {total_companies_in_db}")
        print(f"Markets exported: {len(markets_out)}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
