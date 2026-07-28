"""
holistic_estimator.py — Phase 3 Step 2: the old single-shot LLM scorer,
demoted to "one vote of three" instead of sole source of truth.

Zero changes to scoring_agent.py. The demotion lives entirely in how a
future Reconcile step weights this vote, not in this module -- this is a
thin wrapper that reuses score_company_sync() verbatim, passing PREFETCHED
signals+metadata so nothing is re-gathered (the caller already ran the
formula-scorer's collectors).

WHY THIS VOTE STAYS IN THE ENSEMBLE AT ALL: the old scorer's failure mode
(measured in Phase 2: E +0.246 / S -0.154 / G +0.045 correlation against real
bcorp/upright ground truth) was that it was the SOLE source of truth with no
grounding check. As one vote among several -- always outweighed by the
formula when real evidence exists, and by real peer statistics when it
doesn't -- an LLM's holistic read can still catch gestalt patterns (e.g. "this
company's overall narrative feels evasive") that a claims-based formula
literally cannot see, since the formula only ever sees what the extractor
explicitly tagged.

KNOWN LIMITATION (measured live this session): the same company, same code,
re-scored twice by this LLM can differ by several points even at
temperature=0 -- this is the single noisiest input any ensemble built on top
of it will have, and must be weighted accordingly (permanently capped, never
let outvote a real claim or peer statistic).

CLI:
    python -m agentic_estimation.layer_3.holistic_estimator dry "Nvidia" --industry "Semiconductors" --country USA
"""

import sys
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header

log = get_logger("holistic_estimator")


def holistic_vote(
    company: str,
    industry: str,
    country: Optional[str],
    signals: dict[str, str],
    metadata: Optional[dict] = None,
):
    """
    One holistic LLM vote, reusing scoring_agent.score_company_sync verbatim.
    Returns None on any LLM failure (missing vote -- caller/Reconcile must
    handle this, never fabricate a score in its place).
    """
    from agentic_estimation.layer_3.scoring_agent import score_company_sync

    try:
        result = score_company_sync(
            company, industry=industry, country=country, signals=signals, metadata=metadata,
        )
        if result is None:
            log.warning("[%s] holistic vote failed (LLM call or parse error) -- missing vote", company)
        return result
    except Exception as exc:
        log.warning("[%s] holistic vote raised %s -- missing vote", company, exc)
        return None


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Holistic LLM Estimator — one ensemble vote (dry run)")
    ap.add_argument("mode", choices=["dry"])
    ap.add_argument("company")
    ap.add_argument("--industry", default="")
    ap.add_argument("--country", default=None)
    args = ap.parse_args()

    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata

    log_header(log, "Holistic Estimator — dry run", company=args.company, country=args.country or "auto-detect")

    signals = fetch_company_signals(args.company, args.industry)
    metadata = get_company_metadata(args.company)
    country = args.country or metadata.get("country")

    result = holistic_vote(args.company, args.industry, country, signals, metadata)
    if result is None:
        print("holistic vote: FAILED (missing)")
        sys.exit(1)

    print(f"\nHolistic vote for {args.company} (country={result.country}):")
    print(f"  E: {result.e_score:.1f}/100 -- {result.e_reasoning}")
    print(f"  S: {result.s_score:.1f}/100 -- {result.s_reasoning}")
    print(f"  G: {result.g_score:.1f}/100 -- {result.g_reasoning}")


if __name__ == "__main__":
    _cli()
