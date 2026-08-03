"""
reconcile.py — Phase 3 Step 3: merge up to 3 independent E/S/G estimates into
one score + an honest uncertainty range. Pure code -- no LLM, no DB.

Inputs per pillar:
  - formula   : PillarFormulaScore from formula_estimator.py (now itself
                includes a real peer_anchor contribution when evidence is
                thin -- see peer_anchor.py)
  - holistic  : ESGScore from holistic_estimator.py (the demoted single-shot
                LLM vote) -- the noisiest input, permanently weight-capped.

(The Peer-Analogy vote as its own independent estimator, per the original
Phase 3 plan, is folded directly into formula_estimator.py's peer_anchor.py
rather than kept separate here -- simpler, and the formula's own contribution
list already carries the peer evidence as an auditable line. Reconcile
therefore merges TWO independent estimates today: formula (which itself blends
country baseline + real claims + real peer statistics) and holistic (the LLM
vote). This is a smaller ensemble than the original 3-way design, chosen
because splitting the peer vote back out would double-count evidence that
formula_estimator.py already incorporates.)

MERGE MATH:
  Base weights are a FLAT 0.7/0.3 (formula/holistic) split for ALL THREE
  pillars -- see _PILLAR_WEIGHTS (its own comment there has the full
  decision history/dates; this docstring previously described an abandoned
  per-pillar variant, E .75/.25 / S .45/.55 / G .65/.35, whose backtest never
  completed -- fixed here, DEFECT_FIX_PLAN.md 2.2, to match what the code
  actually does). Two flat splits were compared on the SAME held-out n=30
  seed=101 bcorp sample:
    flat 0.7/0.3: E +0.417  S +0.150  G +0.118  Total +0.009
    flat 0.6/0.4: E +0.286  S +0.235  G +0.152  Total +0.042
  0.7/0.3 is the settled default (_PILLAR_WEIGHTS' own comment, dated
  2026-07-20) -- per-pillar splits are UNDECIDABLE at this sample size
  (n=30 fresh-gather runs carry +/-0.1 Spearman noise per pillar, dwarfing
  any weight-split effect measurable today); revisit in Phase 5 with the
  full ground truth (fixed-evidence offline comparison, not fresh-gather
  backtests).

  Per-vote self-confidence:
    formula:  c_f = min(1.0, 0.4 + 0.04 * M_trust)   [EQUATION_CHANGES.md v8]
              M_trust = sum(w*c*method_trust) over all contributions (claims
              AND the peer_anchor line -- the "trust mass": quantity x quality,
              replacing the old count-based 0.4 + 0.15*n which treated five
              weak claims identically to five strong ones). 0 contributions
              -> c_f=0.4 exactly as before -- even an unsupported baseline
              carries some trust, since it's a real country statistic, not an
              LLM guess. Full trust at M_trust ~= 15 (~four solid claims).
    holistic: c_h = 0.5 FIXED -- never adjusted per-company; this is the
              permanent cap on the noisiest input (measured live: the same
              company/code/temperature=0 combination can swing several
              points between identical re-runs).

  eff_i = base_i * c_i; w_i = eff_i / sum(eff) over available votes;
  score = clamp(sum(w_i * s_i), 0, 100).

  NOTE (DEFECT_FIX_PLAN.md 2.2, re-verified): because c_f scales DOWN with
  thin formula evidence while c_h stays fixed at 0.5, holistic's RELATIVE
  weight already rises automatically as formula confidence shrinks --
  verified numerically: w_holistic swings from ~17.6% (c_f=1.0, strong
  evidence) to ~34.9% (c_f=0.4, the floor -- zero contributions) under the
  flat 0.7/0.3 base split. It never reaches parity or dominance: formula
  stays >=~65% even at formula's worst-case confidence, so "the LLM vote
  dominates thin-evidence companies" does not hold as literally stated.
  Separately, a pillar whose QC verdict is 'thin' (confidence_gate.py) is
  NOT fed back into this merge -- it's flagged mode='range'/needs_review
  downstream instead, after this score is computed. That's a deliberate
  separation (reconcile merges votes; the gate decides whether to trust the
  result), not an oversight -- changing it would mean re-validating the
  settled 0.7/0.3 weights against ground truth, which the note above already
  defers to Phase 5.

  Missing vote (holistic LLM call failed) -> renormalize over what's left.
  If formula is ALSO somehow missing (should not happen -- formula never
  raises, see formula_estimator.py), fall back entirely to holistic, or to
  a bare 50.0 with confidence 'low' if both are missing.

SPREAD / RANGE:
  n==2 (both present): spread = |formula.score - holistic.score|;
    low = clamp(min(scores) - 2, 0, 100); high = clamp(max(scores) + 2, 0, 100).
  n==1 (holistic missing, common case): score passes through as formula.score;
    spread=None; band = score +/- 15 (wide default, honestly flags "only one
    estimator contributed"); confidence='low'.
  Confidence label: 'high' if n==2 and spread<=10; 'low' if n==1 or spread>25;
  else 'medium'.

CLI:
    python -m agentic_estimation.layer_3.reconcile dry "Nvidia" --industry "Semiconductors" --country USA
"""

import sys
from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header

log = get_logger("reconcile")

# Per-pillar base weights, NOT a single global split. Tuned after two held-out
# backtests (n=30, seed=101, bcorp) on the SAME sample:
#   0.7/0.3 flat: E +0.417  S +0.150  G +0.118  Total +0.009
#   0.6/0.4 flat: E +0.286  S +0.235  G +0.152  Total +0.042
# A single global weight can't target the actual asymmetry: Formula's evidence
# coverage is genuinely strong on E (Climate TRACE anchor + widely-disclosed
# emissions data) and reasonable on G (SEC filings, litigation records), but
# thin on S (few structured social-data sources exist -- see pillar_extractors.py
# comments on labor/human-rights evidence being news-derived and sparse). The
# flat 0.6/0.4 tune improved S/G by giving Holistic more say everywhere, but
# cost E's strong result in the process, since Holistic's extra weight isn't
# selective. Per-pillar weights let S trust Holistic more WITHOUT diluting E's
# already-good Formula signal.
_PILLAR_WEIGHTS = {
    # DECISION (2026-07-20): flat 0.7/0.3 was the settled default, because
    # per-pillar splits were UNDECIDABLE at n=30 (fresh-gather noise +/-0.1
    # Spearman/pillar dwarfed any weight effect).
    #
    # REVISED 2026-08-02 for S ONLY, on frozen evidence at n=373 tune /
    # n=94 holdout (calibration/abl_final2_split_*.json, replayed offline via
    # calibration/weight_tuner.py -- no gather noise, so the earlier
    # "undecidable" objection no longer applies at this sample size).
    #
    # S: 0.7 -> 0.1 formula. Measured S Spearman as the formula's share falls:
    #     0.7 -> tune +0.094  holdout +0.095   (previous default)
    #     0.3 -> tune +0.198  holdout +0.268
    #     0.1 -> tune +0.257  holdout +0.360
    #     0.0 -> tune +0.277  holdout +0.375
    #   The curve is MONOTONIC on both splits -- there is no interior optimum,
    #   which is the signature of a component that subtracts signal rather than
    #   one whose weight was merely mis-set. Root cause is coverage, not the
    #   weights: only 14% of tune / 16% of holdout companies have ANY S claim
    #   (51/373, 15/94), so for ~85% of companies the S "formula vote" is a
    #   country baseline that ranks at +0.007 tune -- i.e. noise diluting a
    #   holistic vote that ranks at +0.277/+0.375 on its own.
    #   Kept at 0.1 rather than 0.0 deliberately: the formula still carries the
    #   real claims for the 14% that have them, and a nonzero share means new
    #   S evidence sources (enforcement/BHRRC ingestion) raise this pillar
    #   automatically instead of being ignored by a hard-zeroed vote. Revisit
    #   upward once S claim coverage materially exceeds ~15%.
    #
    # E and G deliberately UNCHANGED at 0.7/0.3:
    #   E -- every alternative blend was worse on BOTH splits (0.6 -> tune
    #        +0.241/holdout +0.198; 0.5 -> +0.229/+0.186; 0.3 -> +0.202/+0.172).
    #   G -- a 52-parameter fit reached tune +0.276 but FELL on holdout
    #        (+0.203 -> +0.158), the classic overfit signature, so it was
    #        rejected. G's real problem is upstream: 94% of G claims are
    #        esg_report_published (rho -0.051) and compliance_certification
    #        (rho +0.030) -- boilerplate "we published a policy" facts that
    #        nearly every company satisfies and that therefore cannot rank
    #        anyone. Fix the evidence, not this constant.
    "E": {"formula": 0.7, "holistic": 0.3},
    "S": {"formula": 0.1, "holistic": 0.9},
    "G": {"formula": 0.7, "holistic": 0.3},
}
_HOLISTIC_CONFIDENCE = 0.5   # fixed cap -- see module docstring


@dataclass
class Vote:
    estimator: str          # 'formula' | 'holistic'
    score: float
    base_weight: float
    self_confidence: float
    detail: str              # e.g. "3 contributions (2 claims + peer_anchor)" / "single-shot llm"


@dataclass
class ReconciledScore:
    pillar: str
    score: float
    low: float
    high: float
    spread: Optional[float]
    votes: list[Vote] = field(default_factory=list)
    weights_used: dict[str, float] = field(default_factory=dict)
    n_votes: int = 0
    confidence: str = "low"   # 'high' | 'medium' | 'low'


def _formula_confidence(pfs) -> float:
    """Evidence-MASS-based trust (EQUATION_CHANGES.md v7/v8), replacing the old
    contribution-COUNT formula min(1.0, 0.4 + 0.15*n). The count version treated
    five weak 0.15-confidence claims identically to five strong dataset-backed
    ones (both hit c_f=1.0). Trust mass fixes that:

        M_trust = sum(w_i * c_i * m(method_i))  over ALL contributions
        c_f     = min(1.0, 0.4 + 0.04 * M_trust)

    Summing over the full contributions list automatically includes the
    _peer_anchor pseudo-contribution (method trust 1.0), giving exactly
    M_claims + w_PA*c_PA -- the v8 "trust mass". This deliberately DIFFERS from
    saturation_score.py's gate mass (claims only): the gate asks "can the claims
    be trusted?", this asks "can the whole deterministic estimate be trusted?" --
    a peer statistic genuinely raises the latter. Anchors preserved: bare
    baseline (no contributions) -> 0.4 exactly as before; peer-anchor-only
    company -> ~0.6 (was 0.55); full trust at M_trust ~= 15 (roughly four
    solid claims, matching the old n=4 -> 1.0 intent, now quality-weighted)."""
    from agentic_estimation.layer_3.saturation_score import _method_trust
    m_trust = sum(c.weight * c.confidence * _method_trust(c.method) for c in pfs.contributions)
    return min(1.0, 0.4 + 0.04 * m_trust)


def _confidence_label(n_votes: int, spread: Optional[float]) -> str:
    if n_votes == 1 or (spread is not None and spread > 25):
        return "low"
    if n_votes == 2 and spread is not None and spread <= 10:
        return "high"
    return "medium"


def reconcile_pillar(pillar: str, formula_score, holistic_score: Optional[float]) -> ReconciledScore:
    """
    formula_score: PillarFormulaScore for this pillar (from formula_estimator.py),
        or None in the (should-not-happen) case formula itself failed.
    holistic_score: the pillar's raw 0-100 value from an ESGScore (e.g.
        holistic.e_score), or None if the holistic LLM call failed/was skipped.
    """
    votes: list[Vote] = []
    pillar_w = _PILLAR_WEIGHTS[pillar]

    if formula_score is not None:
        c_f = _formula_confidence(formula_score)
        votes.append(Vote(
            estimator="formula", score=formula_score.score, base_weight=pillar_w["formula"],
            self_confidence=c_f,
            detail=f"{len(formula_score.contributions)} contribution(s), baseline={formula_score.baseline:.1f} ({formula_score.baseline_source})",
        ))

    if holistic_score is not None:
        votes.append(Vote(
            estimator="holistic", score=holistic_score, base_weight=pillar_w["holistic"],
            self_confidence=_HOLISTIC_CONFIDENCE, detail="single-shot llm vote",
        ))

    if not votes:
        # Both missing -- should only happen if formula_estimator.py itself
        # raised, which it's designed never to do. Honest last resort, not a
        # fabricated confident number.
        log.warning("[%s] no votes available at all -- returning bare 50.0, confidence=low", pillar)
        return ReconciledScore(pillar=pillar, score=50.0, low=20.0, high=80.0, spread=None,
                                votes=[], weights_used={}, n_votes=0, confidence="low")

    eff = {v.estimator: v.base_weight * v.self_confidence for v in votes}
    total_eff = sum(eff.values())
    weights_used = {k: (v / total_eff if total_eff > 0 else 1.0 / len(votes)) for k, v in eff.items()}

    score = sum(weights_used[v.estimator] * v.score for v in votes)
    score = max(0.0, min(100.0, score))

    scores = [v.score for v in votes]
    if len(votes) >= 2:
        spread = max(scores) - min(scores)
        low = max(0.0, min(scores) - 2)
        high = min(100.0, max(scores) + 2)
    else:
        spread = None
        low = max(0.0, score - 15)
        high = min(100.0, score + 15)

    confidence = _confidence_label(len(votes), spread)

    return ReconciledScore(
        pillar=pillar, score=score, low=low, high=high, spread=spread,
        votes=votes, weights_used=weights_used, n_votes=len(votes), confidence=confidence,
    )


def reconcile_all(formula_scores: dict, holistic) -> dict[str, ReconciledScore]:
    """
    formula_scores: dict['E'|'S'|'G', PillarFormulaScore] from
        formula_estimator.compute_formula_scores().
    holistic: ESGScore from holistic_estimator.holistic_vote(), or None.
    """
    results = {}
    for pillar in ("E", "S", "G"):
        holistic_val = getattr(holistic, f"{pillar.lower()}_score", None) if holistic is not None else None
        results[pillar] = reconcile_pillar(pillar, formula_scores.get(pillar), holistic_val)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Reconcile — merge formula + holistic votes (dry run)")
    ap.add_argument("mode", choices=["dry"])
    ap.add_argument("company")
    ap.add_argument("--industry", default="")
    ap.add_argument("--country", default=None)
    args = ap.parse_args()

    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.governance_collector import fetch_governance_signals
    from agentic_estimation.layer_1.facility_extractor import fetch_facility_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    from agentic_estimation.layer_2.pillar_extractors import extract_all_claims
    from agentic_estimation.layer_2.climate_trace_anchor import ct_anchor_claims
    from agentic_estimation.layer_3.formula_estimator import compute_formula_scores
    from agentic_estimation.layer_3.holistic_estimator import holistic_vote

    log_header(log, "Reconcile — dry run", company=args.company, country=args.country or "auto-detect")

    signals = {}
    signals.update(fetch_company_signals(args.company, args.industry))
    signals.update(fetch_governance_signals(args.company))
    signals.update(fetch_facility_signals(args.company, args.industry))
    metadata = get_company_metadata(args.company)
    country = args.country or metadata.get("country")

    claims = extract_all_claims(args.company, signals, metadata)
    claims += ct_anchor_claims(args.company, country=country)
    formula_scores = compute_formula_scores(claims, country, metadata, company_name=args.company, sector=args.industry or None, signals=signals)
    holistic = holistic_vote(args.company, args.industry, country, signals, metadata)

    reconciled = reconcile_all(formula_scores, holistic)
    for pillar, r in reconciled.items():
        print(f"\n=== {pillar}: {r.score:.1f} in [{r.low:.1f}, {r.high:.1f}] "
              f"(spread={r.spread if r.spread is None else round(r.spread,1)}, confidence={r.confidence}) ===")
        for v in r.votes:
            w = r.weights_used[v.estimator]
            print(f"  {v.estimator:10s} {v.score:6.1f}  weight={w:.2f} (base={v.base_weight} x conf={v.self_confidence:.2f})  {v.detail}")


if __name__ == "__main__":
    _cli()
