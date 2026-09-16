"""
evidence_ladder.py — the low-evidence estimation path (see
research/LOW_EVIDENCE_LADDER_PLAN.md). Pure, deterministic, no LLM.

WHAT THIS SOLVES: measured live (niche-10 production demo, 2026-08-18) --
real niche/private companies return ZERO extracted claims from every
collector on 9/10 companies. Today's fallback for a zero-evidence pillar is
peer_anchor (often abstains too) then a country baseline suppressed to 2%
weight (saturation_score.py's tiebreaker mode) -- the result is every
zero-evidence company landing in a flat ~49-60 band regardless of how
differently they actually perform (confirmed: real Upright truth for the
same 10 companies spans -5.9 to +0.3, ALL structurally different).

THE FIX: climb a ladder of INCREASINGLY COARSE, but genuinely predictive,
priors when company-specific evidence is thin -- validated on real,
disjoint holdout (calibration/shoot_out_rungs.py, zero circularity):

    rung              E rho    S rho    G rho     scope
    industry_median   +0.728   +0.634   n/a       Upright's 28 industries (30 minus 2 below the min-sample floor)
    exio_structural   +0.615   n/a      n/a       E only -- 150 EXIOBASE sectors, physical intensity
    peer_upright       +0.498   +0.273   (no G truth to validate against)
    country baseline  -0.061   +0.071   untested  existing floor, kept as last resort

G has NO industry_median or exio rung -- Upright carries zero governance
columns (confirmed via information_schema query), so G's ladder is
structurally shorter: peer_upright -> country only.

BLEND, NOT HARD-SWITCH: a company at the evidence-mass threshold must not
flip discontinuously between "pure evidence" and "pure prior" scoring.
Shrinkage form, same discipline as saturation_score.py's own
_TIEBREAKER_BASELINE_WEIGHT and reconcile.py's confidence weighting:

    w_claims = M / (M + k_blend)
    score    = tiebreaker_center + w_claims * evidence_term
                                  + (1 - w_claims) * prior_term

At M=0 this is a pure prior score; at large M it converges to today's
Path A score exactly (w_claims -> 1). One formula, no cliff. Parameters
(k_blend, A_prior per pillar) are swept on TRAIN only -- see
calibration/sweep_ladder_params.py -- never guessed.
"""

from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("evidence_ladder")

# ── Frozen rung order, per the shoot-out (calibration/rung_shootout*.json) ───
_RUNG_ORDER: dict[str, list[str]] = {
    "E": ["industry_median", "exio_structural", "peer_upright", "country"],
    "S": ["industry_median", "peer_upright", "country"],
    "G": ["peer_upright", "country"],
}

# ── Tunable parameters -- sweep-derived defaults, see sweep_ladder_params.py ──
# k_blend: evidence_mass at which claims and prior contribute equally
# (w_claims=0.5). A_prior: max deviation from centre the prior term can
# apply (Core Framework rule -- coarser signal, SMALLER swing than the
# A_pos/A_neg=40-200 the claims-based evidence_term uses).
@dataclass(frozen=True)
class LadderParams:
    k_blend: float = 2.5
    a_prior: float = 15.0


_DEFAULT_LADDER_PARAMS: dict[str, LadderParams] = {
    "E": LadderParams(),
    "S": LadderParams(),
    "G": LadderParams(k_blend=2.5, a_prior=10.0),   # G's ladder is shorter/weaker evidence -- smaller swing
}

_TIEBREAKER_CENTER = 50.0   # matches saturation_score.py's own constant


@dataclass
class LadderResult:
    pillar: str
    rung: str                 # 'industry_median' | 'exio_structural' | 'peer_upright' | 'country' | 'none'
    prior_pct: Optional[float]   # 0-100, higher = better, or None if no rung answered
    n_basis: int               # peers/companies behind the number, for audit
    detail: dict = field(default_factory=dict)


# Measured holdout Spearman per rung -- the exact table in this module's own
# docstring above, exposed as data so EVIDENCE_ROUTE_PLAN.md sec4.2's
# rung_factor (range width shrinks for a stronger rung) can read it without
# re-deriving or hardcoding a second copy. Source: calibration/shoot_out_rungs.py,
# disjoint holdout, zero circularity with any tuning corpus. G's peer_upright/
# country rungs are UNTESTED (no G truth source to validate against -- Upright
# carries no governance columns) -- reported as None, not a guessed value.
RUNG_HOLDOUT_RHO: dict[str, dict[str, Optional[float]]] = {
    "E": {"industry_median": 0.728, "exio_structural": 0.615, "peer_upright": 0.498, "country": -0.061, "none": None},
    "S": {"industry_median": 0.634, "peer_upright": 0.273, "country": 0.071, "none": None},
    "G": {"peer_upright": None, "country": None, "none": None},
}


def rung_holdout_rho(pillar: str, rung: str) -> Optional[float]:
    """Measured holdout Spearman for one (pillar, rung) pair, or None if
    untested (G's rungs) or the rung name is unrecognized. Callers computing
    a confidence-width term from rung strength should treat None as "no
    stronger than the weakest tested rung", not as zero -- zero is a real,
    worse-than-noise measured value for E/country."""
    return RUNG_HOLDOUT_RHO.get(pillar, {}).get(rung)


# upright_industry_prior is tiny and static (56 rows: ~28 industries x 2
# pillars, re-derived only by an explicit calibration re-run, never by live
# traffic) -- loaded ONCE per process and cached, not re-queried per
# company. Confirmed the cost of NOT doing this: a combine-method test that
# queried it fresh per company (300 companies x 2 pillars = 600 avoidable
# round-trips to Neon) never finished in a reasonable time before being
# killed. Every other rung already follows this pattern (country_baseline_
# agent caches its whole table on first call; peer_anchor's bcorp/upright
# distribution caches are similar) -- this was the one rung that didn't.
_industry_prior_cache: Optional[dict] = None


def _load_industry_prior_table() -> dict:
    global _industry_prior_cache
    if _industry_prior_cache is None:
        from agentic_estimation.layer_1.peer_anchor_collector import _db_conn
        conn = _db_conn()
        cur = conn.cursor()
        cur.execute("SELECT industry, pillar, cross_industry_percentile, n_train, median_val FROM upright_industry_prior")
        rows = cur.fetchall()
        table: dict = {}
        for industry, pillar, pctile, n_train, median_val in rows:
            table.setdefault(industry, {})[pillar] = (pctile, n_train, median_val)
        _industry_prior_cache = table
        log.info("cached upright_industry_prior: %d industries", len(table))
    return _industry_prior_cache


def _industry_median_vote(pillar: str, industry: Optional[str]) -> Optional[LadderResult]:
    if not industry or pillar not in ("E", "S"):
        return None
    table = _load_industry_prior_table()
    entry = table.get(industry, {}).get(pillar)
    if entry is None:
        return None
    pctile, n_train, median_val = entry
    return LadderResult(
        pillar=pillar, rung="industry_median", prior_pct=pctile, n_basis=n_train,
        detail={"industry": industry, "median_val": median_val},
    )


def _exio_vote(pillar: str, sector: Optional[str]) -> Optional[LadderResult]:
    if pillar != "E":
        return None
    from agentic_estimation.layer_3.exio_lookup import exio_e_vote

    vote = exio_e_vote(sector)
    if vote is None:
        return None
    return LadderResult(
        pillar=pillar, rung="exio_structural", prior_pct=vote.percentile, n_basis=1,
        detail={"sector_matched": vote.sector_matched, "similarity": vote.similarity, "basis": vote.basis},
    )


def _peer_upright_vote(pillar: str, company_name: str, sector: Optional[str],
                        country: Optional[str]) -> Optional[LadderResult]:
    from agentic_estimation.layer_3.peer_anchor import peer_anchor_vote

    vote = peer_anchor_vote(pillar, company_name, sector, country, truth_source="upright")
    if vote.percentile is None:
        return None
    return LadderResult(
        pillar=pillar, rung="peer_upright", prior_pct=vote.percentile, n_basis=vote.n_peers,
        detail={"tier": vote.tier, "basis": vote.basis},
    )


def _country_vote(pillar: str, country: Optional[str]) -> Optional[LadderResult]:
    if not country:
        return None
    from agentic_estimation.layer_1.country_baseline_agent import get_country_baseline_with_fallback

    try:
        bl, src = get_country_baseline_with_fallback(country)
    except Exception as exc:
        log.warning("country baseline lookup failed for %r: %s", country, exc)
        return None
    val = {"E": bl.e_score, "S": bl.s_score, "G": bl.g_score}[pillar]
    if val is None:
        return None
    return LadderResult(
        pillar=pillar, rung="country", prior_pct=val, n_basis=bl.indicator_count,
        detail={"country": country, "source": src},
    )


_RUNG_FN = {
    "industry_median": _industry_median_vote,
    "exio_structural": _exio_vote,
    "peer_upright": _peer_upright_vote,
    "country": _country_vote,
}


def climb(pillar: str, company_name: str, sector: Optional[str], industry: Optional[str],
          country: Optional[str]) -> LadderResult:
    """Climb DOWN the frozen rung order (best signal first) for this pillar,
    returning the first rung with real data. `industry` is the Upright-
    vocabulary industry string when known (best match for industry_median);
    `sector` is the company's own free-text industry/sector hint, used by
    exio_structural (fuzzy-matched) and peer_upright (fuzzy + exact tiers).
    Falls back to `sector` for industry_median too if `industry` isn't
    given -- an exact-string DB lookup that simply won't match unless the
    caller already resolved to Upright's own 28 labels, in which case this
    rung will just fall through to the next one, matching the fail-open
    discipline every other rung in this codebase uses.

    Returns rung='none', prior_pct=None only if EVERY rung abstains (no
    country baseline resolvable at all) -- callers must handle this (keep
    whatever score the existing pipeline already produced)."""
    order = _RUNG_ORDER.get(pillar, ["country"])
    for rung_name in order:
        fn = _RUNG_FN[rung_name]
        if rung_name == "industry_median":
            result = fn(pillar, industry or sector)
        elif rung_name == "exio_structural":
            result = fn(pillar, sector)
        elif rung_name == "peer_upright":
            result = fn(pillar, company_name, sector, country)
        else:  # country
            result = fn(pillar, country)
        if result is not None:
            return result

    return LadderResult(pillar=pillar, rung="none", prior_pct=None, n_basis=0)


# ── Controversy overlay (LSEG pattern, ch06) ──────────────────────────────────
# Adjudicated/confirmed negative events (regulatory fines, litigation, real
# violations) are pulled OUT of the main Sigma(w*c'*delta)/Sigma(w) blend and
# applied as a separate, DOWN-ONLY overlay after the base score -- clean
# record leaves the score untouched, a confirmed negative event can only
# pull down, never up. This matters most for exactly the niche/thin-evidence
# companies this ladder targets: a single real enforcement hit is often the
# ONLY company-specific signal available, and today it competes for the same
# denominator as one-sided badge factors (already zeroed, see
# factor_registry.py) and gets diluted rather than trusted.
_CONTROVERSY_FACTORS = frozenset({
    "litigation", "regulatory_fines", "environmental_controversy",
    "governance_controversy", "labor_controversy", "human_rights_incident",
})

_A_CONTRO = 20.0   # max points the overlay can pull down, before saturation
_K_CONTRO = 1.2    # tanh steepness on the severity sum


def controversy_overlay(contributions: list) -> tuple:
    """Down-only adjustment from confirmed negative-event contributions.
    Returns (overlay_points <= 0, matched_factors) -- overlay_points is
    ALWAYS <= 0 (a positive-delta 'controversy' contribution, e.g. a claim
    that found no violation, contributes zero to the overlay -- it is not a
    reward, the absence of bad news is the expected state, not an
    achievement, per LSEG's own documented rationale for this pattern).

    Pure function of the SAME contributions list formula_estimator.py
    already builds -- no new evidence source, just a different reduction
    for factors that are event-shaped rather than disclosure-shaped."""
    severity_sum = 0.0
    matched = []
    for c in contributions:
        if c.factor not in _CONTROVERSY_FACTORS:
            continue
        if c.delta >= 0:
            continue  # no confirmed negative event found for this factor -- not a reward
        severity = c.weight * c.confidence * (-c.delta)   # magnitude only, sign handled below
        severity_sum += severity
        matched.append(c.factor)

    if severity_sum <= 0.0:
        return 0.0, []

    import math
    overlay = -_A_CONTRO * math.tanh(_K_CONTRO * severity_sum / 20.0)  # /20 keeps tanh in its linear-ish range for typical 1-2 event sums
    return overlay, matched


def blend(evidence_term: float, evidence_mass: float, ladder: LadderResult,
          params: Optional[LadderParams] = None) -> tuple:
    """Combine the existing v5 evidence_term (A*tanh(k*delta)*cov_mult, from
    saturation_score.saturate_pillar) with the ladder's prior into one
    score, via the confidence-weighted shrinkage already validated
    elsewhere in this codebase (country-baseline tiebreaker, reconcile.py).

    Returns (score, w_claims) -- w_claims exposed for the audit trail.

    Degradation identities (both exact, not approximate):
      M=0            -> w_claims=0   -> score = center + prior_term  (pure prior)
      M >> k_blend    -> w_claims->1  -> score -> center + evidence_term  (today's Path A, unchanged)
    """
    p = params or LadderParams()

    if ladder.prior_pct is None:
        # No rung answered at all (e.g. no country resolvable either) --
        # degrade to pure evidence_term, i.e. today's existing behavior.
        w_claims = 1.0
        prior_term = 0.0
    else:
        w_claims = evidence_mass / (evidence_mass + p.k_blend) if (evidence_mass + p.k_blend) > 0 else 0.0
        prior_term = p.a_prior * (ladder.prior_pct - 50.0) / 50.0

    score = _TIEBREAKER_CENTER + w_claims * evidence_term + (1.0 - w_claims) * prior_term
    score = max(0.0, min(100.0, score))
    return score, w_claims
