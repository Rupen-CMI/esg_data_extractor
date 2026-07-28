"""
saturation_score.py — v5 aggregate-saturation scoring, an alternative final
aggregation step for the deterministic Formula Estimator (see EQUATION_CHANGES.md
"CHANGES 5.0"). NO LLM. Isolated from formula_estimator.py: it consumes the
SAME per-pillar Contribution list that formula_estimator.py already builds
(claims + the _peer_anchor pseudo-contribution), and only replaces the final
`score = baseline + sum(points)` step. Every upstream mechanism (best-claim-
per-factor, benchmark deltas, evidence_freshness decay) is reused unchanged.

WHY (the one real gap v1-v4 converged on): today's `B + Σ w·c·δ` sums
contributions linearly, so several different high-impact factors firing
together (controversy -10, fine -9, litigation -8, ...) stack into an
unrealistically large swing. This module AVERAGES the evidence direction
instead of piling it up, then applies bounded diminishing-returns saturation.

THE MATH (per pillar, independent):

  0. Method-trust adjusted confidence (reuses _METHOD_RANK ordering as
     multipliers; NOT a new reliability variable -- see EQUATION_CHANGES.md):
        c_i' = c_i * m(method_i)
     Freshness stays inside c_i (already applied upstream); no second decay.

  1. Partition the contributions:
        C_claims = registry-factor contributions (factor != "_peer_anchor")
        PA       = the "_peer_anchor" pseudo-contribution, if present

  2. Evidence gate (CLAIMS ONLY -- peer anchor is exempt, so thin-evidence
     companies keep their peer differentiation instead of collapsing to the
     bare country baseline, the ~47%-tie bug peer_anchor.py exists to fix):
        M = Σ_{i in C_claims} w_i * c_i'
        C = (gate-passed claims) + PA    if M >= THRESHOLD
        C = PA only                      if M <  THRESHOLD

  3. Normalized swing (denominator is Σw_i, NEVER Σ w_i·c_i -- the latter
     cancels confidence for a lone claim, v3's bug):
        Δ = Σ_{i in C} w_i·c_i'·δ_i / Σ_{i in C} w_i        (0 if C empty)

  4. Coverage (registry claims only; PA and gated-out claims EXCLUDED, so
     Coverage can never exceed 1):
        Coverage = Σ_{i in gate-passed C_claims} w_i / Σ_{i in registry_p} w_i
        CovMult  = β + (1-β)·Coverage

  5. Sign-aware saturation:
        A = A_p^- if Δ < 0 else A_p^+
        score = clamp(B + A·tanh(k_p·Δ)·CovMult, 0, 100)

Initial parameters reproduce the registry's stated ±30 design envelope:
A=40, k=1.0 -> 40·tanh(1)·1 ≈ 30. β=0.6, THRESHOLD=2.5 (one strong claim
w·c'≈8 passes; one weak claim w·c'≈1.5 gates).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("saturation_score")

# Method-trust multipliers -- INITIAL calibration values, keyed to the same
# method strings _METHOD_RANK uses (formula_estimator.py). dataset_lookup
# (Climate TRACE, deterministic) is trusted fully; extracted (LLM-read) slightly
# discounted; the peer_ratio/coarse fallbacks (ratio_estimator paths) more so.
_METHOD_TRUST = {
    "dataset_lookup":       1.0,
    "extracted":            0.9,
    "peer_ratio_fallback":  0.7,
    "coarse_bucket":        0.5,
    "peer_anchor":          1.0,   # the peer vote's own confidence already caps low; don't double-discount
}
_DEFAULT_METHOD_TRUST = 0.9        # unknown method -> treat like an LLM extraction

_PEER_ANCHOR_FACTOR = "_peer_anchor"

# ── Default per-pillar saturation parameters (tunable; see calibration) ───────
# A: max deviation from baseline (before tanh/coverage shrink it). k: tanh
# steepness. Start symmetric and identical across pillars -- calibration decides
# any asymmetry / per-pillar split (same discipline as reconcile.py's weights).
@dataclass(frozen=True)
class PillarSatParams:
    a_pos: float = 40.0
    a_neg: float = 40.0
    k: float = 1.0

_DEFAULT_SAT_PARAMS: dict[str, PillarSatParams] = {
    "E": PillarSatParams(),
    # S: A=35 from the ensemble-target (A,k) grid search -- the ONLY tuned value
    # that survived held-out validation (seed42 tune +0.217 vs default +0.196;
    # seed101 held-out +0.282 vs default +0.265). E and G tuned candidates both
    # REGRESSED on held-out (classic n=30 overfit) and stay at defaults; real
    # parameter fitting deferred to Phase 5 against the full ground truth.
    "S": PillarSatParams(a_pos=35.0, a_neg=35.0),
    "G": PillarSatParams(),
}

_BETA = 0.6                 # coverage floor: sparse-but-real evidence still moves the score
_EVIDENCE_THRESHOLD = 2.5   # claim evidence-mass gate (peer anchor exempt)


@dataclass
class SaturationBreakdown:
    """Full audit trail -- every number that produced the score, so a
    saturation score stays as traceable as the linear one's contributions."""
    pillar: str
    baseline: float
    score: float
    delta: float                    # normalized swing Δ in [-1, 1]
    evidence_mass: float            # M (claims only)
    gate_fired: bool                # True -> claims ignored, peer anchor only
    coverage: float                 # 0..1
    coverage_multiplier: float
    a_used: float                   # A_pos or A_neg actually applied
    k_used: float
    n_claim_contribs: int           # gate-passed registry claims in C
    used_peer_anchor: bool


def _method_trust(method: str) -> float:
    return _METHOD_TRUST.get(method, _DEFAULT_METHOD_TRUST)


def saturate_pillar(
    pillar: str,
    baseline: float,
    contributions: list,             # list[Contribution] from formula_estimator
    registry_weight_sum: float,      # Σ w_i over ALL registry factors in this pillar
    params: Optional[PillarSatParams] = None,
    beta: float = _BETA,
    threshold: float = _EVIDENCE_THRESHOLD,
) -> SaturationBreakdown:
    """Re-aggregate an already-built Contribution list with the v5 saturation
    math. Pure/deterministic. `contributions` is exactly what
    formula_estimator.compute_formula_scores assembles for this pillar
    (registry-factor claims + optionally the _peer_anchor pseudo-contribution).
    `registry_weight_sum` is the pillar's Σ w_i over EVERY registered factor
    (from factor_registry), used as the coverage denominator."""
    p = params or _DEFAULT_SAT_PARAMS.get(pillar, PillarSatParams())

    claim_contribs = [c for c in contributions if c.factor != _PEER_ANCHOR_FACTOR]
    peer_anchor = next((c for c in contributions if c.factor == _PEER_ANCHOR_FACTOR), None)

    # Step 0+2: method-adjusted confidence + claim evidence mass (gate input).
    def cadj(c) -> float:
        return c.confidence * _method_trust(c.method)

    evidence_mass = sum(c.weight * cadj(c) for c in claim_contribs)
    gate_fired = evidence_mass < threshold

    # Step 2: assemble the aggregation set C.
    active_claims = [] if gate_fired else claim_contribs
    agg: list = list(active_claims)
    if peer_anchor is not None:
        agg.append(peer_anchor)

    # Step 3: normalized swing (denominator = Σ w_i, confidence in numerator only).
    num = sum(c.weight * cadj(c) * c.delta for c in agg)
    den = sum(c.weight for c in agg)
    delta = (num / den) if den > 0 else 0.0
    delta = max(-1.0, min(1.0, delta))   # guard float drift; math already bounds it

    # Step 4: coverage over gate-passed REGISTRY claims only (PA excluded).
    covered_weight = sum(c.weight for c in active_claims)
    coverage = (covered_weight / registry_weight_sum) if registry_weight_sum > 0 else 0.0
    coverage = max(0.0, min(1.0, coverage))
    cov_mult = beta + (1.0 - beta) * coverage

    # Step 5: sign-aware saturation.
    a_used = p.a_neg if delta < 0 else p.a_pos
    raw = baseline + a_used * math.tanh(p.k * delta) * cov_mult
    score = max(0.0, min(100.0, raw))

    return SaturationBreakdown(
        pillar=pillar, baseline=baseline, score=score, delta=delta,
        evidence_mass=evidence_mass, gate_fired=gate_fired,
        coverage=coverage, coverage_multiplier=cov_mult,
        a_used=a_used, k_used=p.k,
        n_claim_contribs=len(active_claims),
        used_peer_anchor=peer_anchor is not None,
    )
