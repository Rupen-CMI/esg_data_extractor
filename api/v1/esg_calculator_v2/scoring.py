"""
scoring.py -- ESG Calculator v2, Layer 1: turn a CalculatorInput into real
agentic_estimation ExtractedClaim objects.

See plans/ESG_CALCULATOR_V2_PLAN.md section 4.1/4.2 for the full design.

THIS IS NOT AN "EXTRACTION" STAGE. There is no text to extract from -- the
user already gave us structured values. This module is plumbing: it puts
those values into the exact shape agentic_estimation.layer_3.formula_estimator
.compute_formula_scores() consumes, so THAT function -- the real one, not a
reimplementation -- does the actual scoring in scoring_v2's Phase 3.

CONFIDENCE IS SET BY HOW THE VALUE WAS OBTAINED, not by how much was
supplied. This is the calculator's version of the pipeline's own provenance
reasoning (a metered bill is worth more than a self-report). See
_CONFIDENCE table below and plan doc section 4.1.

Blank fields produce NO claim -- never a zero-value claim. A factor with no
claim simply doesn't appear in compute_formula_scores' contribution list,
which means the pillar score sits at the baseline for that factor's
contribution (see formula_estimator.py's own module docstring) -- exactly
the treatment the plan requires: blank widens the range and lowers
confidence, it does not get scored as a negative.
"""

from __future__ import annotations

import math
from typing import Optional

from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.layer_2.factor_registry import FACTORS
from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("esg_calculator_v2.scoring")

from api.v1.esg_calculator_v2.data.emission_factors import (
    FUEL_KG_CO2E_PER_LITRE,
    GLOBAL_AVERAGE_KG_PER_KWH,
)
from api.v1.esg_calculator_v2.schema import CalculatorInput

# ── Confidence by provenance ─────────────────────────────────────────────────
# See plan doc section 4.1's table. Deliberately NOT based on how many fields
# were filled in -- based on what KIND of fact each one is.
_CONF_BILL_DERIVED = 0.85          # metered quantity x a directly-verified published factor
_CONF_DIRECT_METRIC = 0.6          # a real number, self-reported, unverified
_CONF_BOOLEAN_TICK = 0.3           # unverifiable, universally answered "yes" -- deliberately low-impact
_CONF_ADMITTED_NEGATIVE = 0.8      # admission against interest is credible
_CONF_DENIED_NEGATIVE = 0.25       # asymmetric on purpose: denying earns little

_SRC_BILL = "derived_from_bill"
_SRC_METRIC = "self_reported_metric"
_SRC_BOOLEAN = "self_reported_boolean"
_SRC_NEGATIVE = "self_reported_negative"


def _numeric_claim(factor_key: str, value: float, confidence: float, source_tag: str,
                    reasoning: str) -> ExtractedClaim:
    """benchmark_band factors: compute_formula_scores computes delta/polarity
    itself from `value` against the factor's benchmark band (see
    formula_estimator._benchmark_delta) -- polarity=0/strength=1.0 here are
    placeholders only used if that lookup falls back to the event shape
    (no revenue available to normalise an intensity metric)."""
    factor = FACTORS[factor_key]
    return ExtractedClaim(
        factor=factor_key, pillar=factor.pillar, polarity=0, strength=1.0,
        confidence=confidence, value=value, source_tag=source_tag,
        reasoning=reasoning, method="extracted",
    )


def _event_claim(factor_key: str, polarity: int, strength: float, confidence: float,
                  source_tag: str, reasoning: str) -> ExtractedClaim:
    """event factors: compute_formula_scores uses polarity*strength directly
    (see formula_estimator._event_delta) -- value is None, there is no
    quantity here, just a directional fact."""
    factor = FACTORS[factor_key]
    return ExtractedClaim(
        factor=factor_key, pillar=factor.pillar, polarity=polarity, strength=strength,
        confidence=confidence, value=None, source_tag=source_tag,
        reasoning=reasoning, method="extracted",
    )


# ── Positive credential booleans (zero-weight in the registry; see plan doc
# section 2.2/2.3 -- kept because everyone ticks yes, so they carry almost no
# scoring signal by design, not because we forgot to weight them) ──────────
_CREDENTIAL_FACTORS = (
    "net_zero_pledge", "sbti_commitment", "cdp_disclosure",
    "anti_corruption_policy", "whistleblower_mechanism",
    "esg_report_published", "third_party_esg_audit", "compliance_certification",
)

# ── Public-record negatives: admitting costs real points, denying earns
# little (asymmetric confidence, see plan doc section 2.4/4.1) ─────────────
_NEGATIVE_FACTORS = ("environmental_controversy", "regulatory_fines", "litigation")

# ── Direct numeric metrics: user supplies the factor's own unit directly ───
# scope_3_emissions deliberately excluded -- removed from this calculator
# entirely (2026-09, product decision, not a data-quality one). No field, no
# claim built for it, even though it's a real askable registry factor.
_DIRECT_METRIC_FACTORS = (
    "scope_1_emissions", "scope_2_emissions",
    "renewable_energy_pct", "total_energy_consumption", "water_withdrawal",
    "total_waste_generated", "female_employees_pct", "female_board_pct",
    "employee_turnover_rate", "lost_time_injury_rate", "board_independence_pct",
)


def build_claims(inp: CalculatorInput) -> list[ExtractedClaim]:
    """The whole of Layer 1's claim-construction step. Pure function, no
    network/DB/LLM. Order of construction: direct metrics -> bill-derived
    fallbacks (only for scope_1/scope_2, only if not already supplied
    directly) -> credential booleans -> public-record negatives."""
    claims: list[ExtractedClaim] = []

    # -- Direct numeric metrics --
    for factor_key in _DIRECT_METRIC_FACTORS:
        value = getattr(inp, factor_key)
        if value is None:
            continue
        unit = FACTORS[factor_key].metric["unit"]
        claims.append(_numeric_claim(
            factor_key, value, _CONF_DIRECT_METRIC, _SRC_METRIC,
            f"Self-reported: {value:g} {unit}.",
        ))

    # -- Bill-derived fallback: scope_1 (fuel), scope_2 (electricity) --
    # Only fires when the user didn't supply the direct tCO2e figure --
    # never both, to avoid double-counting the same underlying emissions
    # under two claims on the same factor (formula_estimator picks the best
    # single claim per factor anyway, but a bill-derived duplicate would
    # just be silently discarded, wasting the higher-confidence source for
    # no reason -- so we only build it when it's the ONLY source).
    if inp.scope_1_emissions is None and (inp.diesel_litres is not None or inp.petrol_litres is not None):
        kg_co2e = 0.0
        parts = []
        if inp.diesel_litres is not None:
            kg = inp.diesel_litres * FUEL_KG_CO2E_PER_LITRE["diesel"]
            kg_co2e += kg
            parts.append(f"{inp.diesel_litres:g} L diesel x {FUEL_KG_CO2E_PER_LITRE['diesel']} kg CO2e/L")
        if inp.petrol_litres is not None:
            kg = inp.petrol_litres * FUEL_KG_CO2E_PER_LITRE["petrol"]
            kg_co2e += kg
            parts.append(f"{inp.petrol_litres:g} L petrol x {FUEL_KG_CO2E_PER_LITRE['petrol']} kg CO2e/L")
        tco2e = kg_co2e / 1000.0
        claims.append(_numeric_claim(
            "scope_1_emissions", tco2e, _CONF_BILL_DERIVED, _SRC_BILL,
            f"Derived from fuel bill: {' + '.join(parts)} = {tco2e:.2f} tCO2e "
            "(DEFRA 2024 average biofuel blend factors).",
        ))

    if inp.scope_2_emissions is None and inp.electricity_kwh is not None:
        kg_co2 = inp.electricity_kwh * GLOBAL_AVERAGE_KG_PER_KWH
        tco2e = kg_co2 / 1000.0
        claims.append(_numeric_claim(
            "scope_2_emissions", tco2e, _CONF_BILL_DERIVED, _SRC_BILL,
            f"Derived from electricity bill: {inp.electricity_kwh:g} kWh x "
            f"{GLOBAL_AVERAGE_KG_PER_KWH} kg CO2/kWh (IEA 2024 global average) "
            f"= {tco2e:.2f} tCO2e.",
        ))

    # -- Credential booleans: zero-weight, low confidence, positive only --
    for factor_key in _CREDENTIAL_FACTORS:
        value = getattr(inp, factor_key)
        if value is None:
            continue
        polarity = 1 if value else 0  # a "no" here is not a negative signal, just no credit
        if polarity == 0:
            continue  # weight=0 factors: skip "no" entirely, nothing to contribute
        claims.append(_event_claim(
            factor_key, polarity=1, strength=1.0, confidence=_CONF_BOOLEAN_TICK,
            source_tag=_SRC_BOOLEAN,
            reasoning=f"Self-reported: {FACTORS[factor_key].description} (unverified).",
        ))

    # -- Public-record negatives: asymmetric confidence --
    _negative_detail = {
        "environmental_controversy": inp.environmental_controversy_detail,
        "regulatory_fines": inp.regulatory_fines_detail,
        "litigation": inp.litigation_detail,
    }
    for factor_key in _NEGATIVE_FACTORS:
        value = getattr(inp, factor_key)
        if value is None:
            continue
        detail = _negative_detail[factor_key]
        if value:
            reasoning = f"Self-disclosed: {FACTORS[factor_key].description}."
            if detail:
                reasoning += f" User detail: {detail}"
            claims.append(_event_claim(
                factor_key, polarity=-1, strength=1.0, confidence=_CONF_ADMITTED_NEGATIVE,
                source_tag=_SRC_NEGATIVE, reasoning=reasoning,
            ))
        else:
            claims.append(_event_claim(
                factor_key, polarity=1, strength=0.3, confidence=_CONF_DENIED_NEGATIVE,
                source_tag=_SRC_NEGATIVE,
                reasoning=f"Self-reported clean record: no {FACTORS[factor_key].description.lower()} "
                          "disclosed (unverified -- the full pipeline checks this against "
                          "regulator/court records).",
            ))

    return claims


# ── Layer 2: real compute_formula_scores (deterministic) FIRST, then an LLM
# review of that computed result (see _llm_review) -- NOT a blind second
# estimator. See plans/ESG_CALCULATOR_V2_PLAN.md section 4.2 for why: the
# LLM sees the SAME handful of user-supplied numbers the formula already
# used, so a blind independent read would add noise, not signal. Layer 3:
# real confidence gate. reconcile_pillar() degrades correctly to a
# formula-only vote when the LLM review is unavailable/disabled (see
# reconcile.py) -- score() with use_llm=False is still a genuinely complete,
# honest, shippable product: real formula estimator, real QC gate, real
# confidence label, just one estimator voting instead of two.

from dataclasses import dataclass as _dataclass

from agentic_estimation.layer_3.formula_estimator import compute_formula_scores, _registry_weight_sum
from agentic_estimation.layer_3.confidence_gate import qc_assess, gate
from agentic_estimation.layer_3.reconcile import reconcile_all
from agentic_estimation.layer_3.peer_anchor import PeerAnchorVote

# Live peer_anchor_vote() is a real DB round-trip -- measured, ~3-5.5s per
# /score call even after the truth_source="upright" fix below cut it from
# ~15s (see that comment). This calculator debounces and re-scores on every
# field edit, so that cost is paid on every single keystroke-settle, not
# once -- felt as "why is there a delay every time I change a value"
# (measured live, 2026-09). Skip the live query entirely: an abstain vote
# for all three pillars, same shape peer_anchor_override already produces
# when the tiered lookup itself comes up empty. This is not a loss of a
# primary signal -- the real country (World Bank) and industry (EXIOBASE)
# baselines already do the heavy differentiation work (see
# use_saturation=False above and the industry-blend below); peer_anchor was
# only ever a secondary nudge on top (confidence capped at 0.5, weight 10
# alongside registry weights that sum much higher per pillar).
_NO_PEER_ANCHOR = {
    p: PeerAnchorVote(pillar=p, percentile=None, confidence=0.0, n_peers=0,
                       tier="abstain", basis="esg_calculator_v2: peer_anchor disabled for latency, see scoring.py")
    for p in ("E", "S", "G")
}

from api.v1.esg_calculator_v2.data.industry_baselines import industry_baseline_for
from api.v1.esg_calculator_v2.schema import CalculatorResult, ClaimSummary, PillarResult

# reconcile_all() expects a "holistic" object shaped like
# scoring_agent.ESGScore (attributes .e_score/.s_score/.g_score) -- NOT that
# class itself, since ESGScore also carries fields (company, signals_used)
# that make no sense for a review-of-a-computed-result. Minimal stand-in,
# same attribute names, nothing more.
@_dataclass
class _ReviewScore:
    e_score: float
    s_score: float
    g_score: float
    e_reasoning: str
    s_reasoning: str
    g_reasoning: str


_REVIEW_PROMPT = """You are reviewing an ESG (Environmental/Social/Governance) score that has \
already been computed by a deterministic formula. You are NOT scoring from scratch -- you are \
checking whether the computed result looks right given the data, and giving your own read.

Rules:
- 50 is the country/industry baseline before any company-specific data.
- Use ONLY the data given below -- do not invent facts about this company.
- If you agree with the formula's number, say so and give the same number back.
- If you disagree, say why, referencing SPECIFIC inputs below -- not a vague feeling.
- Do not penalise for data the company simply didn't provide.

COMPANY CONTEXT:
  Country: {country}
  Industry: {industry}

FORMULA'S COMPUTED SCORES (0-100, higher is better):
  Environmental: {formula_e:.1f}
  Social:        {formula_s:.1f}
  Governance:    {formula_g:.1f}

DATA THE FORMULA USED (one line per claim, confidence 0-1 reflects how the value was obtained \
-- a metered bill is more trustworthy than a self-reported tick):
{claims_block}

End your response with this JSON object (no markdown, no code fences):
{{
  "e_score": <float 0-100>,
  "s_score": <float 0-100>,
  "g_score": <float 0-100>,
  "e_reasoning": "<1-2 sentences>",
  "s_reasoning": "<1-2 sentences>",
  "g_reasoning": "<1-2 sentences>"
}}"""


def _claims_block(claims: list[ExtractedClaim]) -> str:
    if not claims:
        return "(no data supplied)"
    lines = []
    for c in claims:
        val = f", value={c.value:g}" if c.value is not None else ""
        lines.append(f"[{c.pillar}] {c.factor}: {c.reasoning} (confidence={c.confidence:.2f}{val})")
    return "\n".join(lines)


def _parse_review_response(raw: str) -> Optional[dict]:
    """Same last-valid-JSON-object scan as scoring_agent._parse_llm_response
    -- reasoning models emit chain-of-thought prose before the final JSON."""
    import json
    import re

    if not raw:
        return None
    required = {"e_score", "s_score", "g_score", "e_reasoning", "s_reasoning", "g_reasoning"}
    text = re.sub(r"```(?:json)?|```", "", raw, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and required.issubset(obj.keys()):
            return obj
    except Exception:
        pass
    candidates = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(candidates):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and required.issubset(obj.keys()):
                return obj
        except Exception:
            pass
    return None


def _clamp01_100(v) -> float:
    return max(0.0, min(100.0, float(v)))


def llm_review(
    inp: CalculatorInput, claims: list[ExtractedClaim], formula_scores: dict,
    model: Optional[str] = None,
) -> Optional[_ReviewScore]:
    """One LLM call: review the formula's already-computed scores against
    the same data it used. Returns None on ANY failure (missing vote --
    caller must handle this via reconcile_all's own None-tolerant path,
    never fabricate a score in its place -- same discipline as
    holistic_estimator.holistic_vote()).

    NOT VERIFIED against a live model -- see score()'s docstring."""
    from zen_client import call_with_prompt, DEFAULT_MODEL

    prompt = _REVIEW_PROMPT.format(
        country=inp.country or "Unknown", industry=inp.industry or "Unknown",
        formula_e=formula_scores["E"].score, formula_s=formula_scores["S"].score,
        formula_g=formula_scores["G"].score, claims_block=_claims_block(claims),
    )
    try:
        resp = call_with_prompt(prompt, model=model or DEFAULT_MODEL, max_tokens=1500, timeout=60)
    except Exception as exc:
        log.warning("llm_review call raised %s -- missing vote", exc)
        return None
    if not resp.get("ok"):
        log.warning("llm_review call failed: %s -- missing vote", resp.get("error"))
        return None
    parsed = _parse_review_response(resp.get("raw", ""))
    if parsed is None:
        log.warning("llm_review response could not be parsed -- missing vote")
        return None
    try:
        return _ReviewScore(
            e_score=_clamp01_100(parsed["e_score"]), s_score=_clamp01_100(parsed["s_score"]),
            g_score=_clamp01_100(parsed["g_score"]), e_reasoning=str(parsed["e_reasoning"]),
            s_reasoning=str(parsed["s_reasoning"]), g_reasoning=str(parsed["g_reasoning"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("llm_review response had malformed fields: %s -- missing vote", exc)
        return None

_PILLAR_LOCKED_NOTE = {
    "E": "The full pipeline also checks sector_emissions_intensity from Climate TRACE here.",
    "S": "The full pipeline also checks for labor disputes, human rights incidents, and "
         "consumer harm from news/BHRRC/regulator signals here -- none of which any "
         "company self-reports.",
    "G": "The full pipeline also checks for governance/ethics controversies from news "
         "signals here.",
}


# reconcile.py's _confidence_label() returns "low" whenever n_votes==1,
# UNCONDITIONALLY -- reasonable in production, where a single vote means the
# holistic LLM review failed or was skipped (a real red flag there). In this
# calculator, formula-only (no LLM review) is the NORMAL, always-available
# path (Phase 3) -- treating every single-vote score as automatically
# low-confidence with a flat +-15 misrepresents a score genuinely backed by
# a real country/industry baseline and several real claims. Same story as
# the use_saturation=False fix above: reuse the real formula math, but judge
# confidence on what actually backs THIS score, not on vote count borrowed
# from a different product's assumptions.
def _formula_only_confidence_and_range(inp: CalculatorInput, pfs) -> tuple[str, float, float]:
    """Confidence driven by SOURCE REALITY, not by how many weak signals
    happen to stack up: a real, checkable dataset value (World Bank country
    baseline, or EXIOBASE industry baseline) is high confidence on its own,
    full stop -- it says less about this ONE company than about its
    country/sector, but the NUMBER itself is not in doubt the way a guess
    would be. Only genuinely no data at all (no country, no industry, no
    claims) is low. Real claims on top only ever raise the bar further.
    Returns (confidence, low, high)."""
    has_country = inp.country is not None
    has_industry = inp.industry is not None
    n_claims = len(pfs.contributions)
    claim_conf = (sum(c.confidence for c in pfs.contributions) / n_claims) if n_claims else 0.0

    has_real_source = has_country or has_industry
    if has_real_source:
        # A real, checkable dataset value (World Bank / EXIOBASE) is high
        # confidence on its own -- the NUMBER isn't in doubt, even though it
        # describes the country/sector rather than this one company.
        confidence = "high"
    elif n_claims >= 2 and claim_conf >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    # Width: a real dataset baseline alone narrows a lot (+-8) -- it's a
    # real, sourced number, just about the country/sector rather than this
    # one company. Country+industry together, or several confident claims,
    # narrow further (+-5). Genuinely nothing stays wide (+-50, i.e. the
    # FULL 0-100 range around the neutral 50 baseline) -- +-25 (a 25-75
    # window) implied more certainty than "we have zero information" ever
    # earned; a company we know nothing about could truly be anywhere on
    # the scale, and the range should say so honestly, not hedge toward
    # the middle. Scores backed by real data (below) are NOT affected by
    # this -- they already range freely outside 25-75 when the underlying
    # baseline/claims support it, see score itself (not this range) for that.
    if not has_real_source and n_claims == 0:
        half_width = 50.0
    elif has_country and has_industry:
        half_width = 5.0
    elif has_real_source and n_claims >= 2 and claim_conf >= 0.6:
        half_width = 5.0
    elif has_real_source:
        half_width = 8.0
    else:
        half_width = 15.0

    score = pfs.score
    return confidence, max(0.0, score - half_width), min(100.0, score + half_width)


def _claim_summary(c: ExtractedClaim) -> ClaimSummary:
    return ClaimSummary(
        factor=c.factor, pillar=c.pillar, polarity=c.polarity,
        confidence=c.confidence, source_tag=c.source_tag, reasoning=c.reasoning,
    )


# ── Overflow penalty: extreme-magnitude fix, CALCULATOR-ONLY ────────────────
#
# BUG (found live, 2026-09-15, see problems.md for the shared-code version of
# this): formula_estimator._benchmark_delta clamps delta to [-1.0, +1.0] once
# a value crosses a factor's benchmark "worst" edge. Contribution is
# weight * confidence * delta, and once delta==-1.0 that product is FIXED --
# a value 2x past the worst threshold and a value 2,000,000x past it produce
# the EXACT SAME point swing. Reproduced: 1e13 litres of diesel against $1M
# revenue (an absurd but user-enterable input) moved the E score by only
# -6.8 points, because scope_1_emissions' contribution was already pinned
# at its floor. confidence (0-1, "how sure are we this number is real") and
# delta (meant to be -1..+1, "how bad is it") are two different scales
# multiplied together -- once delta saturates, no amount of "how bad" can
# express itself through the confidence factor either, since confidence
# was never meant to carry magnitude information.
#
# FIX HERE (not in formula_estimator.py -- that's shared production code,
# calibrated against real Spearman correlations; changing its clamp
# behaviour is a bigger, riskier change belonging to the pipeline's own
# backlog, see problems.md): after compute_formula_scores returns, look at
# how far past each claim's OWN benchmark edge its raw value actually sits
# (a real, computable ratio -- "raw / worst-edge", intensity-normalised by
# revenue exactly like _benchmark_delta does internally), and apply an
# ADDITIONAL, UNBOUNDED penalty on top of the already-clamped contribution,
# scaled by log10(overflow ratio) so a 10x-past-threshold value counts for
# real additional damage and a 1,000,000x-past-threshold value counts for
# much more -- without needing to touch or duplicate compute_formula_scores'
# own saturation math.
_OVERFLOW_PENALTY_PER_DECADE = 4.0  # points per 10x past the benchmark's worst edge, per factor


def _overflow_penalty_for_claim(c: ExtractedClaim, revenue_musd: Optional[float]) -> float:
    """Returns an EXTRA point penalty (>=0, to be SUBTRACTED from the
    pillar score) for a claim whose raw value sits far past its factor's
    benchmark 'worst' edge -- 0.0 for anything within or only slightly past
    the normal range (the base clamped contribution already covers that).

    BUG FIXED TWICE (2026-09-16): the benchmark tuple is `(v100, v0)` where
    v100 is ALWAYS the "best" edge and v0 is ALWAYS the "worst" edge,
    regardless of the factor's `direction` -- confirmed against every
    benchmark_band factor in the registry (e.g. scope_1_emissions: v100=15
    best, v0=250 worst, direction='lower'; renewable_energy_pct: v100=75
    best, v0=0 worst, direction='higher').

    Bug 1: originally used `worst = max(v100, v0)`, correct for 'lower'
    factors (worst IS the larger number there) but WRONG for 'higher'
    factors, where v0 (the worst edge) is the SMALLER number -- max()
    silently picked v100 (the BEST value) as "worst" instead. A company
    reporting 100% renewable energy was penalised for being "past the
    worst-case edge." Fixed by using v0 directly.

    Bug 2 (found immediately after fixing bug 1): overflow direction ALSO
    needs to flip with `direction`, not just which edge counts as worst.
    For 'lower' factors, overflow means value > worst (further past the bad
    end than the benchmark even measures). For 'higher' factors, overflow
    would mean value < worst -- but EVERY 'higher' factor in this registry
    is a percentage schema-bounded to [0, 100] (renewable_energy_pct,
    female_employees_pct, female_board_pct, board_independence_pct), so 0%
    is always the natural floor and is never "exceeded downward" -- there
    is no real input that is worse than the benchmark's own worst case
    already accounts for. Concretely: female_employees_pct=45 (worst edge
    is 10) was computing 45 > 10 as "overflow" and applying a penalty to a
    GOOD value, simply because 45 is numerically larger than 10 even though
    45% is far better than the 10% worst case. Fixed: the overflow penalty
    now only ever applies to direction='lower' factors -- structurally the
    only ones where a real disclosed value can sit further past 'bad' than
    the benchmark itself anticipates."""
    factor = FACTORS.get(c.factor)
    if factor is None or factor.metric is None or c.value is None:
        return 0.0
    if factor.direction != "lower":
        return 0.0  # see Bug 2 above -- overflow is only meaningful for direction='lower' factors
    v100, v0 = factor.metric["benchmark"]
    worst = v0  # v0 is ALWAYS the worst/0-score edge, regardless of direction
    if worst <= 0:
        return 0.0

    value = c.value
    if factor.metric.get("intensity") == "annual_revenue":
        if not revenue_musd or revenue_musd <= 0:
            return 0.0  # can't normalise without revenue -- same as the base contribution's own fallback
        value = value / revenue_musd

    if value <= worst:
        return 0.0  # within the benchmark's own range -- base contribution already handles this correctly
    overflow_ratio = value / worst
    decades_past = math.log10(overflow_ratio)
    return max(0.0, decades_past) * _OVERFLOW_PENALTY_PER_DECADE * c.confidence


def score(inp: CalculatorInput, use_llm: bool = False, llm_model: Optional[str] = None) -> CalculatorResult:
    """Layers 1-3 always run (deterministic, real pipeline code). Layer 2's
    LLM-review half runs only when use_llm=True -- see _llm_review below.

    use_llm defaults to False: NOT VERIFIED against a live model as of this
    writing (Ollama daemon not running locally; opencode.ai's hy3-free
    returned 401 Unauthorized -- expired/rotated key, not a model-availability
    issue). The prompt/parse/reconcile wiring is complete and follows the
    exact pattern agentic_estimation/layer_3/scoring_agent.py already uses in
    production, but has not been exercised end-to-end against a real
    response. Flip use_llm=True once a working model is confirmed."""
    claims = build_claims(inp)

    metadata = {}
    if inp.annual_revenue_usd:
        metadata["revenue"] = inp.annual_revenue_usd
    if inp.industry:
        metadata["industry"] = inp.industry

    formula_scores = compute_formula_scores(
        claims, country=inp.country, metadata=metadata or None, sector=inp.industry,
        # peer_anchor_override=_NO_PEER_ANCHOR skips peer_anchor_vote()'s
        # live DB query entirely (see _NO_PEER_ANCHOR's own comment above --
        # ~3-5.5s per /score call, paid on every debounced field edit).
        # truth_source is passed for documentation/consistency with the
        # esg-truth-source-decision memory (bcorp is OUT) but is currently a
        # no-op here: compute_formula_scores only consults truth_source
        # inside the live peer_anchor_vote() branch, which the override
        # above bypasses entirely. Kept so re-enabling the live lookup later
        # doesn't silently regress to the bcorp-contaminated default.
        truth_source="upright",
        peer_anchor_override=_NO_PEER_ANCHOR,
        # use_saturation=False: the calculator wants country/industry to
        # visibly move the score -- this is a calculator, not the production
        # ranking pipeline. use_saturation=True (the pipeline's own default)
        # runs saturation_score.py's "tiebreaker" baseline mode, which
        # deliberately compresses the country baseline to ~2% weight
        # (_TIEBREAKER_BASELINE_WEIGHT) because a backtest found it barely
        # orders COMPANIES against each other (Spearman +0.068) once real
        # evidence exists -- correct for that ranking task, wrong here.
        # use_saturation=False gives the legacy linear reduction instead:
        # score = baseline + sum(contribution points), i.e. the FULL country/
        # industry baseline, with claims still adjusting it normally on top.
        use_saturation=False,
    )

    # compute_formula_scores' baseline is COUNTRY ONLY (get_country_baseline_
    # with_fallback) -- sector/industry is used solely for peer matching
    # inside that call, never as a second baseline source. That leaves this
    # calculator's own industry data (data/industry_baselines.py, real
    # EXIOBASE medians, already used for the E-pillar renewable-tier scaling
    # elsewhere) completely unused whenever a user picks an industry but no
    # country -- exactly the "industry only" case that showed a flat ~50
    # despite Energy & Utilities having a real, low, differentiating
    # baseline (36.7). Blend it in ourselves: average country + industry the
    # same way v1's calculator did, then shift each pillar's score by the
    # same delta applied to its baseline (equivalent under use_saturation=
    # False's linear score = baseline + sum(points), since the baseline term
    # is additive and independent of the contribution points).
    industry_baseline = industry_baseline_for(inp.industry)
    if industry_baseline is not None:
        for p, key in (("E", "e_score"), ("S", "s_score"), ("G", "g_score")):
            industry_val = industry_baseline.get(key)
            if industry_val is None:
                continue
            pfs = formula_scores[p]
            country_val = pfs.baseline if inp.country else None
            blended = ((country_val + industry_val) / 2.0) if country_val is not None else industry_val
            delta = blended - pfs.baseline
            pfs.score = max(0.0, min(100.0, pfs.score + delta))
            pfs.baseline = blended
            pfs.baseline_source = f"{pfs.baseline_source}+industry" if inp.country else "industry"

    # Coverage rescale -- a single genuinely excellent (or terrible) number
    # was barely moving the score: scope_1_emissions at its best possible
    # value (delta=+1.0) only contributes weight(8) x confidence(~0.6-0.85)
    # x 1.0 = ~5-7 points, because weight=8 is a small slice of the ~59-pt
    # E registry total. That ~5-7pt cap applies EVEN WHEN IT IS THE ONLY
    # FIELD REPORTED -- as if 11 other unreported factors were silently
    # pulling toward zero, when in fact they simply were never asked. Fix:
    # when only a fraction of the pillar's registry weight was actually
    # supplied, scale the total contribution up so the one (or few) factors
    # reported represent the FULL swing "all the information we have" is
    # worth, rather than being diluted against factors nobody filled in.
    # Capped at 4x so a single low-weight factor (e.g. renewable_energy_pct,
    # weight 6, out of 59) can't swing a pillar unrealistically far on its
    # own -- still real, sourced information, just not the ENTIRE pillar's
    # worth of evidence. Country/industry baseline terms are NOT rescaled
    # (they are separate, already-real starting points, not claim
    # contributions) -- only the sum of claim-derived points is.
    # BUG FOUND AND FIXED (2026-09-16): pfs.contributions includes
    # SYNTHETIC contributions compute_formula_scores adds internally --
    # "_peer_anchor" and "_industry_median" (weight 10 each; the latter is
    # a real EXIOBASE structural vote, added even when peer_anchor is
    # disabled here). These are NOT things the user supplied, but the
    # rescale below was including their weight in `supplied_weight` (so a
    # single real user claim, e.g. scope_1_emissions weight=8, silently
    # became "18 units of coverage" once _industry_median's weight-10 was
    # added in) AND including their points in `contribution_points`, so
    # compute_formula_scores' own already-correctly-weighted EXIOBASE vote
    # got RE-AMPLIFIED by this calculator's rescale on top of everything
    # else. Reproduced: AFG/Financial Services + 20,000 L combined
    # diesel+petrol (correctly summed into one 45.97 tCO2e claim, delta
    # +0.736 -- NOT maxed, nowhere near the best-case edge of 15) still
    # clamped the E pillar at exactly 100, because the industry-blended
    # baseline (~80.7) plus the re-amplified _industry_median contribution
    # left no real headroom before the rescale even started amplifying the
    # user's own claim. Fixed: filter to claims the user actually supplied
    # (factor name not starting with "_") for BOTH supplied_weight and
    # contribution_points -- the rescale's whole premise is "amplify what
    # the user told us, since they didn't fill in the rest of the pillar,"
    # which was never true of a vote compute_formula_scores adds on its own.
    # SECOND BUG (same root cause, found immediately after fixing the
    # synthetic-contribution one above): even counting ONLY the user's own
    # claims, `extra` was added to pfs.score with no regard for how much
    # the pillar had ALREADY moved from the country+industry blend a few
    # lines up. For AFG/Financial Services, that blend alone pushes the
    # score to ~88.5 (Financial Services' real EXIOBASE e_score is a
    # genuinely clean 89.8) BEFORE the rescale even runs -- so a moderate,
    # not-maxed claim (delta=+0.736, nowhere near the best-case edge) still
    # added a further +15 unclamped, blowing straight through 100. Three
    # independently-reasonable signals (country/industry blend, the real
    # _industry_median vote, this rescale) stacked with no shared headroom
    # awareness. Fixed: apply `extra` as a fraction of REMAINING headroom
    # (tanh-saturating, same pattern saturation_score.py already uses
    # elsewhere in this codebase for "bounded swing without a hard clamp
    # that silently discards magnitude") rather than a flat, unbounded add
    # -- a claim can still move the score a lot when there's room, but
    # can't manufacture more room than exists once other real signals have
    # already claimed most of it.
    _COVERAGE_RESCALE_CAP = 4.0
    for p in ("E", "S", "G"):
        pfs = formula_scores[p]
        user_contributions = [c for c in pfs.contributions if not c.factor.startswith("_")]
        if not user_contributions:
            continue
        supplied_weight = sum(c.weight for c in user_contributions)
        if supplied_weight <= 0:
            continue
        registry_total = _registry_weight_sum(p)
        rescale = min(_COVERAGE_RESCALE_CAP, registry_total / supplied_weight)
        if rescale <= 1.0:
            continue  # already fully (or more than) covered -- no rescale needed
        contribution_points = sum(c.points for c in user_contributions)
        extra = contribution_points * (rescale - 1.0)
        if extra > 0:
            headroom = 100.0 - pfs.score
            pfs.score = pfs.score + headroom * math.tanh(extra / max(headroom, 1e-6))
        elif extra < 0:
            headroom = pfs.score - 0.0
            pfs.score = pfs.score - headroom * math.tanh(-extra / max(headroom, 1e-6))

    # Overflow penalty -- see _overflow_penalty_for_claim's own comment
    # above for the bug this works around. Applied per-claim, summed per
    # pillar, subtracted directly from the already-computed score (additive
    # on top, same as every other adjustment in this function).
    revenue_musd = (inp.annual_revenue_usd / 1_000_000.0) if inp.annual_revenue_usd else None
    overflow_notes: dict[str, list[str]] = {"E": [], "S": [], "G": []}
    for c in claims:
        penalty = _overflow_penalty_for_claim(c, revenue_musd)
        if penalty > 0.01:
            pfs = formula_scores[c.pillar]
            pfs.score = max(0.0, min(100.0, pfs.score - penalty))
            overflow_notes[c.pillar].append(
                f"{c.factor} is far past its benchmark's worst-case edge -- an extra "
                f"{penalty:.1f}pt penalty applied on top of the base contribution "
                f"(see problems.md: formula_estimator's delta clamp otherwise treats "
                f"any value past the edge identically)."
            )

    review: Optional[_ReviewScore] = None
    if use_llm:
        review = llm_review(inp, claims, formula_scores, model=llm_model)
        if review is None:
            log.warning("LLM review unavailable -- falling back to formula-only reconciliation")

    review_reasoning = {
        "E": review.e_reasoning if review else None,
        "S": review.s_reasoning if review else None,
        "G": review.g_reasoning if review else None,
    }
    review_score = {
        "E": review.e_score if review else None,
        "S": review.s_score if review else None,
        "G": review.g_score if review else None,
    }

    pillars: dict[str, PillarResult] = {}
    if review is not None:
        # Two real votes (formula + LLM review) -- this IS the case
        # reconcile.py/confidence_gate.py's vote-count-based rules were
        # designed for, use them as-is.
        qc = qc_assess(formula_scores)
        reconciled = reconcile_all(formula_scores, holistic=review)
        gated = gate(reconciled, qc)
        for p in ("E", "S", "G"):
            g = gated[p]
            pfs = formula_scores[p]
            basis = [
                f"Baseline: {pfs.baseline:.1f}/100 ({pfs.baseline_source}).",
                f"{len(pfs.contributions)} of your inputs contributed to this pillar.",
                f"LLM review: {review_reasoning[p]}",
            ]
            basis.extend(overflow_notes[p])
            if g.mode == "range":
                basis.append(f"Wide range: {g.reason}.")
            basis.append(_PILLAR_LOCKED_NOTE[p])
            pillars[p] = PillarResult(
                score=round(g.score, 1), low=round(g.low, 1), high=round(g.high, 1),
                confidence=g.confidence, formula_score=round(pfs.score, 1),
                llm_score=round(review_score[p], 1), basis=basis,
            )
    else:
        # Formula-only (Phase 3's normal path, or LLM review failed/skipped)
        # -- confidence/range judged on what actually backs the score, not
        # borrowed vote-counting logic that assumes a missing second vote is
        # abnormal. See _formula_only_confidence_and_range.
        for p in ("E", "S", "G"):
            pfs = formula_scores[p]
            confidence, low, high = _formula_only_confidence_and_range(inp, pfs)
            basis = [
                f"Baseline: {pfs.baseline:.1f}/100 ({pfs.baseline_source}).",
                f"{len(pfs.contributions)} of your inputs contributed to this pillar.",
            ]
            basis.extend(overflow_notes[p])
            if use_llm:
                basis.append("LLM review was unavailable for this run -- score reflects the "
                              "formula estimator alone.")
            basis.append(_PILLAR_LOCKED_NOTE[p])
            pillars[p] = PillarResult(
                score=round(pfs.score, 1), low=round(low, 1), high=round(high, 1),
                confidence=confidence, formula_score=round(pfs.score, 1), llm_score=None,
                basis=basis,
            )

    return CalculatorResult(
        E=pillars["E"], S=pillars["S"], G=pillars["G"],
        claims=[_claim_summary(c) for c in claims],
        narrative=None,  # Phase 5
    )
