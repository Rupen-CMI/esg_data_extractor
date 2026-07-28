"""
claim_validators.py — Tier-0 deterministic validators (UPDATED_AGENTIC_WORKFLOW.md's
"Tier 0 — Deterministic validators" node). No LLM, no DB writes, pure code.

WHY THIS EXISTS: this session's live evidence-recovery probe on the seed=314
backtest sample reproduced the "yoga-page" failure mode TWICE with real data --
a factor-targeted search for "Copastur Turismo ... gender pay ... workplace
safety" returned genuine, non-Wikipedia, sufficiently-long web text that was
entirely about a Hawaiian restaurant ("Lilikoi, Kauai's best new Restaurant
and Bar"), and a Google-News search for "MW Enterprises" returned an article
about a person named "Melissa Wyatt" with zero connection to the company.
Both would pass every existing collector-level filter (real text, on-topic-
looking source, long enough) while asserting nothing true about the company
being scored. pillar_extractors.py's Step-1 relevance check is a PROMPT
instruction to the LLM ("does this text actually discuss X") -- the exact
same class of instruction ("if evidence is thin, score near 45-55") that
measurably failed to hold in the original scoring_agent baseline. A
deterministic, code-level check that never depends on the LLM choosing to
follow an instruction is the honest backstop.

This module runs AFTER extract_all_claims() + ct_anchor_claims() return and
BEFORE compute_formula_scores() sees the claims -- see calibration_harness.py
_gather_and_score_formula and graph.py's claim-extraction node for the two
call sites.

Five deterministic rules, all either DROP a claim or CAP its confidence
(never raise it, never fabricate a value):

  a. Lexical relevance   -- the yoga-page catcher. DROP.
  b. Numeric bounds      -- null out impossible/out-of-range values. CAP.
  c. Polarity consistency-- an inherently-negative event factor claimed
                             positive is suspicious, not impossible. CAP.
  d. Corroboration       -- one high-weight negative claim from one source
                             shouldn't swing a pillar alone. CAP.
  e. Known-failure rule  -- high confidence + very short cited text was the
                             observed shape of both live failures above. CAP.

Every action is logged as a ValidationFlag for audit (surfaced as
claims_dropped/claims_capped counts in the calibration harness).
"""

import math
from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.layer_2.factor_registry import get_factor
from agentic_estimation.layer_1.evidence_filters import _compiled_patterns

log = get_logger("claim_validators")

# ── Rule (a): per-factor topic vocabulary ────────────────────────────────────
# Keyed by factor_registry key. Deliberately narrower than evidence_filters.py's
# broad _ESG_RELEVANCE_TERMS (that gate asks "is this ESG-adjacent at all?";
# this one asks "does this text actually support THIS SPECIFIC factor?").
# Unlisted factors fall back to a per-pillar term set below -- additive, never
# a reason to fail closed on a factor we haven't hand-tuned yet.
_FACTOR_TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "scope_1_emissions": ("emission", "co2", "carbon", "ghg", "greenhouse gas", "tco2e"),
    "scope_2_emissions": ("emission", "co2", "carbon", "electricity", "energy", "ghg"),
    "scope_3_emissions": ("emission", "co2", "carbon", "supply chain", "value chain", "ghg"),
    "renewable_energy_pct": ("renewable", "solar", "wind", "energy", "clean power"),
    "total_energy_consumption": ("energy", "electricity", "power", "consumption"),
    "water_withdrawal": ("water", "withdrawal", "wastewater"),
    "total_waste_generated": ("waste", "recycling", "landfill"),
    "net_zero_pledge": ("net zero", "carbon neutral", "climate target", "pledge"),
    "sbti_commitment": ("sbti", "science based target"),
    "cdp_disclosure": ("cdp", "climate disclosure"),
    "environmental_controversy": ("spill", "pollution", "contamination", "environmental violation",
                                   "epa", "fine", "penalty"),
    "female_employees_pct": ("gender", "women", "female", "diversity", "workforce"),
    "female_board_pct": ("gender", "women", "female", "board", "director"),
    "employee_turnover_rate": ("turnover", "attrition", "retention", "workforce"),
    "lost_time_injury_rate": ("injury", "safety", "trir", "osha", "accident", "fatality"),
    "workplace_safety": ("safety", "injury", "trir", "osha", "accident", "workplace"),
    "labor_controversy": ("strike", "union", "labor", "labour", "wage", "layoff", "dispute"),
    "human_rights_incident": ("human rights", "modern slavery", "forced labor", "forced labour",
                               "child labor", "child labour"),
    "board_independence_pct": ("board", "independent director", "director", "governance"),
    "anti_corruption_policy": ("anti-corruption", "bribery", "corruption", "compliance"),
    "whistleblower_mechanism": ("whistleblower", "ethics", "grievance"),
    "esg_report_published": ("esg report", "sustainability report", "disclosure"),
    "third_party_esg_audit": ("audit", "third-party", "assurance", "verification"),
    "regulatory_fines": ("fine", "penalty", "settlement", "sec ", "regulator", "enforcement"),
    "litigation": ("lawsuit", "litigation", "legal", "sued", "settlement"),
    "compliance_certification": ("certification", "compliance", "certified", "audit"),
    "governance_controversy": ("misconduct", "scandal", "governance", "ethics", "fraud"),
}

_PILLAR_FALLBACK_TERMS: dict[str, tuple[str, ...]] = {
    "E": ("environmental", "emission", "carbon", "climate", "energy", "water", "waste", "sustainability"),
    "S": ("labor", "labour", "employee", "worker", "diversity", "safety", "human rights", "community"),
    "G": ("governance", "board", "compliance", "regulator", "litigation", "ethics", "corruption"),
}

# ── Rule (c): factors whose polarity is fixed by definition ─────────────────
# Any claim on these factors carrying the OPPOSITE polarity is suspicious --
# these are "bad news" factors by construction (see factor_registry.py
# direction="lower" event factors).
_INHERENTLY_NEGATIVE_FACTORS = {
    "environmental_controversy", "labor_controversy", "human_rights_incident",
    "regulatory_fines", "litigation", "governance_controversy",
}

# ── Rule (d): corroboration floor ────────────────────────────────────────────
_CORROBORATION_WEIGHT_FLOOR = 9.0  # factor weight at/above which a single-source claim gets capped

# ── Rule (e): known-failure shape (short text + high confidence) ────────────
_SHORT_TEXT_CHARS = 200
_HIGH_CONFIDENCE = 0.7

# Methods whose source_tag is synthetic (not a key in the signals dict) or
# whose text was never LLM-read in the first place -- lexical relevance and
# the short-text rule only make sense for LLM-extracted claims citing real
# gathered text.
_EXEMPT_FROM_TEXT_CHECKS = {"dataset_lookup", "peer_ratio_fallback", "coarse_bucket"}


@dataclass
class ValidationFlag:
    factor: str
    pillar: str
    rule: str          # 'lexical_relevance' | 'numeric_bounds' | 'polarity_consistency'
                        # | 'corroboration' | 'known_failure_shape'
    action: str         # 'dropped' | 'capped'
    detail: str


def _topic_terms_for(factor_key: str, pillar: str) -> tuple[str, ...]:
    return _FACTOR_TOPIC_TERMS.get(factor_key) or _PILLAR_FALLBACK_TERMS.get(pillar, ())


def _check_lexical_relevance(claim: ExtractedClaim, signals: dict) -> Optional[ValidationFlag]:
    """Rule (a) -- the yoga-page / Hawaiian-restaurant catcher. Returns a
    'dropped' flag if the claim should be discarded, else None."""
    if claim.method in _EXEMPT_FROM_TEXT_CHECKS:
        return None
    text = signals.get(claim.source_tag) if signals else None
    if not text:
        # No cited text to check against -- can't confirm relevance, but this
        # isn't this rule's job to police (a missing source_tag mapping is a
        # different failure than irrelevant-but-present text). Leave to caller.
        return None
    terms = _topic_terms_for(claim.factor, claim.pillar)
    if not terms:
        return None
    patterns = _compiled_patterns(terms)
    if any(p.search(text) for p in patterns):
        return None
    return ValidationFlag(
        factor=claim.factor, pillar=claim.pillar, rule="lexical_relevance", action="dropped",
        detail=f"cited signal '{claim.source_tag}' contains none of the topic terms for "
               f"'{claim.factor}' -- likely wrong-entity or off-topic text (source_tag text: "
               f"{text[:120]!r}...)",
    )


def _check_numeric_bounds(claim: ExtractedClaim, country: Optional[str]) -> Optional[ValidationFlag]:
    """Rule (b). Mutates claim.value in place (nulling it) when out of bounds;
    returns a 'capped' flag describing what happened, or None if untouched."""
    if claim.value is None:
        return None

    if not math.isfinite(claim.value) or claim.value < 0:
        detail = f"value={claim.value!r} is non-finite or negative -- nulled"
        claim.value = None
        return ValidationFlag(factor=claim.factor, pillar=claim.pillar, rule="numeric_bounds",
                               action="capped", detail=detail)

    factor = get_factor(claim.factor)
    if factor is not None and factor.key.endswith("_pct") and not (0.0 <= claim.value <= 100.0):
        detail = f"pct value={claim.value!r} outside [0,100] -- nulled"
        claim.value = None
        return ValidationFlag(factor=claim.factor, pillar=claim.pillar, rule="numeric_bounds",
                               action="capped", detail=detail)

    if claim.factor == "scope_1_emissions" and country:
        try:
            from agentic_estimation.layer_2.climate_trace_anchor import get_country_total_emissions
            ceiling = get_country_total_emissions(country)
        except Exception as e:
            log.warning("numeric_bounds: country-total lookup failed (%s) -- skipping check", e)
            ceiling = None
        if ceiling is not None and claim.value > ceiling:
            claim.confidence = min(claim.confidence, 0.3)
            detail = (f"claimed scope_1_emissions={claim.value:,.0f} exceeds {country}'s entire "
                      f"harvested total ({ceiling:,.0f} tCO2e) -- confidence capped at 0.3")
            return ValidationFlag(factor=claim.factor, pillar=claim.pillar, rule="numeric_bounds",
                                   action="capped", detail=detail)
    return None


def _check_polarity_consistency(claim: ExtractedClaim) -> Optional[ValidationFlag]:
    """Rule (c). Caps confidence in place; returns a flag if triggered."""
    if claim.factor in _INHERENTLY_NEGATIVE_FACTORS and claim.polarity == 1:
        old_conf = claim.confidence
        claim.confidence = claim.confidence * 0.5
        return ValidationFlag(
            factor=claim.factor, pillar=claim.pillar, rule="polarity_consistency", action="capped",
            detail=f"'{claim.factor}' is an inherently-negative factor but claim carries polarity=+1 "
                   f"-- confidence {old_conf:.2f} -> {claim.confidence:.2f}",
        )
    return None


def _check_known_failure_shape(claim: ExtractedClaim, signals: dict) -> Optional[ValidationFlag]:
    """Rule (e). Caps confidence in place; returns a flag if triggered."""
    if claim.method in _EXEMPT_FROM_TEXT_CHECKS:
        return None
    if claim.confidence <= _HIGH_CONFIDENCE:
        return None
    text = signals.get(claim.source_tag) if signals else None
    if text is None or len(text) >= _SHORT_TEXT_CHARS:
        return None
    old_conf = claim.confidence
    claim.confidence = min(claim.confidence, 0.5)
    return ValidationFlag(
        factor=claim.factor, pillar=claim.pillar, rule="known_failure_shape", action="capped",
        detail=f"confidence={old_conf:.2f} on a {len(text)}-char cited snippet -- the observed shape "
               f"of both live wrong-entity failures this session -- capped to {claim.confidence:.2f}",
    )


def _check_corroboration(claims: list[ExtractedClaim]) -> list[ValidationFlag]:
    """Rule (d). Operates across the WHOLE claim list (needs to count sources
    per factor), unlike the other rules which are per-claim. Caps in place."""
    flags: list[ValidationFlag] = []
    by_factor: dict[str, list[ExtractedClaim]] = {}
    for c in claims:
        by_factor.setdefault(c.factor, []).append(c)

    for factor_key, factor_claims in by_factor.items():
        factor = get_factor(factor_key)
        if factor is None or factor.weight < _CORROBORATION_WEIGHT_FLOOR:
            continue
        negative = [c for c in factor_claims if c.polarity == -1]
        if not negative:
            continue  # nothing to cap
        # Corroboration is about DISTINCT sources, not claim-object count -- a
        # single extraction call can return the same underlying fact as
        # several near-duplicate claim objects (reworded reasoning) all
        # citing the same source_tag. Counting objects instead of sources
        # let exactly that slip through uncapped in production (Cardinal
        # Health: one google_news_rss article extracted 7x as
        # labor_controversy, all sharing one source_tag, never capped).
        sources = {c.source_tag for c in negative}
        if len(sources) != 1:
            continue  # >=2 distinct sources: genuinely corroborated.
        # Single distinct source, possibly duplicated across N claim objects
        # -- cap all of them, not just a lone claim.
        for claim in negative:
            old_conf = claim.confidence
            claim.confidence = claim.confidence * 0.7
            flags.append(ValidationFlag(
                factor=factor_key, pillar=claim.pillar, rule="corroboration", action="capped",
                detail=f"single source ({claim.source_tag}) for high-weight (w={factor.weight}) negative "
                       f"claim -- confidence {old_conf:.2f} -> {claim.confidence:.2f}",
            ))
    return flags


def validate_claims(
    claims: list[ExtractedClaim],
    signals: Optional[dict] = None,
    country: Optional[str] = None,
) -> tuple[list[ExtractedClaim], list[ValidationFlag]]:
    """Pure function, no LLM, no DB writes (the one numeric-bounds check that
    reads climate_trace_country_emissions is read-only). Returns (kept_claims,
    flags) -- kept_claims is a NEW list (claims that failed lexical relevance
    are dropped entirely); claims that survive may have had `.value` nulled or
    `.confidence` capped in place.

    signals: {source_tag: text}, the same dict claims were extracted from.
    None is safe (all signal-dependent rules degrade to no-ops), but passing
    it is how rules (a) and (e) actually catch anything.
    """
    signals = signals or {}
    flags: list[ValidationFlag] = []
    kept: list[ExtractedClaim] = []

    for claim in claims:
        drop_flag = _check_lexical_relevance(claim, signals)
        if drop_flag is not None:
            flags.append(drop_flag)
            log.info("[%s/%s] DROPPED: %s", claim.pillar, claim.factor, drop_flag.detail)
            continue

        for check in (
            lambda c: _check_numeric_bounds(c, country),
            _check_polarity_consistency,
            lambda c: _check_known_failure_shape(c, signals),
        ):
            flag = check(claim)
            if flag is not None:
                flags.append(flag)
                log.info("[%s/%s] CAPPED (%s): %s", claim.pillar, claim.factor, flag.rule, flag.detail)

        kept.append(claim)

    flags.extend(_check_corroboration(kept))

    return kept, flags
