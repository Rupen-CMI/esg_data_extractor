"""
critic_panel.py — Phase 4's gated adversarial LLM critics (see PHASE_4_PLAN.md
sections 2-4). No DB access.

Three distinct lenses, each one `zen_client.call_with_prompt` call, run only
when `estimate_verifier.py` decides a pillar is genuinely ambiguous (real
evidence exists, but the two independent estimators only moderately agree).
This is NOT the pipeline's first line of defense -- claim_validators.py's
Tier-0 deterministic checks already run on every claim, every company, for
free, and catch the crude case (a claim citing text with zero topic
vocabulary for its factor). These critics exist for the case Tier-0
structurally cannot catch: cited text that is genuinely ON-TOPIC but does not
actually assert what the claim says (e.g. a sustainability-report mention of
"emissions methodology" being read as an environmental_controversy claim).

Each critic returns pass/refute + an optional flagged_factor + an objection.
Majority (>=2 of 3) refute -> the panel result is refuted. Convergence
(>=2 refuters naming the SAME flagged_factor) is the signal
estimate_verifier.py uses to decide whether a bounded retry is even possible
(a converged factor whose winning claim is peer_anchor/dataset_lookup can't
be fixed by re-extraction -- see estimate_verifier.py's retryability guard).

A critic whose LLM call fails or returns unparseable JSON ABSTAINS rather
than counting as either a pass or a refute -- majority is computed over
responders only. Fewer than 2 responders -> fail-open (refuted=False),
logged as a warning, never a hard crash. This mirrors the fail-open,
never-raise discipline claim_validators.py and confidence_gate.py already
follow.
"""

from dataclasses import dataclass, field
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger
from agentic_estimation.shared.llm_json import extract_json_object

log = get_logger("critic_panel")

_MAX_EXCERPT_CHARS = 800
_CRITIC_MAX_TOKENS = 800
_CRITIC_TIMEOUT = 120

_CRITIC_SYSTEM = (
    "You are an adversarial ESG estimate reviewer. Your job is to find reasons this "
    "estimate might be WRONG, not to confirm it. After reasoning, you MUST end with a "
    "single valid JSON object of the exact shape requested."
)


@dataclass
class CriticVerdict:
    critic: str                      # 'evidence_support' | 'peer_plausibility' | 'internal_consistency'
    verdict: str                     # 'pass' | 'refute' | 'abstain'
    flagged_factor: Optional[str]
    objection: str


@dataclass
class CriticPanelResult:
    pillar: str
    verdicts: list = field(default_factory=list)   # list[CriticVerdict]
    refuted: bool = False
    flagged_factor: Optional[str] = None            # set only on convergence (>=2 agree)
    objections: list = field(default_factory=list)  # objection texts from refuting critics


def _valid_factors(formula_score) -> set:
    return {c.factor for c in (formula_score.contributions or [])}


def _winning_claim_source_tags(formula_score, claims) -> dict:
    """Contribution has no source_tag field (see formula_estimator.py's
    Contribution dataclass) -- only the underlying ExtractedClaim does. To
    show Critic A the actual cited text, re-derive which claim WON each
    factor using the exact same selection rule formula_estimator.py's
    _pick_best_claim uses (highest confidence, then method trust) -- not a
    new/different selection, just recovering what already happened."""
    from agentic_estimation.layer_3.formula_estimator import _claim_sort_key

    by_factor: dict = {}
    for c in (claims or []):
        by_factor.setdefault(c.factor, []).append(c)

    tags = {}
    for factor, factor_claims in by_factor.items():
        if factor_claims:
            best = max(factor_claims, key=_claim_sort_key)
            tags[factor] = best.source_tag
    return tags


def _parse_critic_response(raw_text: str, critic_name: str, valid_factors: set) -> CriticVerdict:
    """Fail-closed parse: unparseable/invalid response -> abstain. A
    flagged_factor not in this pillar's own contribution list is nulled --
    a critic can't flag a factor that isn't in the audit trail."""
    parsed = extract_json_object(raw_text) if raw_text else None
    if not parsed:
        return CriticVerdict(critic=critic_name, verdict="abstain", flagged_factor=None,
                              objection="(unparseable critic response)")

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "refute"):
        return CriticVerdict(critic=critic_name, verdict="abstain", flagged_factor=None,
                              objection="(critic returned neither pass nor refute)")

    flagged = parsed.get("flagged_factor")
    if flagged is not None and flagged not in valid_factors:
        log.warning("critic %s flagged unknown factor %r (not in this pillar's contributions) -- nulling",
                    critic_name, flagged)
        flagged = None

    objection = str(parsed.get("objection", ""))[:500]
    return CriticVerdict(critic=critic_name, verdict=verdict, flagged_factor=flagged, objection=objection)


def _run_one_critic(critic_name: str, prompt: str, valid_factors: set) -> CriticVerdict:
    from zen_client import call_with_prompt
    try:
        resp = call_with_prompt(prompt, max_tokens=_CRITIC_MAX_TOKENS, timeout=_CRITIC_TIMEOUT,
                                 system=_CRITIC_SYSTEM)
    except Exception as exc:
        log.warning("critic %s call raised: %s -- abstaining", critic_name, exc)
        return CriticVerdict(critic=critic_name, verdict="abstain", flagged_factor=None,
                              objection=f"(critic call failed: {exc})")

    if not resp.get("ok"):
        log.warning("critic %s call failed: %s -- abstaining", critic_name, resp.get("error"))
        return CriticVerdict(critic=critic_name, verdict="abstain", flagged_factor=None,
                              objection="(critic call did not succeed)")

    raw_text = resp.get("raw", "") or resp.get("reasoning", "")
    return _parse_critic_response(raw_text, critic_name, valid_factors)


# ── Critic A: evidence-support ───────────────────────────────────────────────

def _critic_a_prompt(pillar: str, company: str, formula_score, signals: dict, claims) -> str:
    winning_tags = _winning_claim_source_tags(formula_score, claims)
    lines = [
        f"Review the {pillar} pillar's evidence-based contributions for {company}. "
        f"For EACH contribution below, does the quoted cited text actually assert what "
        f"the claim says? A claim can look on-topic (right vocabulary) while asserting "
        f"nothing of the sort -- e.g. text about a sustainability report's emissions "
        f"methodology is NOT evidence of an environmental controversy. Quote the exact "
        f"mismatch if you refute.",
        "",
    ]
    for c in (formula_score.contributions or []):
        lines.append(f"- factor: {c.factor} | claim_reasoning: {c.claim_reasoning!r} | "
                     f"confidence: {c.confidence:.2f} | method: {c.method}")
    lines.append("")
    lines.append("CITED SIGNAL EXCERPTS (only for extracted claims -- dataset/peer-derived "
                  "contributions have no signal text to check):")
    for c in (formula_score.contributions or []):
        if c.method != "extracted" or not signals:
            continue
        source_tag = winning_tags.get(c.factor)
        text = signals.get(source_tag) if source_tag else None
        if text:
            lines.append(f"[{c.factor} / {source_tag}]\n{text[:_MAX_EXCERPT_CHARS]}")
    lines.append("")
    lines.append('Respond with ONLY this JSON object after your reasoning:\n'
                  '{"verdict": "pass"|"refute", "flagged_factor": "<exact factor key>"|null, '
                  '"objection": "<one sentence, quote the mismatch if refuting>"}')
    return "\n".join(lines)


# ── Critic B: peer-plausibility ──────────────────────────────────────────────

def _critic_b_prompt(pillar: str, company: str, reconciled, formula_score) -> str:
    breakdown = getattr(formula_score, "breakdown", None)
    breakdown_lines = ""
    if breakdown is not None:
        breakdown_lines = (
            f"Evidence mass: {breakdown.evidence_mass:.2f} | Coverage: {breakdown.coverage:.2f} "
            f"| Coverage multiplier: {breakdown.coverage_multiplier:.2f}\n"
        )
    peer_line = ""
    if formula_score.peer_anchor is not None and formula_score.peer_anchor.percentile is not None:
        peer_line = f"Peer anchor: {formula_score.peer_anchor.basis}\n"

    contrib_lines = "\n".join(
        f"  - {c.factor}: {c.points:+.1f} pts (weight={c.weight}, confidence={c.confidence:.2f})"
        for c in (formula_score.contributions or [])
    ) or "  (no contributions)"

    return (
        f"Review the {pillar} pillar's final reconciled score for {company} for arithmetic "
        f"plausibility.\n\n"
        f"Country baseline: {formula_score.baseline:.1f} (source={formula_score.baseline_source})\n"
        f"{peer_line}{breakdown_lines}"
        f"Formula contributions:\n{contrib_lines}\n\n"
        f"Reconciled score: {reconciled.score:.1f} (spread={reconciled.spread}, "
        f"confidence={reconciled.confidence})\n\n"
        f"Is this final score plausible given the baseline, the real peer statistic (if any), "
        f"and how much evidence actually supports it? A large swing on thin/weak evidence is "
        f"implausible -- show the arithmetic if you refute.\n\n"
        f'Respond with ONLY this JSON object after your reasoning:\n'
        f'{{"verdict": "pass"|"refute", "flagged_factor": "<exact factor key>"|null, '
        f'"objection": "<one sentence, show the arithmetic if refuting>"}}'
    )


# ── Critic C: internal-consistency ───────────────────────────────────────────

def _critic_c_prompt(pillar: str, company: str, reconciled, formula_score, holistic) -> str:
    formula_narrative = "\n".join(
        f"  - {c.factor}: {c.claim_reasoning}"
        for c in (formula_score.contributions or [])
    ) or "  (no evidence -- baseline only)"

    holistic_reasoning = "(no holistic vote available)"
    holistic_score_str = "n/a"
    if holistic is not None:
        holistic_reasoning = getattr(holistic, f"{pillar.lower()}_reasoning", "(no reasoning given)")
        holistic_score_str = f"{getattr(holistic, f'{pillar.lower()}_score', float('nan')):.1f}"

    return (
        f"Review the {pillar} pillar's two independent estimates for {company} for internal "
        f"consistency.\n\n"
        f"FORMULA estimator's evidence narrative:\n{formula_narrative}\n\n"
        f"HOLISTIC (LLM) estimator's reasoning: {holistic_reasoning!r} (score={holistic_score_str})\n\n"
        f"Reconciled: {reconciled.score:.1f} (spread={reconciled.spread})\n\n"
        f"Do the two estimators' narratives CONTRADICT each other (e.g. Formula credits strong "
        f"positive evidence while Holistic's reasoning says none was found, or vice versa)? "
        f"A contradiction is a refute.\n\n"
        f'Respond with ONLY this JSON object after your reasoning:\n'
        f'{{"verdict": "pass"|"refute", "flagged_factor": "<exact factor key>"|null, '
        f'"objection": "<one sentence describing the contradiction if refuting>"}}'
    )


def run_critic_panel(
    pillar: str,
    company: str,
    reconciled,
    formula_score,
    holistic,
    claims,
    signals: dict,
    metadata: Optional[dict] = None,
) -> CriticPanelResult:
    """Runs all 3 critics sequentially (each call serializes through
    zen_client's process-wide rate limiter regardless of call order, so
    sequential vs threaded costs the same wall-clock here -- sequential is
    simpler and keeps critic call order deterministic for logging)."""
    valid_factors = _valid_factors(formula_score)

    prompts = {
        "evidence_support": _critic_a_prompt(pillar, company, formula_score, signals, claims),
        "peer_plausibility": _critic_b_prompt(pillar, company, reconciled, formula_score),
        "internal_consistency": _critic_c_prompt(pillar, company, reconciled, formula_score, holistic),
    }

    verdicts = [_run_one_critic(name, prompt, valid_factors) for name, prompt in prompts.items()]

    responders = [v for v in verdicts if v.verdict != "abstain"]
    if len(responders) < 2:
        log.warning("[%s/%s] fewer than 2 critics responded (%d abstained) -- fail-open, refuted=False",
                    company, pillar, len(verdicts) - len(responders))
        return CriticPanelResult(pillar=pillar, verdicts=verdicts, refuted=False,
                                  flagged_factor=None, objections=[])

    refuters = [v for v in responders if v.verdict == "refute"]
    refuted = len(refuters) >= 2

    flagged_factor = None
    if refuted:
        factor_counts: dict = {}
        for v in refuters:
            if v.flagged_factor:
                factor_counts[v.flagged_factor] = factor_counts.get(v.flagged_factor, 0) + 1
        for factor, count in factor_counts.items():
            if count >= 2:
                flagged_factor = factor
                break

    objections = [v.objection for v in refuters] if refuted else []

    log.info("[%s/%s] critic panel: %d/%d refute, converged_factor=%s",
              company, pillar, len(refuters), len(responders), flagged_factor)

    return CriticPanelResult(pillar=pillar, verdicts=verdicts, refuted=refuted,
                              flagged_factor=flagged_factor, objections=objections)
