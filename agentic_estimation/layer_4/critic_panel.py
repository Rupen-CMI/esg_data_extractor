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

# No excerpt truncation constant here on purpose -- removed 2026-09-18.
# Critic A (evidence_support) used to see only the first 800 chars of the
# cited signal, which could cause it to refute a claim purely because the
# real supporting sentence sat past that cutoff, never having actually been
# wrong about the evidence. It now sees the FULL text the original
# extraction was shown (same signal, no re-truncation) -- see
# _critic_a_prompt. _CRITIC_MAX_TOKENS only bounds the critic's OWN
# response length, not how much cited text it reads.
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
    corrected_excerpt: Optional[str] = None   # only 'evidence_support' populates this on refute --
                                               # the exact text the critic found that actually
                                               # supports/contradicts the claim, quoted verbatim
                                               # from the SAME full signal it was shown (see
                                               # _critic_a_prompt). None for the other two lenses
                                               # (they never see cited signal text) and for pass/abstain.


@dataclass
class CriticPanelResult:
    pillar: str
    verdicts: list = field(default_factory=list)   # list[CriticVerdict]
    refuted: bool = False
    flagged_factor: Optional[str] = None            # set only on convergence (>=2 agree)
    objections: list = field(default_factory=list)  # objection texts from refuting critics
    corrected_excerpt: Optional[str] = None         # from the converged refuter that supplied one
                                                     # (evidence_support), for estimate_verifier.py
                                                     # to feed into the retry's extraction prompt
    inconclusive: bool = False                      # True when <2 critics responded (LLM calls
                                                     # failed/timed out, not a real disagreement) --
                                                     # refuted is always False in this case (the
                                                     # fail-open safety behavior is unchanged), but
                                                     # this flag lets callers tell "nobody actually
                                                     # checked" apart from "checked and it passed".
                                                     # Added 2026-09-19 -- previously both cases
                                                     # produced an identical refuted=False result,
                                                     # so an infra outage was indistinguishable from
                                                     # a real, majority "pass" verdict downstream.


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

    # Only meaningful on refute, and only evidence_support ever asks for it
    # (the other two lenses aren't shown cited signal text at all) -- a
    # stray value from another critic or on a pass verdict is discarded.
    corrected_excerpt = None
    if verdict == "refute" and critic_name == "evidence_support":
        raw_excerpt = parsed.get("corrected_excerpt")
        if raw_excerpt:
            corrected_excerpt = str(raw_excerpt).strip() or None

    return CriticVerdict(critic=critic_name, verdict=verdict, flagged_factor=flagged,
                          objection=objection, corrected_excerpt=corrected_excerpt)


def _run_one_critic(critic_name: str, prompt: str, valid_factors: set,
                     model: Optional[str] = None) -> CriticVerdict:
    from zen_client import call_with_prompt, DEFAULT_MODEL
    try:
        resp = call_with_prompt(prompt, model=model or DEFAULT_MODEL,
                                 max_tokens=_CRITIC_MAX_TOKENS, timeout=_CRITIC_TIMEOUT,
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


def _prior_round_block(prior_round: Optional[dict]) -> str:
    """Renders round-1's objections + corrected_excerpt as a short context
    block for round-2 critic prompts. Round 2 re-runs on genuinely NEW
    evidence (new_claims from the retry's re-extraction), so this is not a
    cache of round-1's verdict -- it's telling round-2 what was contested
    and how it was supposedly fixed, so it can check that specific concern
    instead of re-deriving it blind. None (default) reproduces round-1's
    exact prompt text, unchanged."""
    if not prior_round:
        return ""
    objections = prior_round.get("objections") or []
    corrected_excerpt = prior_round.get("corrected_excerpt")
    lines = [
        "",
        "--- CONTEXT: this pillar was refuted on the first pass and has been "
        "re-extracted. Check specifically whether that fix actually resolved "
        "the original concern -- do not just re-run a generic check. ---",
    ]
    if objections:
        lines.append("Original objection(s): " + " | ".join(objections))
    if corrected_excerpt:
        lines.append(f"Evidence was re-extracted using this corrected excerpt: {corrected_excerpt!r}")
    lines.append("---")
    return "\n".join(lines)


# ── Critic A: evidence-support ───────────────────────────────────────────────

def _critic_a_prompt(pillar: str, company: str, formula_score, signals: dict, claims,
                      prior_round: Optional[dict] = None) -> str:
    winning_tags = _winning_claim_source_tags(formula_score, claims)
    lines = [
        f"Review the {pillar} pillar's evidence-based contributions for {company}. "
        f"For EACH contribution below, does the quoted cited text actually assert what "
        f"the claim says? A claim can look on-topic (right vocabulary) while asserting "
        f"nothing of the sort -- e.g. text about a sustainability report's emissions "
        f"methodology is NOT evidence of an environmental controversy. Quote the exact "
        f"mismatch if you refute.\n"
        f"If you refute AND the full signal text below actually contains a passage that "
        f"correctly supports or contradicts the claim (just not the one the original "
        f"extraction cited), quote that passage verbatim in \"corrected_excerpt\" -- it "
        f"will be handed to a re-extraction pass so a real correction doesn't require "
        f"searching the same text blind a second time. Leave it null if no such passage "
        f"exists (i.e. the claim should simply be dropped, not corrected).",
        "",
    ]
    for c in (formula_score.contributions or []):
        lines.append(f"- factor: {c.factor} | claim_reasoning: {c.claim_reasoning!r} | "
                     f"confidence: {c.confidence:.2f} | method: {c.method}")
    lines.append("")
    lines.append("CITED SIGNAL EXCERPTS -- the FULL text each claim was extracted from (only for "
                  "extracted claims -- dataset/peer-derived contributions have no signal text to "
                  "check). Not truncated: this is exactly what the original extraction saw, so a "
                  "refute here means the text genuinely doesn't support the claim, not that "
                  "relevant text was cut off before you could see it.")
    for c in (formula_score.contributions or []):
        if c.method != "extracted" or not signals:
            continue
        source_tag = winning_tags.get(c.factor)
        text = signals.get(source_tag) if source_tag else None
        if text:
            lines.append(f"[{c.factor} / {source_tag}]\n{text}")
    lines.append("")
    lines.append(_prior_round_block(prior_round))
    lines.append('Respond with ONLY this JSON object after your reasoning:\n'
                  '{"verdict": "pass"|"refute", "flagged_factor": "<exact factor key>"|null, '
                  '"objection": "<one sentence, quote the mismatch if refuting>", '
                  '"corrected_excerpt": "<verbatim supporting passage>"|null}')
    return "\n".join(lines)


# ── Critic B: peer-plausibility ──────────────────────────────────────────────

def _critic_b_prompt(pillar: str, company: str, reconciled, formula_score,
                      prior_round: Optional[dict] = None) -> str:
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
        f"implausible -- show the arithmetic if you refute.\n"
        f"{_prior_round_block(prior_round)}\n\n"
        f'Respond with ONLY this JSON object after your reasoning:\n'
        f'{{"verdict": "pass"|"refute", "flagged_factor": "<exact factor key>"|null, '
        f'"objection": "<one sentence, show the arithmetic if refuting>"}}'
    )


# ── Critic C: internal-consistency ───────────────────────────────────────────

def _critic_c_prompt(pillar: str, company: str, reconciled, formula_score, holistic,
                      prior_round: Optional[dict] = None) -> str:
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
        f"A contradiction is a refute.\n"
        f"{_prior_round_block(prior_round)}\n\n"
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
    model: Optional[str] = None,
    prior_round: Optional[dict] = None,
) -> CriticPanelResult:
    """Runs all 3 critics sequentially (each call serializes through
    zen_client's process-wide rate limiter regardless of call order, so
    sequential vs threaded costs the same wall-clock here -- sequential is
    simpler and keeps critic call order deterministic for logging).

    model: optional override forwarded to every critic's call_with_prompt.
    None (default) preserves today's exact behavior.

    prior_round: optional {"objections": [...], "corrected_excerpt": str|None}
    from round 1's CriticPanelResult. Only meaningful for a round-2 call
    (estimate_verifier.py passes this after a retry) -- tells the fresh
    panel what was previously contested and how the evidence was corrected,
    so it checks that specific concern instead of re-deriving it blind.
    None (default) preserves round-1's exact prompt text."""
    valid_factors = _valid_factors(formula_score)

    prompts = {
        "evidence_support": _critic_a_prompt(pillar, company, formula_score, signals, claims,
                                              prior_round=prior_round),
        "peer_plausibility": _critic_b_prompt(pillar, company, reconciled, formula_score,
                                               prior_round=prior_round),
        "internal_consistency": _critic_c_prompt(pillar, company, reconciled, formula_score, holistic,
                                                  prior_round=prior_round),
    }

    # Early exit after 2 calls when their verdicts already mathematically
    # decide `refuted` no matter what the 3rd critic says -- found 2026-09-18:
    # this ran all 3 unconditionally before. With 3 critics and refuted
    # requiring >=2 refuters, the FIRST TWO agreeing (both refute, or both
    # pass) already locks the outcome: two refutes can't drop below 2; two
    # passes leave only 1 remaining critic, who alone can never reach the
    # 2-refuter threshold. Only a 1-1 split (or an abstain in the first two)
    # is genuinely undecided and needs the 3rd call. Order stays deterministic
    # (evidence_support, peer_plausibility, internal_consistency) -- this
    # only skips calls that were already sequential and never reorders them.
    verdicts: list[CriticVerdict] = []
    names = list(prompts.keys())
    for i, name in enumerate(names):
        verdicts.append(_run_one_critic(name, prompts[name], valid_factors, model=model))
        if i == 1:  # just ran the 2nd of 3 -- check for an early, decided outcome
            responded_so_far = [v for v in verdicts if v.verdict != "abstain"]
            if len(responded_so_far) == 2:
                votes = {v.verdict for v in responded_so_far}
                if len(votes) == 1:   # both refute, or both pass -- 3rd can't change it
                    log.info("[%s/%s] critic panel: %s agree (%s) after 2 of 3 -- skipping 3rd call",
                              company, pillar, "/".join(v.critic for v in responded_so_far),
                              responded_so_far[0].verdict)
                    break

    responders = [v for v in verdicts if v.verdict != "abstain"]
    if len(responders) < 2:
        log.warning("[%s/%s] fewer than 2 critics responded (%d abstained) -- fail-open, "
                    "refuted=False, inconclusive=True", company, pillar, len(verdicts) - len(responders))
        return CriticPanelResult(pillar=pillar, verdicts=verdicts, refuted=False,
                                  flagged_factor=None, objections=[], inconclusive=True)

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

    # corrected_excerpt only ever comes from evidence_support (the only lens
    # shown cited signal text -- peer_plausibility/internal_consistency
    # have nothing to quote from and never populate it, see CriticVerdict).
    # No 2-of-3 convergence requirement here, unlike flagged_factor: the
    # other two lenses structurally can't corroborate or contradict a
    # quoted excerpt they were never shown, so requiring their agreement
    # would make this field permanently unreachable rather than add rigor.
    corrected_excerpt = None
    if refuted:
        for v in refuters:
            if v.critic == "evidence_support" and v.corrected_excerpt:
                corrected_excerpt = v.corrected_excerpt
                break

    log.info("[%s/%s] critic panel: %d/%d refute, converged_factor=%s, corrected_excerpt=%s",
              company, pillar, len(refuters), len(responders), flagged_factor,
              "yes" if corrected_excerpt else "no")

    return CriticPanelResult(pillar=pillar, verdicts=verdicts, refuted=refuted,
                              flagged_factor=flagged_factor, objections=objections,
                              corrected_excerpt=corrected_excerpt)
