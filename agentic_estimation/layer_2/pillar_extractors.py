"""
pillar_extractors.py — LLM evidence taggers for E/S/G (Phase 2, Step 3;
see PHASE_2_PLAN.md).

One LLM call PER PILLAR (3/company), not one combined call: a parse failure
loses one pillar's claims, not all three; each prompt carries only that
pillar's ~7-12 factor definitions (better JSON compliance on the free
gateway than a ~27-factor combined prompt); no max_tokens truncation risk on
claim-heavy companies.

The model is an EVIDENCE TAGGER, never a scorer -- it reads the company's
gathered signal text and emits typed claims against the closed factor set in
factor_registry.py, citing which signal each claim came from. It never
converts a value into a score; formula_estimator.py does that deterministically.

Structural fix for the Phase-1 "yoga-page" finding (a DDG governance query for
Blackmores returned real, non-Wikipedia, sufficiently-long text that was
entirely about yoga poses -- passed every collector-level filter while
containing zero governance signal): the prompt requires an explicit
per-signal relevance check BEFORE extraction, and code-level validation drops
any claim whose source_tag isn't one of the signals actually shown to the
model.

extract_pillar_claims() / extract_all_claims() are PURE -- no DB access, just
{source: text} in, list[ExtractedClaim] out. persist_claims() is the only
writer, and the only place that resolves source_tag -> source_signal_id
(collector entry points return {source: text} without row ids, so this must
re-query company_esg_signals).

CLI:
    python -m agentic_estimation.layer_2.pillar_extractors dry "Nvidia" --pillar E
    python -m agentic_estimation.layer_2.pillar_extractors dry "Nvidia"   # all three pillars
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.shared.llm_json import extract_json_object
from agentic_estimation.layer_2.factor_registry import factors_for_pillar, get_factor
from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.layer_1.signal_agent import RateLimitTripped
from zen_client import OllamaRateLimitTripped

log = get_logger("pillar_extractors")

_MAX_CHARS_PER_SIGNAL = 4000
_MAX_TOTAL_CHARS = 60000

_PILLAR_TOPICS = {
    "E": (
        "environmental performance, greenhouse-gas and scope 1/2/3 emissions, "
        "energy use and renewable-energy sourcing, water withdrawal and stewardship, "
        "waste and recycling, biodiversity and deforestation, pollution/spills/"
        "contamination, climate targets (net-zero, SBTi, CDP disclosure) or "
        "environmental controversies, fines and violations"
    ),
    "S": (
        "labor practices and workforce, wages and pay equity (including gender pay "
        "gap), diversity and inclusion, board and workforce gender balance, "
        "workplace health and safety (injury/TRIR/fatality rates), human rights and "
        "modern slavery in operations or supply chain, unions/strikes/disputes, "
        "community relations, product safety, labor controversies"
    ),
    "G": (
        "corporate governance, board composition and independence, executive "
        "compensation, shareholder rights, regulatory compliance and enforcement, "
        "litigation and settlements, fines and penalties, anti-corruption/bribery, "
        "tax transparency, data privacy/breaches, whistleblower and ethics policies, "
        "audit and disclosure quality"
    ),
}

# unit -> multiplier to the metric's native unit (see CORE_METRICS units: tCO2e, GJ, m3, tonnes)
# Includes common natural-language phrasings an LLM is likely to emit verbatim
# from source text (e.g. "metric tons"), not just abbreviations -- an unmatched
# unit silently drops the value (see _coerce_unit_value), so this table being
# thin directly costs real, well-formed claims their numeric value.
_UNIT_MULTIPLIERS = {
    "t": 1.0, "tonnes": 1.0, "tonne": 1.0, "ton": 1.0, "tons": 1.0,
    "metric ton": 1.0, "metric tons": 1.0, "metric tonne": 1.0, "metric tonnes": 1.0,
    "tco2e": 1.0, "tco2": 1.0, "t co2e": 1.0, "t co2": 1.0,
    "kt": 1_000.0, "ktco2e": 1_000.0, "kilotonnes": 1_000.0, "kilotons": 1_000.0,
    "mt": 1_000_000.0, "mtco2e": 1_000_000.0, "megatonnes": 1_000_000.0, "megatons": 1_000_000.0,
    "million tonnes": 1_000_000.0, "million tons": 1_000_000.0, "million metric tons": 1_000_000.0,
    "million": 1_000_000.0,
    "gj": 1.0, "gigajoules": 1.0, "gigajoule": 1.0,
    "twh": 3_600_000.0, "gwh": 3_600.0, "mwh": 3.6, "kwh": 0.0036,
    "m3": 1.0, "m³": 1.0, "cubic meters": 1.0, "cubic metres": 1.0,
    "million m3": 1_000_000.0, "million cubic meters": 1_000_000.0, "million cubic metres": 1_000_000.0,
}

_OBJECTION_TEMPLATE = """
REVIEWER OBJECTION -- address before extracting:
A reviewer disputed the "{flagged_factor}" claim from a previous extraction pass on this \
same evidence. Their objection: {objection_text}
Re-read the signals relevant to "{flagged_factor}" with this objection specifically in mind. \
For that factor, either (a) correct the claim so it accurately reflects what the cited text \
actually says, (b) keep it only if you can point to the exact supporting sentence verbatim in \
"reasoning", or (c) omit it entirely if the objection is valid and no real support exists. \
Do not simply repeat the original claim unchanged without addressing the objection.
{corrected_excerpt_block}"""

_CORRECTED_EXCERPT_BLOCK = """
The reviewer also quoted this passage from the SAME signal text as what they believe actually \
supports or contradicts the "{flagged_factor}" claim:
    "{corrected_excerpt}"
Do NOT accept this passage automatically -- triple-check it against the original signal text \
yourself, the same way you would check any other claim. Confirm the quoted passage genuinely \
appears in the signal and genuinely supports what the reviewer says before using it. If it \
checks out, build the corrected claim from it. If it does NOT hold up on your own re-reading \
(misquoted, taken out of context, or doesn't actually say what the reviewer claims), disregard \
it and decide the factor on the original text alone.
"""

_PROMPT_TEMPLATE = """You are an evidence tagger reviewing signals about a company for the {pillar} \
(Environmental/Social/Governance -- specifically {topics}) pillar. You are NOT a scorer: \
you never invent facts, never convert a value into a 0-100 score, and never guess \
a factor's presence without a specific piece of supporting text.

COMPANY: {company}
{context_block}
SIGNALS (each tagged with its source):
{signals_block}
{objection_block}
STEP 1 -- RELEVANCE CHECK: for each signal above, decide whether it actually discusses \
something relevant to {topics} for THIS company. Some signals may be off-topic, generic, \
or about an unrelated company/subject that happened to match a search query -- extract \
NOTHING from those. Only proceed to Step 2 for signals that are genuinely on-topic.

STEP 1B -- DUPLICATE CHECK: multiple signals can report the SAME underlying event, worded \
differently by different outlets (e.g. "Toyota and IVECO join forces on hydrogen trucks" vs \
"IVECO and Toyota join forces on hydrogen trucks" -- same partnership, same event). Do NOT \
extract the same claim twice from what is really one event reported twice. But be careful: \
headlines that LOOK similar can still be genuinely DIFFERENT events -- "Toyota recalls \
500,000 vehicles" and "Toyota recalls 8,000 vehicles" are two separate recalls even though \
they share the same template, because the number, date, or specific defect differs. Only \
treat two signals as the same event if the SPECIFIC facts (numbers, dates, named parties) \
genuinely match, not just the general topic or sentence shape. When you do merge duplicate \
signals into one claim, cite whichever source_tag has the most complete/specific information.

STEP 2 -- EXTRACT CLAIMS: for each relevant signal, extract claims strictly from this \
closed factor list (do not invent new factor names):

{factor_list}

Rules:
- Only emit a claim if a signal ABOVE actually supports it -- cite that signal's exact \
  source tag in "source_tag".
- If a factor has a stated numeric value in the text (e.g. "1.2 million tCO2e"), put the \
  raw number in "value" and the unit exactly as written in "unit" -- do NOT normalise, \
  rescale, or convert it to a score yourself.
- "polarity": -1 if the claim is negative for the company (e.g. a controversy, a fine), \
  +1 if positive (e.g. a pledge, a certification), 0 if it's a neutral disclosed value.
- "strength" (0-1): how strong/severe/credible the claim is. "confidence" (0-1): how \
  certain you are this claim is accurate given the source text.
- No supporting text for a factor -> omit it entirely. Do not pad the list.

Respond with ONLY this JSON object (no markdown fences), after your reasoning:
{{"claims": [{{"factor": "<exact key from the list>", "polarity": <-1|0|1>, "strength": <0-1>, \
"confidence": <0-1>, "value": <number|null>, "unit": "<string|null>", "source_tag": "<exact tag from signals above>", \
"reasoning": "<one short sentence>"}}, ...]}}"""


# sector_emissions_intensity is the one non-zero-weight factor that is,
# same as the weight==0 badges below, never meant to be LLM-extracted --
# its only real claim source is climate_trace_anchor.py's deterministic
# Climate TRACE sector-percentile lookup (confidence 0.25 by design, see
# that module's docstring), not text evidence. Found + fixed 2026-09-18:
# weight=4 (not 0) meant _factor_list_block's weight==0 filter let it
# through into the live prompt anyway, worded as "(Climate TRACE anchor)"
# -- text the LLM has no actual Climate TRACE data to check against, so it
# can only skip the factor or hallucinate a guess. Worse, formula_estimator.
# _pick_best_claim() ranks claims by confidence FIRST or method-trust
# second (see that module) -- any hallucinated LLM guess with confidence
# above 0.25 would silently outrank and discard the real dataset-grounded
# claim for this exact factor. Excluding it from the prompt removes the
# hallucination risk at the source instead of trying to out-rank it later.
_LLM_EXCLUDED_FACTORS = {"sector_emissions_intensity"}

# Fields worth surfacing to the LLM as company context: country/industry help
# it judge plausibility (a claim about deep-sea drilling is implausible for a
# software company), employees/revenue give a rough size/scale anchor. Every
# other metadata field (qid, lei, sec_cik, website, subsidiary_count, ...) is
# either an identifier with no bearing on evidence judgement or too sparse to
# rely on. Deliberately excludes anything that could be a scoring answer --
# metadata never carries a benchmark/ESG score (see company_metadata.py's own
# return shape), so there is no has_ground_truth_leakage-style risk here.
_CONTEXT_FIELDS = ("country", "industry", "employees", "revenue")


def _context_block(metadata: Optional[dict]) -> str:
    """Renders `metadata` (see company_metadata.get_company_metadata) as a
    short CONTEXT line for the prompt. Was accepted-but-unused in this
    module until 2026-09-18 -- see module history: ~15 real callers across
    the codebase already gather and pass this in, it just never reached the
    prompt. Degrades to "" (an extra blank line, harmless) when metadata is
    None/{}/unmatched or every field is empty, so every existing caller that
    passes no metadata is completely unaffected."""
    if not metadata or not metadata.get("matched"):
        return ""
    parts = [f"{field}: {metadata[field]}" for field in _CONTEXT_FIELDS if metadata.get(field)]
    if not parts:
        return ""
    return "CONTEXT (from public company records, not evidence -- background only): " \
           + "; ".join(parts) + "\n"


def _factor_list_block(pillar: str) -> str:
    """Closed factor list injected into the LLM prompt.

    weight == 0 factors (badge factors: net_zero_pledge, sbti_commitment,
    cdp_disclosure, anti_corruption_policy, whistleblower_mechanism,
    esg_report_published, third_party_esg_audit, compliance_certification --
    see factor_registry.py's BADGE-FACTOR ZEROING note) are excluded here.
    They stay in the registry itself (still a valid target for a dataset-
    lookup claim, still auditable) -- this only stops the LLM from spending
    prompt space and relevance-checking effort finding evidence for factors
    that are mathematically guaranteed to contribute zero swing to the
    score regardless of what's extracted. Confirmed live 2026-09-12: every
    one of these factors was still being actively searched for and
    extracted with no effect on any score.

    _LLM_EXCLUDED_FACTORS (see above) are excluded for a different reason --
    not zero-weight, but never meant to be text-extracted at all.
    """
    lines = []
    for f in factors_for_pillar(pillar):
        if f.weight == 0 or f.key in _LLM_EXCLUDED_FACTORS:
            continue
        lines.append(f"  - {f.key} ({f.description}) [{f.delta_shape}]")
    return "\n".join(lines)


def _signals_block(signals: dict[str, str]) -> str:
    parts = []
    total = 0
    for tag, text in signals.items():
        chunk = text[:_MAX_CHARS_PER_SIGNAL].strip()
        if len(text) > _MAX_CHARS_PER_SIGNAL:
            chunk += "..."
        block = f"[source_tag: {tag}]\n{chunk}"
        if total + len(block) > _MAX_TOTAL_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "(no signals gathered)"


def _coerce_unit_value(value, unit: Optional[str]) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not unit:
        return v
    mult = _UNIT_MULTIPLIERS.get(unit.strip().lower())
    if mult is None:
        log.warning("unknown unit %r -- dropping value, claim degrades to event shape", unit)
        return None
    return v * mult


def extract_pillar_claims(
    pillar: str,
    company: str,
    signals: dict[str, str],
    metadata: Optional[dict] = None,
    objection: Optional[dict] = None,
    model: Optional[str] = None,
) -> list[ExtractedClaim]:
    """Pure: no DB access. One LLM call for one pillar.

    metadata: company context from company_metadata.get_company_metadata()
    (country, industry, employees, revenue, etc.) -- ~15 call sites across
    graph.py, calibration_harness.py, formula_estimator.py, reconcile.py,
    and several calibration/ scripts already gather and pass this in.
    Confirmed 2026-09-18 it was accepted but never read (dead parameter,
    every claim extracted identically with or without it); fixed the same
    day by rendering the size/plausibility-relevant fields (see
    _context_block()) into a CONTEXT line in the prompt, explicitly marked
    "not evidence -- background only" so the model can't cite it as a
    source_tag or treat it as a claim.

    objection: optional {"flagged_factor": str, "objection_text": str,
    "corrected_excerpt": Optional[str]} -- the Phase 4 critic-panel retry
    channel (see estimate_verifier.py). When set, a "REVIEWER OBJECTION"
    section is injected into the prompt asking the model to specifically
    re-examine that one factor against the reviewer's stated concern. This
    is the explicit correction channel -- NOT a signals-dict piggyback,
    which would fabricate a citable source_tag that was never a real
    gathered signal. corrected_excerpt (added 2026-09-18): an optional
    verbatim passage the evidence_support critic quoted from the SAME
    signal text (critic_panel.py now shows it the full, untruncated
    signal specifically so it can quote real text instead of guessing) --
    when present, the model is told to verify the quote against the
    original text itself before trusting it, never to accept it blindly.
    None (the default) is a no-op -- identical prompt/behavior to every
    existing caller.

    model: optional override, passed straight to zen_client.call_with_prompt.
    None (default) uses zen_client's own DEFAULT_MODEL (opencode.ai). Pass
    e.g. "gpt-oss:120b-cloud" to route through local Ollama's cloud backend
    instead (added 2026-08-18: opencode.ai's free tier was intermittently
    429-throttling this session; Ollama's cloud models run on a separate
    backend/quota entirely). Every existing caller is unaffected."""
    from zen_client import call_with_prompt, DEFAULT_MODEL

    if not signals:
        log.info("[%s/%s] no signals gathered -- skipping extraction", company, pillar)
        return []

    objection_block = ""
    if objection:
        flagged_factor = objection.get("flagged_factor", "(unspecified)")
        corrected_excerpt = objection.get("corrected_excerpt")
        corrected_excerpt_block = ""
        if corrected_excerpt:
            corrected_excerpt_block = _CORRECTED_EXCERPT_BLOCK.format(
                flagged_factor=flagged_factor, corrected_excerpt=corrected_excerpt,
            )
        objection_block = _OBJECTION_TEMPLATE.format(
            flagged_factor=flagged_factor,
            objection_text=objection.get("objection_text", "(no detail provided)"),
            corrected_excerpt_block=corrected_excerpt_block,
        )

    prompt = _PROMPT_TEMPLATE.format(
        pillar=pillar,
        topics=_PILLAR_TOPICS[pillar],
        company=company,
        context_block=_context_block(metadata),
        signals_block=_signals_block(signals),
        objection_block=objection_block,
        factor_list=_factor_list_block(pillar),
    )

    log.info("[%s/%s] calling LLM (%d signals, model=%s)...", company, pillar, len(signals),
             model or DEFAULT_MODEL)
    resp = call_with_prompt(
        prompt, model=model or DEFAULT_MODEL, max_tokens=2500, timeout=120,
        system="You are an ESG evidence tagger. After reasoning, you MUST end with a single "
               "valid JSON object of the exact shape requested, containing only claims genuinely "
               "supported by the signals shown to you.",
    )
    if not resp.get("ok"):
        log.warning("[%s/%s] LLM call failed: %s", company, pillar, resp.get("error"))
        return []

    parsed = extract_json_object(resp.get("raw", "")) or extract_json_object(resp.get("reasoning", ""))
    if not parsed or not isinstance(parsed.get("claims"), list):
        log.warning("[%s/%s] failed to parse claims JSON", company, pillar)
        return []

    valid_tags = set(signals.keys())
    claims: list[ExtractedClaim] = []
    for raw_claim in parsed["claims"]:
        if not isinstance(raw_claim, dict):
            continue
        factor_key = raw_claim.get("factor")
        factor = get_factor(factor_key) if factor_key else None
        if not factor or factor.pillar != pillar:
            log.warning("[%s/%s] dropping claim -- unknown/mismatched factor %r", company, pillar, factor_key)
            continue

        claim = _build_claim(raw_claim, factor, valid_tags, company, pillar)
        if claim is not None:
            claims.append(claim)

    log.info("[%s/%s] extracted %d valid claims", company, pillar, len(claims))
    return claims


def _build_claim(raw_claim: dict, factor, valid_tags: set, company: str,
                 pillar: str) -> Optional[ExtractedClaim]:
    """Validate and coerce one raw LLM claim dict into an ExtractedClaim.

    Shared by the per-pillar and merged extraction paths so the two cannot drift
    apart in how they sanitise model output -- the source_tag check in particular
    is the anti-hallucination guard, and having two copies of it would mean one
    could silently lose it.
    """
    source_tag = raw_claim.get("source_tag")
    if source_tag not in valid_tags:
        log.warning("[%s/%s] dropping claim for %s -- hallucinated source_tag %r",
                    company, pillar, factor.key, source_tag)
        return None

    try:
        polarity = int(raw_claim.get("polarity", 0))
    except (TypeError, ValueError):
        polarity = 0
    polarity = polarity if polarity in (-1, 0, 1) else 0

    try:
        strength = max(0.0, min(1.0, float(raw_claim.get("strength", 0.5))))
    except (TypeError, ValueError):
        strength = 0.5
    try:
        confidence = max(0.0, min(1.0, float(raw_claim.get("confidence", 0.4))))
    except (TypeError, ValueError):
        confidence = 0.4

    value = None
    if raw_claim.get("value") is not None:
        value = _coerce_unit_value(raw_claim.get("value"), raw_claim.get("unit"))

    return ExtractedClaim(
        factor=factor.key, pillar=pillar, polarity=polarity, strength=strength,
        confidence=confidence, value=value, source_tag=source_tag,
        reasoning=str(raw_claim.get("reasoning", ""))[:300], method="extracted",
    )


_MERGED_PROMPT_TEMPLATE = """You are an evidence tagger reviewing signals about a company \
across ALL THREE ESG pillars. You are NOT a scorer: you never invent facts, never convert \
a value into a 0-100 score, and never guess a factor's presence without a specific piece \
of supporting text.

COMPANY: {company}
{context_block}
SIGNALS (each tagged with its source):
{signals_block}

STEP 1 -- RELEVANCE CHECK: for each signal above, decide whether it actually discusses \
something relevant to THIS company. Some signals may be off-topic, generic, or about an \
unrelated company/subject that happened to match a search query -- extract NOTHING from those.

STEP 2 -- EXTRACT CLAIMS for every pillar the evidence supports. Work through the pillars \
in order (E, then S, then G) and consider each independently: evidence that yields nothing \
for one pillar may still yield a claim for another. Extract strictly from this closed \
factor list (do not invent new factor names):

=== E -- {topics_e} ===
{factors_e}

=== S -- {topics_s} ===
{factors_s}

=== G -- {topics_g} ===
{factors_g}

Rules:
- Only emit a claim if a signal ABOVE actually supports it -- cite that signal's exact \
  source tag in "source_tag".
- "pillar" MUST be exactly "E", "S" or "G" and MUST match the section the factor came from.
- If a factor has a stated numeric value in the text (e.g. "1.2 million tCO2e"), put the \
  raw number in "value" and the unit exactly as written in "unit" -- do NOT normalise, \
  rescale, or convert it to a score yourself.
- "polarity": -1 if the claim is negative for the company (e.g. a controversy, a fine), \
  +1 if positive (e.g. a pledge, a certification), 0 if it's a neutral disclosed value.
- "strength" (0-1): how strong/severe/credible the claim is. "confidence" (0-1): how \
  certain you are this claim is accurate given the source text.
- No supporting text for a factor -> omit it entirely. Do not pad the list. Emitting \
  nothing for a pillar is a correct answer when the evidence does not support it.

Respond with ONLY this JSON object (no markdown fences), after your reasoning:
{{"claims": [{{"pillar": "<E|S|G>", "factor": "<exact key from that pillar's list>", \
"polarity": <-1|0|1>, "strength": <0-1>, "confidence": <0-1>, "value": <number|null>, \
"unit": "<string|null>", "source_tag": "<exact tag from signals above>", \
"reasoning": "<one short sentence>"}}, ...]}}"""

# One merged call instead of three per-pillar calls. The three prompts sent the
# IDENTICAL signals block (_signals_block does no pillar filtering) and differed
# only in which factor list they asked for, so we were paying to re-send the same
# evidence three times: measured at ~4 LLM calls/company, this is 3 of them.
#
# Off by default. The merged prompt carries all three factor lists at once, which
# is a real precision risk -- a longer closed list gives the model more chances to
# mis-assign a factor to the wrong pillar. Enable only after comparing claim yield
# against the per-pillar path on the SAME frozen corpus; the env flag exists so
# that comparison can be run without editing code.
_MERGE_EXTRACTORS = os.getenv("ESG_MERGE_EXTRACTORS", "0") == "1"


def extract_all_claims_merged(company: str, signals: dict[str, str],
                              metadata: Optional[dict] = None) -> list[ExtractedClaim]:
    """All three pillars in ONE LLM call. Same contract as extract_all_claims."""
    from zen_client import call_with_prompt

    if not signals:
        log.info("[%s] no signals gathered -- skipping extraction", company)
        return []

    prompt = _MERGED_PROMPT_TEMPLATE.format(
        company=company,
        context_block=_context_block(metadata),
        signals_block=_signals_block(signals),
        topics_e=_PILLAR_TOPICS["E"], factors_e=_factor_list_block("E"),
        topics_s=_PILLAR_TOPICS["S"], factors_s=_factor_list_block("S"),
        topics_g=_PILLAR_TOPICS["G"], factors_g=_factor_list_block("G"),
    )

    log.info("[%s] calling LLM once for all pillars (%d signals)...", company, len(signals))
    resp = call_with_prompt(
        prompt, max_tokens=4000, timeout=180,
        system="You are an ESG evidence tagger. After reasoning, you MUST end with a single "
               "valid JSON object of the exact shape requested, containing only claims genuinely "
               "supported by the signals shown to you.",
    )
    if not resp.get("ok"):
        log.warning("[%s] merged LLM call failed: %s", company, resp.get("error"))
        return []

    parsed = extract_json_object(resp.get("raw", "")) or extract_json_object(resp.get("reasoning", ""))
    if not parsed or not isinstance(parsed.get("claims"), list):
        log.warning("[%s] failed to parse merged claims JSON", company)
        return []

    valid_tags = set(signals.keys())
    claims: list[ExtractedClaim] = []
    for raw_claim in parsed["claims"]:
        if not isinstance(raw_claim, dict):
            continue
        factor_key = raw_claim.get("factor")
        factor = get_factor(factor_key) if factor_key else None
        if not factor:
            log.warning("[%s] dropping claim -- unknown factor %r", company, factor_key)
            continue
        # Trust the FACTOR's registered pillar over the model's "pillar" field.
        # The merged prompt shows all three lists at once, so mis-assignment is
        # the specific new failure mode this path introduces; the factor registry
        # is authoritative and makes the mistake harmless rather than silent.
        stated = raw_claim.get("pillar")
        if stated and stated != factor.pillar:
            log.info("[%s] claim %r labelled %s but factor is %s -- using %s",
                     company, factor_key, stated, factor.pillar, factor.pillar)
        claim = _build_claim(raw_claim, factor, valid_tags, company, factor.pillar)
        if claim is not None:
            claims.append(claim)

    by_pillar: dict[str, int] = {}
    for c in claims:
        by_pillar[c.pillar] = by_pillar.get(c.pillar, 0) + 1
    log.info("[%s] merged extraction -> %d claims %s", company, len(claims), by_pillar)
    return claims


def extract_all_claims(company: str, signals: dict[str, str], metadata: Optional[dict] = None,
                        model: Optional[str] = None) -> list[ExtractedClaim]:
    """Pure: no DB access. Runs the three pillar extractions concurrently.

    model: optional override forwarded to extract_pillar_claims (see its
    docstring). None (default) preserves today's exact behavior."""
    if _MERGE_EXTRACTORS:
        return extract_all_claims_merged(company, signals, metadata)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(extract_pillar_claims, p, company, signals, metadata, model=model): p
                   for p in ("E", "S", "G")}
        results: list[ExtractedClaim] = []
        for fut in futures:
            try:
                results.extend(fut.result())
            except (RateLimitTripped, OllamaRateLimitTripped):
                raise   # never swallow the abort signal -- must stop the run
            except Exception as exc:
                log.warning("[%s/%s] extraction raised: %s", company, futures[fut], exc)
    return results


async def persist_claims(company_id, claims: list[ExtractedClaim], produced_by: str = "pillar_extractor_v1") -> int:
    """The only DB writer. Resolves source_tag -> source_signal_id by re-querying
    company_esg_signals (collector entry points don't return row ids)."""
    import os
    import asyncpg

    db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set")

    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT id, source FROM company_esg_signals WHERE company_id = $1", str(company_id),
        )
        tag_to_id = {r["source"]: r["id"] for r in rows}

        written = 0
        async with conn.transaction():
            # No unique constraint on this table to ON CONFLICT against, so a
            # rerun for the same company would otherwise just append on top of
            # every prior run's claims forever. Clear this producer's own rows
            # for this company first -- scoped to produced_by so it never
            # touches rows written by a different producer (e.g.
            # ratio_estimator.save_ratio_estimates()).
            await conn.execute(
                "DELETE FROM company_evidence_claims WHERE company_id = $1 AND produced_by = $2",
                str(company_id), produced_by,
            )
            for c in claims:
                source_signal_id = tag_to_id.get(c.source_tag)
                source_note = None if source_signal_id else f"tag:{c.source_tag} (not found in company_esg_signals)"
                try:
                    await conn.execute(
                        """
                        INSERT INTO company_evidence_claims
                            (company_id, pillar, factor, polarity, strength, confidence,
                             value, reasoning, source_signal_id, source_note, produced_by, method)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        """,
                        str(company_id), c.pillar, c.factor, c.polarity, c.strength, c.confidence,
                        c.value, c.reasoning, source_signal_id, source_note, produced_by, c.method,
                    )
                    written += 1
                except asyncpg.exceptions.ForeignKeyViolationError as exc:
                    # company_id doesn't reference a real, committed companies row
                    # (e.g. threaded through a pipeline stage ahead of its own
                    # insert committing). Fail loudly with context rather than
                    # letting a bare asyncpg error propagate from inside the loop
                    # and abort a whole batch over one bad id -- callers running
                    # a batch (calibration harness, graph fan-out) can catch this
                    # specific error and skip just this company.
                    raise RuntimeError(
                        f"persist_claims: company_id={company_id} does not exist in companies "
                        f"table (FK violation on claim factor={c.factor!r})"
                    ) from exc
    finally:
        await conn.close()

    log.info("persisted %d claims for company_id=%s", written, company_id)
    return written


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Pillar Extractors — LLM evidence tagger (dry run, no DB)")
    ap.add_argument("mode", choices=["dry"])
    ap.add_argument("company")
    ap.add_argument("--pillar", choices=["E", "S", "G"], default=None)
    ap.add_argument("--industry", default="")
    args = ap.parse_args()

    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.governance_collector import fetch_governance_signals
    from agentic_estimation.layer_1.facility_extractor import fetch_facility_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata

    log_header(log, "Pillar Extractors — dry run", company=args.company, pillar=args.pillar or "all")

    signals = {}
    signals.update(fetch_company_signals(args.company, args.industry))
    signals.update(fetch_governance_signals(args.company))
    signals.update(fetch_facility_signals(args.company, args.industry))
    metadata = get_company_metadata(args.company)

    print(f"\ngathered {len(signals)} signals: {list(signals.keys())}\n")

    pillars = [args.pillar] if args.pillar else ["E", "S", "G"]
    for pillar in pillars:
        claims = extract_pillar_claims(pillar, args.company, signals, metadata)
        print(f"\n=== {pillar}: {len(claims)} claims ===")
        for c in claims:
            print(f"  {c.factor:28s} polarity={c.polarity:+d} strength={c.strength:.2f} "
                  f"confidence={c.confidence:.2f} value={c.value}")
            print(f"    source={c.source_tag}  {c.reasoning}")


if __name__ == "__main__":
    _cli()
