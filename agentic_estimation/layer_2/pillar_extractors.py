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

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.shared.llm_json import extract_json_object
from agentic_estimation.layer_2.factor_registry import factors_for_pillar, get_factor
from agentic_estimation.shared.claim_types import ExtractedClaim

log = get_logger("pillar_extractors")

_MAX_CHARS_PER_SIGNAL = 1500
_MAX_TOTAL_CHARS = 12000

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
"""

_PROMPT_TEMPLATE = """You are an evidence tagger reviewing signals about a company for the {pillar} \
(Environmental/Social/Governance -- specifically {topics}) pillar. You are NOT a scorer: \
you never invent facts, never convert a value into a 0-100 score, and never guess \
a factor's presence without a specific piece of supporting text.

COMPANY: {company}

SIGNALS (each tagged with its source):
{signals_block}
{objection_block}
STEP 1 -- RELEVANCE CHECK: for each signal above, decide whether it actually discusses \
something relevant to {topics} for THIS company. Some signals may be off-topic, generic, \
or about an unrelated company/subject that happened to match a search query -- extract \
NOTHING from those. Only proceed to Step 2 for signals that are genuinely on-topic.

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


def _factor_list_block(pillar: str) -> str:
    lines = []
    for f in factors_for_pillar(pillar):
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
) -> list[ExtractedClaim]:
    """Pure: no DB access. One LLM call for one pillar.

    objection: optional {"flagged_factor": str, "objection_text": str} --
    the Phase 4 critic-panel retry channel (see estimate_verifier.py). When
    set, a "REVIEWER OBJECTION" section is injected into the prompt asking
    the model to specifically re-examine that one factor against the
    reviewer's stated concern. This is the explicit correction channel --
    NOT a signals-dict piggyback, which would fabricate a citable
    source_tag that was never a real gathered signal. None (the default)
    is a no-op -- identical prompt/behavior to every existing caller."""
    from zen_client import call_with_prompt

    if not signals:
        log.info("[%s/%s] no signals gathered -- skipping extraction", company, pillar)
        return []

    objection_block = ""
    if objection:
        objection_block = _OBJECTION_TEMPLATE.format(
            flagged_factor=objection.get("flagged_factor", "(unspecified)"),
            objection_text=objection.get("objection_text", "(no detail provided)"),
        )

    prompt = _PROMPT_TEMPLATE.format(
        pillar=pillar,
        topics=_PILLAR_TOPICS[pillar],
        company=company,
        signals_block=_signals_block(signals),
        objection_block=objection_block,
        factor_list=_factor_list_block(pillar),
    )

    log.info("[%s/%s] calling LLM (%d signals)...", company, pillar, len(signals))
    resp = call_with_prompt(
        prompt, max_tokens=2500, timeout=120,
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

        source_tag = raw_claim.get("source_tag")
        if source_tag not in valid_tags:
            log.warning("[%s/%s] dropping claim for %s -- hallucinated source_tag %r",
                        company, pillar, factor_key, source_tag)
            continue

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

        claims.append(ExtractedClaim(
            factor=factor_key, pillar=pillar, polarity=polarity, strength=strength,
            confidence=confidence, value=value, source_tag=source_tag,
            reasoning=str(raw_claim.get("reasoning", ""))[:300], method="extracted",
        ))

    log.info("[%s/%s] extracted %d valid claims", company, pillar, len(claims))
    return claims


def extract_all_claims(company: str, signals: dict[str, str], metadata: Optional[dict] = None) -> list[ExtractedClaim]:
    """Pure: no DB access. Runs the three pillar extractions concurrently."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(extract_pillar_claims, p, company, signals, metadata): p for p in ("E", "S", "G")}
        results: list[ExtractedClaim] = []
        for fut in futures:
            try:
                results.extend(fut.result())
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
