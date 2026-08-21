"""
scoring_agent.py — ESG Scoring Agent (Agent 3)

Takes gathered signals (from signal_agent) + country baseline (from
country_baseline_agent) and asks DeepSeek-v4-flash to produce E/S/G
pillar scores (0–100) with short reasoning for each pillar.

Scores are saved to company_metric_values using the three metric
definitions: esg_e_score, esg_s_score, esg_g_score.

Usage:
    # Score and persist
    result = await score_company(company_name="Bosch", company_id=<uuid>)

    # Score only (no DB write)
    result = score_company_sync("Bosch", signals=prefetched_dict)

CLI:
    python scoring_agent.py score "Bosch" [--industry "Industrial Machinery"] [--country Germany]
    python scoring_agent.py dry "Bosch"        # score only, print result, no DB write
"""

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
log = get_logger("scoring_agent")

# ── Metric definition keys (must exist in esg_metric_definitions table) ───────
_METRIC_KEYS = {
    "E": "esg_e_score",
    "S": "esg_s_score",
    "G": "esg_g_score",
}

# ── Scoring prompt ─────────────────────────────────────────────────────────────
_SCORE_PROMPT = """You are an expert ESG analyst. Your task is to score a company on Environmental (E), Social (S), and Governance (G) pillars based on the evidence below.

Scoring rules:
- Each pillar score: 0–100 (higher = better ESG performance)
- 50 is the country baseline (average for companies in that country)
- Score above 50 means the company outperforms its country peer group
- Score below 50 means it underperforms
- Use ONLY the evidence provided — do not invent facts
- If evidence for a pillar is thin or absent, score at the country baseline (50) — do not penalize for absence of evidence, only for evidence of actual problems
- Use the company metadata to calibrate expectations: a large listed multinational is held to a higher standard than a small private firm

COMPANY: {company}
INDUSTRY: {industry}
COUNTRY: {country}

COMPANY METADATA:
{metadata_block}

COUNTRY ESG BASELINE (World Bank, 0–100 scale):
  Environment score: {country_e:.1f}
  Social score:      {country_s:.1f}
  Governance score:  {country_g:.1f}
  Data year:         {country_year}

SIGNAL EVIDENCE:
{signals_block}

Think through each pillar, then end your response with this JSON object (no markdown, no code fences):
{{
  "e_score": <float 0-100>,
  "s_score": <float 0-100>,
  "g_score": <float 0-100>,
  "e_reasoning": "<1-2 sentences>",
  "s_reasoning": "<1-2 sentences>",
  "g_reasoning": "<1-2 sentences>"
}}"""


@dataclass
class ESGScore:
    company: str
    e_score: float
    s_score: float
    g_score: float
    e_reasoning: str
    s_reasoning: str
    g_reasoning: str
    country: Optional[str] = None
    signals_used: int = 0


def _build_signals_block(signals: dict[str, str], max_chars_per_source: int = 4000) -> str:
    """Format signals dict into a readable block for the LLM prompt."""
    if not signals:
        return "(no signals gathered)"
    parts = []
    for source, text in signals.items():
        truncated = text[:max_chars_per_source].strip()
        if len(text) > max_chars_per_source:
            truncated += "..."
        parts.append(f"[{source.upper()}]\n{truncated}")
    return "\n\n".join(parts)


def _parse_llm_response(raw: str) -> Optional[dict]:
    """
    Extract the ESG score JSON object from LLM output.
    DeepSeek is a reasoning model — it emits chain-of-thought prose first,
    then the final JSON answer. We scan for the LAST valid JSON object that
    contains all six required keys.
    """
    if not raw:
        return None

    REQUIRED = {"e_score", "s_score", "g_score", "e_reasoning", "s_reasoning", "g_reasoning"}

    # Strip code fences
    text = re.sub(r"```(?:json)?|```", "", raw, flags=re.MULTILINE).strip()

    # Try direct parse first (clean output)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and REQUIRED.issubset(obj.keys()):
            return obj
    except Exception:
        pass

    # Find all JSON object candidates, prefer the last one (after reasoning)
    candidates = list(re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL))
    # Also try larger nested objects
    candidates += list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))

    for m in reversed(candidates):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and REQUIRED.issubset(obj.keys()):
                return obj
        except Exception:
            pass

    return None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def score_company_sync(
    company: str,
    industry: str = "",
    country: Optional[str] = None,
    signals: Optional[dict[str, str]] = None,
    metadata: Optional[dict] = None,
    model: Optional[str] = None,
) -> Optional[ESGScore]:
    """
    Compute ESG scores synchronously (no DB write).

    If signals is None, fetches them via signal_agent.fetch_company_signals.
    If metadata is None, fetches it via company_metadata.get_company_metadata.
    If country is None, tries metadata first, then signals['wikipedia'], then
    falls back to no-baseline mode (uses 50/50/50 as neutral baseline).

    model: optional override forwarded to zen_client.call_with_prompt. None
    (default) preserves today's exact behavior (opencode.ai's DEFAULT_MODEL).
    """
    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.country_baseline_agent import (
        get_country_baseline_with_fallback,
        extract_country_from_wikipedia,
    )
    from agentic_estimation.layer_1.company_metadata import get_company_metadata, format_for_prompt
    from zen_client import call_with_prompt, DEFAULT_MODEL

    log_header(log, "Scoring Agent", company=company, industry=industry or "N/A", country=country or "auto-detect")

    # 1. Gather signals if not provided
    if signals is None:
        log.info("[%s] gathering signals...", company)
        signals = fetch_company_signals(company, industry)
    log.info("[%s] signals available: %s", company, list(signals.keys()))

    # 2. Fetch company metadata if not provided
    if metadata is None:
        log.info("[%s] fetching company metadata...", company)
        metadata = get_company_metadata(company)
        log.info("[%s] metadata source: %s", company, metadata.get("source") or "none")

    # 3. Resolve country — metadata first, then Wikipedia signal, then caller-supplied
    resolved_country = country
    if not resolved_country and metadata.get("country"):
        resolved_country = metadata["country"]
        log.info("[%s] country from metadata: %s", company, resolved_country)
    if not resolved_country and "wikipedia" in signals:
        resolved_country = extract_country_from_wikipedia(signals["wikipedia"])
        if resolved_country:
            log.info("[%s] country from Wikipedia signal: %s", company, resolved_country)

    # 4. Get country baseline -- with_fallback never returns None: an
    # unresolved/aliased/regional country still gets a real baseline (alias
    # lookup -> regional parent -> global average) instead of silently
    # dropping to a flat 50/50/50 "no country" neutral (see
    # country_baseline_agent.get_country_baseline_with_fallback's docstring).
    baseline = None
    baseline_source = "no_country"
    if resolved_country:
        baseline, baseline_source = get_country_baseline_with_fallback(resolved_country)
        log.info("[%s] country baseline: E=%.1f S=%.1f G=%.1f (%s, %s, %d)",
                 company, baseline.e_score, baseline.s_score, baseline.g_score,
                 resolved_country, baseline_source, baseline.year)

    country_e = baseline.e_score if baseline else 50.0
    country_s = baseline.s_score if baseline else 50.0
    country_g = baseline.g_score if baseline else 50.0
    country_year = baseline.year if baseline else "N/A"
    country_label = resolved_country or "Unknown"

    # 5. Build and send prompt
    signals_block = _build_signals_block(signals)
    metadata_block = format_for_prompt(metadata)
    prompt = _SCORE_PROMPT.format(
        company=company,
        industry=industry or "Not specified",
        country=country_label,
        metadata_block=metadata_block,
        country_e=country_e,
        country_s=country_s,
        country_g=country_g,
        country_year=country_year,
        signals_block=signals_block,
    )

    log.info("[%s] calling LLM for scoring (model=%s)...", company, model or "default")
    resp = call_with_prompt(
        prompt,
        model=model or DEFAULT_MODEL,
        max_tokens=4000,
        timeout=240,
        system="You are an expert ESG analyst. After your reasoning, you MUST end your response with a valid JSON object containing e_score, s_score, g_score, e_reasoning, s_reasoning, g_reasoning.",
    )

    if not resp.get("ok"):
        log.error("[%s] LLM call failed: %s", company, resp.get("error"))
        return None

    raw = resp["raw"]
    reasoning_raw = resp.get("reasoning", "")
    log.info("[%s] content (%d chars), reasoning (%d chars)", company, len(raw), len(reasoning_raw))
    if raw:
        log.info("[%s] content tail: ...%s", company, raw[-400:].replace("```", ""))

    # Try content first (final answer), then reasoning_content (think block contains JSON too sometimes)
    parsed = _parse_llm_response(raw) or _parse_llm_response(reasoning_raw)
    if not parsed:
        log.error("[%s] failed to parse — content: %s", company, raw[:300])
        log.error("[%s] failed to parse — reasoning tail: %s", company, reasoning_raw[-300:])
        return None

    try:
        return ESGScore(
            company=company,
            e_score=_clamp(parsed["e_score"]),
            s_score=_clamp(parsed["s_score"]),
            g_score=_clamp(parsed["g_score"]),
            e_reasoning=str(parsed.get("e_reasoning", "")),
            s_reasoning=str(parsed.get("s_reasoning", "")),
            g_reasoning=str(parsed.get("g_reasoning", "")),
            country=country_label,
            signals_used=len(signals),
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.error("[%s] invalid score fields: %s — raw: %s", company, exc, parsed)
        return None


async def _get_metric_ids(conn: asyncpg.Connection) -> dict[str, UUID]:
    """Fetch UUIDs for the three ESG pillar metric definitions."""
    keys = list(_METRIC_KEYS.values())
    rows = await conn.fetch(
        "SELECT id, key FROM esg_metric_definitions WHERE key = ANY($1::text[])",
        keys,
    )
    mapping = {row["key"]: UUID(str(row["id"])) for row in rows}
    missing = set(keys) - set(mapping.keys())
    if missing:
        raise RuntimeError(
            f"Missing metric definitions in DB: {missing}. "
            "Run the SQL seed to insert esg_e_score, esg_s_score, esg_g_score."
        )
    return mapping


async def _upsert_score(
    conn: asyncpg.Connection,
    company_id: UUID,
    metric_id: UUID,
    score: float,
    reasoning: str,
    source: str = "agentic_scoring_v1",
) -> None:
    await conn.execute(
        """
        INSERT INTO company_metric_values
            (company_id, metric_id, numeric_value, value, reasoning, source, reporting_year)
        VALUES ($1, $2, $3, $4, $5, $6, EXTRACT(YEAR FROM now())::int)
        ON CONFLICT (company_id, metric_id, reporting_year, source)
        DO UPDATE SET
            numeric_value = EXCLUDED.numeric_value,
            value         = EXCLUDED.value,
            reasoning     = EXCLUDED.reasoning
        """,
        str(company_id),
        str(metric_id),
        score,
        f"{score:.1f}/100",
        reasoning,
        source,
    )


async def score_company(
    company_name: str,
    company_id: UUID,
    industry: str = "",
    country: Optional[str] = None,
    signals: Optional[dict[str, str]] = None,
    metadata: Optional[dict] = None,
) -> Optional[ESGScore]:
    """
    Score a company and persist the three pillar scores to company_metric_values.

    company_id: UUID of the companies row (required for DB write).
    """
    result = score_company_sync(
        company=company_name,
        industry=industry,
        country=country,
        signals=signals,
        metadata=metadata,
    )
    if result is None:
        return None

    db_url = os.environ.get("ASYNC_DB_URL", "")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set in environment")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    try:
        metric_ids = await _get_metric_ids(conn)
        pillar_map = [
            ("E", result.e_score, result.e_reasoning),
            ("S", result.s_score, result.s_reasoning),
            ("G", result.g_score, result.g_reasoning),
        ]
        for pillar, score, reasoning in pillar_map:
            metric_key = _METRIC_KEYS[pillar]
            await _upsert_score(conn, company_id, metric_ids[metric_key], score, reasoning)
        log.info(
            "[%s] saved E=%.1f S=%.1f G=%.1f to DB",
            company_name, result.e_score, result.s_score, result.g_score,
        )
    finally:
        await conn.close()

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_result(result: ESGScore) -> None:
    print(f"\nCompany : {result.company}")
    print(f"Country : {result.country or 'Unknown'}")
    print(f"Signals : {result.signals_used} sources")
    print(f"\nE score : {result.e_score:.1f}/100")
    print(f"          {result.e_reasoning}")
    print(f"\nS score : {result.s_score:.1f}/100")
    print(f"          {result.s_reasoning}")
    print(f"\nG score : {result.g_score:.1f}/100")
    print(f"          {result.g_reasoning}")


def _cli() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python scoring_agent.py score <company> [--industry <...>] [--country <...>]")
        print("  python scoring_agent.py dry   <company> [--industry <...>] [--country <...>]")
        sys.exit(1)

    mode = sys.argv[1]
    company = sys.argv[2]

    industry = ""
    country = None
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--industry" and i + 1 < len(args):
            industry = args[i + 1]
            i += 2
        elif args[i] == "--country" and i + 1 < len(args):
            country = args[i + 1]
            i += 2
        else:
            i += 1

    if mode == "dry":
        result = score_company_sync(company, industry=industry, country=country)
        if result:
            _print_result(result)
        else:
            print("ERROR: scoring failed")
            sys.exit(1)

    elif mode == "score":
        # Need company_id from DB
        async def _run():
            db_url = os.environ.get("ASYNC_DB_URL", "")
            if not db_url:
                raise RuntimeError("ASYNC_DB_URL not set")
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            conn = await asyncpg.connect(db_url)
            try:
                row = await conn.fetchrow(
                    "SELECT id FROM companies WHERE name ILIKE $1 LIMIT 1",
                    f"%{company}%",
                )
                if not row:
                    print(f"ERROR: company '{company}' not found in DB")
                    sys.exit(1)
                company_id = UUID(str(row["id"]))
                # Get actual name from DB
                name_row = await conn.fetchrow("SELECT name FROM companies WHERE id=$1", str(company_id))
                actual_name = name_row["name"] if name_row else company
            finally:
                await conn.close()

            result = await score_company(
                company_name=actual_name,
                company_id=company_id,
                industry=industry,
                country=country,
            )
            if result:
                _print_result(result)
                print("\nSaved to database.")
            else:
                print("ERROR: scoring failed")
                sys.exit(1)

        asyncio.run(_run())

    else:
        print(f"Unknown mode: {mode}. Use 'score' or 'dry'.")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
