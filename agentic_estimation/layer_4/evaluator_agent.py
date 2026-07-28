"""
evaluator_agent.py — ESG Evaluator Agent (Agent 5)

Fact-checks the three pillar scores from the scoring agent against the raw
signals. If scores are inconsistent with the evidence, the agent corrects them
in the same LLM call. At most ONE correction round is attempted.

Flow:
    evaluate_company_sync(score, signals) → EvaluationResult
        - pass  → scores confirmed, no change
        - fix   → corrected ESGScore + explanation of what changed

    evaluate_company(score, company_id, signals, industry) → async, saves to DB
        - on fix: overwrites the three pillar scores + summary in company_metric_values

CLI:
    python -m agentic_estimation.evaluator_agent dry "Bosch" [--industry ...] [--country ...]
    python -m agentic_estimation.evaluator_agent eval "Bosch" [--industry ...] [--country ...]
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_3.scoring_agent import ESGScore, _clamp, _METRIC_KEYS

log = get_logger("evaluator_agent")

# ── Prompt ─────────────────────────────────────────────────────────────────────

_EVAL_PROMPT = """You are a senior ESG auditor. Your job is to review a set of ESG pillar scores and verify whether they are consistent with the raw evidence provided.

COMPANY: {company}
INDUSTRY: {industry}
COUNTRY: {country}

PROPOSED SCORES:
  Environment (E): {e_score:.1f}/100  — "{e_reasoning}"
  Social (S):      {s_score:.1f}/100  — "{s_reasoning}"
  Governance (G):  {g_score:.1f}/100  — "{g_reasoning}"

RAW SIGNAL EVIDENCE:
{signals_block}

TASK:
1. For each pillar, decide if the proposed score is CONSISTENT with the evidence.
   - A score is CONSISTENT if the evidence plausibly supports it (±10 points is fine).
   - A score is INCONSISTENT if the evidence clearly contradicts it or the score is implausible given the evidence strength.
2. If ALL three scores are consistent → verdict = "pass".
3. If ANY score is inconsistent → verdict = "fix" and provide corrected scores.
   - Only correct a pillar if you are confident the evidence justifies a different score.
   - Keep changes conservative; do not overcorrect.

Return ONLY this JSON (no markdown, no code fences):
{{
  "verdict": "pass" or "fix",
  "issues": ["<brief description of each inconsistency found, or empty list>"],
  "e_score": <float 0-100>,
  "s_score": <float 0-100>,
  "g_score": <float 0-100>,
  "e_reasoning": "<updated reasoning or same as before>",
  "s_reasoning": "<updated reasoning or same as before>",
  "g_reasoning": "<updated reasoning or same as before>",
  "evaluator_note": "<one sentence explaining what you changed and why, or 'No changes needed'>"
}}"""


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    verdict: str                    # "pass" or "fix"
    original_score: ESGScore
    final_score: ESGScore           # same as original on pass
    issues: list = field(default_factory=list)
    evaluator_note: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_signals_block(signals: dict[str, str], max_chars_per_source: int = 500) -> str:
    if not signals:
        return "(no signals available)"
    parts = []
    for source, text in signals.items():
        truncated = text[:max_chars_per_source].strip()
        if len(text) > max_chars_per_source:
            truncated += "..."
        parts.append(f"[{source.upper()}]\n{truncated}")
    return "\n\n".join(parts)


def _parse_eval_response(raw: str) -> Optional[dict]:
    REQUIRED = {"verdict", "e_score", "s_score", "g_score",
                "e_reasoning", "s_reasoning", "g_reasoning", "evaluator_note"}

    text = re.sub(r"```(?:json)?|```", "", raw, flags=re.MULTILINE).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and REQUIRED.issubset(obj.keys()):
            return obj
    except Exception:
        pass

    candidates = list(re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL))
    candidates += list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(candidates):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and REQUIRED.issubset(obj.keys()):
                return obj
        except Exception:
            pass

    return None


# ── Core sync function ─────────────────────────────────────────────────────────

def evaluate_company_sync(
    score: ESGScore,
    signals: dict[str, str],
    industry: str = "",
) -> Optional[EvaluationResult]:
    """
    Evaluate ESG scores against signals. Returns EvaluationResult or None on LLM failure.
    No DB interaction.
    """
    from zen_client import call_with_prompt

    log_header(log, "Evaluator Agent",
               company=score.company,
               proposed=f"E={score.e_score:.0f} S={score.s_score:.0f} G={score.g_score:.0f}",
               signals=len(signals))

    signals_block = _build_signals_block(signals)
    prompt = _EVAL_PROMPT.format(
        company=score.company,
        industry=industry or "Not specified",
        country=score.country or "Unknown",
        e_score=score.e_score,
        s_score=score.s_score,
        g_score=score.g_score,
        e_reasoning=score.e_reasoning,
        s_reasoning=score.s_reasoning,
        g_reasoning=score.g_reasoning,
        signals_block=signals_block,
    )

    log.info("[%s] calling LLM for evaluation...", score.company)
    resp = call_with_prompt(
        prompt,
        max_tokens=2000,
        timeout=180,
        system=(
            "You are a senior ESG auditor. Review the proposed scores critically. "
            "Return ONLY valid JSON with the required keys."
        ),
    )

    if not resp.get("ok"):
        log.error("[%s] LLM call failed: %s", score.company, resp.get("error"))
        return None

    raw = resp["raw"]
    reasoning_raw = resp.get("reasoning", "")
    log.info("[%s] content (%d chars), reasoning (%d chars)", score.company, len(raw), len(reasoning_raw))

    parsed = _parse_eval_response(raw) or _parse_eval_response(reasoning_raw)
    if not parsed:
        log.error("[%s] failed to parse evaluator response: %s", score.company, raw[:300])
        return None

    verdict = str(parsed.get("verdict", "pass")).lower().strip()
    issues = parsed.get("issues", [])
    evaluator_note = str(parsed.get("evaluator_note", ""))

    if verdict == "pass":
        log.info("[%s] verdict=PASS — %s", score.company, evaluator_note)
        return EvaluationResult(
            verdict="pass",
            original_score=score,
            final_score=score,
            issues=issues,
            evaluator_note=evaluator_note,
        )

    # verdict == "fix"
    try:
        corrected = ESGScore(
            company=score.company,
            e_score=_clamp(parsed["e_score"]),
            s_score=_clamp(parsed["s_score"]),
            g_score=_clamp(parsed["g_score"]),
            e_reasoning=str(parsed.get("e_reasoning", score.e_reasoning)),
            s_reasoning=str(parsed.get("s_reasoning", score.s_reasoning)),
            g_reasoning=str(parsed.get("g_reasoning", score.g_reasoning)),
            country=score.country,
            signals_used=score.signals_used,
        )
    except (KeyError, TypeError, ValueError) as exc:
        log.error("[%s] corrected score fields invalid: %s", score.company, exc)
        return None

    log.info(
        "[%s] verdict=FIX — E: %.1f→%.1f  S: %.1f→%.1f  G: %.1f→%.1f",
        score.company,
        score.e_score, corrected.e_score,
        score.s_score, corrected.s_score,
        score.g_score, corrected.g_score,
    )
    if issues:
        for issue in issues:
            log.info("[%s] issue: %s", score.company, issue)
    log.info("[%s] evaluator note: %s", score.company, evaluator_note)

    return EvaluationResult(
        verdict="fix",
        original_score=score,
        final_score=corrected,
        issues=issues,
        evaluator_note=evaluator_note,
    )


# ── Async DB write ─────────────────────────────────────────────────────────────

async def evaluate_company(
    score: ESGScore,
    company_id: UUID,
    signals: dict[str, str],
    industry: str = "",
) -> Optional[EvaluationResult]:
    """
    Evaluate scores and, if a fix is needed, overwrite them in company_metric_values.
    Returns EvaluationResult or None on failure.
    """
    result = evaluate_company_sync(score, signals, industry=industry)
    if result is None:
        return None

    if result.verdict == "pass":
        log.info("[%s] no DB update needed (pass)", score.company)
        return result

    # Save corrected scores
    db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set")

    conn = await asyncpg.connect(db_url)
    try:
        keys = list(_METRIC_KEYS.values())
        rows = await conn.fetch(
            "SELECT id, key FROM esg_metric_definitions WHERE key = ANY($1::text[])", keys
        )
        metric_ids = {row["key"]: UUID(str(row["id"])) for row in rows}
        missing = set(keys) - set(metric_ids.keys())
        if missing:
            raise RuntimeError(f"Missing metric definitions: {missing}")

        source = "agentic_evaluator_v1"
        pillar_map = [
            ("E", result.final_score.e_score, result.final_score.e_reasoning),
            ("S", result.final_score.s_score, result.final_score.s_reasoning),
            ("G", result.final_score.g_score, result.final_score.g_reasoning),
        ]
        for pillar, final_score, reasoning in pillar_map:
            metric_key = _METRIC_KEYS[pillar]
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
                str(metric_ids[metric_key]),
                final_score,
                f"{final_score:.1f}/100",
                reasoning,
                source,
            )

        log.info(
            "[%s] corrected scores saved to DB (E=%.1f S=%.1f G=%.1f)",
            score.company,
            result.final_score.e_score,
            result.final_score.s_score,
            result.final_score.g_score,
        )
    finally:
        await conn.close()

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _print_result(result: EvaluationResult) -> None:
    orig = result.original_score
    final = result.final_score
    print(f"\nCompany  : {orig.company}")
    print(f"Verdict  : {result.verdict.upper()}")
    if result.issues:
        for issue in result.issues:
            print(f"  Issue  : {issue}")
    print(f"Note     : {result.evaluator_note}")
    print(f"\n{'Pillar':<6}  {'Original':>10}  {'Final':>10}")
    print(f"{'E':<6}  {orig.e_score:>9.1f}  {final.e_score:>9.1f}")
    print(f"{'S':<6}  {orig.s_score:>9.1f}  {final.s_score:>9.1f}")
    print(f"{'G':<6}  {orig.g_score:>9.1f}  {final.g_score:>9.1f}")
    if result.verdict == "fix":
        print(f"\nUpdated reasonings:")
        print(f"  E: {final.e_reasoning}")
        print(f"  S: {final.s_reasoning}")
        print(f"  G: {final.g_reasoning}")


def _cli() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m agentic_estimation.evaluator_agent dry  <company> [--industry ...] [--country ...]")
        print("  python -m agentic_estimation.evaluator_agent eval <company> [--industry ...] [--country ...]")
        sys.exit(1)

    mode = sys.argv[1]
    company = sys.argv[2]
    industry, country = "", None
    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--industry" and i + 1 < len(args):
            industry = args[i + 1]; i += 2
        elif args[i] == "--country" and i + 1 < len(args):
            country = args[i + 1]; i += 2
        else:
            i += 1

    # Gather signals + score first
    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_3.scoring_agent import score_company_sync

    log.info("Gathering signals for %s...", company)
    signals = fetch_company_signals(company, industry)

    log.info("Scoring %s...", company)
    score = score_company_sync(company, industry=industry, country=country, signals=signals)
    if score is None:
        print("ERROR: scoring failed")
        sys.exit(1)

    if mode == "dry":
        result = evaluate_company_sync(score, signals, industry=industry)
        if result:
            _print_result(result)
        else:
            print("ERROR: evaluation failed")
            sys.exit(1)

    elif mode == "eval":
        async def _run():
            db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
            if not db_url:
                raise RuntimeError("ASYNC_DB_URL not set")
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
            finally:
                await conn.close()

            result = await evaluate_company(
                score, company_id, signals, industry=industry
            )
            if result:
                _print_result(result)
                if result.verdict == "fix":
                    print("\nCorrected scores saved to database.")
            else:
                print("ERROR: evaluation failed")
                sys.exit(1)

        asyncio.run(_run())

    else:
        print(f"Unknown mode: {mode}. Use 'dry' or 'eval'.")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
