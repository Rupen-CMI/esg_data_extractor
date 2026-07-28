"""
explainability_agent.py — ESG Explainability Agent (Agent 4)

Takes the three pillar scores + per-pillar reasoning from the scoring agent
and writes a single plain-English summary paragraph a non-expert can read.

The summary is saved to company_metric_values as metric key 'esg_summary',
with the paragraph stored in the `reasoning` column and value = 'summary'.

Usage:
    # In-memory (no DB read needed if you pass an ESGScore object)
    result = explain_company_sync(score)

    # Full async with DB write
    result = await explain_company(score, company_id=<uuid>)

CLI:
    python explainability_agent.py explain "Bosch" --industry "Industrial Machinery" --country Germany
    python explainability_agent.py dry     "Bosch" --industry "Industrial Machinery" --country Germany
"""

import asyncio
import json
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
log = get_logger("explainability_agent")

_METRIC_KEY = "esg_summary"

# ── Prompt ────────────────────────────────────────────────────────────────────
_EXPLAIN_PROMPT = """You are an ESG communications expert writing for a business audience — clear, direct, no jargon.

Write a single paragraph (4-6 sentences) summarising the ESG performance of the company below.
The paragraph should:
- Open with the company name and an overall ESG characterisation (strong / average / weak)
- Mention all three pillar scores and what drives them (use the reasoning provided)
- Compare to the country peer group where relevant
- Close with the single biggest ESG risk or strength to watch

COMPANY:  {company}
INDUSTRY: {industry}
COUNTRY:  {country}

SCORES (0-100, 50 = country average):
  Environment : {e_score:.1f}/100 — {e_reasoning}
  Social      : {s_score:.1f}/100 — {s_reasoning}
  Governance  : {g_score:.1f}/100 — {g_reasoning}

Write ONLY the paragraph. No headings, no bullet points, no JSON, no preamble."""


@dataclass
class ESGSummary:
    company: str
    summary: str
    e_score: float
    s_score: float
    g_score: float


def explain_company_sync(
    score,  # ESGScore dataclass from scoring_agent
    industry: str = "",
) -> Optional[ESGSummary]:
    """
    Generate a plain-English ESG summary from an ESGScore object.
    No DB interaction — pure LLM call.
    """
    from zen_client import call_with_prompt

    log_header(log, "Explainability Agent",
               company=score.company,
               country=score.country or "Unknown",
               scores=f"E={score.e_score:.0f} S={score.s_score:.0f} G={score.g_score:.0f}")

    prompt = _EXPLAIN_PROMPT.format(
        company=score.company,
        industry=industry or "Not specified",
        country=score.country or "Unknown",
        e_score=score.e_score,
        s_score=score.s_score,
        g_score=score.g_score,
        e_reasoning=score.e_reasoning,
        s_reasoning=score.s_reasoning,
        g_reasoning=score.g_reasoning,
    )

    log.info("[%s] calling LLM for summary...", score.company)
    resp = call_with_prompt(
        prompt,
        # 400 was too tight -- deepseek-v4-flash-free ignores disable_thinking
        # and burns tokens on reasoning_content before writing the answer,
        # truncating `content` mid-sentence (or to nothing) well short of a
        # 4-6 sentence paragraph. 2000 matches other prose-generating callers
        # (evaluator_agent, metric_estimation_agent) and leaves headroom for
        # the reasoning preamble plus the actual paragraph.
        max_tokens=2000,
        timeout=120,
        system="You are an ESG communications expert. Write clear, concise prose. No JSON, no lists.",
    )

    if not resp.get("ok"):
        log.error("[%s] LLM call failed: %s", score.company, resp.get("error"))
        return None

    summary = (resp["raw"] or "").strip()
    if not summary:
        log.error("[%s] LLM returned empty summary (model=%s)", score.company, resp.get("model_used"))
        return None

    # Strip any accidental JSON or markdown that leaked through
    summary = re.sub(r"^```.*?```$", "", summary, flags=re.DOTALL).strip()
    summary = re.sub(r'^\s*[\[{].*?[}\]]\s*$', "", summary, flags=re.DOTALL).strip()

    log.info("[%s] summary (%d chars): %s", score.company, len(summary), summary[:120] + "...")

    return ESGSummary(
        company=score.company,
        summary=summary,
        e_score=score.e_score,
        s_score=score.s_score,
        g_score=score.g_score,
    )


async def _upsert_summary(
    conn: asyncpg.Connection,
    company_id: UUID,
    metric_id: UUID,
    summary: str,
    source: str = "agentic_explainability_v1",
) -> None:
    await conn.execute(
        """
        INSERT INTO company_metric_values
            (company_id, metric_id, value, reasoning, source, reporting_year)
        VALUES ($1, $2, $3, $4, $5, EXTRACT(YEAR FROM now())::int)
        ON CONFLICT (company_id, metric_id, reporting_year, source)
        DO UPDATE SET
            value     = EXCLUDED.value,
            reasoning = EXCLUDED.reasoning
        """,
        str(company_id),
        str(metric_id),
        "summary",
        summary,
        source,
    )


async def explain_company(
    score,
    company_id: UUID,
    industry: str = "",
) -> Optional[ESGSummary]:
    """
    Generate ESG summary and persist to company_metric_values.
    """
    result = explain_company_sync(score, industry=industry)
    if result is None:
        return None

    db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set")

    conn = await asyncpg.connect(db_url)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM esg_metric_definitions WHERE key = $1",
            _METRIC_KEY,
        )
        if not row:
            raise RuntimeError(
                f"Metric definition '{_METRIC_KEY}' not found in DB. "
                "Insert it via SQL: INSERT INTO esg_metric_definitions ..."
            )
        metric_id = UUID(str(row["id"]))
        await _upsert_summary(conn, company_id, metric_id, result.summary)
        log.info("[%s] summary saved to DB", score.company)
    finally:
        await conn.close()

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python explainability_agent.py explain <company> [--industry <...>] [--country <...>]")
        print("  python explainability_agent.py dry     <company> [--industry <...>] [--country <...>]")
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
    return mode, company, industry, country


def _cli():
    mode, company, industry, country = _parse_args()

    # Both modes need a score first — run scoring agent
    from agentic_estimation.layer_3.scoring_agent import score_company_sync
    log.info("Running scoring agent first...")
    score = score_company_sync(company, industry=industry, country=country)
    if score is None:
        print("ERROR: scoring failed, cannot explain")
        sys.exit(1)

    if mode == "dry":
        result = explain_company_sync(score, industry=industry)
        if result:
            print(f"\nCompany : {result.company}")
            print(f"E={result.e_score:.0f}  S={result.s_score:.0f}  G={result.g_score:.0f}\n")
            print(result.summary)
        else:
            print("ERROR: explainability failed")
            sys.exit(1)

    elif mode == "explain":
        async def _run():
            db_url = os.environ.get("ASYNC_DB_URL", "").replace("postgresql+asyncpg://", "postgresql://")
            if not db_url:
                raise RuntimeError("ASYNC_DB_URL not set")
            conn = await asyncpg.connect(db_url)
            try:
                row = await conn.fetchrow(
                    "SELECT id, name FROM companies WHERE name ILIKE $1 LIMIT 1",
                    f"%{company}%",
                )
                if not row:
                    print(f"ERROR: company '{company}' not found in DB")
                    sys.exit(1)
                company_id = UUID(str(row["id"]))
            finally:
                await conn.close()

            result = await explain_company(score, company_id=company_id, industry=industry)
            if result:
                print(f"\nCompany : {result.company}")
                print(f"E={result.e_score:.0f}  S={result.s_score:.0f}  G={result.g_score:.0f}\n")
                print(result.summary)
                print("\nSaved to database.")
            else:
                print("ERROR: explainability failed")
                sys.exit(1)

        asyncio.run(_run())

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
