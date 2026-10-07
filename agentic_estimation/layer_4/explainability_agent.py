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
- Mention all three pillar scores and what drives them (use the reasoning and cited evidence provided)
- Compare to the country peer group where relevant
- Close with the single biggest ESG risk or strength to watch
- For any pillar marked "UNRESOLVED" below, say so plainly (e.g. "the Governance \
score is still under review and should be treated as provisional") rather than \
presenting it with the same confidence as a resolved pillar
- For any pillar marked "THIN EVIDENCE" below, hedge accordingly (e.g. "little public \
disclosure exists, so this is closer to an industry/country estimate than a measured \
score") -- do NOT write about a thin-evidence pillar with the same certainty as a \
RICH EVIDENCE one, even if the numeric scores look similarly confident

COMPANY:  {company}
INDUSTRY: {industry}
COUNTRY:  {country}

SCORES (0-100, 50 = country average):
  Environment : {e_score:.1f}/100{e_review_flag}{e_route_flag} — {e_reasoning}
{e_evidence_block}
  Social      : {s_score:.1f}/100{s_review_flag}{s_route_flag} — {s_reasoning}
{s_evidence_block}
  Governance  : {g_score:.1f}/100{g_review_flag}{g_route_flag} — {g_reasoning}
{g_evidence_block}

Write ONLY the paragraph. No headings, no bullet points, no JSON, no preamble."""

_MAX_EVIDENCE_ITEMS_PER_PILLAR = 8


@dataclass
class ESGSummary:
    company: str
    summary: str
    e_score: float
    s_score: float
    g_score: float


def _format_evidence_block(pillar_label: str, items: Optional[list]) -> str:
    """Renders this pillar's cited evidence (Contribution.claim_reasoning
    strings, one per factor that actually contributed to the score) as an
    indented bullet list under the pillar's score line. None/empty -> a
    single "(no cited evidence...)" line rather than an empty gap, so the
    prompt's line count/shape stays stable whether or not evidence was
    passed (keeps every existing caller's un-evidenced summaries looking
    the same as before this was added)."""
    if not items:
        return "      (no cited evidence available for this pillar)"
    shown = items[:_MAX_EVIDENCE_ITEMS_PER_PILLAR]
    lines = [f"      - {text}" for text in shown if text]
    if not lines:
        return "      (no cited evidence available for this pillar)"
    if len(items) > len(shown):
        lines.append(f"      - (+{len(items) - len(shown)} more, omitted for length)")
    return "\n".join(lines)


def explain_company_sync(
    score,  # ESGScore dataclass from scoring_agent
    industry: str = "",
    evidence_by_pillar: Optional[dict] = None,
    routing_by_pillar: Optional[dict] = None,
) -> Optional[ESGSummary]:
    """
    Generate a plain-English ESG summary from an ESGScore object.
    No DB interaction — pure LLM call.

    evidence_by_pillar: optional {"E"|"S"|"G": [claim_reasoning, ...]} --
    the actual cited-evidence text behind each pillar's score (e.g. each
    Contribution.claim_reasoning from formula_scores), so the summary is
    grounded in what was actually found rather than only the compressed
    vote-breakdown string already in e/s/g_reasoning. None (default, every
    caller before 2026-09-21) omits the evidence block entirely -- same
    prompt shape as before this was added.

    routing_by_pillar: optional {"E"|"S"|"G": "rich"|"thin"} (see
    graph.py's pillar_routing / EVIDENCE_ROUTE_PLAN.md) -- tells the LLM
    which pillars are backed by real evidence vs. an industry/country
    estimate, so it hedges a thin pillar's language even when its number
    alone wouldn't signal that. None (default) omits the route tag.
    """
    from zen_client import call_with_prompt

    log_header(log, "Explainability Agent",
               company=score.company,
               country=score.country or "Unknown",
               scores=f"E={score.e_score:.0f} S={score.s_score:.0f} G={score.g_score:.0f}")

    evidence_by_pillar = evidence_by_pillar or {}
    routing_by_pillar = routing_by_pillar or {}

    # getattr-with-default: callers may pass an ESGScore-shaped object
    # predating the e/s/g_needs_review fields (added 2026-09-18), or any
    # other object satisfying this "duck-typed" contract -- absence means
    # "not flagged", never a crash.
    def _review_flag(pillar: str) -> str:
        return " (UNRESOLVED, under review)" if getattr(score, f"{pillar}_needs_review", False) else ""

    def _route_flag(pillar_key: str) -> str:
        route = routing_by_pillar.get(pillar_key)
        if route == "thin":
            return " (THIN EVIDENCE)"
        if route == "rich":
            return " (RICH EVIDENCE)"
        return ""  # unknown/not routed (e.g. formula/llm scorer callers) -- say nothing, not a guess

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
        e_review_flag=_review_flag("e"),
        s_review_flag=_review_flag("s"),
        g_review_flag=_review_flag("g"),
        e_route_flag=_route_flag("E"),
        s_route_flag=_route_flag("S"),
        g_route_flag=_route_flag("G"),
        e_evidence_block=_format_evidence_block("Environment", evidence_by_pillar.get("E")),
        s_evidence_block=_format_evidence_block("Social", evidence_by_pillar.get("S")),
        g_evidence_block=_format_evidence_block("Governance", evidence_by_pillar.get("G")),
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
    evidence_by_pillar: Optional[dict] = None,
    routing_by_pillar: Optional[dict] = None,
) -> Optional[ESGSummary]:
    """
    Generate ESG summary and persist to company_metric_values.
    """
    result = explain_company_sync(score, industry=industry, evidence_by_pillar=evidence_by_pillar,
                                   routing_by_pillar=routing_by_pillar)
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
            from agentic_estimation.shared.db_company_lookup import resolve_company_id_standalone

            match = await resolve_company_id_standalone(company)
            if match is None:
                print(f"ERROR: company '{company}' not found in DB")
                sys.exit(1)
            company_id, _actual_name = match

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
