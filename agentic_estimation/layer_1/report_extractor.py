"""
report_extractor.py -- retrieved report context -> structured ESG facts.

WHAT THE LLM IS ASKED TO DO, AND WHAT IT IS NOT
    It maps labelled report content onto a CLOSED list of core metric keys and
    copies the value verbatim. It does not estimate, infer, convert units, or
    score. That division is the founding constraint of this pipeline: the LLM
    extracts, the formula scores.

    Concretely, for Toyota page 37 the row is

        | Water withdrawal | 4,548 km3 |

    and the expected output is {"metric": "water_withdrawal", "value": "4,548",
    "unit": "km3", "quote": "Water withdrawal 4,548 km3", "page": 37}.
    No arithmetic, no normalisation -- downstream code owns those, and a model
    that silently converts units produces facts that cannot be verified against
    the source.

VERIFIABILITY IS THE POINT
    Every fact carries `quote`, which MUST appear in the retrieved context.
    verify_facts() re-checks each quote against the PDF text, so a hallucinated
    number is detectable mechanically rather than by reading. Facts whose quote
    cannot be found are dropped, and the drop is reported -- a silent pass would
    reintroduce exactly the class of error this pipeline keeps getting bitten by.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from agentic_estimation.layer_1.report_rag import PillarContext, retrieve
from agentic_estimation.layer_1.report_structure import (
    METRIC_ALIASES,
    METRIC_PILLAR,
)
from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("report_extractor")

_MODEL = "deepseek-v4-flash-free"

_PROMPT = """Extract ESG facts for the {pillar} pillar from this company report extract.

ONLY extract facts that are explicitly stated in the text below. Copy values
EXACTLY as written -- do not convert units, do not do arithmetic, do not round.

Return JSON: {{"facts": [
  {{"metric": "<one of the allowed keys, or null>",
    "label": "<the report's own name for this figure>",
    "value": "<the figure exactly as printed>",
    "unit": "<unit exactly as printed, or null>",
    "period": "<year or period if stated, else null>",
    "quote": "<the exact substring from the text that contains this fact>",
    "page": <page number if shown in the extract, else null>}}
]}}

Allowed metric keys for {pillar} (use null if a fact matches none of them):
{keys}

Rules:
- "quote" must be copied character-for-character from the text below.
- If a figure has no explicit label, skip it.
- Prefer figures from DATA TABLES over narrative claims.
- Return at most 25 facts. If nothing qualifies, return {{"facts": []}}.

TEXT:
{context}
"""


@dataclass
class Fact:
    pillar: str
    metric: Optional[str]
    label: str
    value: str
    unit: Optional[str]
    period: Optional[str]
    quote: str
    page: Optional[int]
    verified: bool = False
    source: str = "report"


def _keys_for(pillar: str) -> list[str]:
    return [k for k, p in METRIC_PILLAR.items() if p == pillar]


def _norm_ws(s: str) -> str:
    """Whitespace-insensitive form for quote matching.

    PDF text carries line breaks and doubled spaces at unpredictable places;
    an exact substring test would fail on quotes that ARE genuinely present.
    Digits, letters and units are preserved, so a fabricated number still
    fails to match.
    """
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def extract_pillar_facts(ctx: PillarContext, company: str,
                         max_chars: int = 14000) -> list[Fact]:
    from zen_client import call_with_prompt

    context = ctx.as_prompt_text(max_chars=max_chars)
    if not context.strip():
        return []
    prompt = _PROMPT.format(pillar=ctx.pillar, keys=", ".join(_keys_for(ctx.pillar)),
                            context=context)
    try:
        res = call_with_prompt(prompt, model=_MODEL, max_tokens=2500, timeout=180)
    except Exception as exc:
        log.warning("[%s/%s] LLM failed: %s", company, ctx.pillar, exc)
        return []

    # zen_client returns {"ok","raw","reasoning","error","model_used",
    # "latency_s"} -- the payload is "raw". There is NO "content" key; reading
    # one returns None on every call and yields a silent zero-fact run that
    # looks like "the report had nothing in it".
    if not (res or {}).get("ok"):
        log.warning("[%s/%s] LLM not ok: %s", company, ctx.pillar,
                    (res or {}).get("error"))
        return []
    raw = (res or {}).get("raw") or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        log.warning("[%s/%s] no JSON in response (%d chars)",
                    company, ctx.pillar, len(raw))
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        log.warning("[%s/%s] unparseable JSON", company, ctx.pillar)
        return []

    out: list[Fact] = []
    for f in (data.get("facts") or [])[:25]:
        if not isinstance(f, dict) or not f.get("value"):
            continue
        met = f.get("metric")
        if met and met not in METRIC_PILLAR:
            met = None                     # model invented a key
        out.append(Fact(
            pillar=ctx.pillar, metric=met, label=str(f.get("label") or "")[:120],
            value=str(f.get("value"))[:60],
            unit=(str(f["unit"])[:24] if f.get("unit") else None),
            period=(str(f["period"])[:24] if f.get("period") else None),
            quote=str(f.get("quote") or "")[:300],
            page=f.get("page") if isinstance(f.get("page"), int) else None))
    return out


def verify_facts(facts: list[Fact], haystacks: list[str]) -> tuple[list[Fact], list[Fact]]:
    """Split facts into (verified, rejected) by locating each quote in source text.

    Two-stage: the quote must appear, AND the value must appear inside the
    matched quote. A model that copies a real sentence but swaps the number
    passes the first test and fails the second.
    """
    hay = [_norm_ws(h) for h in haystacks]
    ok: list[Fact] = []
    bad: list[Fact] = []
    for f in facts:
        q = _norm_ws(f.quote)
        v = _norm_ws(f.value)
        found = bool(q) and any(q in h for h in hay)
        if found and v and v not in q:
            found = False                  # quote real, value not in it
        f.verified = found
        (ok if found else bad).append(f)
    return ok, bad


def extract_from_report(pdf: Path, company: str, pages: str | None = None
                        ) -> dict:
    """Full pipeline for one report: retrieve -> extract -> verify."""
    ctxs = retrieve(pdf, pages=pages)
    all_facts: list[Fact] = []
    contexts: list[str] = []
    for pillar, ctx in ctxs.items():
        contexts.append(ctx.as_prompt_text())
        all_facts += extract_pillar_facts(ctx, company)

    verified, rejected = verify_facts(all_facts, contexts)
    log.info("[%s] %d facts, %d verified, %d rejected",
             company, len(all_facts), len(verified), len(rejected))
    return {
        "company": company,
        "pdf": str(pdf),
        "facts": [asdict(f) for f in verified],
        "rejected": [asdict(f) for f in rejected],
        "metrics_found": sorted({f.metric for f in verified if f.metric}),
        "context_chars": {p: len(c) for p, c in zip(ctxs, contexts)},
    }


if __name__ == "__main__":
    import sys

    out = extract_from_report(Path(sys.argv[1]), sys.argv[2],
                              pages=(sys.argv[3] if len(sys.argv) > 3 else None))
    print(json.dumps(out, indent=2)[:4000])
