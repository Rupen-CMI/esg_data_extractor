"""
market_climate_trace_mapper.py — ONE-TIME batch mapping: market name ->
Climate TRACE sector/subsector.

Not a live pipeline component. LLM-based (not keyword matching) because this
is a one-shot batch classification, not a repeated live call — the usual
"no LLM, deterministic" rule for pipeline collectors doesn't apply here,
since there's no repeated-call cost/latency/non-determinism to avoid, and
the accuracy benefit is real: an LLM can correctly classify long-tail market
names ("Nano Silicon Drug Delivery Platform Market") that a keyword list
can't anticipate, whereas a keyword prototype only looked good against a
handful of hand-picked examples (see conversation).

CLOSED-SET classification, same discipline as every other classifier in this
codebase: the LLM must pick from the REAL, fixed list of Climate TRACE
sector/subsector pairs already harvested into climate_trace_country_emissions
(64 pairs, derived live from the DB — never hand-typed, so it can't drift out
of sync), or explicitly answer "no match". Most markets (healthcare, tech,
pharma, and anything non-physical) SHOULD get "no match" -- Climate TRACE only
tracks physical emitters (verified earlier: Bosch/Nvidia owner search 404'd).
The prompt makes "no match is a correct, expected answer" explicit so the
model doesn't force a weak match onto something like "Ambulance Drone Market".

Results are written to a new market_climate_trace_mapping table -- a
one-time batch write, re-run only if markets are added/changed.

CLI:
    python -m agentic_estimation.market_climate_trace_mapper run --limit 20   # test on a sample
    python -m agentic_estimation.market_climate_trace_mapper run              # all markets
    python -m agentic_estimation.market_climate_trace_mapper spot-check --n 15
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.climate_trace_harvester import _db_conn, _write_rows

log = get_logger("market_climate_trace_mapper")

# Pre-filter: skip the LLM call for markets that are UNAMBIGUOUSLY non-physical
# -- deterministic and conservative by design, since a false skip permanently
# loses real grounding data while a false LLM-call only costs one API call.
# Built from real market names sampled from the DB (see conversation), not
# guessed -- markets.sasb_sector is NULL for 99% of rows (9,475/9,580 as of
# this session), so a sector-metadata filter would barely skip anything; this
# has to work off the market NAME itself. Only markets matching one of these
# keywords skip the LLM -- everything else (including ambiguous cases) still
# gets a real classification call.
_NO_MATCH_KEYWORDS = [
    # pharma / biologics / clinical
    "drug", "vaccine", "therapeutic", "therapy", "antibody", "biologic",
    "oncology", "diagnostic", "clinical", "pharma", "mab market",  # mAb suffix, e.g. "Atezolizumab"
    "disease", "treatment", "syndrome", "tumor", "cancer",
    # medical devices / healthcare services
    "implant", "prosthetic", "surgical", "healthcare", "hospital", "patient",
    "wheelchair", "catheter", "stent", "pacemaker", "dialysis", "dental",
    "veterinary", "medical device",
    # software / IT / digital services
    "software", "saas", "cloud computing", "cybersecurity", "app market",
    "platform market", "digital twin", "blockchain", "artificial intelligence",
    "data analytics", "e-commerce",
    # financial / insurance / professional services
    "insurance", "banking", "asset management", "wealth management",
    "payment", "fintech", "accounting", "consulting", "legal services",
    "reporting software",
]

_NO_MATCH_RE = re.compile("|".join(re.escape(k) for k in _NO_MATCH_KEYWORDS), re.IGNORECASE)


def _prefilter_no_match(market_name: str) -> bool:
    """True if the name unambiguously indicates a non-physical market --
    the LLM call should be skipped and this should be recorded as a
    confidence-0 no-match without ever contacting the API."""
    return bool(_NO_MATCH_RE.search(market_name))

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_climate_trace_mapping (
    market_id       UUID PRIMARY KEY REFERENCES markets(id) ON DELETE CASCADE,
    ct_sector       VARCHAR(60),
    ct_subsector    VARCHAR(60),
    confidence      FLOAT NOT NULL DEFAULT 0.0,
    reasoning       TEXT,
    mapped_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_MAPPING_SQL = """
    INSERT INTO market_climate_trace_mapping
        (market_id, ct_sector, ct_subsector, confidence, reasoning, mapped_at)
    VALUES (%s, %s, %s, %s, %s, now())
    ON CONFLICT (market_id) DO UPDATE SET
        ct_sector = EXCLUDED.ct_sector,
        ct_subsector = EXCLUDED.ct_subsector,
        confidence = EXCLUDED.confidence,
        reasoning = EXCLUDED.reasoning,
        mapped_at = now()
"""

_PROMPT_TEMPLATE = """You are classifying a market/industry name against Climate TRACE's emissions
tracking taxonomy, to find which INDUSTRIAL PROCESS produces this market's
product -- so we can borrow that process's real-world emissions-intensity
data as a comparison baseline. You are NOT checking whether this specific
market listing is itself a tracked facility -- you ARE checking whether the
underlying manufacturing/production process for this product falls within
one of Climate TRACE's tracked physical-emitter categories.

Example of correct reasoning: "Cheese Powder Market" is a product market, not
a facility -- but cheese powder IS PRODUCED by food processing plants, which
fall under sector=manufacturing, subsector=food-beverage-tobacco. The right
answer is that pair, NOT no_match -- the product's underlying production
process is what matters, not whether the market listing names a specific
facility.

Climate TRACE tracks PHYSICAL, asset-heavy production processes (factories,
power plants, mines, farms, refineries, transport). It does NOT track
software, financial services, pure R&D/lab techniques with no bulk industrial
production, healthcare/medical services, or retail/distribution-only
activities with no manufacturing of their own.

VALID (sector, subsector) PAIRS -- you MUST pick one of these exactly if the
product's production process genuinely matches one, or answer "no_match":
true only if NO industrial production process for this item exists in the
list below (e.g. software, pure services, lab research with no bulk
manufacturing). Do not withhold a real match just because the market NAME
itself doesn't literally say "factory" or "plant" -- infer the production
process from what the product actually is.

{pairs_list}

MARKET NAME: "{market_name}"

Think briefly about what industrial process actually produces this item, then
respond with ONLY this JSON object (no markdown, no prose):
{{"no_match": true|false, "sector": "<exact sector from list, or null>", "subsector": "<exact subsector from list, or null>", "confidence": <0.0-1.0>, "reasoning": "<one short sentence>"}}"""


def _get_valid_pairs() -> list[tuple[str, str]]:
    """Fetch the real, current list of CT sector/subsector pairs from the
    harvested data -- never hand-typed, so this can't drift out of sync with
    what Climate TRACE actually covers."""
    conn = _db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT sector, subsector FROM climate_trace_country_emissions "
            "WHERE subsector IS NOT NULL ORDER BY sector, subsector"
        )
        return [(r[0], r[1]) for r in cur.fetchall()]
    finally:
        conn.close()


def _pairs_list_text(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"  - sector={s}, subsector={sub}" for s, sub in pairs)


def _extract_json(raw: str) -> Optional[dict]:
    """Same last-balanced-object scan pattern used across the pipeline
    (metric_estimation_agent._extract_json_object) -- reasoning models emit
    prose before the final JSON answer."""
    if not raw:
        return None
    text = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    best = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            best = obj
                    except Exception:
                        pass
    return best


def classify_market(market_name: str, valid_pairs: list[tuple[str, str]]) -> dict:
    """One closed-set LLM classification. Returns
    {ct_sector, ct_subsector, confidence, reasoning} -- ct_sector/subsector
    are None on no_match or any parse/validation failure (fails closed, not
    open -- an unparseable response must never silently become a fabricated
    mapping)."""
    from zen_client import call_with_prompt

    prompt = _PROMPT_TEMPLATE.format(
        pairs_list=_pairs_list_text(valid_pairs), market_name=market_name,
    )
    # 300 was too tight -- found live via truncated responses under real load
    # (e.g. '{"no_match":false,"sector":"manufacturing","subsector":"other-
    # manufacturing","confidence":0.9' cut off mid-JSON, no closing brace, no
    # reasoning field at all). 600 gives headroom for the reasoning sentence.
    resp = call_with_prompt(prompt, max_tokens=600, timeout=60)
    if not resp.get("ok"):
        log.warning("[%s] LLM call failed: %s", market_name, resp.get("error"))
        return {"ct_sector": None, "ct_subsector": None, "confidence": 0.0,
                "reasoning": f"LLM call failed: {resp.get('error')}"}

    parsed = _extract_json(resp.get("raw", "")) or _extract_json(resp.get("reasoning", ""))
    if not parsed:
        log.warning("[%s] failed to parse LLM response", market_name)
        return {"ct_sector": None, "ct_subsector": None, "confidence": 0.0,
                "reasoning": "failed to parse LLM response"}

    if parsed.get("no_match"):
        return {"ct_sector": None, "ct_subsector": None,
                "confidence": 0.0, "reasoning": str(parsed.get("reasoning", ""))[:300]}

    sector = parsed.get("sector")
    subsector = parsed.get("subsector")
    # VALIDATE against the real pair list -- never trust the LLM's string
    # verbatim; an invented or slightly-off pair must fail closed to
    # no-match, not silently pollute the table with a fake CT category.
    if (sector, subsector) not in valid_pairs:
        log.warning("[%s] LLM returned invalid pair (%s, %s) -- treating as no_match",
                    market_name, sector, subsector)
        return {"ct_sector": None, "ct_subsector": None, "confidence": 0.0,
                "reasoning": f"invalid pair returned: ({sector}, {subsector})"}

    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return {"ct_sector": sector, "ct_subsector": subsector,
            "confidence": confidence, "reasoning": str(parsed.get("reasoning", ""))[:300]}


def run(limit: Optional[int] = None, skip_done: bool = True, workers: int = 4) -> int:
    """
    Two speed levers, applied in order (pre-filter first -- it's free, no LLM
    call at all; then a bounded thread pool for what's left):
      1. Deterministic keyword pre-filter (_prefilter_no_match) -- skips the
         LLM entirely for markets unambiguously non-physical (pharma, medical
         devices, software, financial services). Conservative by design.
      2. ThreadPoolExecutor(workers=4) for the remaining ambiguous/physical
         markets -- confirmed safe at this concurrency: the free gateway
         (zen_client.py) handled 8 concurrent calls cleanly with zero fallback
         to a different model and zero errors (verified live this session).

    skip_done=True (default): skip markets that already have a mapping row --
    lets a full run resume after an interruption without re-classifying
    markets already done. Mirrors climate_trace_harvester.py's resume pattern.
    """
    log_header(log, "Market -> Climate TRACE Mapper", limit=limit or "all", workers=workers)

    conn = _db_conn()
    try:
        cur = conn.cursor()
        cur.execute(_CREATE_TABLE_SQL)
        conn.commit()
        query = "SELECT id, name FROM markets"
        if skip_done:
            query += " WHERE id NOT IN (SELECT market_id FROM market_climate_trace_mapping)"
        query += " ORDER BY name"
        if limit:
            query += f" LIMIT {int(limit)}"
        cur.execute(query)
        markets = cur.fetchall()
    finally:
        conn.close()

    valid_pairs = _get_valid_pairs()
    log.info("loaded %d markets to process, %d valid CT sector/subsector pairs (skip_done=%s)",
              len(markets), len(valid_pairs), skip_done)

    # Pass 1: pre-filter, no LLM calls at all.
    prefiltered_rows = []
    to_classify = []
    for market_id, name in markets:
        if _prefilter_no_match(name):
            prefiltered_rows.append((str(market_id), None, None, 0.0, "pre-filtered: non-physical market name"))
        else:
            to_classify.append((market_id, name))

    log.info("pre-filter: %d/%d skipped (no LLM call), %d remain for classification",
              len(prefiltered_rows), len(markets), len(to_classify))
    if prefiltered_rows:
        _write_rows(_MAPPING_SQL, prefiltered_rows)

    # Pass 2: bounded concurrency for the rest.
    matched = 0
    done = 0
    buffer = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(classify_market, name, valid_pairs): (market_id, name)
                   for market_id, name in to_classify}
        for fut in as_completed(futures):
            market_id, name = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                log.warning("[%s] classify_market raised: %s", name, exc)
                result = {"ct_sector": None, "ct_subsector": None, "confidence": 0.0,
                          "reasoning": f"exception: {exc}"}
            if result["ct_sector"]:
                matched += 1
            buffer.append((str(market_id), result["ct_sector"], result["ct_subsector"],
                            result["confidence"], result["reasoning"]))
            done += 1
            if len(buffer) >= 20:
                _write_rows(_MAPPING_SQL, buffer)
                buffer = []
                log.info("progress %d/%d classified, %d matched so far", done, len(to_classify), matched)

    if buffer:
        _write_rows(_MAPPING_SQL, buffer)

    total_matched = matched
    log.info("done: %d/%d classified markets matched to a real CT sector/subsector "
             "(+%d pre-filtered no-match)", total_matched, len(to_classify), len(prefiltered_rows))
    return len(markets)


def spot_check(n: int) -> None:
    """Print a random sample of mapping results for manual review."""
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.name, mm.ct_sector, mm.ct_subsector, mm.confidence, mm.reasoning
        FROM market_climate_trace_mapping mm
        JOIN markets m ON m.id = mm.market_id
        ORDER BY random() LIMIT %s
        """,
        (n,),
    )
    for name, sector, subsector, conf, reasoning in cur.fetchall():
        print(f"[{name}]")
        print(f"  -> sector={sector} subsector={subsector} conf={conf}")
        print(f"     {reasoning}")
        print()
    conn.close()


def _cli() -> None:
    ap = argparse.ArgumentParser(description="One-time market -> Climate TRACE subsector mapper (LLM-based, closed-set).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--limit", type=int, default=None, help="cap number of markets (for testing)")
    run_p.add_argument("--workers", type=int, default=4, help="concurrent LLM classification calls (default 4)")
    run_p.add_argument("--no-skip-done", action="store_true",
                        help="re-classify markets that already have a mapping row (default: skip them)")
    sc_p = sub.add_parser("spot-check")
    sc_p.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    if args.cmd == "run":
        n = run(limit=args.limit, skip_done=not args.no_skip_done, workers=args.workers)
        print(f"Processed {n} markets.")
    elif args.cmd == "spot-check":
        spot_check(args.n)


if __name__ == "__main__":
    _cli()
