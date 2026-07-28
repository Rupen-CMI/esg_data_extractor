"""
metric_estimation_agent.py — Core Metric Estimation Agent

Estimates the "star" set of universal ESG disclosure metrics (physical
quantities like emissions, energy, water, workforce diversity, governance
policies) for a company that does not publicly disclose them.

Grounding: the estimate is anchored to company SIZE (employees / revenue),
SECTOR norms, and COUNTRY context — a 4k-employee firm cannot carry a
400k-employee firm's absolute emissions. We estimate the metric's real-world
value in its native unit (tCO2e, m3, %, Yes/No), never a fabricated "score".

Estimates are saved to company_metric_values with source="agentic_metrics_v1".
build_esg_json prefers real disclosed values over these estimates (real always
wins), so running this for a reported company only fills the gaps it never
disclosed.

Reuses signals + metadata + baseline already gathered by the pipeline — no
extra web calls when run as a pipeline stage.

CLI:
    python -m agentic_estimation.metric_estimation_agent dry "Clarion" --industry "Automotive Infotainment"
"""

import asyncio
import json
import logging
import os
import re
import sys
from typing import Optional
from uuid import UUID

import asyncpg
from dotenv import load_dotenv

load_dotenv()

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
log = get_logger("metric_estimation")

ESTIMATE_SOURCE = "agentic_metrics_v1"        # gap-fill estimate (loses to real data)
CORRECTION_SOURCE = "agentic_metrics_eval_v1"  # correction of an implausible real value (beats real data)


# ── The "star" core metric set — single source of truth ──────────────────────
# kind:      "numeric" (absolute quantity), "pct" (0-100), "boolean" (Yes/No)
# direction: "higher" (more = better), "lower" (less = better), "neutral" (not scored)
# intensity: denominator key for size-normalisation, or None
# benchmark: (value_for_100, value_for_0) fixed scoring band, or None.
#            Applied to the (intensity-normalised) value — size-independent, so a
#            startup and an MNC with the same carbon-per-$revenue score identically.
#            Intensity bands are per USD-million of revenue. Values grounded in
#            broad real-world ESG intensity ranges (sector-agnostic).
CORE_METRICS = [
    # Environmental (absolute quantities intensity-normalised by revenue → per $M)
    {"key": "scope_1_emissions",        "name": "Scope 1 GHG Emissions",       "unit": "tCO2e",         "category": "E", "kind": "numeric", "direction": "lower",  "intensity": "annual_revenue", "benchmark": (15, 250)},
    {"key": "scope_2_emissions",        "name": "Scope 2 GHG Emissions",       "unit": "tCO2e",         "category": "E", "kind": "numeric", "direction": "lower",  "intensity": "annual_revenue", "benchmark": (15, 250)},
    {"key": "scope_3_emissions",        "name": "Scope 3 GHG Emissions",       "unit": "tCO2e",         "category": "E", "kind": "numeric", "direction": "lower",  "intensity": "annual_revenue", "benchmark": (150, 3000)},
    {"key": "renewable_energy_pct",     "name": "Renewable Energy %",          "unit": "%",             "category": "E", "kind": "pct",     "direction": "higher", "intensity": None,             "benchmark": (75, 0)},
    {"key": "total_energy_consumption", "name": "Total Energy Consumption",    "unit": "GJ",            "category": "E", "kind": "numeric", "direction": "lower",  "intensity": "annual_revenue", "benchmark": (300, 3500)},
    {"key": "water_withdrawal",         "name": "Total Water Withdrawal",      "unit": "m3",            "category": "E", "kind": "numeric", "direction": "lower",  "intensity": "annual_revenue", "benchmark": (100, 3000)},
    {"key": "total_waste_generated",    "name": "Total Waste Generated",       "unit": "tonnes",        "category": "E", "kind": "numeric", "direction": "lower",  "intensity": "annual_revenue", "benchmark": (5, 120)},
    # Social
    {"key": "employee_count",           "name": "Total Employees",             "unit": "#",             "category": "S", "kind": "numeric", "direction": "neutral", "intensity": None,             "benchmark": None},
    {"key": "female_employees_pct",     "name": "Female Employees %",          "unit": "%",             "category": "S", "kind": "pct",     "direction": "higher", "intensity": None,             "benchmark": (50, 10)},
    {"key": "female_board_pct",         "name": "Female Board Members %",      "unit": "%",             "category": "S", "kind": "pct",     "direction": "higher", "intensity": None,             "benchmark": (40, 5)},
    {"key": "employee_turnover_rate",   "name": "Employee Turnover Rate",      "unit": "%",             "category": "S", "kind": "pct",     "direction": "lower",  "intensity": None,             "benchmark": (5, 30)},
    {"key": "lost_time_injury_rate",    "name": "Lost Time Injury Rate",       "unit": "per 200k hours","category": "S", "kind": "numeric", "direction": "lower",  "intensity": None,             "benchmark": (0.2, 5)},
    {"key": "annual_revenue",           "name": "Annual Revenue",              "unit": "USD millions",  "category": "G", "kind": "numeric", "direction": "neutral", "intensity": None,             "benchmark": None},
    # Governance
    {"key": "board_independence_pct",   "name": "Board Independence %",        "unit": "%",             "category": "G", "kind": "pct",     "direction": "higher", "intensity": None,             "benchmark": (75, 20)},
    {"key": "anti_corruption_policy",   "name": "Anti-Corruption Policy",      "unit": "Yes/No",        "category": "G", "kind": "boolean", "direction": "higher", "intensity": None,             "benchmark": None},
    {"key": "whistleblower_mechanism",  "name": "Whistleblower Mechanism",     "unit": "Yes/No",        "category": "G", "kind": "boolean", "direction": "higher", "intensity": None,             "benchmark": None},
    {"key": "esg_report_published",     "name": "ESG Report Published",        "unit": "Yes/No",        "category": "G", "kind": "boolean", "direction": "higher", "intensity": None,             "benchmark": None},
    {"key": "third_party_esg_audit",    "name": "Third-Party ESG Assurance",   "unit": "Yes/No",        "category": "G", "kind": "boolean", "direction": "higher", "intensity": None,             "benchmark": None},
]

CORE_METRIC_KEYS = [m["key"] for m in CORE_METRICS]
_CORE_BY_KEY = {m["key"]: m for m in CORE_METRICS}


# ── Estimation prompt ────────────────────────────────────────────────────────
_ESTIMATE_PROMPT = """You are an ESG data estimator and validator. For EACH metric below, produce the most plausible real-world value for this company, using its size, sector, and country context plus any evidence.

Hard rules:
- Anchor every value to company SIZE. Employee count and revenue bound what is physically possible — a small firm cannot have a large multinational's absolute emissions, energy, water, or waste.
- Use sector norms: emissions/energy/water intensity per employee differs greatly by industry (heavy manufacturing >> software).
- Use the country context for governance/policy likelihood and grid renewable mix.
- Give the value in the metric's NATIVE UNIT shown. For Yes/No metrics return true or false.
- Round sensibly — don't invent precision. If a metric is truly un-estimable, use null.
- Give each value a confidence 0.0–1.0.

VALIDATING REPORTED VALUES:
- Some metrics may already be REPORTED (listed below). Treat these as trusted UNLESS a value is clearly implausible for a company of this size/sector — e.g. off by orders of magnitude, or an obvious unit error (a 400,000-employee firm reporting 0.02 GJ total energy).
- If a reported value is plausible: return it unchanged with "corrected": false.
- If a reported value is clearly implausible: OVERRIDE it with your corrected best estimate and set "corrected": true. Explain the correction in the reasoning.
- For metrics NOT reported: estimate them and set "corrected": false.

COMPANY: {company}
INDUSTRY: {industry}
COUNTRY: {country}

COMPANY METADATA:
{metadata_block}

COUNTRY CONTEXT (World Bank ESG baseline, 0–100):
  Environment: {country_e}   Social: {country_s}   Governance: {country_g}

EVIDENCE SIGNALS:
{signals_block}

REPORTED VALUES (validate these; correct only if clearly implausible):
{reported_block}

METRICS TO PRODUCE:
{metric_list}

Think it through, then END your response with a single JSON object (no markdown fences) mapping each metric key to an object with "value", "confidence", "reasoning", and "corrected":
{{
{example_lines}
}}"""


def _metric_list_block() -> str:
    lines = []
    for m in CORE_METRICS:
        kind = "Yes/No" if m["kind"] == "boolean" else m["unit"]
        lines.append(f"- {m['key']} ({m['name']}) — value in {kind}")
    return "\n".join(lines)


def _example_lines_block() -> str:
    # Show two representative shapes so the model matches the schema
    return (
        '  "scope_1_emissions": {"value": <number|null>, "confidence": <0-1>, "reasoning": "<short>", "corrected": <true|false>},\n'
        '  "anti_corruption_policy": {"value": <true|false|null>, "confidence": <0-1>, "reasoning": "<short>", "corrected": <true|false>},\n'
        '  ...one entry per metric key above...'
    )


def _reported_block(existing_reals: Optional[dict]) -> str:
    """Render the company's already-reported core values for validation."""
    if not existing_reals:
        return "(none reported — estimate all metrics)"
    lines = []
    for key, numeric in existing_reals.items():
        meta = _CORE_BY_KEY.get(key)
        if not meta or numeric is None:
            continue
        lines.append(f"- {key}: {_display_value(meta, float(numeric))}")
    return "\n".join(lines) if lines else "(none reported — estimate all metrics)"


def _extract_json_object(raw: str) -> Optional[dict]:
    """
    Scan LLM output for the LAST balanced top-level JSON object (reasoning
    models emit prose first, then the answer). Returns the parsed dict or None.
    """
    if not raw:
        return None
    text = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Fast path: whole thing is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Scan for balanced { ... } blocks, keep the last one that parses to a dict
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


def _coerce_value(metric: dict, raw_value) -> Optional[float]:
    """Coerce an LLM value to a stored numeric, respecting the metric kind."""
    if raw_value is None:
        return None
    kind = metric["kind"]
    if kind == "boolean":
        if isinstance(raw_value, bool):
            return 1.0 if raw_value else 0.0
        s = str(raw_value).strip().lower()
        if s in ("true", "yes", "y", "1"):
            return 1.0
        if s in ("false", "no", "n", "0"):
            return 0.0
        return None
    # numeric / pct
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        return None
    if val < 0:
        val = 0.0
    if kind == "pct" and val > 100:
        val = 100.0
    return val


def _display_value(metric: dict, numeric: float) -> str:
    kind = metric["kind"]
    if kind == "boolean":
        return "Yes" if numeric >= 0.5 else "No"
    # Plain number, no scientific notation, thousands-separated for big values.
    if abs(numeric) >= 1000:
        num_str = f"{numeric:,.0f}"
    elif numeric == int(numeric):
        num_str = f"{int(numeric)}"
    else:
        num_str = f"{numeric:.2f}".rstrip("0").rstrip(".")
    if kind == "pct":
        return f"{num_str}%"
    return f"{num_str} {metric['unit']}"


def estimate_metrics_sync(
    company: str,
    industry: str,
    country: Optional[str],
    signals: dict,
    metadata: dict,
    baseline=None,
    existing_reals: Optional[dict] = None,
) -> dict[str, dict]:
    """
    Estimate/validate the core metric set. Returns
    {metric_key: {numeric, value, confidence, reasoning, corrected}}.

    existing_reals: {core_key: numeric} of already-reported values. The model
    validates these and only overrides ones it judges clearly implausible
    (corrected=True). Only keys the model returned with a usable value are
    included.
    """
    from agentic_estimation.layer_3.scoring_agent import _build_signals_block
    from agentic_estimation.layer_1.company_metadata import format_for_prompt
    from zen_client import call_with_prompt

    log_header(log, "Metric Estimation", company=company, industry=industry or "N/A", country=country or "Unknown")

    country_e = f"{baseline.e_score:.0f}" if baseline else "50"
    country_s = f"{baseline.s_score:.0f}" if baseline else "50"
    country_g = f"{baseline.g_score:.0f}" if baseline else "50"

    prompt = _ESTIMATE_PROMPT.format(
        company=company,
        industry=industry or "Not specified",
        country=country or "Unknown",
        metadata_block=format_for_prompt(metadata) if metadata else "(none)",
        country_e=country_e, country_s=country_s, country_g=country_g,
        signals_block=_build_signals_block(signals) if signals else "(no signals)",
        reported_block=_reported_block(existing_reals),
        metric_list=_metric_list_block(),
        example_lines=_example_lines_block(),
    )

    log.info("[%s] calling LLM for metric estimation...", company)
    resp = call_with_prompt(
        prompt,
        max_tokens=4000,
        timeout=240,
        system="You are an ESG data estimator. After reasoning, you MUST end with a single valid JSON object mapping each metric key to {value, confidence, reasoning}.",
    )
    if not resp.get("ok"):
        log.error("[%s] LLM call failed: %s", company, resp.get("error"))
        return {}

    parsed = _extract_json_object(resp.get("raw", "")) or _extract_json_object(resp.get("reasoning", ""))
    if not parsed:
        log.error("[%s] failed to parse metric JSON — tail: %s", company, (resp.get("raw") or "")[-300:])
        return {}

    out: dict[str, dict] = {}
    for key, metric in _CORE_BY_KEY.items():
        entry = parsed.get(key)
        if not isinstance(entry, dict):
            continue
        numeric = _coerce_value(metric, entry.get("value"))
        if numeric is None:
            continue
        try:
            confidence = float(entry.get("confidence", 0.4))
        except (TypeError, ValueError):
            confidence = 0.4
        confidence = max(0.0, min(1.0, confidence))
        out[key] = {
            "numeric": numeric,
            "value": _display_value(metric, numeric),
            "confidence": confidence,
            "reasoning": str(entry.get("reasoning", ""))[:500],
            "corrected": bool(entry.get("corrected", False)),
        }
    n_corrected = sum(1 for e in out.values() if e["corrected"])
    log.info("[%s] produced %d/%d core metrics (%d corrections)", company, len(out), len(CORE_METRICS), n_corrected)
    return out


async def _get_core_metric_ids(conn: asyncpg.Connection) -> dict[str, UUID]:
    rows = await conn.fetch(
        "SELECT id, key FROM esg_metric_definitions WHERE key = ANY($1::text[])",
        CORE_METRIC_KEYS,
    )
    mapping = {row["key"]: UUID(str(row["id"])) for row in rows}
    missing = set(CORE_METRIC_KEYS) - set(mapping.keys())
    if missing:
        log.warning("Core metric definitions missing from DB (will skip): %s", missing)
    return mapping


async def _upsert_estimate(conn, company_id, metric_id, numeric, value, confidence, reasoning, source):
    await conn.execute(
        """
        INSERT INTO company_metric_values
            (company_id, metric_id, numeric_value, value, confidence, reasoning, source, reporting_year)
        VALUES ($1, $2, $3, $4, $5, $6, $7, EXTRACT(YEAR FROM now())::int)
        ON CONFLICT (company_id, metric_id, reporting_year, source)
        DO UPDATE SET
            numeric_value = EXCLUDED.numeric_value,
            value         = EXCLUDED.value,
            confidence    = EXCLUDED.confidence,
            reasoning     = EXCLUDED.reasoning
        """,
        str(company_id), str(metric_id), numeric, value, confidence, reasoning, source,
    )


async def _load_existing_reals(conn, company_id: UUID) -> dict:
    """Latest real (non-agentic) core metric values for a company: {key: numeric}."""
    rows = await conn.fetch(
        """
        SELECT d.key AS key, v.numeric_value AS numeric_value
        FROM company_metric_values v
        JOIN esg_metric_definitions d ON d.id = v.metric_id
        WHERE v.company_id = $1
          AND d.key = ANY($2::text[])
          AND v.source <> ALL($3::text[])
          AND v.numeric_value IS NOT NULL
        ORDER BY v.reporting_year DESC NULLS LAST
        """,
        str(company_id), CORE_METRIC_KEYS, [ESTIMATE_SOURCE, CORRECTION_SOURCE],
    )
    reals = {}
    for r in rows:
        if r["key"] not in reals:  # first = latest year
            reals[r["key"]] = r["numeric_value"]
    return reals


async def estimate_and_save(
    company_name: str,
    company_id: UUID,
    industry: str,
    country: Optional[str],
    signals: dict,
    metadata: dict,
    baseline=None,
) -> dict[str, dict]:
    """
    Validate + estimate core metrics and persist them:
      - gap-fill estimates      → source 'agentic_metrics_v1'  (loses to real data)
      - corrections of a bad
        reported value          → source 'agentic_metrics_eval_v1' (beats real data)
      - plausible reported values are left untouched (not re-stored)

    Returns the estimate dict. Non-fatal: logs and returns {} on failure.
    """
    db_url = os.environ.get("ASYNC_DB_URL", "")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set in environment")
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    conn = await asyncpg.connect(db_url)
    try:
        existing_reals = await _load_existing_reals(conn, company_id)
        if existing_reals:
            log.info("[%s] validating %d reported core values", company_name, len(existing_reals))

        estimates = estimate_metrics_sync(
            company_name, industry, country, signals, metadata, baseline,
            existing_reals=existing_reals,
        )
        if not estimates:
            return {}

        metric_ids = await _get_core_metric_ids(conn)
        saved = corrected = 0
        for key, est in estimates.items():
            metric_id = metric_ids.get(key)
            if metric_id is None:
                continue
            is_real = key in existing_reals
            if is_real and not est["corrected"]:
                continue  # plausible reported value — leave the real row as-is
            source = CORRECTION_SOURCE if (is_real and est["corrected"]) else ESTIMATE_SOURCE
            await _upsert_estimate(
                conn, company_id, metric_id,
                est["numeric"], est["value"], est["confidence"], est["reasoning"], source,
            )
            saved += 1
            if source == CORRECTION_SOURCE:
                corrected += 1
        log.info("[%s] saved %d metric rows to DB (%d corrections of bad reported data)",
                 company_name, saved, corrected)
    finally:
        await conn.close()

    return estimates


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "dry":
        print("Usage: python -m agentic_estimation.metric_estimation_agent dry <company> [--industry ...] [--country ...]")
        sys.exit(1)

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

    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    from agentic_estimation.layer_1.country_baseline_agent import get_country_baseline_with_fallback

    signals = fetch_company_signals(company, industry)
    metadata = get_company_metadata(company)
    resolved = country or metadata.get("country")
    baseline = get_country_baseline_with_fallback(resolved)[0] if resolved else None

    estimates = estimate_metrics_sync(company, industry, resolved, signals, metadata, baseline)
    print(f"\nEstimated {len(estimates)} / {len(CORE_METRICS)} core metrics for {company}:\n")
    for key, est in estimates.items():
        print(f"  {key:28s} = {est['value']:<18s} (conf {est['confidence']:.2f})  {est['reasoning']}")


if __name__ == "__main__":
    _cli()
