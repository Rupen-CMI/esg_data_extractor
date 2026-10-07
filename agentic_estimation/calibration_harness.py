"""
calibration_harness.py — Backtest the ESG estimation pipeline against real,
third-party ESG assessments already in the DB.

WHY THIS EXISTS
    The pipeline estimates ESG for companies that don't disclose. "Is it
    accurate?" is unanswerable without ground truth. This harness measures the
    pipeline against companies whose ESG *is* independently assessed —
    B Corp Impact scores (bcorp_lookup, 10k+ firms) and Upright net-impact
    (upright_lookup, 10k+ firms) — so every change to the pipeline can be
    proven to raise or lower accuracy instead of argued about.

    It writes NOTHING to production tables. Pure read + measure.

WHAT IT VALIDATES (and what it does NOT)
    Validates: the 0-100 E/S/G PILLAR SCORES (scoring_agent, optionally after
    evaluator_agent). Because B Corp / Upright use their own scales (a B Corp
    overall score can exceed 100; Upright is a percentile), absolute error is
    meaningless. We validate by RANK AGREEMENT:
        - Spearman rank correlation  — does the estimator order firms like the
          real assessor does?  (the headline number; 1.0 = perfect order,
          0 = no relationship, <0 = inverted)
        - Percentile MAE            — convert both pred and truth to within-
          sample percentiles; average |pred_pctile - truth_pctile|. Interpretable
          as "the estimate is on average X percentile-points off."
    Does NOT validate: absolute physical metrics (tCO2e, m3, %). B Corp / Upright
    don't publish those; that needs disclosed-figure ground truth (Wikirate),
    which is currently too sparse to backtest. See module TODO.

HONEST CAVEATS baked into the report
    - Evidence split: a company with no web signals can't be scored better than
      its country baseline, so its error reflects missing evidence, not a bad
      estimator. The report splits results by "had signals" vs "no signals".
    - Selection bias: B Corp firms are all certified-sustainable (range is
      truncated high), which depresses correlation power. Upright spans all
      large public firms and is the better range test. Pick with --source.

CLI
    python -m agentic_estimation.calibration_harness --n 20 --source bcorp
    python -m agentic_estimation.calibration_harness --n 40 --source upright --workers 4
    python -m agentic_estimation.calibration_harness --n 20 --with-evaluator --out calibration/report.csv
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from agentic_estimation.shared.pipeline_logger import get_logger, log_header

log = get_logger("calibration_harness")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_DEFAULT_CACHE = _PROJECT_ROOT / "calibration" / "cache"

# v5 aggregate-saturation scoring toggle + optional per-pillar param override.
# Module-level (set from CLI) so it reaches _gather_and_score_formula without
# threading through run_backtest/estimate_one/_estimate_one_via_* signatures.
# DEFAULT True since 2026-07-21 (v5 is the pipeline default; --no-saturation
# gets the legacy linear reduction for A/B).
# {pillar: PillarSatParams} or None -> saturation_score.py defaults.
_USE_SATURATION = True
_SAT_PARAMS = None

# Phase 4 --verify opt-in (ensemble scorer only): runs estimate_verifier.py's
# gated critic panel + bounded retry on top of the already-built Confidence
# Gate. Module-level for the same reason as _USE_SATURATION above -- reaches
# _estimate_one_via_ensemble without threading through every call site.
# DEFAULT False -- Phase 4's critic panel adds real LLM cost; opt-in only.
_VERIFY = False

# Sustainability-report PDF text as an evidence source (report_collector.py).
# Module-level for the same reason as _USE_SATURATION -- reaches
# _gather_and_score_formula without threading through every call site.
#
# DEFAULT True: the documents are already on disk and cost nothing to read.
# --no-report-pdf turns it off, which is what an A/B comparison needs -- and
# that comparison MUST also pass --no-cache or a different --cache-prefix,
# because the prediction cache is keyed on company name and has no idea this
# flag exists (see the cache-prefix comment further down).
_USE_REPORT_PDF = os.getenv("ESG_REPORT_PDF", "1") != "0"


# ── Console visibility ───────────────────────────────────────────────────────
# The pipeline loggers are FILE-ONLY by design (pipeline_logger writes to
# agentic_estimation/logs/*.log, never console, so they don't pollute the API
# server's stdout). When running the harness by hand you want to SEE progress,
# so we print directly to stdout and optionally mirror the agent logs to console.

def _p(msg: str) -> None:
    """Print a progress line to stdout immediately (flush so it shows even when piped).
    Company names can carry arbitrary Unicode (macrons, accents, CJK...) that
    Windows' default console codepage (cp1252) can't encode -- found live when
    a name containing U+014D crashed a 40-company background dump on its LAST
    company, after ~35 min of real gathering, because the exception ITSELF
    tried to print the name again in the failure handler. Never let a display
    encoding problem discard completed work; degrade to '?'-replaced ASCII
    instead of raising."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _attach_console_logging(verbose: bool) -> None:
    """
    Mirror the (normally file-only) pipeline logs to the console so DDG
    rate-limits, HTTP errors, and per-source results are visible live.
      - default: WARNING+ only (rate-limits / errors — the stuff you care about)
      - --verbose: INFO firehose (every source hit, every score)

    IMPORTANT: must go through pipeline_logger.get_logger() FIRST, not
    logging.getLogger() directly. get_logger() short-circuits ("if log.handlers:
    return log") when a logger already has ANY handler attached — so if a bare
    console StreamHandler got added before get_logger() ever ran for that name,
    get_logger() would see handlers already present and skip attaching the file
    handler entirely, silently killing file logging for that module. Calling
    get_logger() here first guarantees the file handler exists before we layer
    the console handler on top.
    """
    level = logging.INFO if verbose else logging.WARNING
    fmt = logging.Formatter("      %(levelname)-7s [%(name)s] %(message)s")
    for name in ("signal_agent", "scoring_agent", "evaluator_agent",
                 "company_metadata", "calibration_harness"):
        lg = get_logger(name)  # ensures file handler is attached first
        if any(getattr(h, "_harness_console", False) for h in lg.handlers):
            continue  # already attached (safe on re-invocation)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        h.setFormatter(fmt)
        h._harness_console = True  # type: ignore[attr-defined]
        lg.addHandler(h)


# ── Environment / DB ─────────────────────────────────────────────────────────

def _resolve_db_url() -> str:
    """
    Resolve the Postgres URL. load_dotenv(override=True) so an empty shell var
    (which does NOT trigger os.getenv's default) can't shadow the real value in
    .env — the failure mode that silently points psycopg2 at localhost.
    """
    load_dotenv(_ENV_PATH, override=True)
    url = os.getenv("ASYNC_DB_URL") or os.getenv("DB_URL") or ""
    if not url:
        from dotenv import dotenv_values
        vals = dotenv_values(_ENV_PATH)
        url = vals.get("ASYNC_DB_URL") or vals.get("DB_URL") or ""
    return url.replace("postgresql+asyncpg://", "postgresql://")


def _db_conn():
    """psycopg2 connection, mirroring the ?ssl= → sslmode= handling used across the pipeline."""
    parsed = urlparse(_resolve_db_url())
    qs = parse_qs(parsed.query)
    sslmode = qs.pop("ssl", [None])[0]
    clean_qs = urlencode({k: v[0] for k, v in qs.items()})
    if sslmode:
        clean_qs = (clean_qs + "&" if clean_qs else "") + f"sslmode={sslmode}"
    return psycopg2.connect(urlunparse(parsed._replace(query=clean_qs)))


# ── Ground-truth records ─────────────────────────────────────────────────────

@dataclass
class TruthRecord:
    name: str
    country: Optional[str]
    industry: Optional[str]
    # Ground-truth pillar values on the ASSESSOR's own scale (rank-comparable only)
    truth_e: Optional[float] = None
    truth_s: Optional[float] = None
    truth_g: Optional[float] = None
    truth_total: Optional[float] = None
    # Peer-matching metadata. `industry` above is bcorp's FREE-TEXT column (163
    # distinct values, ~6 companies per value inside a scored corpus) and peer_anchor
    # matches sector strings with SQL '=' against industry_category / sasb_sector --
    # so passing `industry` almost never hits an exact-match tier. Measured
    # 2026-08-04 (n=25, production peer_anchor_vote called unmodified):
    #     sector=industry           -> 32% abstain, 15 sector_country votes
    #     sector=industry_category  ->  0% abstain, 57 sector_country votes
    #     sector=sasb_sector        ->  3% abstain, 72 sector_country votes
    # The anchor is the strongest single scoring component where it fires (held-out
    # n=193: anchor-only G +0.364 vs claims-only +0.071; S +0.215 vs -0.021), and it
    # fires for only 63-65% of company-pillars today -- so the abstentions are pure
    # lost signal, caused by a vocabulary mismatch rather than by missing data.
    industry_category: Optional[str] = None
    sasb_sector: Optional[str] = None
    size: Optional[str] = None
    ownership: Optional[str] = None

    @property
    def peer_sector(self) -> Optional[str]:
        """The sector string to hand to peer_anchor_vote.

        industry_category first (bcorp's own 22-value vocabulary, which the crosswalk
        and sector_country tiers match exactly), then sasb_sector, then the free-text
        industry as a last resort so nothing regresses for rows lacking the columns.
        """
        return self.industry_category or self.sasb_sector or self.industry


def load_bcorp_truth(n: int, seed: int) -> list[TruthRecord]:
    """
    Sample n B Corp companies. Pillar mapping:
        E     ← impact_area_environment
        G     ← impact_area_governance
        S     ← mean(workers, community, customers)   (B Corp's social areas)
        total ← overall_score
    """
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT company_name, country, industry,
               overall_score, impact_area_environment, impact_area_governance,
               impact_area_workers, impact_area_community, impact_area_customers,
               industry_category, sasb_sector, size, ownership
        FROM bcorp_lookup
        WHERE overall_score IS NOT NULL
          AND company_name IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()

    rng = random.Random(seed)
    rng.shuffle(rows)

    out: list[TruthRecord] = []
    for (name, country, industry, overall, env, gov, workers, community, customers,
         industry_category, sasb_sector, size, ownership) in rows:
        social_parts = [v for v in (workers, community, customers) if v is not None]
        social = sum(social_parts) / len(social_parts) if social_parts else None
        out.append(TruthRecord(
            name=name, country=country, industry=industry,
            truth_e=env, truth_s=social, truth_g=gov, truth_total=overall,
            industry_category=industry_category, sasb_sector=sasb_sector,
            size=size, ownership=ownership,
        ))
        if len(out) >= n:
            break
    return out


def load_upright_truth(n: int, seed: int) -> list[TruthRecord]:
    """
    Sample n Upright companies, biased toward larger firms (they have real web
    signal, so the backtest measures estimator skill rather than missing evidence).
    Pillar mapping (Upright is a positive/negative impact model, not E/S/G, so this
    is approximate — documented in the report):
        E     ← net environmental impact  = Σ e*_positive − Σ e*_negative
        S     ← net social+health impact  = Σ (s*,h*)_positive − Σ (s*,h*)_negative
        G     ← not cleanly defined in Upright → left None (skipped in report)
        total ← net_impact_ratio_percentile  (already 0-100)
    """
    conn = _db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT name, country, industry, revenue_usd, net_impact_ratio_percentile,
               e1_ghg_positive, e2_non_ghg_positive, e3_scarce_resources_positive,
               e4_biodiversity_positive, e5_waste_positive,
               e1_ghg_negative, e2_non_ghg_negative, e3_scarce_resources_negative,
               e4_biodiversity_negative, e5_waste_negative,
               h1_physical_diseases_positive, h2_mental_diseases_positive,
               h3_nutrition_positive, h4_relationships_positive, h5_meaning_joy_positive,
               s1_jobs_positive, s3_societal_infra_positive, s4_societal_stability_positive,
               s5_equality_positive,
               h1_physical_diseases_negative, h2_mental_diseases_negative,
               h4_relationships_negative, h5_meaning_joy_negative,
               s4_societal_stability_negative, s5_equality_negative
        FROM upright_lookup
        WHERE net_impact_ratio_percentile IS NOT NULL
          AND name IS NOT NULL
        ORDER BY revenue_usd DESC NULLS LAST
        """
    )
    rows = cur.fetchall()
    conn.close()

    # Take a large head of biggest firms, then sample within it for web-visible names.
    head = rows[: max(n * 8, 200)]
    rng = random.Random(seed)
    rng.shuffle(head)

    def _net(pos_vals, neg_vals):
        pos = sum(v for v in pos_vals if v is not None)
        neg = sum(v for v in neg_vals if v is not None)
        if not any(v is not None for v in pos_vals + neg_vals):
            return None
        return pos - neg

    out: list[TruthRecord] = []
    for r in head:
        (name, country, industry, _rev, pctile,
         e1p, e2p, e3p, e4p, e5p, e1n, e2n, e3n, e4n, e5n,
         h1p, h2p, h3p, h4p, h5p, s1p, s3p, s4p, s5p,
         h1n, h2n, h4n, h5n, s4n, s5n) = r
        e_net = _net([e1p, e2p, e3p, e4p, e5p], [e1n, e2n, e3n, e4n, e5n])
        s_net = _net([h1p, h2p, h3p, h4p, h5p, s1p, s3p, s4p, s5p],
                     [h1n, h2n, h4n, h5n, s4n, s5n])
        out.append(TruthRecord(
            name=name, country=country, industry=industry,
            truth_e=e_net, truth_s=s_net, truth_g=None, truth_total=pctile,
        ))
        if len(out) >= n:
            break
    return out


# ── Prediction (pipeline under test) ─────────────────────────────────────────

@dataclass
class BacktestRow:
    name: str
    country: Optional[str]
    industry: Optional[str]
    pred_e: Optional[float] = None
    pred_s: Optional[float] = None
    pred_g: Optional[float] = None
    spread_e: Optional[float] = None   # ensemble scorer only -- reconcile.py's uncertainty signal
    spread_s: Optional[float] = None
    spread_g: Optional[float] = None
    # reconcile.py's raw 'high'|'medium'|'low' confidence LABEL -- distinct from
    # gate_* below (which only says point/range framing). This is what a
    # calibration analysis buckets on: does 'high' actually mean lower error?
    confidence_e: Optional[str] = None
    confidence_s: Optional[str] = None
    confidence_g: Optional[str] = None
    # Confidence Gate outputs (ensemble scorer only -- confidence_gate.py).
    # 'point' | 'range'; low_*/high_* are always populated (reconcile's own
    # band) regardless of mode -- gate_* just says which framing was chosen.
    gate_e: Optional[str] = None
    gate_s: Optional[str] = None
    gate_g: Optional[str] = None
    low_e: Optional[float] = None
    low_s: Optional[float] = None
    low_g: Optional[float] = None
    high_e: Optional[float] = None
    high_s: Optional[float] = None
    high_g: Optional[float] = None
    # Tier-0 validator audit counts (claim_validators.py; ensemble/formula scorers)
    claims_dropped: int = 0
    claims_capped: int = 0
    # opposite-polarity same-factor conflicts seen by Tier-0's polarity_consistency
    # rule -- the Contradiction-Resolution precondition counter (see
    # EVALUATION_STRATEGIES.md #4: CR stays deferred unless this shows real
    # conflict volume; near-zero here is itself informative).
    polarity_conflicts: int = 0
    # Evidence-utilization telemetry (from PillarFormulaScore.breakdown, when
    # use_saturation=True; None on the linear path or when no breakdown exists).
    coverage_e: Optional[float] = None
    coverage_s: Optional[float] = None
    coverage_g: Optional[float] = None
    evidence_mass_e: Optional[float] = None
    evidence_mass_s: Optional[float] = None
    evidence_mass_g: Optional[float] = None
    # Phase 4 --verify outputs (estimate_verifier.py; ensemble scorer only).
    # verdict_*: 'skipped'|'passed'|'passed_after_retry'|'refuted'|'inconclusive'.
    # None when --verify wasn't used -- distinguishes "not verified" from any
    # real verdict.
    verdict_e: Optional[str] = None
    verdict_s: Optional[str] = None
    verdict_g: Optional[str] = None
    critic_calls: int = 0
    signals_count: int = 0
    metadata_source: Optional[str] = None
    resolved_country: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None
    # ground truth copied in for the report
    truth_e: Optional[float] = None
    truth_s: Optional[float] = None
    truth_g: Optional[float] = None
    truth_total: Optional[float] = None

    @property
    def pred_total(self) -> Optional[float]:
        parts = [p for p in (self.pred_e, self.pred_s, self.pred_g) if p is not None]
        # Match build_esg_json weighting (E .40 / S .35 / G .25) when all present.
        if self.pred_e is not None and self.pred_s is not None and self.pred_g is not None:
            return 0.40 * self.pred_e + 0.35 * self.pred_s + 0.25 * self.pred_g
        return sum(parts) / len(parts) if parts else None


def _cache_path(cache_dir: Path, name: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in name.lower())[:80]
    return cache_dir / f"{safe}.json"


def _write_cache(cpath: Optional[Path], row: BacktestRow) -> None:
    if not (cpath and row.ok):
        return
    try:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({
            "pred_e": row.pred_e, "pred_s": row.pred_s, "pred_g": row.pred_g,
            "spread_e": row.spread_e, "spread_s": row.spread_s, "spread_g": row.spread_g,
            "confidence_e": row.confidence_e, "confidence_s": row.confidence_s, "confidence_g": row.confidence_g,
            "gate_e": row.gate_e, "gate_s": row.gate_s, "gate_g": row.gate_g,
            "low_e": row.low_e, "low_s": row.low_s, "low_g": row.low_g,
            "high_e": row.high_e, "high_s": row.high_s, "high_g": row.high_g,
            "claims_dropped": row.claims_dropped, "claims_capped": row.claims_capped,
            "polarity_conflicts": row.polarity_conflicts,
            "coverage_e": row.coverage_e, "coverage_s": row.coverage_s, "coverage_g": row.coverage_g,
            "evidence_mass_e": row.evidence_mass_e, "evidence_mass_s": row.evidence_mass_s,
            "evidence_mass_g": row.evidence_mass_g,
            "verdict_e": row.verdict_e, "verdict_s": row.verdict_s, "verdict_g": row.verdict_g,
            "critic_calls": row.critic_calls,
            "signals_count": row.signals_count, "metadata_source": row.metadata_source,
            "resolved_country": row.resolved_country, "ok": row.ok,
        }), encoding="utf-8")
    except Exception as exc:
        log.debug("cache write failed: %s", exc)


def _estimate_one_via_graph(truth: TruthRecord, with_evaluator: bool, row: BacktestRow, cpath: Optional[Path]) -> BacktestRow:
    """
    Phase-0 verification path: same estimation task as estimate_one, but routed
    through agentic_estimation.graph's LangGraph scaffold instead of calling
    signal_agent/scoring_agent directly. with_evaluator is always effectively
    True here since the graph's dry path always runs the evaluator node — this
    matches what the scaffold actually does end-to-end.
    """
    from agentic_estimation.graph import run_company_dry_graph

    try:
        industry = truth.industry or ""
        _p(f"  [{truth.name}] (via graph) running full dry pipeline...")
        result = run_company_dry_graph(truth.name, industry=industry, country=truth.country)
        if result.error or result.e_score is None:
            row.error = result.error or "graph pipeline returned no score"
            _p(f"  [{truth.name}] FAILED: {row.error}")
            return row

        row.pred_e = result.e_score
        row.pred_s = result.s_score
        row.pred_g = result.g_score
        row.signals_count = result.signals_count
        row.resolved_country = result.country
        row.ok = True
        _p(f"  [{truth.name}] pred E={result.e_score:.0f} S={result.s_score:.0f} G={result.g_score:.0f} (via graph)")
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        log.warning("[%s] graph estimate failed: %s", truth.name, row.error)
        _p(f"  [{truth.name}] FAILED: {row.error}")

    _write_cache(cpath, row)
    return row


def _gather_and_score_formula(truth: TruthRecord, row: BacktestRow, capture: Optional[dict] = None):
    """
    Shared by _estimate_one_via_formula and _estimate_one_via_ensemble: gather
    evidence from ALL Layer-1 collectors (signal, governance, facility -- the
    old direct-call harness path only used signal_agent; the extractors need
    governance/facility signals too), tag it into typed claims via the E/S/G
    LLM extractors, add the deterministic Climate TRACE anchor claims, then
    compute scores with formula_estimator.py (which itself now blends in a
    real bcorp peer-anchor contribution when evidence is thin -- see
    peer_anchor.py). No LLM asked to output a 0-100 score anywhere in this path.

    Returns (signals, metadata, country, claims, formula_scores) so callers
    needing the same signals for a second estimator (e.g. the holistic LLM
    vote in the ensemble path) don't re-gather them.

    capture: optional dict the caller pre-creates; when given, this function
    stuffs "raw_claims" (post-extraction, PRE-Tier-0), "flags" (Tier-0's
    validate_claims output), and "kept_claims" (post-Tier-0) into it -- lets
    the ablation/route-comparison harness (calibration/ablation_replay.py)
    freeze evidence at the pre-Tier-0 stage so a "no_tier0" variant can be
    replayed later without re-gathering. None (default): no-op, zero behavior
    change for the two existing callers.
    """
    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.governance_collector import fetch_governance_signals
    from agentic_estimation.layer_1.facility_extractor import fetch_facility_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    from agentic_estimation.layer_2.pillar_extractors import extract_all_claims
    from agentic_estimation.layer_2.climate_trace_anchor import ct_anchor_claims
    from agentic_estimation.layer_3.formula_estimator import compute_formula_scores

    industry = truth.industry or ""
    _p(f"  [{truth.name}] gathering signals (signal+governance+facility)...")
    signals = {}
    # truth.country (ground truth's own real country) is known upfront -- pass
    # it straight into the signal fetch so a mapped country gets its localized
    # ESG query on the FIRST gather, not just on a later re-run once metadata
    # resolves a country. Falls back to metadata's resolved country below only
    # when the ground truth itself has no country (metadata isn't known yet
    # at fetch time either way, so this is the best country hint available).
    signals.update(fetch_company_signals(truth.name, industry, country=truth.country))
    signals.update(fetch_governance_signals(truth.name))
    signals.update(fetch_facility_signals(truth.name, industry))
    # Sustainability-report PDF text, from documents already downloaded to
    # raw_esg_data/ (report_coverage.py + the Wayback recovery). Purely local:
    # no network, no rate-limit exposure, ~10s of parsing on a cache miss and
    # ~0 on a hit.
    #
    # ADDED LAST, DELIBERATELY. pillar_extractors._signals_block iterates
    # signals in insertion order and `break`s -- not `continue`s -- once the
    # cumulative prompt passes _MAX_TOTAL_CHARS. Anything inserted after a
    # large source is therefore at risk of being dropped from the prompt with
    # no log line. Report text is the biggest source we have, so it goes last,
    # where it can only cost itself. Do not move this above the three
    # collectors.
    if _USE_REPORT_PDF:
        try:
            from agentic_estimation.layer_1.report_collector import fetch_report_signals
            signals.update(fetch_report_signals(truth.name))
        except Exception as exc:      # a bad PDF must never kill a company
            _p(f"  [{truth.name}] report signals failed: {type(exc).__name__}: {exc}")
    metadata = get_company_metadata(truth.name)
    # Seed the peer-matching sector into metadata as well as passing it explicitly
    # below. formula_estimator falls back to metadata["industry"] when no sector= is
    # given, and ablation_replay._dump_one takes exactly that path -- so without this,
    # replayed dumps would silently use a different (worse) sector vocabulary than the
    # live scoring run. Only fills a gap; never overwrites a resolved value.
    if truth.peer_sector and not metadata.get("industry"):
        metadata["industry"] = truth.peer_sector
    row.signals_count = len(signals)
    row.metadata_source = metadata.get("source")

    country = truth.country or metadata.get("country")
    row.resolved_country = country

    _p(f"  [{truth.name}] {len(signals)} signals, extracting claims...")
    claims = extract_all_claims(truth.name, signals, metadata)
    claims += ct_anchor_claims(truth.name, country=country)
    if capture is not None:
        capture["raw_claims"] = list(claims)

    from agentic_estimation.layer_2.claim_validators import validate_claims
    n_before = len(claims)
    claims, flags = validate_claims(claims, signals=signals, country=country)
    if capture is not None:
        capture["flags"] = list(flags)
        capture["kept_claims"] = list(claims)
    row.claims_dropped = n_before - len(claims)
    row.claims_capped = sum(1 for f in flags if f.action == "capped")
    row.polarity_conflicts = sum(1 for f in flags if f.rule == "polarity_consistency")
    if flags:
        _p(f"  [{truth.name}] Tier-0: {row.claims_dropped} claim(s) dropped, {row.claims_capped} capped")

    # truth_source intentionally NOT passed here (see agentic_estimation/
    # layer_3/peer_anchor.py's "BCORP REMOVAL" note) -- this harness can
    # backtest against EITHER source (run_backtest(source=...)), and
    # TruthRecord doesn't carry which one down to this function today. That
    # used to mean a bcorp backtest could leak bcorp peer votes regardless
    # of intent; it's now moot because find_peers()'s own include_bcorp
    # default is False, so bcorp peers don't leak in here even without a
    # truth_source restriction. If bcorp peer-anchor tiers are ever
    # re-enabled, thread truth_source=source through estimate_one ->
    # _estimate_one_via_formula/_estimate_one_via_ensemble -> here first.
    formula_scores = compute_formula_scores(
        claims, country, metadata, company_name=truth.name, sector=truth.peer_sector, signals=signals,
        use_saturation=_USE_SATURATION, sat_params=_SAT_PARAMS,
    )

    # Evidence-utilization telemetry (only when a saturation breakdown exists --
    # None on the linear --no-saturation path).
    for pillar, cov_attr, mass_attr in (("E", "coverage_e", "evidence_mass_e"),
                                         ("S", "coverage_s", "evidence_mass_s"),
                                         ("G", "coverage_g", "evidence_mass_g")):
        bd = getattr(formula_scores.get(pillar), "breakdown", None)
        if bd is not None:
            setattr(row, cov_attr, bd.coverage)
            setattr(row, mass_attr, bd.evidence_mass)

    return signals, metadata, country, claims, formula_scores


def _estimate_one_via_formula(truth: TruthRecord, row: BacktestRow, cpath: Optional[Path]) -> BacktestRow:
    """
    Phase 2 scorer path: formula only, no LLM asked to output a 0-100 score
    anywhere in this path. Nothing persisted (no company_id available/needed
    here) -- consistent with the harness's "writes nothing" contract.
    """
    try:
        signals, metadata, country, claims, scores = _gather_and_score_formula(truth, row)
        row.pred_e = scores["E"].score
        row.pred_s = scores["S"].score
        row.pred_g = scores["G"].score
        row.ok = True

        def _fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        _p(f"  [{truth.name}] (formula) pred E={row.pred_e:.0f} S={row.pred_s:.0f} G={row.pred_g:.0f}"
           f"  |  truth E={_fmt(truth.truth_e)} S={_fmt(truth.truth_s)} G={_fmt(truth.truth_g)}"
           f"  ({len(claims)} claims, baseline_source={scores['E'].baseline_source})")
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        log.warning("[%s] formula estimate failed: %s", truth.name, row.error)
        _p(f"  [{truth.name}] FAILED: {row.error}")

    _write_cache(cpath, row)
    return row


def _estimate_one_via_ensemble(truth: TruthRecord, row: BacktestRow, cpath: Optional[Path]) -> BacktestRow:
    """
    Phase 3 scorer path: formula (which already blends country baseline +
    real claims + real peer statistics) reconciled with ONE additional
    independent vote -- the demoted holistic LLM scorer -- via reconcile.py.
    Spread between the two becomes an honest uncertainty signal
    (row.spread_e/s/g) instead of an asserted one. Nothing persisted.
    """
    from agentic_estimation.layer_3.holistic_estimator import holistic_vote
    from agentic_estimation.layer_3.reconcile import reconcile_all
    from agentic_estimation.layer_3.confidence_gate import qc_assess, gate

    try:
        signals, metadata, country, claims, formula_scores = _gather_and_score_formula(truth, row)

        _p(f"  [{truth.name}] (ensemble) holistic LLM vote...")
        holistic = holistic_vote(truth.name, truth.industry or "", country, signals, metadata)

        reconciled = reconcile_all(formula_scores, holistic)
        row.pred_e = reconciled["E"].score
        row.pred_s = reconciled["S"].score
        row.pred_g = reconciled["G"].score
        row.spread_e = reconciled["E"].spread
        row.spread_s = reconciled["S"].spread
        row.spread_g = reconciled["G"].spread
        row.confidence_e = reconciled["E"].confidence
        row.confidence_s = reconciled["S"].confidence
        row.confidence_g = reconciled["G"].confidence

        qc = qc_assess(formula_scores)
        gated = gate(reconciled, qc)
        row.gate_e, row.low_e, row.high_e = gated["E"].mode, gated["E"].low, gated["E"].high
        row.gate_s, row.low_s, row.high_s = gated["S"].mode, gated["S"].low, gated["S"].high
        row.gate_g, row.low_g, row.high_g = gated["G"].mode, gated["G"].low, gated["G"].high

        if _VERIFY:
            from agentic_estimation.layer_4.estimate_verifier import verify_reconciled
            _p(f"  [{truth.name}] (verify) running critic panel where gated...")
            verified = verify_reconciled(
                truth.name, reconciled, formula_scores, holistic, claims, signals, metadata, country,
            )
            # verify_reconciled recomputes qc_assess/gate internally (composes
            # them, doesn't replace them) -- overwrite row.gate_*/low_*/high_*
            # with its output so a retry that changed the routing is reflected,
            # not just the pre-verification gate snapshot above.
            row.gate_e, row.low_e, row.high_e = verified["E"].mode, verified["E"].low, verified["E"].high
            row.gate_s, row.low_s, row.high_s = verified["S"].mode, verified["S"].low, verified["S"].high
            row.gate_g, row.low_g, row.high_g = verified["G"].mode, verified["G"].low, verified["G"].high
            row.verdict_e, row.verdict_s, row.verdict_g = (
                verified["E"].verdict, verified["S"].verdict, verified["G"].verdict,
            )
            row.critic_calls = sum(v.critic_calls for v in verified.values())
            _p(f"  [{truth.name}] verdicts: E={row.verdict_e} S={row.verdict_s} G={row.verdict_g}"
               f"  (critic_calls={row.critic_calls})")

        row.ok = True

        def _fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        _p(f"  [{truth.name}] (ensemble) pred E={row.pred_e:.0f} S={row.pred_s:.0f} G={row.pred_g:.0f}"
           f"  |  truth E={_fmt(truth.truth_e)} S={_fmt(truth.truth_s)} G={_fmt(truth.truth_g)}"
           f"  (n_votes E={reconciled['E'].n_votes}/S={reconciled['S'].n_votes}/G={reconciled['G'].n_votes},"
           f" confidence E={reconciled['E'].confidence})")
        _p(f"  [{truth.name}] gate: E={row.gate_e} S={row.gate_s} G={row.gate_g}")
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        log.warning("[%s] ensemble estimate failed: %s", truth.name, row.error)
        _p(f"  [{truth.name}] FAILED: {row.error}")

    _write_cache(cpath, row)
    return row


def estimate_one(
    truth: TruthRecord,
    with_evaluator: bool,
    cache_dir: Optional[Path],
    via_graph: bool = False,
    scorer: str = "llm",
) -> BacktestRow:
    """
    Run the pipeline's scoring path for one company WITHOUT its ground truth.
    Results are disk-cached by company name so re-runs during development are cheap.

    scorer:
        "llm"     (default) — signals → metadata → score_company_sync →
                  (optional) evaluator. The original single-shot LLM scorer.
        "formula" — Phase 2 path: signals from all Layer-1 collectors →
                  pillar_extractors (LLM tags evidence, never scores) +
                  climate_trace_anchor (deterministic) → formula_estimator
                  (deterministic score). --with-evaluator is ignored under
                  this scorer -- Phase 2 isolates the formula's own skill.

    via_graph: route estimation through agentic_estimation.graph's LangGraph
    scaffold (Phase 0) instead of calling signal_agent/scoring_agent directly.
    Used to confirm the orchestration rebuild alone doesn't regress the
    calibration baseline before any estimation logic changes (Phase 0
    verification step in the rebuild plan). Not compatible with scorer="formula"
    (the graph's formula path is wired in Step 6, not here).
    """
    row = BacktestRow(
        name=truth.name, country=truth.country, industry=truth.industry,
        truth_e=truth.truth_e, truth_s=truth.truth_s,
        truth_g=truth.truth_g, truth_total=truth.truth_total,
    )

    # Disk cache (predictions only — never ground truth). Graph-routed and
    # formula/ensemble-scored runs each get their own cache namespace so
    # predictions from different scorers never mix. v5 saturation is the
    # default, so it uses the plain namespace; the legacy linear A/B path
    # (--no-saturation) gets a distinct 'linear_' prefix so the two reductions'
    # predictions never collide. 't0_' prefix added 2026-07-21 when Tier-0
    # claim validators + the Confidence Gate landed -- both change what
    # formula/ensemble predictions actually ARE (dropped/capped claims, gated
    # low/high), so pre-Tier-0 cache entries must never be silently reused.
    cache_prefix = "graph_" if via_graph else {"formula": "formula_", "ensemble": "ensemble_"}.get(scorer, "")
    if not _USE_SATURATION:
        cache_prefix = "linear_" + cache_prefix
    if scorer in ("formula", "ensemble"):
        cache_prefix = "t0_" + cache_prefix
    if scorer == "ensemble" and _VERIFY:
        # A successful critic retry can change a pillar's claims (and
        # therefore its score) relative to the same company scored without
        # --verify -- these must never share a cache namespace with
        # unverified ensemble predictions.
        cache_prefix = "verified_" + cache_prefix
    if _USE_REPORT_PDF and scorer in ("formula", "ensemble"):
        # Report PDF text adds a whole evidence source, so a company scored
        # with it is a DIFFERENT prediction from the same company scored
        # without -- exactly the situation the 't0_' prefix above was added
        # for. Without this, the with/without A/B that justifies the feature
        # would compare fresh predictions against stale cached ones and report
        # a difference of zero.
        cache_prefix = "rpdf_" + cache_prefix
    cpath = _cache_path(cache_dir, cache_prefix + truth.name) if cache_dir else None
    if cpath and cpath.exists():
        try:
            cached = json.loads(cpath.read_text(encoding="utf-8"))
            row.pred_e = cached.get("pred_e")
            row.pred_s = cached.get("pred_s")
            row.pred_g = cached.get("pred_g")
            row.spread_e = cached.get("spread_e")
            row.spread_s = cached.get("spread_s")
            row.spread_g = cached.get("spread_g")
            row.confidence_e = cached.get("confidence_e")
            row.confidence_s = cached.get("confidence_s")
            row.confidence_g = cached.get("confidence_g")
            row.gate_e = cached.get("gate_e")
            row.gate_s = cached.get("gate_s")
            row.gate_g = cached.get("gate_g")
            row.low_e = cached.get("low_e")
            row.low_s = cached.get("low_s")
            row.low_g = cached.get("low_g")
            row.high_e = cached.get("high_e")
            row.high_s = cached.get("high_s")
            row.high_g = cached.get("high_g")
            row.claims_dropped = cached.get("claims_dropped", 0)
            row.claims_capped = cached.get("claims_capped", 0)
            row.polarity_conflicts = cached.get("polarity_conflicts", 0)
            row.coverage_e = cached.get("coverage_e")
            row.coverage_s = cached.get("coverage_s")
            row.coverage_g = cached.get("coverage_g")
            row.evidence_mass_e = cached.get("evidence_mass_e")
            row.evidence_mass_s = cached.get("evidence_mass_s")
            row.evidence_mass_g = cached.get("evidence_mass_g")
            row.verdict_e = cached.get("verdict_e")
            row.verdict_s = cached.get("verdict_s")
            row.verdict_g = cached.get("verdict_g")
            row.critic_calls = cached.get("critic_calls", 0)
            row.signals_count = cached.get("signals_count", 0)
            row.metadata_source = cached.get("metadata_source")
            row.resolved_country = cached.get("resolved_country")
            row.ok = cached.get("ok", False)
            if row.ok:
                _p(f"  [{truth.name}] cache hit (E={row.pred_e} S={row.pred_s} G={row.pred_g})")
                return row
        except Exception:
            pass  # fall through to recompute

    if scorer == "ensemble":
        return _estimate_one_via_ensemble(truth, row, cpath)

    if scorer == "formula":
        return _estimate_one_via_formula(truth, row, cpath)

    if via_graph:
        return _estimate_one_via_graph(truth, with_evaluator, row, cpath)

    from agentic_estimation.layer_1.signal_agent import fetch_company_signals
    from agentic_estimation.layer_1.company_metadata import get_company_metadata
    from agentic_estimation.layer_3.scoring_agent import score_company_sync

    try:
        industry = truth.industry or ""
        _p(f"  [{truth.name}] gathering web signals...")
        signals = fetch_company_signals(truth.name, industry)
        metadata = get_company_metadata(truth.name)
        row.signals_count = len(signals)
        row.metadata_source = metadata.get("source")
        _p(f"  [{truth.name}] {len(signals)} signals, metadata={metadata.get('source') or 'none'} — scoring...")

        score = score_company_sync(
            truth.name, industry=industry, country=truth.country,
            signals=signals, metadata=metadata,
        )
        if score is None:
            row.error = "scoring returned None"
            _p(f"  [{truth.name}] FAILED: scoring returned None")
            return row

        if with_evaluator:
            from agentic_estimation.layer_4.evaluator_agent import evaluate_company_sync
            ev = evaluate_company_sync(score, signals, industry=industry)
            if ev is not None:
                score = ev.final_score

        row.pred_e = score.e_score
        row.pred_s = score.s_score
        row.pred_g = score.g_score
        row.resolved_country = score.country
        row.ok = True

        def _fmt(v):
            return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"
        _p(f"  [{truth.name}] pred E={score.e_score:.0f} S={score.s_score:.0f} G={score.g_score:.0f}"
           f"  |  truth E={_fmt(truth.truth_e)} S={_fmt(truth.truth_s)} G={_fmt(truth.truth_g)}")
    except Exception as exc:
        row.error = f"{type(exc).__name__}: {exc}"
        log.warning("[%s] estimate failed: %s", truth.name, row.error)
        _p(f"  [{truth.name}] FAILED: {row.error}")

    _write_cache(cpath, row)
    return row


# ── Metrics ──────────────────────────────────────────────────────────────────

def _spearman(df: pd.DataFrame, pred_col: str, truth_col: str) -> tuple[Optional[float], int]:
    """Spearman ρ between two columns over rows where both are present. Returns (rho, n)."""
    sub = df[[pred_col, truth_col]].dropna()
    # Need variance on both sides and >=3 points for a meaningful coefficient
    if len(sub) < 3 or sub[pred_col].nunique() < 2 or sub[truth_col].nunique() < 2:
        return None, len(sub)
    # Spearman = Pearson on the rank-transformed columns (avoids the scipy dependency
    # pandas' method="spearman" pulls in).
    rho = sub[pred_col].rank().corr(sub[truth_col].rank(), method="pearson")
    return (round(float(rho), 3) if pd.notna(rho) else None), len(sub)


def _percentile_mae(df: pd.DataFrame, pred_col: str, truth_col: str) -> Optional[float]:
    """Convert both columns to within-sample percentile ranks (0-100), return MAE."""
    sub = df[[pred_col, truth_col]].dropna()
    if len(sub) < 3:
        return None
    p = sub[pred_col].rank(pct=True) * 100
    t = sub[truth_col].rank(pct=True) * 100
    return round(float((p - t).abs().mean()), 1)


def compute_report(df: pd.DataFrame) -> dict:
    """Build the aggregate report card from the per-company backtest DataFrame."""
    ok = df[df["ok"]].copy()
    ok["pred_total"] = ok.apply(
        lambda r: (0.40 * r.pred_e + 0.35 * r.pred_s + 0.25 * r.pred_g)
        if pd.notna(r.pred_e) and pd.notna(r.pred_s) and pd.notna(r.pred_g) else None,
        axis=1,
    )

    def _block(frame: pd.DataFrame) -> dict:
        pairs = [("E", "pred_e", "truth_e"), ("S", "pred_s", "truth_s"),
                 ("G", "pred_g", "truth_g"), ("Total", "pred_total", "truth_total")]
        res = {}
        for label, pc, tc in pairs:
            if tc not in frame or frame[tc].dropna().empty:
                continue
            rho, n = _spearman(frame, pc, tc)
            res[label] = {"spearman": rho, "n": n, "pct_mae": _percentile_mae(frame, pc, tc)}
        return res

    with_sig = ok[ok["signals_count"] > 0]
    report = {
        "n_total": int(len(df)),
        "n_ok": int(len(ok)),
        "n_with_signals": int(len(with_sig)),
        "n_failed": int((~df["ok"]).sum()),
        "overall": _block(ok),
        "evidence_only": _block(with_sig),  # companies the estimator actually had signals for
    }

    # Ensemble-only diagnostic (NOT a gate): does the reported spread actually
    # correlate with real error? If uncertainty-by-construction (Phase 3's
    # premise) works, companies with a wide formula/holistic disagreement
    # SHOULD tend to have larger |pred - truth| rank error too.
    if "spread_e" in ok.columns and ok["spread_e"].notna().any():
        spread_stats = {}
        for label, sc, pc, tc in [("E", "spread_e", "pred_e", "truth_e"), ("S", "spread_s", "pred_s", "truth_s"),
                                    ("G", "spread_g", "pred_g", "truth_g")]:
            sub = ok[[sc, pc, tc]].dropna()
            if len(sub) < 3:
                continue
            spread_stats[label] = {
                "mean_spread": round(float(sub[sc].mean()), 1),
                "median_spread": round(float(sub[sc].median()), 1),
                "n_with_spread": int(len(sub)),
            }
            rank_err = (sub[pc].rank() - sub[tc].rank()).abs()
            if sub[sc].nunique() >= 2 and rank_err.nunique() >= 2:
                corr = sub[sc].rank().corr(rank_err.rank())
                spread_stats[label]["spread_vs_rank_error_corr"] = round(float(corr), 3) if pd.notna(corr) else None
        report["spread_diagnostic"] = spread_stats

    # Confidence Gate split (ensemble scorer only, confidence_gate.py): the
    # BOTH-views report the user asked for. "overall" above is UNCHANGED
    # (every ok row, point scores as always -- so it's comparable to every
    # prior CSV/report this session). "gated" recomputes Spearman per pillar
    # over ONLY the rows the gate called 'point' for that pillar, plus counts
    # and the names of range-flagged (needs_review) companies -- the honest
    # split between "companies we had real grounds to score" and "companies
    # we're flagging as too thin for a point estimate".
    if "gate_e" in ok.columns and ok["gate_e"].notna().any():
        gated_block = {}
        pillar_cols = [("E", "pred_e", "truth_e", "gate_e"), ("S", "pred_s", "truth_s", "gate_s"),
                       ("G", "pred_g", "truth_g", "gate_g")]
        for label, pc, tc, gc in pillar_cols:
            if gc not in ok.columns:
                continue
            point_rows = ok[ok[gc] == "point"]
            range_rows = ok[ok[gc] == "range"]
            entry = {"n_point": int(len(point_rows)), "n_range": int(len(range_rows))}
            if tc in ok and not point_rows.empty:
                rho, n = _spearman(point_rows, pc, tc)
                entry["spearman"] = rho
                entry["n"] = n
                entry["pct_mae"] = _percentile_mae(point_rows, pc, tc)
            entry["range_flagged_companies"] = range_rows["name"].tolist() if "name" in range_rows else []
            gated_block[label] = entry
        report["gated"] = gated_block

    # Confidence LABEL calibration (reconcile.py's raw 'high'/'medium'/'low' --
    # distinct from the point/range gate split above, which only tells you
    # WHETHER a score was flagged, not whether the three-way label itself is
    # meaningful). This answers the actual question: does 'high' really mean
    # lower error than 'medium'/'low', or is the label decorative? Reports
    # BOTH percentile-rank MAE (comparable to the rest of this report) and
    # raw |pred-truth| on the native 0-100 scale (the number that answers
    # "how far off is a 'high confidence' score, in points" -- rank error
    # alone can hide this on a small sample where ranks compress).
    if "confidence_e" in ok.columns and ok["confidence_e"].notna().any():
        calib_block = {}
        pillar_cols = [("E", "pred_e", "truth_e", "confidence_e"),
                       ("S", "pred_s", "truth_s", "confidence_s"),
                       ("G", "pred_g", "truth_g", "confidence_g")]
        for label, pc, tc, cc in pillar_cols:
            if cc not in ok.columns or tc not in ok.columns:
                continue
            by_bucket = {}
            for bucket in ("high", "medium", "low"):
                sub = ok[ok[cc] == bucket][[pc, tc]].dropna()
                entry = {"n": int(len(sub))}
                if len(sub) >= 3:
                    entry["mean_abs_error"] = round(float((sub[pc] - sub[tc]).abs().mean()), 1)
                    entry["median_abs_error"] = round(float((sub[pc] - sub[tc]).abs().median()), 1)
                    rank_err = (sub[pc].rank() - sub[tc].rank()).abs()
                    entry["mean_rank_error"] = round(float(rank_err.mean()), 1)
                elif len(sub) > 0:
                    # Too few rows for a stable mean -- still report the raw
                    # errors so a n=1/2 bucket isn't silently hidden as empty.
                    entry["abs_errors"] = [round(float(v), 1) for v in (sub[pc] - sub[tc]).abs().tolist()]
                by_bucket[bucket] = entry
            # Monotonicity check: is high <= medium <= low mean error, as the
            # label claims? False/None (not enough buckets populated) is
            # itself the finding -- it means the labels aren't discriminating.
            means = {b: by_bucket[b].get("mean_abs_error") for b in ("high", "medium", "low")}
            populated = {b: v for b, v in means.items() if v is not None}
            if len(populated) >= 2:
                ordered = [populated[b] for b in ("high", "medium", "low") if b in populated]
                calib_block_monotonic = all(ordered[i] <= ordered[i + 1] for i in range(len(ordered) - 1))
            else:
                calib_block_monotonic = None
            calib_block[label] = {"buckets": by_bucket, "monotonic_high_to_low": calib_block_monotonic}
        report["confidence_calibration"] = calib_block

    # Phase 4 --verify block (estimate_verifier.py): per-pillar verdict
    # counts (skipped/passed/passed_after_retry/refuted), % skipped (the
    # cost-refined trigger's effectiveness -- most companies should skip),
    # needs_review rate, and the diagnostic split of needs_review-flagged
    # vs unflagged mean rank error (flagged rows SHOULD be worse -- not a
    # gate, same framing as spread_diagnostic above).
    if "verdict_e" in ok.columns and ok["verdict_e"].notna().any():
        verify_block = {}
        pillar_cols = [("E", "pred_e", "truth_e", "verdict_e", "gate_e"),
                       ("S", "pred_s", "truth_s", "verdict_s", "gate_s"),
                       ("G", "pred_g", "truth_g", "verdict_g", "gate_g")]
        for label, pc, tc, vc, gc in pillar_cols:
            if vc not in ok.columns:
                continue
            verified_rows = ok[ok[vc].notna()]
            n_verified = int(len(verified_rows))
            counts = verified_rows[vc].value_counts().to_dict() if n_verified else {}
            entry = {"n_verified": n_verified, "verdict_counts": {str(k): int(v) for k, v in counts.items()}}
            if n_verified:
                entry["pct_skipped"] = round(100.0 * counts.get("skipped", 0) / n_verified, 1)
            if gc in verified_rows.columns and tc in verified_rows:
                flagged = verified_rows[verified_rows[gc] == "range"]
                unflagged = verified_rows[verified_rows[gc] == "point"]
                for name, sub in (("flagged", flagged), ("unflagged", unflagged)):
                    sub2 = sub[[pc, tc]].dropna()
                    if len(sub2) >= 3:
                        rank_err = (sub2[pc].rank() - sub2[tc].rank()).abs()
                        entry[f"mean_rank_error_{name}"] = round(float(rank_err.mean()), 1)
            verify_block[label] = entry
        report["verify"] = verify_block

    # RANGE COVERAGE (does the range actually contain truth?): the API now
    # returns [low, high] as the only value, never the point score alone --
    # so the calibration question that matters isn't just "is the point close"
    # (Spearman/pct_MAE above, unaffected -- truth has no range to rank
    # against, so ranking stays point-based), it's "would a user who only saw
    # the range have been right." Coverage rate = % of companies where
    # truth_e/s/g actually falls inside [low_e, high_e] etc. If the range is
    # honest, coverage should roughly match the confidence it implies (e.g. a
    # range meant to be ~80% reliable should contain truth ~80% of the time);
    # too low means ranges are overconfident (too narrow), too high (near
    # 100%) means they're so wide they're not informative.
    if "low_e" in ok.columns and ok["low_e"].notna().any():
        coverage_block = {}
        pillar_cols = [("E", "low_e", "high_e", "truth_e"), ("S", "low_s", "high_s", "truth_s"),
                       ("G", "low_g", "high_g", "truth_g")]
        for label, lc, hc, tc in pillar_cols:
            if lc not in ok.columns or hc not in ok.columns or tc not in ok.columns:
                continue
            sub = ok[[lc, hc, tc]].dropna()
            if sub.empty:
                continue
            inside = (sub[tc] >= sub[lc]) & (sub[tc] <= sub[hc])
            coverage_block[label] = {
                "n": int(len(sub)),
                "coverage_pct": round(100.0 * float(inside.mean()), 1),
                "mean_width": round(float((sub[hc] - sub[lc]).mean()), 1),
                "median_width": round(float((sub[hc] - sub[lc]).median()), 1),
            }
        report["range_coverage"] = coverage_block

    return report


# ── Orchestration ────────────────────────────────────────────────────────────

def run_backtest(
    source: str = "bcorp",
    n: int = 20,
    seed: int = 42,
    workers: int = 3,
    with_evaluator: bool = False,
    cache_dir: Optional[Path] = _DEFAULT_CACHE,
    out_csv: Optional[Path] = None,
    console_verbose: bool = False,
    via_graph: bool = False,
    scorer: str = "llm",
) -> dict:
    if scorer not in ("llm", "formula", "ensemble"):
        raise ValueError(f"unknown scorer '{scorer}' (use 'llm', 'formula', or 'ensemble')")
    if scorer in ("formula", "ensemble") and via_graph:
        raise ValueError(f"--scorer {scorer} and --via-graph are not compatible yet")

    load_dotenv(_ENV_PATH, override=True)  # ensure downstream modules see real creds
    _attach_console_logging(console_verbose)  # mirror file logs to stdout (kept, not replaced)
    log_header(log, "Calibration Harness",
               source=source, n=n, workers=workers, evaluator=with_evaluator, via_graph=via_graph, scorer=scorer)

    loader = {"bcorp": load_bcorp_truth, "upright": load_upright_truth}.get(source)
    if loader is None:
        raise ValueError(f"unknown source '{source}' (use 'bcorp' or 'upright')")

    truths = loader(n, seed)
    log.info("Loaded %d ground-truth companies from %s", len(truths), source)
    _p(f"\n=== Calibration Harness: {len(truths)} companies from {source} "
       f"(workers={workers}, scorer={scorer}, via_graph={via_graph}) ===")
    _p(f"    File logs still writing to agentic_estimation/logs/*.log as usual.\n")

    rows: list[BacktestRow] = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(estimate_one, t, with_evaluator, cache_dir, via_graph, scorer): t for t in truths}
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                rows.append(BacktestRow(name=t.name, country=t.country, industry=t.industry,
                                        error=str(exc)))
                _p(f"  [{t.name}] EXCEPTION: {exc}")
            done += 1
            elapsed = time.monotonic() - t0
            _p(f"  --> progress {done}/{len(truths)}  ({elapsed:.0f}s elapsed)")
            log.info("progress %d/%d — %s", done, len(truths), t.name)

    df = pd.DataFrame([asdict(r) for r in rows])
    report = compute_report(df)
    report["source"] = source
    report["elapsed_s"] = round(time.monotonic() - t0, 1)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        report["csv"] = str(out_csv)

    return report


# ── Reporting ────────────────────────────────────────────────────────────────

def _print_report(report: dict) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print(f"  ESG Pipeline Calibration - ground truth: {report['source'].upper()}")
    print(sep)
    print(f"  companies: {report['n_total']}   scored ok: {report['n_ok']}   "
          f"with signals: {report['n_with_signals']}   failed: {report['n_failed']}")
    print(f"  elapsed: {report['elapsed_s']}s")
    if report.get("csv"):
        print(f"  per-company CSV: {report['csv']}")

    def _table(title: str, block: dict) -> None:
        print(f"\n  {title}")
        print(f"  {'pillar':<8}{'spearman':>10}{'pct_MAE':>10}{'n':>6}")
        print("  " + "-" * 34)
        if not block:
            print("  (insufficient data — need ≥3 companies with both pred & truth)")
            return
        for label in ("E", "S", "G", "Total"):
            if label not in block:
                continue
            b = block[label]
            rho = f"{b['spearman']:+.3f}" if b["spearman"] is not None else "  n/a"
            mae = f"{b['pct_mae']:.1f}" if b["pct_mae"] is not None else " n/a"
            print(f"  {label:<8}{rho:>10}{mae:>10}{b['n']:>6}")

    _table("ALL scored companies:", report["overall"])
    _table("EVIDENCE-ONLY (companies that had web signals):", report["evidence_only"])

    if report.get("spread_diagnostic"):
        print(f"\n  ENSEMBLE SPREAD DIAGNOSTIC (not a gate -- sanity check on uncertainty-by-construction):")
        print(f"  {'pillar':<8}{'mean':>8}{'median':>8}{'n':>6}{'spread-vs-error corr':>24}")
        print("  " + "-" * 54)
        for label, s in report["spread_diagnostic"].items():
            corr = s.get("spread_vs_rank_error_corr")
            corr_str = f"{corr:+.3f}" if corr is not None else "n/a"
            print(f"  {label:<8}{s['mean_spread']:>8.1f}{s['median_spread']:>8.1f}{s['n_with_spread']:>6}{corr_str:>24}")
        print(f"  (positive corr = wider disagreement DOES predict larger real error, as intended)")

    if report.get("gated"):
        print(f"\n  CONFIDENCE GATE SPLIT (confidence_gate.py -- 'overall' above always uses the")
        print(f"  point score; this shows the honest split into point-scored vs range-flagged):")
        print(f"  {'pillar':<8}{'spearman':>10}{'pct_MAE':>10}{'n_point':>9}{'n_range':>9}")
        print("  " + "-" * 46)
        for label in ("E", "S", "G"):
            g = report["gated"].get(label)
            if not g:
                continue
            rho = f"{g['spearman']:+.3f}" if g.get("spearman") is not None else "  n/a"
            mae = f"{g['pct_mae']:.1f}" if g.get("pct_mae") is not None else " n/a"
            print(f"  {label:<8}{rho:>10}{mae:>10}{g['n_point']:>9}{g['n_range']:>9}")
        for label in ("E", "S", "G"):
            g = report["gated"].get(label)
            if g and g.get("range_flagged_companies"):
                names = ", ".join(g["range_flagged_companies"])
                print(f"  {label} range-flagged (needs_review): {names}")

    if report.get("confidence_calibration"):
        print(f"\n  CONFIDENCE LABEL CALIBRATION (reconcile.py's 'high'/'medium'/'low' --")
        print(f"  does the label actually track real error, on the native 0-100 scale?):")
        print(f"  {'pillar':<8}{'bucket':<8}{'n':>4}{'mean_abs_err':>14}{'median_abs_err':>16}{'mean_rank_err':>15}")
        print("  " + "-" * 65)
        for label in ("E", "S", "G"):
            c = report["confidence_calibration"].get(label)
            if not c:
                continue
            for bucket in ("high", "medium", "low"):
                b = c["buckets"].get(bucket, {})
                n = b.get("n", 0)
                if n == 0:
                    print(f"  {label:<8}{bucket:<8}{0:>4}{'n/a':>14}{'n/a':>16}{'n/a':>15}")
                elif "mean_abs_error" in b:
                    print(f"  {label:<8}{bucket:<8}{n:>4}{b['mean_abs_error']:>14.1f}"
                          f"{b['median_abs_error']:>16.1f}{b['mean_rank_error']:>15.1f}")
                else:
                    errs = ", ".join(str(v) for v in b.get("abs_errors", []))
                    print(f"  {label:<8}{bucket:<8}{n:>4}  (n<3, raw errors: {errs})")
            mono = c.get("monotonic_high_to_low")
            mono_str = "YES (labels track error as claimed)" if mono is True \
                else "NO (labels do NOT reliably track error)" if mono is False \
                else "n/a (not enough populated buckets to judge)"
            print(f"    {label} monotonic high<=medium<=low error: {mono_str}")
        print(f"  (mean_abs_error is |pred-truth| in points, 0-100 scale -- this is the number")
        print(f"   that tells you whether trusting a 'high confidence' score is actually justified.)")

    if report.get("verify"):
        print(f"\n  PHASE 4 VERIFY (estimate_verifier.py -- critic panel gated on medium+QC-ok pillars):")
        print(f"  {'pillar':<8}{'n_verif':>8}{'pct_skip':>9}{'rank_err_flag':>14}{'rank_err_unflag':>16}")
        print("  " + "-" * 55)
        for label in ("E", "S", "G"):
            v = report["verify"].get(label)
            if not v:
                continue
            skip = f"{v['pct_skipped']:.1f}%" if v.get("pct_skipped") is not None else "n/a"
            rf = v.get("mean_rank_error_flagged")
            ru = v.get("mean_rank_error_unflagged")
            rf_str = f"{rf:.1f}" if rf is not None else "n/a"
            ru_str = f"{ru:.1f}" if ru is not None else "n/a"
            print(f"  {label:<8}{v['n_verified']:>8}{skip:>9}{rf_str:>14}{ru_str:>16}")
            print(f"    verdicts: {v['verdict_counts']}")
        print(f"  (rank_err_flag should be >= rank_err_unflag -- confirms needs_review actually")
        print(f"   marks the worse estimates. pct_skip is the cost-refined trigger's effectiveness.)")

    if report.get("range_coverage"):
        print(f"\n  RANGE COVERAGE (does [low, high] actually contain truth? -- the API now")
        print(f"  returns ranges only, so this is the honesty check for that output):")
        print(f"  {'pillar':<8}{'n':>4}{'coverage_%':>12}{'mean_width':>12}{'median_width':>14}")
        print("  " + "-" * 50)
        for label in ("E", "S", "G"):
            c = report["range_coverage"].get(label)
            if not c:
                continue
            print(f"  {label:<8}{c['n']:>4}{c['coverage_pct']:>11.1f}%{c['mean_width']:>12.1f}{c['median_width']:>14.1f}")
        print(f"  (coverage_% = % of companies where truth actually fell inside our range --")
        print(f"   too low means ranges are overconfident/too narrow, near 100% with a huge")
        print(f"   mean_width means they're wide enough to be uninformative.)")

    print(f"\n  Reading it: spearman +1.0 = estimator orders firms exactly like the")
    print(f"  assessor, 0 = no relationship, negative = inverted. pct_MAE = avg")
    print(f"  percentile-points the estimate is off. Trust EVIDENCE-ONLY as the")
    print(f"  estimator's real skill; the gap vs ALL shows the cost of missing signals.")
    print(f"{sep}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    ap = argparse.ArgumentParser(description="Backtest the ESG pipeline against B Corp / Upright ground truth.")
    ap.add_argument("--source", choices=["bcorp", "upright"], default="bcorp")
    ap.add_argument("--n", type=int, default=20, help="number of ground-truth companies")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (reproducible)")
    ap.add_argument("--workers", type=int, default=3, help="concurrent companies")
    ap.add_argument("--with-evaluator", action="store_true", help="include evaluator correction step")
    ap.add_argument("--no-cache", action="store_true", help="ignore/skip the disk prediction cache")
    ap.add_argument("--out", default=None, help="path to write per-company CSV")
    ap.add_argument("--verbose", "-v", action="store_true",
                     help="mirror full INFO-level agent logs to console (every source hit, not just warnings/results)")
    ap.add_argument("--via-graph", action="store_true",
                     help="route estimation through agentic_estimation.graph's LangGraph scaffold "
                          "instead of calling signal_agent/scoring_agent directly (Phase 0 verification)")
    ap.add_argument("--scorer", choices=["llm", "formula", "ensemble"], default="llm",
                     help="'llm' = original single-shot scorer (default); "
                          "'formula' = Phase 2 evidence-claims + deterministic formula path; "
                          "'ensemble' = Phase 3: formula reconciled with a demoted holistic LLM vote, "
                          "spread between them reported as an uncertainty signal")
    ap.add_argument("--no-saturation", action="store_true",
                     help="use the LEGACY linear final aggregation (baseline+sum(points)) "
                          "instead of the v5 saturation default. For A/B comparison only.")
    ap.add_argument("--sat-ak", default=None,
                     help="per-pillar (A,k) override for saturation tuning, e.g. "
                          "'E:40,1.0;S:30,1.5;G:35,1.0' (A symmetric; a_pos=a_neg=A). "
                          "Pillars omitted use module defaults.")
    ap.add_argument("--verify", action="store_true",
                     help="Phase 4: run the gated critic panel + bounded retry "
                          "(estimate_verifier.py) on top of the Confidence Gate. "
                          "Ensemble scorer only; adds real LLM cost on pillars "
                          "landing medium-confidence + QC-ok (see PHASE_4_PLAN.md).")
    ap.add_argument("--no-report-pdf", action="store_true",
                     help="exclude sustainability-report PDF text from the evidence "
                          "gather (report_collector.py). Default is to include it. "
                          "Use for the with/without A/B -- and pass --no-cache too, "
                          "since the two runs otherwise share a cache namespace.")
    args = ap.parse_args()

    global _USE_SATURATION, _SAT_PARAMS, _VERIFY, _USE_REPORT_PDF
    _USE_SATURATION = not args.no_saturation
    _VERIFY = args.verify
    _USE_REPORT_PDF = not args.no_report_pdf
    if _VERIFY and args.scorer != "ensemble":
        raise ValueError("--verify requires --scorer ensemble")
    if args.sat_ak:
        from agentic_estimation.layer_3.saturation_score import PillarSatParams
        _SAT_PARAMS = {}
        for chunk in args.sat_ak.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            pillar, ak = chunk.split(":")
            a_str, k_str = ak.split(",")
            a, k = float(a_str), float(k_str)
            _SAT_PARAMS[pillar.strip()] = PillarSatParams(a_pos=a, a_neg=a, k=k)

    report = run_backtest(
        source=args.source, n=args.n, seed=args.seed, workers=args.workers,
        with_evaluator=args.with_evaluator,
        cache_dir=None if args.no_cache else _DEFAULT_CACHE,
        out_csv=Path(args.out) if args.out else None,
        console_verbose=args.verbose,
        via_graph=args.via_graph,
        scorer=args.scorer,
    )
    _print_report(report)


if __name__ == "__main__":
    _cli()
