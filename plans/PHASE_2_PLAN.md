# Phase 2 Implementation Plan — Extractors + Formula Estimator

*The core accuracy fix: replace "LLM guesses a score" with "LLM tags evidence,
a deterministic formula computes the score."*

## Why

The current scorer (`agentic_estimation/scoring_agent.py`) asks one LLM call to
freehand E/S/G scores. Backtested against real ground truth (bcorp/upright, via
`calibration_harness.py`): Spearman ρ E +0.246, S −0.154, G +0.045, Total −0.132.
Negative/near-zero correlation = measurable guessing. Phase 2 builds the layer
that fixes this.

## What exists already (Phase 0–1 + Climate TRACE work, all verified live)

- `company_evidence_claims` table with the DB-enforced "no source, no claim"
  constraint (`db_migrations/002_evidence_claims.sql`).
- All Layer-1 collectors persist evidence text into `company_esg_signals`
  (id UUID PK, company_id, source VARCHAR(60), signal_text, gathered_at):
  signal_agent (news_api, google_news_rss, bhrrc, sbti, cdp, gri, wikipedia,
  sustainability_report, net_zero, controversies), governance_collector
  (gov_board_sec, gov_litigation_sec, gov_board_count, gov_board, gov_fines,
  gov_compliance, gov_litigation), facility_extractor (sec_10k_properties,
  facility_web). Collector entry points return `{source: text}` WITHOUT row ids
  — claim persistence must re-query to resolve `source_signal_id` FKs.
- `CORE_METRICS` in `metric_estimation_agent.py`: 17 metrics with
  pillar/kind/direction/intensity/benchmark bands — the numeric factor backbone.
- `country_baseline_agent.get_country_baseline(country)` — the formula's B term
  (210 countries, DB-cached).
- `ratio_estimator.estimate_missing_factors()` — deterministic peer-median
  back-fill for employee_count / annual_revenue / etc.
- **Climate TRACE grounding (new)**: `climate_trace_owners` (14,513),
  `climate_trace_owner_emissions` (harvest in progress, ~41%),
  `climate_trace_country_emissions` (18,900 rows, complete),
  `market_climate_trace_mapping` (25/9,580 mapped; full batch pending).
  Join path: companies → market_company_link → markets →
  market_climate_trace_mapping → CT sector/subsector emissions. Plus direct
  company-name → owner match for REAL facility emissions.
- `zen_client.call_with_prompt(...) -> {ok, raw, reasoning, error, ...}` and the
  last-balanced-JSON-object parse pattern (`market_climate_trace_mapper._extract_json`).

## Formula

```
pillar_score = clamp( B_country_pillar + Σ_i  w_i · c_i · δ_i ,  0, 100 )
```

- `B` = country baseline (World Bank), per pillar
- `w_i` = hand-set factor weight (registry; regression-fit later in Phase 5)
- `c_i` = claim confidence (0 when no evidence → factor contributes nothing)
- `δ_i` = signed delta in [−1, +1], computed per factor shape (below)

No evidence → score stays at country baseline. No more fake-confident 45s.

---

## Deliverables

| File | Role |
|---|---|
| `db_migrations/004_claim_method_dataset_lookup.sql` | new | extend `method` CHECK |
| `agentic_estimation/factor_registry.py` | new | single source of truth: factors, weights, shapes |
| `agentic_estimation/climate_trace_anchor.py` | new | deterministic CT-grounded claims |
| `agentic_estimation/pillar_extractors.py` | new | E/S/G LLM evidence taggers |
| `agentic_estimation/formula_estimator.py` | new | deterministic score computation |
| `agentic_estimation/calibration_harness.py` | modified | `--scorer formula` |
| `agentic_estimation/graph.py` | modified | new nodes, scorer routing |

---

## Step 1 — Migration 004 + `factor_registry.py`

**Migration:** `company_evidence_claims.method` CHECK currently allows only
`('extracted','peer_ratio_fallback','coarse_bucket')`. Add `'dataset_lookup'` for
deterministic Climate-TRACE-derived claims. Migrating (not reusing `'extracted'`)
keeps dataset claims distinguishable from LLM extractions — they're categorically
more trustworthy and Phase-5 weight fitting must be able to tell them apart.

**Registry:**

```python
@dataclass(frozen=True)
class Factor:
    key: str
    pillar: str              # 'E' | 'S' | 'G'
    weight: float            # pillar points of swing at c=1, |δ|=1
    delta_shape: str         # 'benchmark_band' | 'event'
    direction: str           # 'higher' | 'lower'
    metric: Optional[dict]   # the CORE_METRICS entry when benchmark-backed
    description: str         # one line; also injected into extractor prompts
```

- Numeric backbone **imported** from `CORE_METRICS` — never duplicated.
  `direction == "neutral"` entries (employee_count, annual_revenue) are context,
  not scored factors.
- Added qualitative factors (hand-set weights, documented; a fully-evidenced
  pillar can swing roughly ±30 around baseline):

| Factor | Pillar | Weight | Shape | Notes |
|---|---|---|---|---|
| net_zero_pledge | E | 5 | event | strength = credibility (target year, interim goals) |
| sbti_commitment | E | 6 | event | validated > committed (strength) |
| cdp_disclosure | E | 4 | event | |
| environmental_controversy | E | 10 | event | polarity −1, strength = severity |
| sector_emissions_intensity | E | 4 | event | CT sector anchor, polarity −1 |
| labor_controversy | S | 10 | event | |
| human_rights_incident | S | 12 | event | bhrrc-backed |
| workplace_safety | S | 6 | event | plus lost_time_injury_rate band from CORE_METRICS |
| regulatory_fines | G | 9 | event | strength scaled by magnitude vs company size |
| litigation | G | 7 | event | |
| compliance_certification | G | 5 | event | |

(CORE_METRICS supplies: scope 1/2/3 emissions w=8 each, renewable_energy_pct 6,
energy/water/waste bands, female_employees_pct/female_board_pct 6,
board_independence_pct 8, boolean policy factors 4–5.)

**Verify:** apply migration live; confirm a `dataset_lookup` insert passes and an
invalid method still rejects. Registry smoke test: all scored CORE_METRICS keys
wrapped, no key collisions, print the weight table.

## Step 2 — `climate_trace_anchor.py` (deterministic, no LLM)

- `ct_owner_match(company) -> Optional[(owner_id, matched_name)]` — normalise
  both sides (lowercase, strip punctuation and legal suffixes gmbh/inc/ltd/ag/
  corp/plc/llc/co/sa); accept ONLY exact normalised equality, or prefix match
  where the residue is purely legal-suffix tokens. **No edit-distance fuzzing**
  — with 14,513 owners, asserting the wrong company's real emissions at 0.85
  confidence is far worse than a miss.
- On hit: `SUM(emissions_quantity)` over the owner's latest harvested year
  (gas LIKE 'co2e%') → ONE claim: factor `scope_1_emissions`, value = tCO2e,
  confidence 0.85, method `dataset_lookup`, source_note = owner_id + facility
  count + year. Matched owner with zero harvested rows → NO claim (harvest is
  ~41% done; absence ≠ zero).
- Sector anchor (needs company_id): company → market_company_link → markets →
  `market_climate_trace_mapping` (mapping confidence ≥ 0.6) → the sector's
  emissions percentile among that country's sectors in
  `climate_trace_country_emissions` → claim `sector_emissions_intensity`,
  polarity −1, strength = percentile, confidence 0.25, method `dataset_lookup`.
- Country name → ISO3 via the World Bank Excel Metadata sheet already parsed by
  `country_baseline_agent` (same country vocabulary as the rest of the pipeline;
  pycountry is NOT installed — no new dependency).
- **Graceful degradation is a requirement**: no company_id / no market link /
  unmapped market / owner not yet harvested → fewer or zero claims, never an
  exception. The module improves automatically as the harvest and full mapper
  batch complete — zero code change needed.

**Verify live:** one true owner hit; one near-miss that must NOT match
(false-positive guard); one no-data company returning `[]`.

## Step 3 — `pillar_extractors.py` (LLM evidence tagger)

**One LLM call per pillar (3/company), not one combined call:**
fail-closed isolation (one parse failure loses one pillar, not all three);
smaller closed factor lists per prompt → better JSON compliance on the free
gateway; no max_tokens truncation risk. 3-way `ThreadPoolExecutor` inside
`extract_all_claims`.

```python
@dataclass
class ExtractedClaim:
    factor: str; pillar: str; polarity: int; strength: float; confidence: float
    value: Optional[float]; source_tag: str; reasoning: str
    method: str = "extracted"

def extract_pillar_claims(pillar, company, signals: dict[str, str], metadata=None)
    -> list[ExtractedClaim]              # PURE — no DB
def extract_all_claims(company, signals, metadata) -> list[ExtractedClaim]
async def persist_claims(company_id, claims, produced_by="pillar_extractor_v1") -> int
```

- `persist_claims` is the ONLY DB writer: re-queries
  `SELECT id, source FROM company_esg_signals WHERE company_id=...` to map
  source_tag → source_signal_id FK; unmapped tag → source_note fallback (the
  provenance CHECK is satisfied either way). Harness runs never persist.
- **Prompt structure** (per pillar): "You are an evidence tagger, NOT a scorer";
  signals as `[tag: xyz]` blocks (~1,500 chars each, ~12k total);
  **STEP 1 — explicit per-signal relevance gate** ("does this text actually
  discuss {pillar topics} for {company}? Extract NOTHING from irrelevant
  signals") — the structural fix for the Phase-1 yoga-page finding;
  STEP 2 — closed factor list with descriptions; numeric factors: "put the
  stated number in `value` with `unit` verbatim — do NOT normalise or score it";
  no supporting text → omit the factor entirely; end with JSON
  `{"claims":[...]}` parsed via the existing last-balanced-object scan.
  `call_with_prompt(max_tokens=2500)`.
- **Code validation, fails closed per claim:** factor in registry AND pillar
  matches; source_tag in the company's ACTUAL signal keys (hallucinated
  attribution → dropped); polarity coerced to {−1,0,1}; strength/confidence
  clamped [0,1]; unit conversion table (kt/Mt CO2e → t, GWh → GJ,
  "million" → 1e6); unknown unit → value nulled, claim degrades to event shape.

**Verify live, in-memory:** Nvidia (SEC-rich), Adidas (EU web-heavy), Blackmores
(must survive the yoga-page trap — irrelevant text → zero claims). Manual
inspection: valid factors, real source_tags, values verbatim not scored.

## Step 4 — `formula_estimator.py` (deterministic, no LLM)

```python
@dataclass
class Contribution:
    factor: str; weight: float; confidence: float; delta: float
    points: float; claim_reasoning: str; method: str

@dataclass
class PillarFormulaScore:
    pillar: str; baseline: float; score: float
    contributions: list[Contribution]     # full audit trail

def compute_formula_scores(claims, country, metadata) -> dict[str, PillarFormulaScore]
async def load_claims(company_id) -> list[ExtractedClaim]    # DB path for graph runs
```

- Baseline from `get_country_baseline(country)`; unresolvable country → 50,
  flagged in output.
- **δ rules:**
  - `benchmark_band` with value: intensity-normalise by revenue ($M, from
    metadata or the ratio-estimator's annual_revenue back-fill; NO revenue →
    degrade to event shape with confidence × 0.5). Then with band `(v100, v0)`:
    `s01 = clamp((v − v0)/(v100 − v0), 0, 1)`, `δ = 2·s01 − 1` — band ordering
    handles both directions automatically.
  - `benchmark_band` without value (qualitative mention only) → event fallback.
  - `event`: `δ = polarity × strength`.
- **Multiple claims on one factor: pick ONE, never blend** —
  argmax(confidence, then method rank `dataset_lookup > extracted >
  peer_ratio_fallback > coarse_bucket`, then recency). Averaging dilutes real
  measured values with guesses; summing lets repeated news mentions of one
  controversy stack unboundedly. One factor = one bounded contribution.
- `score = clamp(baseline + Σ w·c·δ, 0, 100)` per pillar.

**Verify:** synthetic claim sets with hand-checked arithmetic (empty → exactly
baseline; one max-negative controversy → baseline − w; benchmark value at
v100 → +w), then the real Step-3 claims.

## Step 5 — Calibration wiring (`calibration_harness.py`)

- New flag `--scorer {llm,formula}` (default llm — old behavior untouched).
- Formula branch inside `estimate_one` (replaces the `score_company_sync` call
  site, ~line 400): merge signals from ALL collectors
  (`fetch_company_signals` + `fetch_governance_signals` +
  `fetch_facility_signals` — governance/facility weren't in the old harness path
  but the extractors need them); fetch metadata; `estimate_missing_factors` for
  revenue back-fill; `extract_all_claims(...)` + `ct_anchor_claims(name)`
  (owner-match works name-only; sector-anchor skips gracefully without
  company_id); `compute_formula_scores(...)` → pred_e/s/g.
- Nothing persisted — consistent with the harness's "writes nothing" contract.
- Cache namespace prefix `formula_` (mirrors the existing `graph_` pattern) so
  LLM and formula prediction caches never mix. `--with-evaluator` ignored under
  formula (Phase 2 isolates the formula).

## Step 6 — Graph integration (`graph.py`)

- `PipelineState` gains `scorer` ('llm'|'formula'), `claims`, `formula_scores`.
- New nodes: `extract_claims_{dry,persist}` (merges gov/facility signals via
  their DB-cached `get_or_fetch_*` entry points when company_id present;
  3-pillar parallelism INSIDE the node via ThreadPoolExecutor — LangGraph
  fan-out adds state-merge complexity for zero benefit at 3 tasks),
  `ct_anchor`, `formula_score`.
- `_route_after_metadata` routes on (dry_run, scorer). **The old LLM scoring
  path stays untouched** — both paths coexist for A/B comparison until the
  Phase 6 cutover.

**Verify:** full graph dry + persist runs on Nvidia; confirm
`company_evidence_claims` rows land with resolved `source_signal_id` FKs.

## Step 7 — Phase 2 gate: calibration backtest

```
python -m agentic_estimation.calibration_harness --scorer formula \
    --n 30 --seed 42 --workers 3 --source bcorp
# and again with --source upright
```

**Gate:** Spearman ρ positive on all three pillars AND materially above the
documented baseline (E +0.246, S −0.154, G +0.045). Inspect the worst-5
residuals per pillar before declaring Phase 2 done — the Step-4 contributions
audit trail makes every point of every score traceable to a claim, a weight,
and a source.

---

## Build order

Steps 1 → 7, strictly. Each step is verified against live data before the next
begins (same discipline as Phases 0–1: real bugs found live, documented, fixed).

**Not Phase-2 blockers** (run independently in the background, CT anchor
degrades gracefully and improves as they fill):
- the owner-emissions harvest completing (currently ~41%)
- the full market-mapper batch (9,555 markets remaining; needs its threadpool +
  pre-filter built first)
