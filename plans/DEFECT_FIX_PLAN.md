# ESG Pipeline — Defect Fix Plan

## Context

A full audit of the production ensemble pipeline (`agentic_estimation/graph.py` and every layer it calls) surfaced ~30 defects ranked Critical→Low. The pipeline's purpose is producing near-correct, backtestable ESG scores; the audit found (a) the backtest itself is contaminated (ground-truth leakage), (b) production silently degrades in ways the calibration harness never sees (country resolution), (c) an operational trap that permanently orphans companies ("processing" forever), and (d) a long tail of accuracy/correctness bugs. This plan sequences the fixes so measurement is trustworthy **first**, production-breaking bugs second, accuracy improvements third, hygiene last. Each phase ends with a harness run so every change is falsifiable. Work can stop after any phase.

**Key principle:** fixes that change what predictions *are* invalidate the calibration disk cache — each phase notes which `calibration/cache` namespaces to purge.

**Independent cross-check:** the manual audit's core findings (country state-threading, stuck-"processing" lifecycle, peer-anchor leakage) were re-verified with the code-review-graph MCP tools (static call-graph queries, centrality/bridge analysis, `tests_for` coverage queries) rather than taken on faith. That pass also surfaced two coverage gaps not in the original audit, folded into Phases 0 and 4 below.

---

## Phase 0 — Measurement integrity (do first; nothing else is trustworthy until this lands)

### 0.1 Fix backtest ground-truth leakage (C1)
- **Files:** `agentic_estimation/shared/company_name_utils.py`, `agentic_estimation/layer_1/peer_anchor_collector.py`
- Add canonical `normalize_company_name(name) -> str` to `company_name_utils.py`: lowercase, punctuation→space, tokenize, strip `LEGAL_SUFFIXES` from the **tail only**, keep single-char tokens (this is `climate_trace_anchor._normalise` semantics — the safer of the two existing variants). Returns `""` for all-suffix names.
- In `peer_anchor_collector.py`: keep the SQL `!= %s` clauses as cheap pre-filters, add a Python-side `_drop_self(peers, exclude_name)` post-filter using `normalize_company_name` (with a `.strip().lower()` fallback when the normal form is empty). Apply in `_find_bcorp_peers`, `_find_upright_peers`, **and** `_find_real_metric_peers` (which currently has NO exclusion at all — thread `exclude_name` through `find_peers`).
- **Cache purge:** delete all of `calibration/cache` except un-prefixed (legacy llm) entries; peer anchors feed every formula/ensemble prediction.
- **Expect:** backtest numbers may *worsen slightly* — that delta is the removed leakage. Record the new honest baseline; do not "fix" it back.

### 0.1a Add test coverage for `find_peers` / `_drop_self` (new — graph-review finding)
- **File:** new `tests/test_peer_anchor_collector.py`
- Independent graph-based review (code-review-graph `tests_for` query + direct grep of `tests/`) confirmed `find_peers` and its `_find_bcorp_peers`/`_find_upright_peers`/`_find_real_metric_peers` callers have **zero test coverage anywhere** — the exact module carrying the leakage bug and identified as the top lever for the baseline-tie problem (2.1) is otherwise only exercised by slow, DB-backed full calibration runs.
- Write this test file in the same change as 0.1, not after: cover `_drop_self` self-exclusion (exact match, casing variant, legal-suffix variant, empty-normal-form fallback) and a `find_peers` smoke test against a fixture/mocked cursor. This gives the leakage fix a fast regression check instead of relying solely on a full harness run to catch a regression.

### 0.2 Re-baseline
- Run: `python -m agentic_estimation.calibration_harness --source bcorp --scorer ensemble --n 30 --no-cache --out calibration/baseline_postfix.csv` (and once with `--source upright`). This is the reference every later phase compares against.

---

## Phase 1 — Production criticals

### 1.1 Country resolution on the ensemble path (C2 + C3)
- **Files:** `agentic_estimation/graph.py`, `api/v1/esg_data/routes.py`, plus 3 baseline consumers.
- `graph.py`: make `node_metadata` the single resolution point — it runs on all scorer routes, dry and full, before every country consumer. Add helper `_resolve_state_country(input_country, metadata)`: `""`→None, fall back to `metadata["country"]`, canonicalize via existing `resolve_country_name()` (`country_baseline_agent.py`), keep raw string when unresolvable (downstream `get_country_baseline_with_fallback` has its own alias/regional/global chain). `node_metadata` returns `{"metadata": ..., "country": ...}` unconditionally. `PipelineState` already has the `country` field; legacy llm nodes' later `{"country": score.country}` writes still win (existing behavior).
- `routes.py:195`: `company.country or ""` → `company.country or None`.
- **C3 (alias resolution in 3 of 4 consumers):** switch `scoring_agent.py:207`, `orchestrator._resolve_baseline` (~line 150), and `metric_estimation_agent.py:451` from bare `get_country_baseline()` to `get_country_baseline_with_fallback()` (already exists, already used by `formula_estimator`). Keep return-shape compatibility (`_with_fallback` returns the same baseline object; check call sites for the source-label field).
- **Cache purge:** `graph_*` namespaces (harness-internal paths computed country themselves).

### 1.2 Status lifecycle — no more stuck "processing" (C4)
- **Files:** `agentic_estimation/graph.py`, `api/v1/esg_data/routes.py`, `agentic_estimation/orchestrator.py` (minor).
- In `run_company_graph`: wrap `graph.ainvoke` — on raised exception `await _set_esg_scoring_status(company_id, "failed")` and re-raise; after return, if `final_state.get("error")` set `"failed"`. `node_mark_estimated` stays sole writer of `"estimated"`; `run_company_graph` becomes sole writer of `"failed"`. (`_set_esg_scoring_status` is already non-fatal on DB error.)
- `routes.py` `get_market_esg` (~122-130): treat `"failed"` like `"pending"` (re-enqueue full run — safe, all writes are upserts). Stale-processing recovery: only `continue` on `"processing"` when the name is in `_inflight`; otherwise re-enqueue (server died mid-run). Comment the multi-worker caveat (per-worker `_inflight`, duplicates are upsert-safe).
- `get_estimation_status` (~277): map `"failed"` → `"pending"` in the API response for now (zero frontend change); expose a real `failed` state later.
- Optional: same one-liner before `orchestrator.run_company`'s early-return error paths (~305/331/337) — legacy/CLI only.

### 1.3 Nested `asyncio.run` in graph nodes (C6 — downgraded after re-verification, see note)
- **File:** `agentic_estimation/graph.py`
- **Re-verification correction (sanity-check pass):** the original audit flagged this as "verify at runtime — if it fires, CRITICAL." Empirically tested against the installed `langgraph==1.2.9`: a sync node calling `asyncio.run()` internally, invoked via `await graph.ainvoke(...)`, does **not** raise — LangGraph runs sync nodes on a separate worker thread from the event-loop thread, confirmed by printing `threading.get_ident()` inside the node vs the caller. **This is not currently an active bug**; downgrade from Critical to a code-quality/robustness item, not a production incident.
- Still worth doing, at lower priority: convert the 8 persisting/status nodes (`mark_processing`, `mark_estimated`, `extract_claims_persist`, `scoring_persist`, `evaluator_persist`, `metrics_persist`, `explainability_persist`, `persist_ensemble_scores`) to `async def`, deleting the `async def _do()` + `asyncio.run(_do())` wrappers and `await`ing the coroutines directly. Reason to still do it: the current pattern relies on an undocumented LangGraph execution-strategy detail (sync nodes get a worker thread) rather than an explicit contract — a future LangGraph version or a different graph-execution mode could change that. LangGraph supports mixed sync/async nodes natively; verified routing guarantees the sync `invoke()` dry path never reaches these nodes. Add a comment stating that invariant on `run_company_dry_graph` and `_route_after_metadata`.

### 1.4 Forward `country` into signal fetch — revive localized ESG source (C5)
- **Files:** `agentic_estimation/layer_1/signal_agent.py`, `agentic_estimation/graph.py`
- `get_or_fetch_signals(company_id, company, industry, country=None)` — accept and forward to `fetch_company_signals`. Update `node_signals`/`_run_signals` call chain to pass the (input) country; after 1.1, consider a second-chance localized fetch only if trivial — otherwise note that metadata-resolved countries still miss localized signals on first gather (same limitation the harness documents).
- **Note:** signal cache means already-cached companies won't refetch — acceptable; new companies benefit.

### 1.5 Verify-then-fix: ensemble persistence `confidence` column (H8)
- Check schema (`db_migrations/`, base schema) for `company_metric_values.confidence` nullability. If NOT NULL: add `confidence` to `_upsert_ensemble_score` in `ensemble_persistence.py` (derive from the reconciled confidence label or the numeric formula confidence). If nullable: write it anyway for consistency with `metric_estimation_agent._upsert_estimate`.

**Phase 1 verification:** see Verification section — graph dry + full CLI runs, forced-failure status check, harness `--via-graph` run showing populated `resolved_country`.

---

## Phase 2 — High-severity accuracy fixes (each A/B'd against the Phase 0 baseline)

### 2.1 Peer coverage — attack the baseline-tie collapse (H2)
- **Files:** `agentic_estimation/layer_1/peer_anchor_collector.py`, `layer_1/sector_matcher.py`
- Country matching: normalize the `country` argument to the lookup tables' vocabulary before the SQL `=` (small crosswalk dict or reuse `resolve_country_name` + a bcorp/upright spelling map; probe distinct `country` values in both tables first).
- Sector matching: route bcorp queries through the same fuzzy `sector_matcher` fallback already used for upright, so unmatched sector strings degrade to fuzzy instead of silent 0-peer.
- **Measure:** add a harness log/CSV counter for "0-peer companies" and per-pillar baseline-tie rate; compare before/after.

### 2.2 Reconcile: stop the LLM vote dominating thin-evidence companies (H1)
- **File:** `agentic_estimation/layer_3/reconcile.py`
- Structural change, numbers backtest-gated: make holistic's effective weight **scale down** with formula confidence shortfall instead of up (e.g. `eff_holistic = base * c_h * min(1, c_f / 0.7)`) and/or gate the holistic vote out when the pillar's QC verdict is thin. Add per-pillar base-weight override capability (keep flat 0.7/0.3 defaults; tuning deferred to the Phase 5 regression the code already plans). Fix the docstring/code mismatch (docstring describes per-pillar weights the code doesn't use).
- Do **not** silently change the holistic prompt's "45–55" instruction yet — it's shared with the legacy scorer; A/B it separately via the harness if desired.
- **Cache purge:** `t0_*`, `ensemble_*`, `verified_*`.

### 2.3 Metadata correctness (H3 + H4)
- **File:** `agentic_estimation/layer_1/company_metadata.py`
- Wikidata QID: apply the same `_name_overlap >= 0.6` guard GLEIF already uses; on failure fall through the chain instead of accepting `hits[0]`.
- Revenue/assets currency: fetch the P2139/P2403 currency qualifier in the SPARQL; convert to USD via a small static rate table (or skip/blank non-USD values with a `revenue_currency` field) — never label unconverted values "(USD)". `formula_estimator`'s intensity normalization then only uses USD-safe revenue.

### 2.4 Climate TRACE anchor liveness (H5)
- **File:** `agentic_estimation/layer_2/climate_trace_anchor.py`
- Replace hardcoded `_YEAR = 2024` with "latest year present in harvested data" (single query, cached). Log a one-line coverage stat (owner match / sector anchor / nothing) so the anchor's inertness is visible in harness runs. Expanding market-mapper coverage (25/9,580) is a separate data job — out of scope here; note it as follow-up.

### 2.5 Cache hygiene: TTL + negative caching (H7, M11)
- **Files:** `layer_1/signal_agent.py`, `layer_1/governance_collector.py`, `layer_1/facility_extractor.py`, `layer_1/company_metadata.py`
- Add a `fetched_at`-based TTL (e.g. 30d, module constant) to `get_or_fetch_signals`/`get_or_fetch_governance_signals`/`get_or_fetch_facility_signals` — requires reading the existing signals table timestamp column (verify it exists; if not, add a marker row).
- Negative caching: on a completed-but-empty gather, write a sentinel row (e.g. `gov__none`/`facility__none`, filtered out on read) so empty results stop re-hammering and partial gathers stop passing as full cache hits. Same pattern for unmatched metadata (`matched:false` row with TTL).

### 2.6 Name-normalization convergence (H6) — incremental
- Swap `climate_trace_anchor._normalise/_strip_suffixes` to the canonical `normalize_company_name` (behavior-identical by construction in 0.1). Leave `company_metadata._norm` for its fuzzy-overlap scoring until that code is next touched; add a comment pointing at the canonical function.

**Phase 2 verification:** harness A/B per item (same seed/sample, `--no-cache`): 2.1 and 2.2 are expected to move Spearman ρ visibly (report per-pillar); 2.3/2.4 are correctness fixes measured by spot-checks (a known JPY company's revenue, a CT-covered company getting an owner claim).

---

## Phase 3 — Medium fixes

- **3.1 Formula full-run persistence gap (M1):** `graph.py` — route `scorer='formula'` full runs through a minimal score persist (reuse `persist_ensemble_scores` with formula-only reconciled shapes, or block full formula runs with a clear error). Decide by use: it's calibration-only today — blocking with `error` is the cheap correct fix.
- **3.2 `extract_json_object` last-object bug (M2):** `shared/llm_json.py` — prefer the *last object containing the expected key* (add optional `required_key` param; `pillar_extractors` passes `"claims"`), falling back to current behavior. Mirrors `zen_client._extract_json_array`'s existing `"companies"` preference.
- **3.3 Unit table + magnitude guards (M3):** `layer_2/pillar_extractors.py` — extend `_UNIT_MULTIPLIERS` (tonnes co2e, mt co2e, kt co2, tco2e/year, GJ variants…); log every unrecognized-unit drop at WARNING with the raw unit so the table's gaps become visible; add per-factor plausible-magnitude caps in `claim_validators` (reuse the scope_1 country-ceiling pattern).
- **3.4 Tier-0 validator gaps (M4):** `layer_2/claim_validators.py` — extend `_INHERENTLY_NEGATIVE_FACTORS` coverage check to positive-marked-negative direction; add numeric ceilings for the other quantitative factors (percent bounds already exist; add revenue-intensity sanity bounds from the benchmark bands); keep text-check exemptions but log them.
- **3.5 Sector-aware benchmark bands (M5):** `layer_3/metric_estimation_agent.py` — add optional per-sector band overrides for the highest-variance metrics (scope_1/2, water, waste) keyed on the existing `manufacturing_classification`; default to current bands. Backtest-gated.
- **3.6 Spread false-confidence (M8):** `layer_3/reconcile.py` — when holistic is missing OR formula evidence is thin, cap confidence label at `medium` even if spread ≤10 (spread vs a noisy/absent vote is not corroboration). One-line rule + docstring fix.
- **3.7 Claim provenance (M9):** `layer_2/pillar_extractors.py` `persist_claims` — store a content hash of the signal text the claim was extracted from in `source_note` when the current DB row's text differs (cheap drift detector), rather than silently linking a different revision.
- **3.8 `_compute_baselines` contract (M10):** `layer_1/country_baseline_agent.py` — fix annotation to the actual tuple (or return a small dataclass), populate `_iso3_to_name` on the DB path, and build the country regex from the union of both sources.
- **3.9 Dry/full parity note (M12):** document (module docstring) that dry ensemble is not a full-run preview and the three persist nodes are calibration-uncovered; optionally add a `--persist-smoke` CLI flag later. No behavior change.

---

## Phase 4 — Low / hygiene (batch in one pass)

- Dedup fingerprints: strip `[date]`/`source:` prefixes before fingerprinting (`signal_agent.py` + `evidence_filters._fingerprint` usage).
- `run_company_graph`/`run_company_dry_graph`: default `scorer="ensemble"` (production default) — verify no caller depends on the llm default (routes passes explicitly; harness passes explicitly).
- `ensemble_persistence.py`: keep year-scoped rows (display rank already prefers newest) but add a comment; optionally reuse the latest existing ensemble row's year on re-runs within a config window.
- Delete `layer_2/ratio_estimator.py` (confirmed dead; grep for imports first).
- `_claim_sort_key`: honest comment (already partially there); leave recency for the freshness module.
- Throttle race: wrap `_LAST_CALL` access in a `threading.Lock` (`company_metadata.py`).
- Remove duplicate `qc_assess`/`gate` recompute in `graph.node_persist_ensemble_scores` by threading gated output through state (or leave with a comment — zero-risk option).
- **Guard the golden-regression test against silent skip (new — graph-review finding):** `tests/test_golden_rescore.py` was flagged by the code-review-graph tool as the single highest-betweenness node in the entire codebase (the most structurally central regression check) — yet it `pytest.mark.skipif`s cleanly to green when its offline dump fixture (`calibration/sat_ens_seed42_t0.json`) is absent, with nothing in the repo regenerating or verifying that file in CI. The dump exists in the current working copy, so this isn't failing today, but a fresh clone or CI runner without it gets a fully green suite while the one check that would catch a formula/reconcile regression (directly relevant to every Phase 2 change above) never runs. Fix: either check the dump fixture into the repo (it's a static JSON, not a secret), or add a CI step that fails loudly (not skips) when the file is missing on `main`/release branches, distinguishing that from a normal local dev skip.

---

## Verification (end-to-end)

1. **Unit-level:** add small pytest module(s) under a new `tests/` dir for: `normalize_company_name` (suffix-tail, single-char, all-suffix cases), `_drop_self` exclusion variants ("ALPKIT LTD" vs "Alpkit"), `_resolve_state_country` (`""`, None, metadata fallback, alias), `extract_json_object` with trailing schema echo.
2. **Graph dry (sync path unaffected by async nodes):**
   `python -m agentic_estimation.graph dry "Patagonia" --industry "Outdoor Apparel" --scorer ensemble --verify`
3. **Country fix live check:**
   `python -m agentic_estimation.graph dry "Thomson & Scott" --industry "Beverages" --scorer ensemble` — reasoning must show a country-sourced baseline, not `(no_country)` 50.0.
4. **Full-run + failure lifecycle:** `python -m agentic_estimation.graph run "Bosch" --id <uuid> --scorer ensemble` → status `estimated`; then force an error (bad DB URL for persistence) → status `failed`, and next `POST /esg/market-esg` re-enqueues it.
5. **Leakage check:** `find_peers(sector=..., exclude_name='<bcorp name with suffix variant>')` returns no self row. Run the new `tests/test_peer_anchor_collector.py` (0.1a) — must pass before 0.2's re-baseline.
6. **Golden regression guard:** confirm `pytest tests/test_golden_rescore.py` actually executes (not skips) locally after 0.1/2.x changes — `-rs` flag surfaces skip reasons if it silently no-ops.
7. **Calibration A/B after each phase:** `python -m agentic_estimation.calibration_harness --source bcorp --scorer ensemble --n 30 --no-cache --out calibration/phaseN.csv` (fixed seed), compare Spearman/pct-MAE per pillar to `baseline_postfix.csv`. Phase 0 may lower numbers (honest); Phases 1–2 should raise evidence-only ρ, and the 0-peer/tie-rate counters should drop.

## Explicitly deferred (documented, not planned here)

- Regression-fitting all hand-set weights/saturation params (codebase's own "Phase 5", blocked on ground-truth density).
- Climate TRACE market-mapper coverage expansion (data job, 25/9,580 markets).
- Layer-4 critics gaining correction power (design decision, not a bug).
- Holistic prompt "45–55" instruction change (A/B separately; shared with legacy scorer).
