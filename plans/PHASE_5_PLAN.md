# Phase 5 — Accuracy & Calibration Program

*Make every number fitted, measured, or abstained. The plan every other document
defers to ("pending Phase 5 regression") — plus the prerequisites that make that
regression actually decidable, and the evidence/route upgrades it unblocks.*

## 1. Context & the evidence-strength ground rule

The pipeline (see `ESG AGENTIC PIPELINE.md`) is architecturally settled: LLM as
evidence tagger, deterministic formula + v5 saturation as primary scorer, one
capped holistic vote, bounded verification. What is NOT settled is almost every
number in it: 28 factor weights, method-trust values, saturation params for E/G,
coverage floor β, gate threshold, peer pseudo-weight, blend weights, confidence
labels — all hand-set or tuned on samples too small to decide anything.

**Ground rule (why this plan is sequenced the way it is):** every existing
tuning/route verdict rests on n=30–40 samples where per-pillar Spearman noise is
±0.10–0.17. That includes:

- the flat **0.7/0.3 blend** (decided on n=30 held-out, margin +0.009 total —
  well inside noise);
- the seed-777 **n=40 ablation verdicts** — including "no alternative route
  beat 0.7/0.3 base with a CI excluding zero." At n=40 this means "underpowered
  to detect a difference," NOT "the current route is best." **Treat every
  seed-777 verdict as a hypothesis, not a conclusion.**

Therefore: **no production route/weight change ships on n≤40 evidence, and no
existing n≤40 decision is treated as final.** Everything re-decides on the
Stage-0 large corpus with the tune/holdout seed discipline already in
`ESG AGENTIC PIPELINE.md` §11.2.

**Cache principle** (inherited from DEFECT_FIX_PLAN.md): any change that alters
what predictions *are* invalidates `calibration/cache` — each stage notes purges.

**Work can stop after any stage.** Each stage ends with a falsifiable
measurement.

---

## 2. Stage 0 — Measurement infrastructure (nothing later is decidable without this)

### 0.1 `--via-graph` for formula/ensemble scorers
- **File:** `agentic_estimation/calibration_harness.py` (the
  `raise ValueError("--scorer ... and --via-graph are not compatible yet")`
  guard in `run_backtest`), `estimate_one`.
- Production runs `run_company_graph(scorer="ensemble")` through the compiled
  LangGraph; calibration of formula/ensemble currently bypasses the graph via
  direct function calls. Any bug in `_route_after_metadata` /
  `_route_after_formula_score`, state threading, or node ordering is invisible
  to every backtest. Wire `estimate_one` to `run_company_dry_graph` for
  `scorer in ("formula","ensemble")` and delete the guard.
- **Acceptance:** n=10 ensemble run `--via-graph` vs direct-call on the same
  cached evidence produces identical scores (deterministic parts) / within-noise
  (holistic). Any mismatch is a real routing bug — fix before proceeding.

### 0.2 Mid-graph "resume" entry points (state injection)
- **File:** `agentic_estimation/graph.py` (entry conditional).
- Route on pre-populated state: `signals` present → skip to `metadata`;
  `claims` present → skip to `formula_score`; `reconciled` present → skip to
  `verify_estimate`. Dry-run only at first (no persist semantics to design).
- Payoff: (a) graph-level tests from frozen fixtures, (b) direct entry to the
  verify node with fabricated medium-confidence scores — the only practical way
  to exercise every §2.3 branch (panel-2 refute, non-retryable factor,
  <2-responder fail-open), (c) later production reuse for cache-hit companies.
- **Acceptance:** new `tests/test_graph_resume_entries.py` — each injection
  point reaches exactly the expected node sequence (assert via node logs or a
  state-trace field); verify-node branch matrix covered with stubbed critics.

### 0.3 Gather-affecting quick fixes (BEFORE the big corpus gather, so the
corpus benefits — these change what evidence gets collected)
- **Metadata before signals:** swap `node_signals`/`node_metadata` order (or
  add a second localized-gather pass) so a metadata-resolved country gets
  localized ESG queries on the first gather. Today, with no cache TTL, missing
  it once is permanent per company. Files: `graph.py`, check
  `_resolve_state_country` threading.
- **Batch country forwarding:** `fetch_signals_for_companies` drops `country`
  entirely (`signal_agent.py`) — forward it.
- **Cache TTL + negative caching** (DEFECT_FIX_PLAN.md 2.5): implement now so
  the corpus gather can be re-run/refreshed deliberately rather than never.
- **Acceptance:** a company with country resolvable only via metadata gets
  localized queries on first gather (log assertion in a dry run).
- **Cache purge:** full `calibration/cache` purge after landing (signal
  composition changes).

### 0.4 The large frozen corpus — the keystone
- **Tool:** `calibration/ablation_replay.py gather` (exists, proven at n=40).
- **Target:** n=300–500 companies, stratified across bcorp AND upright, by
  sector × country × expected evidence richness (public names vs small
  private). At n=300 the bootstrap CI is ~2.7× tighter than n=40 — this is
  what converts UNDECIDABLE verdicts into decisions.
- **Seeds:** do NOT reuse 42/101/7/314/777/888. Draw two fresh seeds for
  corpus-tune and corpus-holdout; the documented virgin seed stays in reserve
  for the final Stage-2 validation only. Record all three in the §11.2 seed
  ledger.
- **Operational constraints (plan for a multi-day gather):**
  - NewsAPI: 100 req/day free → n=400 alone needs ≥4 days for that source, or
    accept `news_api` absent for part of the corpus (record which — stratify
    the split so absence isn't correlated with sector).
  - DDG limiter 2.0s+jitter and Zen free-tier LLM throughput bound wall-clock;
    run with checkpointing — extend `gather` to append/resume into an existing
    dump (skip already-dumped companies) so a crash doesn't restart the run.
  - Holistic vote + extraction LLM calls are the main cost; estimate from the
    seed-777 run's logs before launching, and log a projected finish time.
- **Dump versioning:** keep the v2 dump schema (stores `peer_anchor`, raw
  claims, Tier-0 flags, holistic vote) — required for every variant. Name:
  `calibration/corpus_v1_seed<A>_n<N>.json` + holdout twin.
- **Acceptance:** dump loads; `replay --variants base` reproduces live scores
  for a 10-company spot check; per-pillar coverage stats (how many companies
  have ≥1 claim per pillar) recorded in a README next to the dump.

---

## 3. Stage 1 — Re-decide the route questions with real power (replay only, no new gathering)

All of Stage 1 is free replay against the Stage-0 corpus. Discipline: explore
on corpus-tune seed, confirm on corpus-holdout; ship only if the holdout CI
excludes zero.

### 1.1 Re-run the full variant matrix
- All 12 existing variants in `ablation_replay.py` (`base`, `no_holistic`,
  `holistic_only`, `formula_only`, `blend_60_40`, `blend_50_50`,
  `no_peer_anchor`, `baseline_only`, `peer_baseline`, `peer_baseline_blend`,
  `no_tier0`, `no_freshness`) on the large corpus.
- Every seed-777 verdict gets re-judged: expected to *confirm* directionally
  (`baseline_only` worse for E, peer anchor load-bearing for S, formula
  load-bearing for G) but now with CIs that can also settle `no_tier0` /
  `no_freshness` (UNDECIDABLE at effective n=22).

### 1.2 Re-open the blend weights — including per-pillar
- The 0.7/0.3 flat split is an n=30 decision with a +0.009 margin; explicitly
  up for re-decision, not defense. Grid: flat {0.8/0.2 … 0.5/0.5 step 0.05} ×
  per-pillar independent splits. Per-pillar was previously UNDECIDABLE at
  n=30 by design of the noise floor — this is the first time it's decidable.
- **Ship rule:** a new split replaces 0.7/0.3 only if holdout CI excludes zero
  for at least one pillar and no pillar is confirmed-worse.

### 1.3 Per-pillar baseline routing (peer-first S / country-first G)
- Seed-777 hypothesis: `peer_baseline` promising for S (+0.291 vs +0.165, CI
  spanned zero), worse for G. Decide it properly on the corpus; if confirmed,
  production change = per-pillar route config consulted by
  `formula_estimator.compute_formula_scores` (no new graph nodes).

### 1.4 Evidence-conditional routing analysis
- Partition corpus companies by `gate_fired` / evidence mass; compare variant
  winners per stratum. If different routes win in thin vs rich strata, design
  a gate-conditional route (thin → peer/baseline-weighted + forced range;
  rich → formula-dominant). Analysis only in this stage; implementation goes
  through the same ship rule.

### 1.5 Oracle claim-audit (error decomposition — informs Stage 2 & 4 priorities)
- Hand-correct the extracted claims for ~30 corpus companies (stratified);
  replay corrected vs original. The gap = extraction-fault error; the residual
  = scoring-fault error. Per-pillar decomposition decides where Stage 4 effort
  goes (better extraction/validators vs better anchors vs formula fitting).
- **Deliverable:** `calibration/oracle_audit_report.md` with the split.

**Stage-1 exit:** a decided route configuration (possibly unchanged — that is
a valid, now-actually-supported outcome) + the error decomposition.
**Cache purge:** only if a route change ships.

---

## 4. Stage 2 — Fit the hand-set constants (the Phase-5 regression proper)

Run on the Stage-1-decided route. All fitting on corpus-tune, validation on
corpus-holdout, final one-shot check on the reserved virgin seed. Fit in this
order (coarse-to-fine, refit downstream after upstream changes):

### 2.1 Factor weights (per pillar)
- Frozen contributions make pillar score linear in the weight vector
  (`points = w·c·δ` per factor, plus baseline/saturation reparametrization —
  fit pre-saturation Δ contributions against ground-truth percentile).
- Method: ridge regression with positivity constraints (weights ≥0, direction
  fixed by registry), target = within-sample ground-truth percentile.
  Regularize toward current hand-set values (they're informed priors, and at
  ~28 weights/pillar with n≈300, unregularized fits will overfit).
- Start with E (most factors, most anchor data, widest ground-truth range).
- **Guard:** golden-rescore test (`tests/test_golden_rescore.py`) gets a new
  expected dump; keep the old one as the pre-Phase-5 reference.

### 2.2 Saturation parameters
- Grid `a_pos/a_neg ∈ {25..50}`, `k ∈ {0.7..1.5}`, `β ∈ {0.4..0.8}`, gate
  threshold `∈ {1.5..4}` per pillar, on the corpus. The S-pillar a=35 result
  (the one value that ever survived holdout) is the prior; E/G finally get a
  decidable tune.

### 2.3 Method-trust values
- {dataset_lookup, extracted, peer_ratio_fallback, coarse_bucket, peer_anchor}
  trust vector — small grid, fitted after 2.1/2.2 (they interact).

### 2.4 Monotone output recalibration (fixes clustering-near-50)
- Isotonic map: predicted score → ground-truth percentile, fitted on
  corpus-tune, applied as a final display-layer transform. Directly widens
  spread (the known "predictions cluster near 50" ρ-cap) without touching
  formula internals. Ship only if holdout percentile-MAE improves.

**Stage-2 exit:** every constant in `ESG AGENTIC PIPELINE.md` §9 re-labeled
from "hand-set" to "fitted (corpus vX)" or "hand-set, fit attempted, prior
retained." **Cache purge:** full (predictions change).

---

## 5. Stage 3 — Honest uncertainty (ranges that mean something)

### 3.1 Measured confidence calibration
- Replace the asserted `high/medium/low → 0.9/0.6/0.3` mapping
  (`ensemble_persistence.py`) with measured per-bucket error from the corpus.
- Fit the range width (currently `±2` around vote spread / flat `±15`) so the
  emitted [low, high] band achieves a target empirical coverage (e.g. 80% of
  ground-truth percentiles inside the band) — conformal-style, per confidence
  bucket. The harness's confidence-label calibration report (§11.3) is the
  measurement tool; it becomes a *fitting* tool.
- **Acceptance:** monotonicity verdict passes (high bucket lowest error) AND
  measured coverage within ±5pts of target on holdout.

### 3.2 Explicit abstain route
- Below a fitted evidence threshold, emit `insufficient_evidence` + peer/
  baseline range instead of the current `score=50, range 20–80` (which looks
  like an estimate). Terminal abstain state in reconcile/gate output;
  `build_esg_json` + frontend render it distinctly; calibration metrics
  exclude abstentions (report abstention rate separately so it can't be gamed
  to inflate ρ).

---

## 6. Stage 4 — Evidence upgrades (raise the input floor)

Priority within this stage is set by the Stage-1.5 oracle decomposition.
Deterministic anchors (4.1, 4.2) are computable offline from DB + metadata —
**verify them by injecting their claims into the frozen corpus and replaying**
(no re-gather needed); ship only on the usual holdout CI rule.

### 4.1 Structured dataset anchors (dataset_lookup claims, trust 1.0)
- New `layer_2/dataset_anchors.py`, same conservative exact-name matching
  discipline as `climate_trace_anchor` (suffix-strip, no fuzzy).
- Sources: SBTi public target dashboard export → `sbti_commitment`;
  `bcorp_lookup` membership → `third_party_esg_audit` / `esg_report_published`
  (a B Corp certification is a third-party audit — confirm mapping
  semantics before wiring); CDP public disclosure lists if obtainable →
  `cdp_disclosure`. Each replaces a DDG-text→LLM-extraction path at ~0.5
  confidence with a deterministic claim at 0.85–0.95.
- **Leakage check:** `bcorp_lookup` is also ground truth — using *membership*
  (a public fact) as an evidence claim while backtesting against bcorp
  *scores* is borderline. Rule: when backtesting against bcorp, exclude the
  bcorp-membership anchor for bcorp-truth companies (mirror the existing
  `_drop_self` discipline); membership stays enabled in production and in
  upright-truth backtests.

### 4.2 Activity × intensity physical estimates
- Upgrade the Climate TRACE sector anchor from a 0.25-confidence percentile
  vote to an actual `benchmark_band` value claim:
  `estimated_scope1 = revenue_musd × country_sector_intensity` from
  `climate_trace_country_emissions` + metadata revenue. Method
  `dataset_lookup`-derived but flagged (`method="activity_intensity"`, own
  trust value fitted in 2.3). Also the first step toward absolute-metric
  backtesting once Wikirate coverage grows.
- Fix the `_YEAR=2024` hardcode while in there (DEFECT_FIX_PLAN.md 2.4).

### 4.3 Evidence Recovery (bounded, gate-triggered re-gather)
- The confidence gate stops being observational: a `thin` pillar triggers ONE
  targeted re-gather (pillar-specific collectors only — e.g. thin G → the
  governance_collector tiers that returned nothing), then re-extract/re-score
  that pillar. Bound exactly like the verifier (1 attempt, no loop). Wire as
  internal control flow of a node (consistent with §2.3's design language).
- **Measure:** abstention/thin rate before vs after on a corpus subsample;
  ship only if thin-rate drops without accuracy regression.

---

## 7. Stage 5 — Production learning loop

### 5.1 Champion/challenger logging
- On every production run, also compute 1–2 deterministic challenger routes on
  the same in-memory evidence (zero extra LLM/network cost) and persist their
  scores to a side table (`route_challenger_scores`). Every production run
  becomes paired backtest data; route re-decisions stop depending purely on
  offline samples.

### 5.2 Holistic-predicts-residuals (v6 recommendation)
- Add as a 13th replay variant FIRST (holistic output reinterpreted as capped
  ±10 residual on the formula score). Ship through the standard Stage-1 CI
  rule. Only then change the production prompt/plumbing.

### 5.3 Surface `failed` status honestly
- `estimation-status` currently maps `failed`→`pending`. Expose it (with
  retry count) so operational failures are visible instead of masked.

---

## 8. What this plan deliberately does NOT do

- No new LangGraph nodes for routing experiments (route config + replay
  variants suffice until something is confirmed).
- No fuzzy entity matching anywhere new (inherits the "Iran ≠ Iraq" and
  Climate-TRACE-exact-only discipline).
- No per-company LLM cost increases before Stage 4.3 (everything in Stages
  0–3 is replay/refit on frozen evidence).
- No shipping on tune-seed results — ever. Tune → holdout → (Stage 2 final:
  virgin seed), same ledger discipline as before, at a sample size where it
  finally has power.

## 9. Dependency graph & suggested order

```
0.1 via-graph ─┐
0.2 resume     ├─ independent, do in parallel
0.3 gather fixes ─┘
        ↓
0.4 corpus gather  (multi-day, start early, checkpointed)
        ↓
1.1–1.4 route re-decisions ── 1.5 oracle audit (parallel)
        ↓                          ↓
2.1–2.4 constant fitting     (priorities for Stage 4)
        ↓
3.1–3.2 uncertainty calibration
        ↓
4.1–4.3 evidence upgrades (anchors injectable → replay-verified)
        ↓
5.1–5.3 production loop
```

Stage 0 items are small code changes; the corpus gather is the long pole —
kick it off as soon as 0.3 lands and let Stages 1+ proceed the moment the
dump is complete.
