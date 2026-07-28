# ESG Pipeline Evaluation & Verification Strategy (refined for this codebase)

## Objective

Evaluate the ESG pipeline at multiple levels to ensure correctness, robustness,
reproducibility, explainability, and scalability — with every strategy grounded
in what this codebase actually has, what has already been measured, and what is
deferred.

**Status tags used throughout:**
- **[DONE]** — already implemented and run at least once
- **[READY]** — infrastructure exists; needs a script/run, no new machinery
- **[ACTION]** — concrete gap to close (listed in the Action Queue at the end)
- **[DEFERRED]** — evaluates a module that is deliberately not built yet
  (Evidence Recovery, Contradiction Resolution, Retry Planner, LLM critics —
  see UPDATED_AGENTIC_WORKFLOW.md "Build Status & Verdicts")
- **[DROPPED]** — not applicable to this pipeline; reason given inline

**Naming note:** this document's stages are pipeline *layers*, deliberately NOT
numbered "Phase 1–6" — that vocabulary is already taken by the build phases
(Phase 0–6 of the rebuild plan) and the collision has caused real confusion.

```
Stage A  Collection      (layer_1: signal/governance/facility/peer collectors)
Stage B  Extraction      (layer_2: pillar_extractors + climate_trace_anchor
                                    + claim_validators [Tier-0])
Stage C  Formula         (layer_3: formula_estimator + saturation_score
                                    + peer_anchor)
Stage D  Holistic        (layer_3: holistic_estimator -> scoring_agent)
Stage E  Reconciliation  (layer_3: reconcile [v8] + confidence_gate)
Stage F  Verification    (Tier-0 validators exist; LLM critics DEFERRED)
End-to-End               (calibration_harness vs bcorp/upright)
```

---

# 0. The Three Methodology Rules (learned the hard way, non-negotiable)

These were established empirically this project and override any generic
advice below:

1. **Fixed-evidence comparisons only, for config decisions.** At n=30, two
   fresh-gather backtests differ by ±0.10–0.17 Spearman per pillar from
   evidence-gathering noise alone (live DDG results change, sources time out,
   the holistic vote is non-deterministic even at temperature 0 — measured
   ±8 pts on identical reruns). Config effects are typically ≤0.05. Therefore:
   **never judge a formula/weight/validator change by comparing two fresh
   runs.** Dump evidence once (`calibration/tune_saturation.py dump`), then
   re-score offline under both configs on IDENTICAL claims. This is how
   v5-vs-linear was decided; it is the only trustworthy method at our n.

2. **Two-population reality.** Random draws from `bcorp_lookup` are dominated
   by web-invisible micro-SMEs (live probe result: searches return nothing or
   wrong-entity garbage — a Hawaiian restaurant, a "Melissa Wyatt" profile).
   On that population, end-to-end correlation measures *evidence
   availability*, not estimator skill (seed=314: 29/30 companies correctly
   range-flagged by the Confidence Gate). Every end-to-end metric must be
   reported per population: web-visible vs thin. The gate split
   (point-scored vs range-flagged) is the built-in lens for this.

3. **Seed ledger — never tune and validate on the same sample.**
   | seed | role | status |
   |---|---|---|
   | 42 | training / tuning | used heavily (tuning sweeps) |
   | 101 | held-out validation | used (v5/v8 final validation) |
   | 7 | phase-2 spot check | used once (n=25) |
   | 314 | fresh test | burned (defaults test + Tier-0/gate A/B) |
   | *fresh, undisclosed until run* | final evaluation | **reserve; n≥60** |
   Any parameter fit (including Phase-5 regression) follows:
   tune on 42 → validate on 101 → confirm on a virgin seed. The n=30 tuning
   overfit lesson is on record: E/G tuned params REGRESSED on held-out; only
   S:A=35 survived. Look for plateaus, not peaks.

---

# 1. Deterministic Testing

## Unit tests — [DONE as scripts / ACTION to make permanent]

35 synthetic tests exist and pass, but live in the session scratchpad — they
must move into a repo `tests/` directory (pytest) or they are lost:

| Suite | Covers | Count |
|---|---|---|
| saturation tests | `saturate_pillar`: gate, coverage, tanh, sign-aware A | 6 |
| reconcile tests | weighted merge, missing-vote passthrough, clamps, labels | 5 |
| validator tests | all 5 Tier-0 rules incl. exemptions + layering | 14 |
| gate tests | qc_assess (breakdown + linear fallback), gate routing | 10 |

Functions with locked-down behavior: `saturate_pillar`,
`_registry_weight_sum`, `_percentile_rank` (peer anchor),
`_formula_confidence` (v8 mass-trust), `_confidence_label`,
`validate_claims` (each rule), `qc_assess`, `gate`.

## Formula verification (synthetic archetypes) — [PARTIAL / ACTION]

Exists: empty→exactly-baseline, single-max-negative, benchmark-at-v100,
gate-fired-peer-anchor-only. Add archetypes: conflicting evidence
(opposite-polarity same factor — also exercises the Tier-0 polarity rule),
sparse-but-strong (1 high-confidence claim: coverage multiplier visible),
all-positive saturation (Δ→1: score ≈ B + A·tanh(k)·CovMult). Each asserts
score AND breakdown fields (evidence_mass, coverage, gate_fired) AND the
contribution audit trail.

## Regression / Golden dataset tests — [READY / ACTION]

Our golden mechanism is the **dump→rescore** path, not a separate fixture
format: `sat_ens_seed42.json` / `sat_ens_seed101.json` hold real captured
evidence (formula contributions + holistic votes for 30 companies each).

- Golden test = load dump → rescore with current library code → assert
  scores identical to stored expected values. Any formula change that
  intends a difference updates the expected file *explicitly* in the same
  commit — silent drift becomes impossible.
- **[ACTION]** the existing dumps predate Tier-0 validation (captured before
  claim_validators existed). Refresh dumps on the next full run so golden
  data reflects the post-Tier-0 pipeline; keep the old dumps for
  before/after ablations.

---

# 2. Non-Deterministic Testing (LLM components)

## Repeatability — [READY, scoped down]

Generic advice says "20 runs"; our gateway serializes calls at a 2 s min-gap,
so scope honestly: **5–10 reruns of `holistic_vote` on ONE fixed, cached
signal set** (no re-gathering — variance must be measured on fixed inputs or
it conflates LLM noise with evidence noise, Rule 1).

Known baseline: ±8 pts swing between identical reruns at temperature 0.
Acceptance is defined downstream, not on the raw vote: with the holistic
base weight 0.3 and fixed confidence 0.5, a ±8 raw swing moves the
reconciled score ≤ ~±3. Criterion: **reconciled-score sensitivity to a
holistic rerun ≤ ±3 pts**; if the raw variance grows enough to break that,
cap the holistic weight further rather than chasing prompt tweaks.

Same protocol applies to the pillar extractors: rerun
`extract_pillar_claims` 5× on one cached signal set; measure claim-set
overlap (Jaccard on factor keys) and confidence spread per factor.

## Prompt robustness — [DROPPED for now]

Real but low-yield at our scale: every variant run costs serialized API time,
and we have no evidence prompt fragility is a live failure mode (the live
failure mode was wrong-entity *retrieval*, now Tier-0-guarded). Revisit if
extractor repeatability (above) shows instability.

## Self-consistency (multiple reasoning paths) — [DROPPED]

The pipeline deliberately runs ONE extraction path per pillar (the 3-way
pillar split is fail-isolation, not self-consistency) and caps the holistic
vote's influence structurally. Adding N-path sampling multiplies serialized
LLM cost for a component whose weight is already bounded. The ensemble's
cross-checking happens between *independent methods* (formula vs holistic),
which is stronger than same-model self-agreement.

## Critic agreement — [DEFERRED]

No critics exist (Phase 4 proper). When built, measure per-lens refutation
rate, 2-of-3 agreement rate, and false-refutation rate via the synthetic
bad-claim injection suite already specified in PHASE_4_PLAN.md.

---

# 3. Stage-Level Evaluation

## Stage A — Collection — [READY]

Measurable today from existing logs/CSVs, no labels needed:
- `signals_count` distribution per run (already a CSV column).
- Per-source hit rate: signal_agent already logs "n/18 sources returned
  data" and per-source OK/empty lines.
- Cross-source duplicate rate: evidence_filters fingerprint dedup —
  **[ACTION]** log the count of deduped lines so the rate is a metric, not
  just a behavior.
- Search-failure rate: DDG/mojeek/startpage timeout warnings per run
  (heavy in recent runs — worth tracking as an availability metric).
- Relevant-document rate has no label set; the **proxy** is the Tier-0
  lexical-relevance drop rate per source (a drop = a retrieved-but-
  irrelevant document that survived collection filters). 19 drops / 30
  companies on the last seed=314 run is the baseline number.

## Stage B — Extraction — [PARTIAL; no annotated corpus exists]

Precision/recall/F1 against manually annotated ESG documents requires an
annotation effort we don't have. Honest substitutes, in order of value:
1. **Synthetic injection suite [READY]** — inject known-bad claims
   (yoga-page-shaped: real source_tag, irrelevant text) and known-good
   claims into fixed signal sets; measure Tier-0 catch rate and extractor
   behavior. Generalizes the Blackmores yoga-page finding into a permanent
   test.
2. **Manual spot-audit [ACTION, recurring]** — sample ~30 claims from each
   major backtest CSV; check factor correct / cited text supports claim /
   value verbatim. Tracks hallucination + missing-claim rate approximately.
3. **Structural guarantees already enforced in code** (not metrics):
   hallucinated attribution is impossible (source_tag must be an actual
   signal key or the claim is dropped); factor must exist in the registry
   and match the pillar.
4. Confidence calibration: bucket audited claims by stated confidence;
   the audit pass/fail rate per bucket IS the calibration curve. Needs (2).

## Stage C — Formula — [READY via dumps]

- Benchmark correlation: fixed-evidence rescoring (Rule 1), never fresh A/B.
- Rank metrics only: Spearman (midrank) + percentile-MAE. Raw MAE/RMSE
  against bcorp/upright are **[DROPPED]** — their scales aren't 0-100 and
  aren't ours; absolute error across scales is meaningless (doc'd in
  calibration_harness header).
- Evidence utilization [ACTION]: log per-pillar `coverage`, `evidence_mass`,
  and gate_fired rate to the backtest CSV (breakdown is now threaded through
  PillarFormulaScore — the data exists; add the columns).
- Synthetic edge cases: §1 suite must stay green.

## Stage D — Holistic — [READY via dumps]

Dumps store holistic votes. Compute holistic-alone Spearman offline from the
same dump used for formula-alone — same evidence, directly comparable. Plus
the repeatability protocol from §2.

## Stage E — Reconciliation + Confidence Gate — [READY]

- Ensemble improvement: reconciled vs formula-alone vs holistic-alone
  Spearman **on the same dump** (tune_saturation's sweep already reproduces
  the reconciled path offline; add the two solo scorers).
- Spread diagnostic: already in the harness report (spread vs rank-error
  correlation, per pillar, explicitly non-gating).
- **Gate validity [READY — the new key metric]**: on any population with a
  non-trivial point/range split, compare mean |rank error| of range-flagged
  vs point-scored companies. The flag is working iff flagged companies are
  measurably worse. (On pure-SME samples the split degenerates to ~100%
  range — expected, not informative; run this on a mixed/web-visible
  sample.)
- Confidence-label accuracy: same comparison keyed by the
  high/medium/low label instead of the gate mode.

## Stage F — Verification — [Tier-0 part READY; critic part DEFERRED]

Tier-0 (exists):
- Drop/cap counts per run: already CSV columns (`claims_dropped`,
  `claims_capped`).
- **False-drop rate [ACTION, recurring]**: spot-audit dropped claims from
  each major run — was the dropped claim actually bad? (Target: near-zero
  false drops; a lexical vocab gap shows up here first.)
- Score impact: fixed-evidence rescore with validators on/off (ablation §7).

Critic-layer metrics (false review rate, missed errors, recovery success):
**[DEFERRED]** until Phase 4 builds them; acceptance tests already specified
in PHASE_4_PLAN.md.

---

# 4. Recovery / Contradiction / Retry Evaluation — [DEFERRED, with pre-registered criteria]

All three modules are deferred (see UPDATED_AGENTIC_WORKFLOW.md). Keep these
as *acceptance criteria for when they're built*, refined by what the live
probe already established:

- **Evidence Recovery**: must be evaluated on the web-visible population
  ONLY — the probe proved broad keyword/localized reformulation returns
  wrong-entity garbage for invisible SMEs, so "evidence gained" on that
  population measures garbage ingestion, not recovery. Pre-registered
  baseline: on random bcorp, trigger frequency would be ~97% (29/30
  range-flagged on seed=314) — a per-batch invocation cap is mandatory
  before any live evaluation. Primary metrics: net NEW claims that survive
  Tier-0 validation (not raw text volume), coverage delta, wrong-entity
  rate of recovered text (spot-audit).
- **Contradiction Resolution**: precondition metric first — measure how
  often opposite-polarity same-factor claim pairs actually occur in real
  runs (Tier-0's polarity rule already flags them, so this is countable
  today **[ACTION: log it]**). If the rate stays near zero, the module
  stays deferred; no point evaluating a resolver of conflicts that don't
  occur.
- **Retry Planner**: only meaningful once Recovery exists (its own Critical
  Rule: no retry without a changed evidence pool). Metrics as originally
  listed, plus wasted-retry rate.

---

# 5. End-to-End Evaluation

Ground truth: `bcorp_lookup` (10,337) and `upright_lookup` (10,086) — the
only answer keys we have. WikiRate disclosed metrics remain future
metric-level (not pillar-level) ground truth — currently too sparse.
"Manually labelled companies" — **[DROPPED]**: no labeling budget; bcorp +
upright cover the need.

Metrics:
- **Spearman (midrank)** — the headline; implementation already shared
  across harness + offline tools (verified identical this session).
- Percentile-MAE — secondary, scale-free.
- Kendall τ — optional, cheap to add, rarely decisive at n≤60. [OPTIONAL]
- Pearson / raw MAE / RMSE — **[DROPPED]**, cross-scale invalid (see §3C).

Protocol (refined by Rules 1–3):
1. Scorer comparisons (formula-only vs holistic-only vs reconciled) run
   OFFLINE on one dump — never as three fresh gathers.
2. Fresh-gather end-to-end runs are for *population-level* claims only, at
   the documented noise band (±0.10–0.17 per pillar at n=30). Differences
   inside the band are noise — say so in the report rather than narrating
   them. n≥60 for anything finer.
3. Report BOTH views every run (already built into the harness):
   overall Spearman + the gate split (point-scored-only Spearman, n_point /
   n_range per pillar, flagged-company list).
4. Standing baselines to beat (documented, fixed-evidence-verified):
   - Original single-shot LLM: E +0.246, S −0.154, G +0.045, Total −0.132
   - v5 > linear on all six pillar/sample combos (fixed evidence, seeds
     42 + 101); upright clean n=60 formula: E +0.272, Total +0.184.

---

# 6. Stress Testing — [READY, all offline on dumps: free, deterministic]

All three stressors run against dumped evidence with offline rescoring — no
API cost, fully repeatable:

- **Missing data**: randomly delete 25/50/75% of claims per company;
  measure Spearman degradation AND gate flag-rate increase. The second is a
  gate-validity check: starved evidence SHOULD flip pillars to range — if
  flag rate doesn't rise with deletion, the gate thresholds are wrong.
- **Noisy data**: inject synthetic wrong-entity claims (yoga-page-shaped,
  real source tags) and duplicate-mention claims; measure Tier-0 leakage
  rate (injected claims surviving validation) and score drift caused by
  survivors. Corroboration + lexical rules are the units under test.
- **Adversarial / greenwashing**: inject positive puff claims
  (net_zero_pledge, esg_report_published with marketing-only cited text);
  measure score inflation and whether confidence caps engage. Note the
  structural defense already in place: benchmark_band factors need numeric
  values to move far, event factors are strength-bounded, and coverage
  multiplies the swing down when evidence is one-sided-thin.

---

# 7. Ablation Studies — [READY offline; two already done]

Method: ONE dump → N offline rescores → Spearman table. Already executed
this way: linear-vs-v5 (v5 won everywhere), count-based-vs-v8 confidence
(neutral-to-positive). Remaining ablations, all feasible now:

| Ablate | How (offline) |
|---|---|
| peer anchor | drop `_peer_anchor` contributions from dump rows |
| coverage multiplier | β=1.0 |
| evidence gate | threshold=0 |
| method trust | m(·)=1.0 for all methods |
| freshness decay | multiplier=1.0 |
| holistic vote | reconcile formula-only |
| Tier-0 validators | rescore pre-validation dump vs post-validation dump |
| sector_emissions_intensity (CT anchor) | drop dataset_lookup claims |

Not applicable (module absent): recovery, contradiction resolution, critics.

**Caveat from Rule 3**: an ablation "win" on one seed is adopted only if it
survives the held-out seed — same discipline as tuning.

---

# 8. Sensitivity Analysis — [PARTIALLY DONE; extend offline]

Already swept: (A, k) per pillar via `tune_saturation.py sweep` —
outcome: only S:A=35 survived held-out; E/G tuned values overfit and were
rejected. That result is the template: **the goal of sensitivity analysis
here is demonstrating FLATNESS around defaults (stability), not finding
peaks (peaks at n=30 are overfitting).**

Still to sweep (same offline infra, extend the sweep tool):
β (0.6), evidence threshold (2.5), reconcile base weights (0.7/0.3),
method-trust values, peer-anchor weight (10) + tier confidences, v8 slope
(0.04). All constants are catalogued in FORMULAS.md (~35 knobs) — sweep the
high-leverage ones, document the rest as hand-set until Phase 5 regression.

---

# 9. Calibration Discipline — [ACTIVE POLICY]

Exactly the seed ledger + rules in §0. Additions:
- Grid search happens OFFLINE on training-seed dumps (never fresh-gather).
- Phase-5 weight regression follows the same split, with the extra guard
  that regression targets are percentile-normalized (bcorp pillar scales
  are not 0-100 — established in peer_anchor.py).
- The final evaluation seed stays untouched until the pipeline is frozen.

---

# 10. Explainability Verification — [MOSTLY BUILT; one test to add]

The audit trail already exists end-to-end in memory:
`Contribution` (factor/weight/confidence/delta/points/method/reasoning) →
`SaturationBreakdown` (now threaded onto PillarFormulaScore) →
`PeerAnchorVote.basis` → `ReconciledScore.votes/weights_used` →
`ValidationFlag` (Tier-0) → `GatedOutput.reason`.

- **Reconstructability test [ACTION]**: from a dump row alone, recompute the
  final reconciled score with library code and match to the recorded value —
  this is the same artifact as the §1 golden test; implement once, counts
  for both.
- Known persistence gaps (accepted, Phase-6 scope): the harness stores flag
  *counts* not full ValidationFlags (add a sidecar JSON if audit demands);
  the graph persist path stores claims but not breakdown/gate verdicts;
  PipelineResult has no numeric band fields (gate verdict rides in the
  reasoning string until cutover).

---

# 11. Scalability Evaluation — [reality-checked targets]

Measured baseline (seed=314 runs, 6 workers): n=30 ensemble ≈ 810 s
(~27 s/company wall); n=10 ≈ 430 s (startup-dominated). Per company:
4 serialized LLM calls (3 pillar extractions + 1 holistic) at a 2 s min-gap
gateway, ~25 rate-limited search requests (18 signal + governance +
facility) through a shared 2 s DDG limiter.

Consequences:
- Runtime is **rate-limit-bound, not compute-bound**; it scales linearly in
  company count by construction (independent companies), and the DB signal
  cache makes re-runs cheap.
- Honest throughput ceiling on current infra: **~100–200 companies/hour**.
  The original 10k/100k batch targets are **[DROPPED]** until infrastructure
  changes (paid search APIs, non-serialized LLM capacity, distributed
  workers) — testing them now would only measure the rate limiter.
- Realistic tiers: 30 → 100 → 1,000 (cache-warm). Track: wall-time/company,
  LLM calls/company, search requests/company, cache hit rate, timeout rate.
- Token consumption is not currently metered — **[ACTION]** log it from
  gateway responses if the API exposes usage.

---

# Success Criteria (rewritten to be checkable)

- ✅ Deterministic components byte-stable: §1 suites green in CI. **(exists;
  needs the ACTION move into tests/)**
- ✅ LLM variance bounded: reconciled-score sensitivity to a holistic rerun
  ≤ ±3 pts on fixed signals.
- ✅ **PROVEN**: formula beats the single-shot LLM baseline (upright n=60:
  E +0.272 vs +0.246; Total +0.184 vs −0.132) and v5 beats linear on fixed
  evidence, all pillars, both seeds.
- Ensemble ≥ best individual estimator on fixed-evidence comparison
  (standing check per dump refresh).
- Tier-0 precision: near-zero false drops on spot-audit; injected-bad-claim
  catch rate high on the synthetic suite.
- Gate validity: on mixed populations, range-flagged companies show higher
  mean rank error than point-scored ones; under evidence deletion (§6),
  flag rate rises monotonically.
- Recovery improves low-evidence companies — **[DEFERRED]**, and scoped to
  the web-visible population only (recovery cannot conjure evidence that
  does not exist; the gate emitting a range IS the correct output for
  truly invisible companies).
- End-to-end correlation: no pillar regresses vs the documented baselines
  under fixed-evidence comparison; fresh-run deltas inside the noise band
  are reported as noise.
- Every prediction reconstructable from its dump row (golden test green).

---

# Action Queue (in order)

1. **Move the 35 scratchpad tests into `tests/`** (pytest) — they currently
   die with the session.
2. **Refresh dumps post-Tier-0** (seed 42 + 101) and store expected outputs
   → golden/regression/reconstructability test in one artifact.
3. **Offline ablation + stress suite** on the refreshed dumps (§6/§7 tables
   — one script, no API cost).
4. **Holistic repeatability mini-study** (5–10 reruns on cached signals) to
   confirm the ≤±3 reconciled-sensitivity criterion.
5. **Add evidence-utilization + polarity-conflict-rate columns** to the
   harness CSV (data already computed, just not logged).
6. **Recurring spot-audits** (extraction claims + Tier-0 drops) attached to
   each major backtest run.
7. Final-eval protocol when pipeline freezes: virgin seed, n≥60, both-views
   reporting, per-population split.

---

# Evaluation Philosophy (unchanged, with one addition)

1. **Correctness** — Is the computation mathematically and logically correct?
2. **Robustness** — Does the system remain stable under uncertainty, sparse
   data, and adversarial inputs?
3. **Performance** — Does each stage and the overall pipeline improve
   agreement with trusted ESG benchmarks?
4. **Scalability** — Can the pipeline process large numbers of companies
   efficiently while preserving reproducibility and auditability?
5. **Honesty** — Does the system refuse to assert what it cannot support
   (ranges instead of forced point scores), and do our own evaluations
   refuse the same (noise reported as noise, populations not conflated,
   held-out seeds kept virgin)?

No single metric is sufficient. Confidence in the pipeline comes from
consistent performance across all five dimensions.
