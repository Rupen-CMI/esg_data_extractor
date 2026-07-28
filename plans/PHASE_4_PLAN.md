# Phase 4 — Verification Layer + Bad-Estimate Handling

> Implementation plan for the adversarial verification layer of the agentic
> estimation pipeline. Sits after Phase 3's Reconcile step. See
> `UPDATED_AGENTIC_WORKFLOW.md` for the full target architecture this slots
> into (that document supersedes the original `AGENTIC_WORKFLOW.md` and
> records the routing/deferral decisions this plan builds on).

## 1. Context & Status

Reconcile (Phase 3) produces a per-pillar `ReconciledScore` with an honest
spread-based confidence label, but a hallucinated/misread claim (the
Phase-1 "yoga page" class of error — a real, cited source that is
topically irrelevant, or subtly misread) can still flow into the final
score. Phase 4 adds the verification back-stop in two layers: a
**deterministic Tier-0 pass** (already built) and a **gated adversarial
LLM critic panel** (this plan's remaining build).

**Live evidence this session that motivated the deterministic layer**: a
recovery probe against the seed=314 backtest sample reproduced the
yoga-page failure twice with real data — a factor-targeted search for
"Copastur Turismo ... gender pay ... workplace safety" returned genuine,
on-topic-looking web text that was entirely about a Hawaiian restaurant; a
Google News query for "MW Enterprises" returned an unrelated person,
"Melissa Wyatt". Both would have passed source-attribution alone.

### Status table

| Component | Status | Where |
|---|---|---|
| QC evidence-sufficiency (`qc_assess`) | ✅ **DONE** | `agentic_estimation/layer_3/confidence_gate.py` |
| Tier-0 deterministic validators (lexical relevance, numeric bounds, polarity consistency, corroboration, known-failure shape) | ✅ **DONE** — 14 synthetic tests pass | `agentic_estimation/layer_2/claim_validators.py` |
| Confidence Gate (point vs range + needs_review) | ✅ **DONE** — 10 synthetic tests pass | `agentic_estimation/layer_3/confidence_gate.py` (`gate()`) |
| Saturation breakdown threaded through | ✅ **DONE** | `PillarFormulaScore.breakdown`, `agentic_estimation/layer_3/formula_estimator.py` |
| Harness + graph wiring for the above | ✅ **DONE**, live-verified (Nvidia + thin-SME spot-checks, n=10 and n=30 backtest runs) | `calibration_harness.py`, `graph.py` |
| **Critic panel** (3 gated LLM lenses) | ⏳ **TO BUILD** (this plan) | new `agentic_estimation/layer_4/critic_panel.py` |
| **Estimate verifier** (orchestration + bounded retry) | ⏳ **TO BUILD** (this plan) | new `agentic_estimation/layer_4/estimate_verifier.py` |
| Objection channel on extractors | ⏳ **TO BUILD** (this plan) | `agentic_estimation/layer_2/pillar_extractors.py` |
| Harness/graph wiring for the verifier | ⏳ **TO BUILD** (this plan) | `calibration_harness.py`, `graph.py` |

**IMPORTANT — do not re-build what's done.** `qc_assess()` and `gate()` are
finished, tested, and wired. This plan's verifier **composes** them (calls
them, does not replace them) — the only genuinely new code is the critic
panel, the retry orchestration, and the objection channel.

**Deferred, not in this plan** (see `UPDATED_AGENTIC_WORKFLOW.md`'s "Build
Status & Verdicts" section for the evidence): Evidence Recovery,
Contradiction Resolution, Targeted Retry Planner. The probe showed the
population's problem is evidence *not existing* online, not retrieval or
reconciliation quality — those three modules bet on a data reality this
project doesn't have yet.

---

## 2. Locked Design Decisions

1. **GATED, and now cost-refined.** Critics run per-pillar only when there
   is genuine ambiguity to adjudicate:
   - `qc == 'thin'` OR `reconciled.confidence == 'low'` → **already** routed
     to range + needs_review by the built Confidence Gate. Critics add
     nothing here — a critic can refute or pass, but it cannot manufacture
     evidence or shrink a spread that's real. Running the panel on an
     already-honest range would only burn LLM calls with no possible
     upgrade. **Skip.**
   - `reconciled.confidence == 'high'` AND `qc == 'ok'` → Formula and
     Holistic already agree closely on real evidence. **Skip** (original
     locked decision, unchanged).
   - `reconciled.confidence == 'medium'` AND `qc == 'ok'` → the one zone
     where two independent estimators partially disagree on real evidence.
     This is exactly where a third, adversarial check earns its cost.
     **Run the panel.**

   Measured invocation baseline this session: thin populations (seed=314
   SMEs) → ~0 critic calls/company (everything is already thin/low). Rich
   companies → 1-2 of 3 pillars land in the medium+ok zone (Nvidia spot
   check: E and G were medium-confidence + QC-ok; S was QC-thin and
   skipped straight to range).

   **Known, accepted risk — agreement-by-corruption.** The 'high' skip
   (present since the original locked decision) has a theoretical hole: a
   bogus NEGATIVE claim lowers the formula score, typically *toward* the
   holistic vote (which tends to sit mid-40s), so a fake can SHRINK the
   spread, flip the label to 'high', and skip the panel entirely.
   Mitigations already in place: Tier-0's lexical gate (crude fakes never
   reach scoring) and the corroboration cap (a single-source high-weight
   negative claim is confidence-capped ×0.7 before it can swing the
   formula far). Fully closing the hole would mean always-on critics —
   rejected on cost. Documented here so a future incident isn't a
   surprise.

2. **DISTINCT LENSES** — the 3 critics check different failure modes
   (evidence-support / peer-plausibility / internal-consistency), not the
   same prompt three times. Diversity catches failure modes redundancy
   can't. (Unchanged from the original plan.)

3. **RE-CRITIQUE ONCE** — after the bounded retry, the corrected estimate
   passes through the panel one more time; a second refute is final (range
   + human review), no second retry. (Unchanged.)

4. **Refute-type routing simplified to convergence** (new — replaces the
   original 3-way contradiction/insufficient/implausible split, which
   assumed Contradiction Resolution existed to receive one of the
   branches). Since Contradiction Resolution is deferred:
   - **Converged flagged_factor** (≥2 critics name the same factor) = a
     specific, fixable claim → bounded retry — **but ONLY if the factor is
     actually retryable**: the flagged factor's winning claim must have
     `method == 'extracted'`. Contributions also include `_peer_anchor`
     (a computed peer statistic) and `dataset_lookup` factors (Climate
     TRACE anchors) — re-extraction cannot change either, so a converged
     flag on a non-retryable factor is treated as no-convergence → Range +
     Review (with the objection recorded; a disputed dataset/peer input is
     reviewer material, not an extraction fix).
   - **Everything else** (no convergence — insufficient evidence,
     implausibility, or a genuine contradiction between estimators) →
     Range + Review directly, immediately, no retry. Retrying can't
     manufacture evidence, and there is no Contradiction Resolution module
     to route a contradiction-type refute to.
   - Log which case fired. A refuted-without-convergence event where the
     underlying cause was a genuine claim conflict is exactly the signal
     that would justify eventually building Contradiction Resolution — see
     `EVALUATION_STRATEGIES.md` §4 for the precondition counter this
     feeds.

5. **The critic retry is legitimate, not a rerun.** The deferred Targeted
   Retry Planner's rule ("no retry without changed evidence") does not
   block this: the objection text injected into re-extraction IS new
   input — a reviewer's specific dispute of a specific reading, not a
   blind rerun on identical evidence. Formula recompute after retry is
   free (deterministic, no LLM cost). The holistic vote is **never**
   re-run (unchanged evidence → same vote would just add noise, per the
   "Holistic should never be rerun on identical evidence" rule).

---

## 3. Per-Pillar Flow

```
reconcile_all()                              [DONE]
      │
      ▼
qc_assess(formula_scores)                    [DONE]
      │
      ▼
   ┌──────────────────────────────────────────────────────────┐
   │ qc=='thin' OR confidence=='low'?                          │
   │   YES → gate() emits RANGE + needs_review   [DONE]        │
   │         verdict='skipped' (no critics ever invoked)        │
   │   NO, confidence=='high' AND qc=='ok'                      │
   │         → gate() emits POINT                [DONE]         │
   │         verdict='skipped'                                  │
   │   NO, confidence=='medium' AND qc=='ok'                    │
   │         → CRITIC PANEL                       [BUILD]       │
   └──────────────────────────────────────────────────────────┘
                              │
                     run_critic_panel(...)
                              │
              ┌───────────────┴───────────────┐
              │                                │
        majority PASS                    majority REFUTE
              │                                │
      verdict='passed'              converged flagged_factor?
      POINT, unchanged                         │
                              ┌─────────────────┴─────────────────┐
                              │ YES                                │ NO
                        Bad-Estimate Handler                 RANGE + needs_review
                        (bounded retry, 1x):                 verdict='refuted'
                        - re-extract pillar w/ objection      (insufficient evidence /
                        - recompute formula (free)             implausibility /
                        - re-reconcile vs SAME holistic         contradiction between
                        - re-run panel ONCE                     estimators)
                              │
                    ┌─────────┴─────────┐
                 round-2 PASS       round-2 REFUTE
                    │                    │
          verdict='passed_after_retry'  RANGE + needs_review
          POINT                        verdict='refuted'
                                        confidence forced 'low'
```

---

## 4. New Files

Both in `agentic_estimation/layer_4/` (alongside the existing
`evaluator_agent.py` / `explainability_agent.py`, which the ensemble path
does not use).

### `critic_panel.py` — the 3 lenses + majority logic (no DB)

```python
run_critic_panel(pillar, company, reconciled, formula_score, holistic,
                 claims, signals, metadata)
    -> CriticPanelResult(verdicts: list[CriticVerdict], refuted: bool,
                          flagged_factor: Optional[str], objections: list[str])
```

- Each critic = one `zen_client.call_with_prompt(prompt, max_tokens=800,
  timeout=120, system=...)` — same import pattern as
  `pillar_extractors.extract_pillar_claims` (`from zen_client import
  call_with_prompt`, line ~173 of that file). Every call serializes through
  `zen_client`'s process-wide `_throttle()` min-gap (see `zen_client.py`
  `_MIN_GAP_S` / `_throttle()`) — no new rate-limit exposure, same shared
  budget the extractors and holistic estimator already use.
- Forced-JSON output: `{"verdict": "pass"|"refute", "flagged_factor":
  str|null, "objection": str}`, parsed with the existing
  `extract_json_object` (`agentic_estimation/shared/llm_json.py`) and
  code-validated in the same fail-closed style as `pillar_extractors.py`'s
  claim validation: verdict coerced to the enum; `flagged_factor` must be
  one of THIS pillar's actual `PillarFormulaScore.contributions[].factor`
  values, else nulled (a critic can't flag a factor that isn't in the
  audit trail).

- **Critic A — evidence-support**: input = each contribution's `factor`,
  `claim_reasoning`, `confidence`, AND the cited signal's text excerpt
  (~800 chars from the `signals` dict, looked up via
  `signals.get(claim.source_tag)` — the exact same lookup pattern Tier-0's
  `_check_lexical_relevance` and `formula_estimator._contribution_for_factor`
  already use). Task: *"for each contribution, does the quoted signal text
  actually assert what the claim says? Quote the exact mismatch."*

  This is the layer that catches what Tier-0 structurally cannot: Tier-0's
  lexical-relevance rule is a coarse word-boundary vocabulary match (does
  the text contain ANY topic term for this factor) — it catches the
  wrong-ENTITY case (Hawaiian restaurant text has zero labor-topic
  vocabulary). It cannot catch a **subtle misread**: text that genuinely
  discusses the right topic but does not actually assert the claimed
  fact (e.g. cited text: *"the company published a sustainability report
  discussing its emissions methodology"*; claim:
  `environmental_controversy`, polarity −1, confidence 0.9 — passes the
  lexical gate, since "emissions" is on-topic, but asserts nothing
  controversy-shaped). This is Critic A's actual reason to exist, and the
  case its acceptance test targets (§7).

- **Critic B — peer-plausibility**: input = reconciled score, country
  baseline + `baseline_source`, `peer_anchor` basis line (real peer
  percentile from `PillarFormulaScore.peer_anchor.basis`), contribution
  point totals, **plus the QC breakdown numbers** (`evidence_mass`,
  `coverage`, `coverage_multiplier` from `PillarFormulaScore.breakdown`,
  now available since the breakdown-threading change; `breakdown` is
  `None` on the legacy linear path — omit those lines from the prompt
  gracefully rather than erroring). Task: *"is the
  final score arithmetically plausible given the baseline, the real peer
  statistic, and how much evidence actually supports it? A large swing on
  thin/weak evidence is implausible — show the arithmetic."*

- **Critic C — internal-consistency**: input = Formula's contribution
  reasonings vs Holistic's per-pillar reasoning text
  (`ESGScore.e_reasoning` / `.s_reasoning` / `.g_reasoning` — see
  `holistic_estimator.py`) + both vote values + `reconciled.spread`. Task:
  *"do the two estimators' narratives contradict each other (e.g. Formula
  credits strong positive evidence while Holistic's reasoning says none
  was found)? Contradiction = refute."*

- **Majority**: ≥2 of 3 refute → `refuted=True`. **Convergence**: ≥2
  refuters name the SAME `flagged_factor` → that becomes
  `CriticPanelResult.flagged_factor`. A critic whose LLM call
  fails/parses empty ABSTAINS (majority computed over responders; <2
  responders → panel returns `refuted=False` with a logged warning —
  fail-open but visible, never a hard crash — consistent with how
  `claim_validators.py` and `confidence_gate.py` both fail open/observable
  rather than raising).

### `estimate_verifier.py` — orchestration: gate composition + bounded retry loop

```python
verify_reconciled(company, reconciled, formula_scores, holistic,
                  claims, signals, metadata, country)
    -> dict[str, VerifiedScore]

VerifiedScore(pillar, mode: 'point'|'range', score, low, high,
              confidence: str, needs_review: bool,
              verdict: 'skipped'|'passed'|'passed_after_retry'|'refuted',
              retried: bool, objections: list[str], reason: str)
```

`VerifiedScore` is the final output object for a verified run — it wraps
(not replaces) today's `GatedOutput` from `confidence_gate.gate()`.
`estimate_verifier.py` imports and calls `qc_assess` + `gate` exactly as
the harness/graph do today; **`confidence_gate.py` itself needs no
changes.**

Per pillar:

1. Call `qc_assess({pillar: formula_scores[pillar]})` and
   `gate({pillar: reconciled[pillar]}, qc)` — reuse the built functions
   directly.
2. If the resulting `GatedOutput.mode == 'range'` → wrap straight through:
   `VerifiedScore(..., verdict='skipped', mode='range', ...)`. No critic
   call. (Covers both the thin-QC and low-confidence skip cases from §2.)
3. If `mode == 'point'` and `reconciled[pillar].confidence == 'high'` →
   also wrap straight through, `verdict='skipped'`, `mode='point'`.
   (The high-confidence skip case.)
4. If `mode == 'point'` and `confidence == 'medium'` → run
   `run_critic_panel(...)`:
   - **pass** → `VerifiedScore(verdict='passed', mode='point',
     score/low/high from the ORIGINAL gate output, unchanged)`.
   - **refuted WITH converged flagged_factor** (and the factor is
     retryable per §2 decision 4 — winning claim `method == 'extracted'`;
     otherwise fall through to the no-convergence branch below) →
     Bad-Estimate Handler:
     - Re-run `extract_pillar_claims(pillar, company, signals, metadata,
       objection=...)` for THAT pillar only. `objection` carries the
       flagged factor + the critics' objection texts + the instruction:
       *"a reviewer disputed this factor's claim; re-read the cited
       signals and either correct it, defend it by quoting the exact
       supporting sentence verbatim, or drop it."*
     - **Replace ONLY that pillar's `method == 'extracted'` claims** with
       the re-extraction output — dataset_lookup (Climate TRACE anchor)
       and any other non-extracted claims for the pillar are PRESERVED
       (re-extraction cannot regenerate them; a blind replace would
       silently delete real dataset evidence). Then re-run
       `validate_claims(...)` (Tier-0 — the retry's new claims must pass
       the same gate as any other) and recompute
       `compute_formula_scores` for that pillar (deterministic, no LLM
       cost).
     - **Re-extraction failure fails closed**: if the retry's
       `extract_pillar_claims` call errors or returns `[]` (LLM failure,
       parse failure), do NOT rescore the gutted pillar as if corrected —
       treat it as a round-2 refute: `verdict='refuted'`, range +
       `needs_review=True`. A "correction" built from an empty
       re-extraction is not a correction.
     - Re-reconcile that pillar via `reconcile_all` against the
       **UNCHANGED** holistic vote (no new holistic call — identical
       evidence rule for Holistic still holds; only Formula's inputs
       changed).
     - Re-run `qc_assess` + `gate` + the panel **ONCE** on the new
       reconciled pillar.
     - Round-2 pass → `verdict='passed_after_retry'`, `mode='point'`.
     - Round-2 refute (or round-2 QC/confidence itself routes to range) →
       final: `verdict='refuted'`, `mode='range'` using the round-2
       reconcile's low/high, `needs_review=True`, `confidence` forced to
       `'low'`.
   - **refuted WITHOUT factor convergence** (insufficient evidence /
     implausibility / genuine estimator contradiction — §2 decision 4) →
     immediately `verdict='refuted'`, `mode='range'` (the ORIGINAL gate's
     low/high), `needs_review=True`, `retried=False`. Log the refutation
     reason (feeds the Contradiction-Resolution precondition counter in
     `EVALUATION_STRATEGIES.md`).

Max 1 retry, max 2 panel rounds per pillar — bounded by construction,
matching the original locked decisions.

---

## 5. Modified Files

- **`agentic_estimation/layer_2/pillar_extractors.py`** — add optional
  `objection: Optional[str] = None` parameter to `extract_pillar_claims`
  (current signature at line ~166: `extract_pillar_claims(pillar, company,
  signals, metadata=None)`); when set, a "REVIEWER OBJECTION — address
  before extracting" section is appended into `_PROMPT_TEMPLATE.format(...)`
  (the format call is at line ~179). No behavior change when `None` — all
  existing callers (harness, graph, `extract_all_claims`) unaffected. This
  is the explicit channel — NOT the signals-dict piggyback, which would
  fabricate a citable `source_tag` that was never a real gathered signal.

- **`agentic_estimation/calibration_harness.py`** — new `--verify` flag
  (opt-in, ensemble scorer only). Insertion point confirmed at the current
  line numbers: inside `_estimate_one_via_ensemble`, right after the
  existing `qc = qc_assess(formula_scores)` / `gated = gate(reconciled,
  qc)` calls (~lines 488-489) — `signals`, `metadata`, `country`, `claims`,
  `formula_scores`, `holistic`, `reconciled`, `qc`, `gated` are ALL already
  in local scope, no re-fetching needed. When `--verify` is set, call
  `verify_reconciled(...)` instead of using `gated` directly (or wrap
  `gated` as the no-critics case — verify_reconciled should accept a
  pre-computed `qc`/`gated` pair to avoid recomputing).
  - `BacktestRow` gains `verdict_e/s/g` (str) and `critic_calls` (int, per
    company) columns — flow to CSV automatically via the existing
    `asdict(row)` mechanism.
  - Cache: `--verify` gets its own prefix `verified_` (stacked with the
    existing `t0_`/`ensemble_` prefixes per the pattern in
    `estimate_one`'s cache-prefix block) — a verified run's claims can
    differ from an unverified one after a successful retry, so predictions
    must never be silently shared between the two cache namespaces.
  - Report: new section — verdict counts (skipped/passed/passed_after_retry
    /refuted) per pillar, % skipped (gate effectiveness — should track the
    thin/high-confidence share), needs_review rate, and (diagnostic, not a
    gate, same framing as the existing spread diagnostic) Spearman/mean-
    rank-error comparison between needs_review-flagged and unflagged rows.

- **`agentic_estimation/graph.py`** — `PipelineState += verified: dict`.
  New node `verify_estimate` inserted between `reconcile`
  (`node_reconcile`, wired at line ~477/494) and the existing conditional
  edge (`lambda state: "mark_estimated" if not state.get("dry_run", True)
  else "end"`, lines ~496-498) — dry and persist share this node,
  verification itself writes nothing to the DB.
  `_populate_result_from_ensemble` (line ~581) renders
  verdict/mode/needs_review into the result reasoning the same way it
  already renders the Confidence Gate's RANGE annotation — this plan
  extends that existing rendering, not a separate one.

- **NO new DB tables/columns in Phase 4** — the ensemble persist path
  today only flips the status flag; persisting verdicts + ranges to
  `company_metric_values` is Phase 6 (cutover) work, same as the
  Confidence Gate's bands today.

- **Evaluation/testing infrastructure is explicitly OUT of this plan** —
  moving the 35 existing synthetic tests into a permanent `tests/`
  directory, dump refreshes, telemetry columns, and the broader
  certification suite are tracked separately in
  `EVALUATION_STRATEGIES.md`'s Action Queue. This plan's own acceptance
  tests (§7) are self-contained and do not depend on that queue.

---

## 6. Cost Budget

`zen_client` serializes all calls at its process-wide min-gap (`_throttle()`):

| Case | Extra LLM calls | Wall-clock |
|---|---|---|
| Skipped pillar (thin/low, or high+ok) | 0 | 0 |
| Gated pillar (medium+ok), panel passes | 3 | ~6-10s |
| Worst case (retry + re-critique) | 3 + 1 + 3 = 7 | ~15-25s |

Measured invocation baseline this session (not estimated — observed):
- Thin/invisible-SME populations (seed=314-style): **~0 critic calls per
  company** — every pillar routes to range before the panel is reached.
- Evidence-rich companies (Nvidia spot-check): **1-2 of 3 pillars** land in
  the medium+ok zone that triggers the panel; the rest skip.

This means the cost-refined trigger (§2, decision 1) makes Phase 4's
real-world LLM cost population-dependent in the same way
`UPDATED_AGENTIC_WORKFLOW.md`'s Success Criteria section already documents
for invocation rates generally — cheap on thin populations (by construction
of the skip), bounded on rich ones (at most 2 pillars × 7 worst-case calls
per company). Log `critic_calls` per company (harness column, §5) to
verify this holds at scale rather than trusting the estimate.

---

## 7. Verification (the phase gate, in build order)

1. **Tier-0 precondition (crude fake)**: a fabricated claim whose cited
   text has ZERO topic vocabulary for its factor (the original yoga-page
   shape) — assert `claim_validators.validate_claims` drops it BEFORE
   `estimate_verifier` or any critic ever runs. This is not a new test;
   it asserts the existing validator suite's behavior as this plan's
   entry precondition, since the critic panel must never be the only
   thing standing between a crude fake and the final score.

2. **Critic injection test (subtle fake)** — the panel's actual reason to
   exist: gather real signals for one company once; extract real claims;
   INJECT a claim that PASSES Tier-0 (cited text contains real topic
   vocabulary for its factor) but does not actually assert the claim
   (e.g. text discussing a sustainability report's emissions methodology,
   claim = `environmental_controversy` polarity −1 confidence 0.9). Run
   `verify_reconciled`. Expected: Critic A refutes, naming
   `environmental_controversy`; the Bad-Estimate Handler retries with the
   objection; re-extraction drops or corrects the bogus claim; round-2
   panel passes; final score recovers toward the clean-run value. Assert
   each step from the returned `VerifiedScore` + `objections`.

3. **Clean-estimate control**: same company, no injection — expect
   majority pass or skip, BUT (found during live verification, Nvidia/E,
   2026-07-21): real unmodified extractions can legitimately contain a
   genuinely weak claim critics correctly refute. Live case: an
   `sbti_commitment` claim sourced from an SBTi database LISTING (Nvidia
   under "Manufacturing") was refuted by Critic A — a directory listing is
   not evidence of a validated commitment, and Critic A was right to say
   so. This refuted without convergence (Critic B separately flagged
   `_peer_anchor` for an arithmetic inconsistency) and correctly routed to
   Range + Review, no retry (`_peer_anchor` is non-retryable anyway). This
   is the critic panel WORKING, not a false positive — treat a refute as
   acceptable in this test if the objection is substantively correct on
   inspection; only a refute with a clearly wrong/nonsensical objection is
   a real regression.

4. **Insufficient-evidence / no-convergence path**: a thin-metadata
   company where the panel DOES run (medium confidence, QC ok) but
   refutes without factor convergence — expect immediate `refuted`, range
   + `needs_review=True`, `retried=False` (no retry attempted, per §2
   decision 4).

5. **Thin-company routing (skip verification entirely)**: a seed=314-style
   invisible SME — expect `critic_calls == 0` for that company, all
   pillars `verdict='skipped'`, straight to range. Confirms the
   cost-refined trigger doesn't waste calls on companies the Confidence
   Gate has already correctly flagged.

6. **Harness A/B**: `--scorer ensemble --verify --n 30 --seed 101 --source
   bcorp` vs the same run without `--verify` (seed 101 — the held-out
   validation seed, per the seed ledger in `EVALUATION_STRATEGIES.md`
   §0; NOT 42 or 314, both already used this session for other purposes).
   Gate: no point-pillar Spearman regression on any pillar vs the
   unverified run, AND needs_review-flagged companies show higher mean
   rank error than unflagged ones (diagnostic confirmation the flag marks
   genuinely worse estimates, not a formal gate).

   **Population caveat — set expectations before running**: a random
   bcorp draw is thin-SME-dominated (measured: 29/30 range-flagged on
   seed=314), so the expected outcome here is `critic_calls ≈ 0` and
   verify ≈ no-verify. That makes this A/B a *safety* check (no
   regression, no cost on thin populations — both worth proving), NOT a
   behavioral test of the critics. Critic behavior coverage comes from
   tests 1-5 (synthetic) plus:

6b. **Rich-company behavioral batch**: run `verify_reconciled` end-to-end
   on 3-5 known evidence-rich companies (e.g. Nvidia, Adidas, and other
   large caps with real medium+ok pillars). Expect: panel actually fires
   on 1-2 pillars each, verdicts are majority-pass on clean estimates,
   `critic_calls` matches the §6 budget. This — not the bcorp A/B — is
   where live critic behavior is actually observed.

7. **Cost check**: log per-company `critic_calls`; confirm skipped
   pillars add zero calls, thin companies add zero calls, and the median
   evidence-rich company stays within the §6 budget.

---

## 8. Deferred & Out of Scope

- **Evidence Recovery, Contradiction Resolution, Targeted Retry Planner**
  — all three remain deferred; see `UPDATED_AGENTIC_WORKFLOW.md`'s "Build
  Status & Verdicts" section for the live-probe evidence behind each
  verdict. This plan's refuted-without-convergence path (§2 decision 4)
  deliberately logs enough to feed the Contradiction-Resolution
  precondition counter (`EVALUATION_STRATEGIES.md` §4) — if genuine
  claim-conflict refutations turn out to be common once critics are live,
  that is the signal to revisit building it.
- ~~**DB persistence of verdicts/ranges** to `company_metric_values`~~ —
  **DONE (2026-07-21, Phase 6 cutover)**. `db_migrations/005_ensemble_cutover.sql`
  adds `low_value/high_value/confidence_label/verdict/needs_review`;
  `agentic_estimation/layer_3/ensemble_persistence.py` +
  `graph.node_persist_ensemble_scores` write them under source
  `agentic_ensemble_v1`. `api/v1/esg_data/routes.py` now calls the ensemble
  path directly for every full-pipeline estimation.
- **PipelineResult schema changes** (numeric band/verdict fields) — still
  NOT done; the verifier's output rides in the reasoning string on the
  in-memory `PipelineResult` object (the DB row itself now has the
  structured columns — see above; this is only a display-object gap).
- **Evaluation infrastructure** (permanent `tests/`, dump refreshes,
  injection-suite generalization beyond §7, telemetry columns beyond
  `critic_calls`/`verdict_e/s/g`) — tracked in `EVALUATION_STRATEGIES.md`'s
  Action Queue as a separate workstream, per explicit decision when this
  plan was scoped.
