# ESG Pipeline — Formula & Numeric-Scoring Reference

> A complete map of every formulaic / mathematical / numeric-scoring section in
> the agentic estimation pipeline. Built to coordinate formula changes across
> layers rather than touch them piecemeal. Verified against live code — where a
> docstring and the code disagree, the **code value** is documented here and
> flagged.
>
> **How to read this:** each entry gives the formula, its inputs, its output
> range, why it exists, and its hardcoded constants. Constants are the tunable
> knobs; the summary table at the end lists every one in a single place.

**Pipeline flow (where formulas sit):**

```
Layer 1  Collectors ──── country baseline, peer medians, sector matching, upright proxy
   ↓
Layer 2  Extractors ──── factor weights, benchmark bands, freshness decay, climate-trace anchor
   ↓
Layer 3  Estimators ──── formula score (linear OR v5 saturation), peer anchor, reconcile
   ↓
Layer 4  Verification ── evaluator clamp (LLM-driven), explainability (no math)
```

---

# LAYER 1 — COLLECTORS

## 1.1 Country Baseline (`layer_1/country_baseline_agent.py`)

The single largest numeric module. Produces per-country E/S/G baselines (0–100)
that every formula score starts from.

**What it does:** turns raw World Bank indicators into one 0–100 score per pillar
per country.

**The math, in order:**

1. **Direction maps** — each indicator tagged `+1` (higher-is-better) or `−1`
   (lower-is-better). E has 34 indicators, S has 23, G has 16.
2. **Latest value** — most recent non-null year ≥ 2015 per indicator.
3. **Min-max normalization to 0–100** (per indicator, across all countries):
   ```
   direction +1:  (col - lo) / (hi - lo) * 100
   direction -1:  (hi - col) / (hi - lo) * 100     # inverted
   degenerate (hi==lo):  50.0
   ```
4. **Pillar score = mean of normalized indicators**, gated on ≥50% coverage:
   ```
   if fewer than half the pillar's indicators present  →  NaN
   else  →  mean of available normalized indicators
   ```
5. **NaN → 50.0** default per pillar (if others exist); rounded to 2 dp.

- **Input:** World Bank indicator table + a country string.
- **Output:** `CountryBaseline(e_score, s_score, g_score)`, each 0–100.
- **Why:** gives every score an evidence-free starting point that is a *real
  statistic about the country*, not a fabricated midpoint. This is what fixed the
  old scorer's "if unsure, guess 45–55" failure.

**Fallback chain** (`get_country_baseline_with_fallback`): exact → resolved
(alias/ISO3) → regional (parent country) → **global average** (mean E/S/G over
all ~210 countries). Each tier returns a real baseline; no weighting.

- **Constants:** coverage gate = half the pillar's indicator count; degenerate
  column → 50.0; NaN pillar → 50.0; fallback year → 2020; alias table ~22 rows;
  regional table 4 territory→parent rows.

## 1.2 Peer Median Collector (`layer_1/peer_anchor_collector.py`)

**What it does:** pure-SQL peer lookup + robust central statistic.

- **`peer_median(peers, field)`** — `statistics.median` of a field across peers.
  Median (not mean) deliberately, to resist outliers in messy real peer data
  (a single giant conglomerate or miscoded row can't wreck it).
- **bcorp social proxy** — bcorp has no single social score, so:
  `social = mean(impact_area_workers, community, customers)` (equal weights).
- **`peer_sample_size(peers, field)`** — counts peers with a *non-null* value for
  that specific field, NOT `len(peers)`. Load-bearing: using the raw union size
  would overstate support (e.g. 205 sector peers but only 5 with employee_count).
- **Dedup** — one row per company (most recent year) before any median, so a
  company with 5 years on file doesn't get 5× weight.
- **Constants:** `_MIN_PEERS_FOR_SECTOR_COUNTRY/ONLY = 5` — **defined but unused**
  here (the live floors live in `ratio_estimator.py` and `peer_anchor.py`); treat
  as dead legacy knobs.

## 1.3 Sector Matcher (`layer_1/sector_matcher.py`) — TF-IDF cosine

**What it does:** fuzzy-matches a free-text sector string to a fixed label list
(bcorp's / upright's vocabularies) without embeddings (no torch available).

**The math:**
```
tf     = raw token counts (after stopword drop + light suffix stemming)
idf    = ln((1 + N) / (1 + df)) + 1                      # smoothed, floor 1.0
vec    = {term: tf * idf}
cosine = dot(a,b) / (||a|| * ||b||)                       # 0..1
matched = cosine >= 0.30
```

- **Input:** a query sector string + candidate labels.
- **Output:** `SectorMatch(label, similarity 0–1, matched bool)`.
- **Why:** bcorp's 4 coarse buckets vs upright's ~30 fine labels never exact-match;
  this bridges word-overlap cases ("Beverages"↔"Food and Beverage").
- **Limitation:** zero-token-overlap synonyms ("Car"↔"Automotive") score 0 — the
  reason the `sector_crosswalk.py` exact-mapping tier now sits *above* this.
- **Constants:** `_MIN_SIMILARITY = 0.30`; stopwords (11 terms incl.
  "manufacturing"); stem aliases (auto/car→"auto"); suffix rules.

## 1.4 Upright Pillar Proxy (`layer_1/upright_pillar_proxy.py`)

**What it does:** derives real per-pillar E/S percentile proxies for upright
companies (which have no e/s/g columns, only raw impact sub-components).

**The math:**
```
per column:   percentile = midrank of raw value in that column's full distribution
              (count_below + 0.5*count_equal) / N * 100
"_negative" columns:   inverted  →  100 - percentile     # higher = better everywhere
pillar proxy = plain mean of its component percentiles
```

- **Input:** a company name + pillar (E or S only — **no G proxy exists**).
- **Output:** 0–100 percentile, or None if no components.
- **Why:** upright's raw sub-components are on wildly different scales and can't be
  summed; percentile-ranking makes them unit-free and averageable. Honestly
  labeled a *derived proxy*, not upright's real methodology.
- **Constants:** 10 E columns (e1–e5 ±), 17 S columns (h1–h5 + s1–s5); equal
  weights (no principled basis to weight one impact dimension over another).

## 1.5 Company Metadata (`layer_1/company_metadata.py`) — name matching

- **Bidirectional token overlap** (fuzzy name match):
  ```
  fwd = shared / |query|;  rev = shared / |candidate|
  if fwd < 0.8 or rev < 0.5:  →  0.0     # gated
  else  →  (fwd + rev) / 2
  ```
- **GLEIF/LEI match floor:** 0.6.
- **Manufacturing/services classifier confidences:** manufacturing-only 0.8,
  services-only 0.8, mixed 0.5, unknown 0.0 (deterministic keyword hits).

---

# LAYER 2 — EXTRACTORS

## 2.1 Factor Registry (`layer_2/factor_registry.py`) — the weights

**What a weight is:** pillar-score points of swing at `confidence=1, |delta|=1`.
Hand-set (NOT fitted — Phase 5 to regress against ground truth), chosen so a
fully-evidenced pillar swings roughly **±30 points** around baseline.

**Benchmark-band factors (CORE_METRICS-backed):**

| Factor | Weight | | Factor | Weight |
|---|---|---|---|---|
| scope_1/2/3_emissions | 8 each | | female_board_pct | 6 |
| renewable_energy_pct | 6 | | employee_turnover_rate | 4 |
| total_energy_consumption | 5 | | lost_time_injury_rate | 6 |
| water_withdrawal | 5 | | board_independence_pct | 8 |
| total_waste_generated | 5 | | anti_corruption_policy | 5 |
| female_employees_pct | 5 | | whistleblower_mechanism | 4 |
| | | | esg_report_published | 4 |
| | | | third_party_esg_audit | 5 |

**Event factors (qualitative/boolean):**

| Factor | Pillar | Weight | Dir | | Factor | Pillar | Weight | Dir |
|---|---|---|---|---|---|---|---|---|
| net_zero_pledge | E | 5 | + | | human_rights_incident | S | **12** | − |
| sbti_commitment | E | 6 | + | | workplace_safety | S | 6 | + |
| cdp_disclosure | E | 4 | + | | regulatory_fines | G | 9 | − |
| environmental_controversy | E | 10 | − | | litigation | G | 7 | − |
| sector_emissions_intensity | E | 4 | − | | compliance_certification | G | 5 | + |
| labor_controversy | S | 10 | − | | governance_controversy | G | 8 | − |

- **Note:** highest weight is `human_rights_incident` (12); negative events weigh
  heavier than positive pledges by design (a scandal hurts more than praise helps).
- `delta_shape` = `benchmark_band` iff the CORE_METRIC has a band, else `event`.
  `direction="neutral"` metrics (employee_count, annual_revenue) are excluded.

## 2.2 Benchmark Bands (`layer_3/metric_estimation_agent.py::CORE_METRICS`)

Each scorable metric has a `(value_for_100, value_for_0)` band. Intensity metrics
are normalized per $M revenue first.

| Metric | Dir | Intensity | Band (v100, v0) |
|---|---|---|---|
| scope_1_emissions | lower | per $M rev | (15, 250) |
| scope_2_emissions | lower | per $M rev | (15, 250) |
| scope_3_emissions | lower | per $M rev | (150, 3000) |
| renewable_energy_pct | higher | — | (75, 0) |
| total_energy_consumption | lower | per $M rev | (300, 3500) |
| water_withdrawal | lower | per $M rev | (100, 3000) |
| total_waste_generated | lower | per $M rev | (5, 120) |
| female_employees_pct | higher | — | (50, 10) |
| female_board_pct | higher | — | (40, 5) |
| employee_turnover_rate | lower | — | (5, 30) |
| lost_time_injury_rate | lower | — | (0.2, 5) |
| board_independence_pct | higher | — | (75, 20) |

- **How a band becomes a delta** (in formula_estimator, §3.1): a value at v100
  scores +1, at v0 scores −1, linearly between. Band ordering (`v0 > v100` for
  lower-is-better) makes the same formula handle both directions.

## 2.3 Evidence Freshness (`layer_2/evidence_freshness.py`) — recency decay

**The math:**
```
decay = max(0.5, exp(-days_old / 365))        # e-folding, NOT literal half-life
undated signal:  0.6  (flat penalty)
```

- **Input:** a signal's text (parses the most recent embedded date).
- **Output:** 0.5–1.0 multiplier applied to claim confidence.
- **Why:** a 3-year-old controversy matters less than last month's — but old
  evidence never goes to zero (floor 0.5), and undated ≠ fresh (penalty 0.6).
- **Scope:** applied ONLY to `event`-shaped `extracted` claims. Disclosed data
  (emissions figures, board %) is NEVER decayed — ESG disclosure is annual, so a
  year-old report figure is usually the newest that exists.
- **Constants:** `_HALF_LIFE_DAYS=365`, `_FRESHNESS_FLOOR=0.5`,
  `_UNDATED_FRESHNESS=0.6` (hits floor at ~8.4 months).

## 2.4 Climate TRACE Anchor (`layer_2/climate_trace_anchor.py`)

- **Owner emissions** (deterministic, real data): `SUM(emissions_quantity)` over a
  matched owner's facilities → a real `scope_1_emissions` value. **confidence 0.85**
  (hardcoded high — it's measured data, not a guess). Exact/prefix name match only,
  no fuzzy scoring (a wrong match against 14.5k owners would be worse than a miss).
- **Sector emissions percentile** (weak proxy):
  ```
  percentile = rank_of_company_sector / (n_sectors - 1)     # 0=cleanest, 1=dirtiest
  ```
  Needs ≥3 sectors to rank against; mapping-confidence gate ≥0.6.
  Claim: `sector_emissions_intensity`, polarity −1, strength = percentile,
  **confidence 0.25** (hardcoded low).

## 2.5 Pillar Extractors (`layer_2/pillar_extractors.py`) — numeric coercion

The LLM tags evidence; the code enforces numeric hygiene:
- **Unit normalization:** kt→×1000, Mt/"million"→×1e6, TWh→×3.6e6, GWh→×3600,
  kWh→×0.0036, etc. Unknown unit → value dropped (claim degrades to event shape).
- **Clamps/defaults:** polarity coerced to {−1,0,1} (default 0); strength clamped
  0–1 (default 0.5); confidence clamped 0–1 (default 0.4).

## 2.6 Ratio Estimator (`layer_2/ratio_estimator.py`) — Tier-3 back-fill

**Ordered fallback chain**, each step a real peer median with a hand-set confidence:
```
sector+country peer median   →  confidence 0.3   (needs ≥3 bcorp/upright peers)
sector-only peer median      →  confidence 0.2
coarse size-bucket median    →  confidence 0.15  (needs ≥1)
absent (no claim)            →  confidence 0.0
```
- **Constants:** `_MIN_PEERS_BCORP_UPRIGHT=3`, `_MIN_PEERS_REAL_METRIC=15` (the
  sparse wikirate path needs a much higher floor — a 5-point sample once produced
  a nonsensical median). Uses `peer_sample_size`, not `len(peers)`, for the gate.

---

# LAYER 3 — ESTIMATORS

## 3.1 Formula Estimator (`layer_3/formula_estimator.py`)

The deterministic core. Governing equation:
```
pillar_score = clamp( baseline + Σ_i  weight_i · confidence_i · delta_i ,  0, 100 )
```

**Per-factor delta computation:**
- **benchmark_band** (has a value): normalize by revenue if intensity metric, then
  ```
  s01   = clamp((value - v0) / (v100 - v0), 0, 1)
  delta = 2·s01 - 1                                    # -1..+1
  ```
  No revenue for an intensity metric → degrade to event shape, **confidence ×0.5**.
- **event:** `delta = polarity · strength` (−1..+1).

**Per-factor points:** `points = weight · confidence · delta`. Freshness multiplier
(§2.3) applied to confidence for event/extracted claims first.

**Best-claim selection:** ONE claim per factor, ranked by `(confidence, _METHOD_RANK)`
where `_METHOD_RANK = {dataset_lookup:3, extracted:2, peer_ratio_fallback:1,
coarse_bucket:0}`. Never blends or sums multiple claims on the same factor.

**Peer anchor contribution** (folded in as a pseudo-factor):
```
anchor_delta  = 2·(percentile/100) - 1                 # -1..+1
anchor_points = 10.0 · anchor_confidence · anchor_delta # weight fixed at 10
```

**Final aggregation — two paths:**
- **Linear (default):** `clamp(baseline + Σ points, 0, 100)`.
- **v5 Saturation (`use_saturation=True`):** delegates to §3.2. Every upstream step
  identical; only the final reduction changes.

- **Constants:** `_DEFAULT_BASELINE=50.0`, peer-anchor weight `10.0`,
  intensity-no-revenue penalty `×0.5`.

## 3.2 Saturation Score — v5 (`layer_3/saturation_score.py`)

Alternative final aggregation that **averages** evidence direction and applies
bounded diminishing-returns, instead of the linear sum. Per pillar:

```
Step 0  method-trust confidence:   c'_i = c_i · m(method_i)
Step 1  partition:                  claims  vs  peer-anchor (PA)
Step 2  evidence gate:              M = Σ_claims w_i·c'_i
                                    if M < threshold → drop claims, keep PA only
Step 3  normalized swing:           Δ = Σ_C w_i·c'_i·δ_i / Σ_C w_i        (-1..+1)
Step 4  coverage:                   Cov = Σ_claims w_i / Σ_registry w_i    (PA excluded)
                                    CovMult = β + (1-β)·Cov
Step 5  sign-aware saturation:      A = A_neg if Δ<0 else A_pos
                                    score = clamp(baseline + A·tanh(k·Δ)·CovMult, 0, 100)
```

- **Why each step:** method-trust (real data > LLM-read), gate (ignore flimsy lone
  claims but keep peer differentiation), average (four bad factors don't stack to
  −34, they average to one bounded swing), coverage (a company with 12 evidenced
  factors outweighs one with a single factor), tanh (diminishing returns), sign-aware
  A (bad news can weigh more than good).
- **Denominator is `Σ w_i` — NOT `Σ w_i·c_i`.** The latter cancels confidence for a
  lone claim; this was an explicit bug fixed in the design (v3→v5).
- **Constants:** `_METHOD_TRUST = {dataset_lookup:1.0, extracted:0.9,
  peer_ratio_fallback:0.7, coarse_bucket:0.5, peer_anchor:1.0}`, default 0.9;
  `A_pos=A_neg=40.0`, `k=1.0` per pillar; `β=0.6`; `threshold=2.5`.
  Design anchor: `40·tanh(1)·1 ≈ 30` reproduces the registry's ±30 envelope.
- **Status:** opt-in, flag-gated. Untuned defaults beat the linear formula on every
  pillar in the held-out backtest (E +0.351, S +0.281, G +0.185 vs linear +0.279 /
  +0.259 / +0.145). See EQUATION_CHANGES.md for the full derivation (v1→v5).

## 3.3 Peer Anchor (`layer_3/peer_anchor.py`)

Turns real peer-company scores into a formula vote via percentile-normalization.

```
percentile = midrank of peer median in the full bcorp column distribution
           = (count_below + 0.5·count_equal) / N · 100      # 0..100
```

**Tiered fallback** (first tier clearing its sample floor wins), each with a
hand-set confidence:

| Tier | Confidence | Min peers | Match type |
|---|---|---|---|
| sector_country | 0.5 | 5 | exact |
| sector_only | 0.3 | 8 | exact |
| crosswalk_country | 0.55 | 5 | exact (hand-built crosswalk) |
| crosswalk_global | 0.4 | 5 | exact |
| bcorp_fuzzy_country | 0.45 | 5→8 | TF-IDF fuzzy |
| bcorp_fuzzy_global | 0.35 | 8 | TF-IDF fuzzy |
| upright_fuzzy (E/S proxy) | 0.3 | 8 | fuzzy + upright proxy |
| upright_fuzzy (G) | 0.25 | 8 | fuzzy + net-impact median |
| abstain | 0.0 | — | percentile=None |

- **Order:** exact tiers → crosswalk (exact, hand-built) → fuzzy tiers → abstain.
- **Why percentile:** peer medians live on different scales (bcorp env 0–78.2, gov
  4–25); percentile puts them all on 0–100 to enter the formula.
- **Constant:** empty distribution → percentile 50.0.

## 3.4 Reconcile (`layer_3/reconcile.py`) — ensemble merge

Merges the formula vote and the (demoted) holistic LLM vote per pillar.

```
formula confidence:   c_f = min(1.0, 0.4 + 0.15·n_contributions)   # 0 contribs → 0.4
holistic confidence:  0.5  (fixed cap — the noisiest input)
effective weight:     eff_i = base_weight_i · confidence_i
final score:          Σ (eff_i/Σeff · score_i)   clamped 0–100
```

**Uncertainty from spread:**
```
n≥2:  spread = max - min;  band = [min-2, max+2]
n==1: spread = None;       band = score ± 15
label: 'low' if n==1 or spread>25; 'high' if n==2 and spread≤10; else 'medium'
```

- **`_PILLAR_WEIGHTS` (LIVE VALUE):** flat **0.7/0.3** (formula/holistic) for ALL
  three pillars. ⚠️ **The module docstring advertises per-pillar weights (E 0.75/0.25,
  S 0.45/0.55, G 0.65/0.35) but the executed constant is currently flat 0.7/0.3**,
  marked "TEMP — re-validating post peer-anchor/crosswalk changes." Trust the code.
- **Both-votes-missing fallback:** score 50.0, band [20, 80], confidence 'low'.

## 3.5 Holistic Estimator (`layer_3/holistic_estimator.py`)

No math — a thin wrapper around the old single-shot LLM scorer, returning its E/S/G
vote (or None on failure). Its only numeric footprint is external: reconcile caps its
confidence at 0.5.

---

# LAYER 4 — VERIFICATION

## 4.1 Evaluator (`layer_4/evaluator_agent.py`)

LLM-driven fact-check with at most one correction round. **No formula computes the
corrections** — the LLM returns corrected scores; the code only **clamps them to
[0, 100]** and persists. The "±10 points is fine" consistency tolerance is prompt
text to the LLM, not executed code.

## 4.2 Explainability (`layer_4/explainability_agent.py`)

No numeric corrections, adjustments, or thresholds — pure LLM prose generation.
Scores are only string-formatted (`{score:.1f}/100`) and passed through unchanged.

---

# Every Numeric Knob, in One Place

| Constant | Value | Location | Role |
|---|---|---|---|
| default baseline | 50.0 | formula_estimator | no-country pillar start |
| `_METHOD_RANK` | 3/2/1/0 | formula_estimator | best-claim tiebreak |
| intensity-no-revenue penalty | ×0.5 | formula_estimator | degrade to event shape |
| peer anchor weight | 10.0 | formula_estimator | anchor contribution |
| `_METHOD_TRUST` | 1.0/0.9/0.7/0.5/1.0 | saturation_score | c' multiplier |
| A_pos / A_neg | 40.0 / 40.0 | saturation_score | tanh amplitude |
| k | 1.0 | saturation_score | tanh steepness |
| β (coverage floor) | 0.6 | saturation_score | sparse-evidence floor |
| evidence threshold | 2.5 | saturation_score | claim gate |
| `_PILLAR_WEIGHTS` (live) | 0.7 / 0.3 all pillars | reconcile | formula/holistic split |
| holistic confidence | 0.5 | reconcile | fixed noisy-input cap |
| formula confidence | 0.4 + 0.15·n | reconcile | evidence-count trust |
| confidence labels | spread 10 / 25 | reconcile | high/med/low |
| band pads | ±2 (n≥2), ±15 (n==1) | reconcile | uncertainty range |
| both-missing fallback | 50.0, [20,80] | reconcile | last resort |
| min peers (anchor) | 5 / 8 / 8 / 5 | peer_anchor | sample floors |
| tier confidences | 0.5/0.3/0.55/0.4/0.45/0.35/0.3/0.25 | peer_anchor | vote confidence |
| empty-dist percentile | 50.0 | peer_anchor / upright_proxy | midrank fallback |
| sector-match threshold | 0.30 | sector_matcher | fuzzy accept floor |
| name-overlap gates | fwd 0.8 / rev 0.5 | company_metadata | fuzzy name match |
| LEI/GLEIF floor | 0.6 | company_metadata | entity match |
| classifier confidences | 0.8 / 0.5 / 0.0 | company_metadata | mfg/svc/mixed |
| freshness half-life | 365 days | evidence_freshness | recency decay |
| freshness floor | 0.5 | evidence_freshness | old-evidence min |
| undated freshness | 0.6 | evidence_freshness | no-date penalty |
| climate-trace owner conf | 0.85 | climate_trace_anchor | measured emissions |
| climate-trace sector conf | 0.25 | climate_trace_anchor | weak proxy |
| ratio-estimator confidences | 0.3 / 0.2 / 0.15 / 0.0 | ratio_estimator | fallback tiers |
| ratio-estimator min peers | 3 / 15 | ratio_estimator | sample floors |
| extractor defaults | strength 0.5 / conf 0.4 | pillar_extractors | LLM-output fallback |
| unit multipliers | kt×1e3 … kWh×0.0036 | pillar_extractors | value normalization |
| country coverage gate | ≥50% of indicators | country_baseline | reliability gate |

---

# Change-Coordination Notes

When editing any formula, these cross-dependencies matter:

- **Confidence flows through multiple stages.** A claim's confidence is set by the
  extractor, decayed by freshness, method-trust-adjusted in saturation, then
  aggregated into the formula-vote confidence in reconcile. Changing how confidence
  is computed anywhere ripples downstream.
- **The ±30 envelope is a shared assumption.** Factor weights, the saturation
  A/k defaults, and reconcile's blend all assume a pillar swings ~±30 around
  baseline. Change one and re-check the others.
- **Percentile-normalization is used in 4 places** (peer_anchor, upright_proxy,
  climate_trace sector, country baseline min-max) — all to put heterogeneous scales
  onto a common 0–100. Consistent technique; consistent edge cases (empty → 50).
- **Hand-set vs fitted.** Nearly every constant here is hand-set, explicitly
  deferred to a future regression pass (Phase 5) against bcorp/upright ground truth.
  `calibration_harness.py` is the standing objective function — no formula change
  ships without re-running it and confirming Spearman doesn't regress.
- **Live vs documented divergence to fix:** `reconcile.py` runs flat 0.7/0.3 but its
  docstring claims per-pillar weights — reconcile these before the next tuning pass.
