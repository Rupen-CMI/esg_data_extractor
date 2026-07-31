# The evidence-quantification problem — where we're stuck, what we tried, what's next

Written 2026-07-31 as a standalone brief for a fresh chat to pick up and
explore alternatives. Supersedes/extends `evidence_quantification.md` (the
original problem framing) with everything tried and measured since.

## The problem, stated precisely

The pipeline reliably does three things with a piece of evidence text:
1. Assigns it to a **pillar** (E/S/G)
2. Assigns it to a **factor** (e.g. `labor_controversy`, `litigation`)
3. Assigns it a **polarity** (+1 good / -1 bad / 0 neutral)

It cannot objectively answer the fourth question: **how much should this
specific piece of evidence move the score?** Given "Company X was fined for
toxic waste dumping," is that a -3 point event or a -15 point event? There is
no ground truth at the evidence level to answer this — bcorp/upright only
publish a company's *final* pillar score, never the evidence they considered
or how much each piece of evidence contributed. See `evidence_quantification.md`
for the original, more detailed walkthrough of why this makes standard
supervised ML (regression, IRT, learning-to-rank) inapplicable: they all need
`evidence -> known contribution` pairs, and none exist.

## How this actually hurts us (concrete, not theoretical)

1. **The extractor's `strength` field (0-1, "how severe is this claim") is
   currently an LLM guess with no calibration.** Two claims that are
   objectively very different in severity can get nearly identical strength
   values, or vice versa, because the LLM has no yardstick to measure against
   — it's pattern-matching from training data, not computing anything.
2. **This directly caps how well the pipeline can rank companies against real
   ground truth.** Measured this session via `calibration_harness.py` /
   `ablation_replay.py` on the frozen tune/holdout corpora (bcorp+upright):
   Spearman rank correlation between our predicted pillar scores and real
   ground truth is currently:
   - E: ~+0.30 (tune), ~+0.16-0.19 (holdout)
   - S: ~+0.21 (tune), ~-0.10 (holdout — worse than random)
   - G: ~+0.14 (tune), ~0.00 (holdout — no signal at all)

   The user's target is +0.5 or better per pillar. We are far from that,
   worst in G, and the holdout numbers (the honest test — companies never
   used to fit anything) are meaningfully worse than tune everywhere, which
   itself signals some amount of overfitting risk in anything tuned against
   the small tune corpus.
3. **A related, harder-to-see symptom**: because severity is a guess, a
   trivial ESG-report-published mention and a securities-fraud lawsuit could
   receive the same or similar weight contribution. This isn't hypothetical —
   we caught it directly (see below).

## What we've tried

### Attempt 1: Clustering as severity (this session, implemented and shipped)

**Idea**: group similar evidence claims by text similarity, label each GROUP
once with a severity level (instead of guessing severity per individual
claim), and have every claim inherit its cluster's label.

**Implementation** (`agentic_estimation/layer_2/evidence_clusters.py`):
- Encoding: TF-IDF (word + bigram counts, sparse vectors), NOT a trained
  embedding model — deliberately formula-based/deterministic per project
  constraint (no model weights, fully auditable).
- Clustering: k-means, fixed at k=3 per pillar (9 clusters total) after an
  A/B sweep showed silhouette-picked k (which came out 4/5/7 = 16 total)
  wasn't earning its extra complexity — k=3 matched or beat it on measured
  Spearman on both tune and holdout.
- Centroid initialization: **keyword-seeded**, not random. Each pillar's 3
  clusters are seeded from hand-written keyword phrases anchored to known
  severity levels (e.g. G's severe seed: "million settlement class action
  fraud fine penalty deceptive alleging death"). This was a real fix for a
  real bug — see below.
- Labeling: one LLM call per pillar labels all 3 clusters at once with a
  severity level (0.25/0.5/0.75/1.0) + a one-line rationale. ~9 labels total
  instead of ~700 individual per-claim guesses.
- Split: cluster **label** -> claim's severity/strength. Cosine
  **membership** (how typical the claim is of its cluster) -> a confidence
  multiplier, NOT severity. This split matters: membership measures
  typicality, not badness — a textbook-ordinary event has HIGH membership; a
  catastrophic, unusual one would have LOW membership. Using membership as
  severity would invert exactly the cases that matter most.
- Fitted on the tune corpus only (never holdout), same discipline as every
  other tuning step in this codebase.

**A real bug found and fixed mid-session**: with unseeded k=3 (random
k-means++ init), ALL 3 of G's clusters ended up labeled severity=0.5 —
meaning a routine "published an ESG report" claim and a "securities fraud
lawsuit" claim would get the identical severity multiplier. Root cause,
confirmed by direct inspection: `board_independence_pct` claims ("director,"
"independent," "committee") have essentially ZERO TF-IDF similarity to any of
the 3 seed phrases (they share no vocabulary with fraud, compliance, or
report language). With nothing to attract them, ~59 board-independence claims
landed on the fraud/litigation seed by pure floating-point tiebreak, diluted
that cluster's centroid, and the label step correctly (but unhelpfully)
called the resulting mix "moderate." Fixed by explicitly folding
board-independence vocabulary into the ROUTINE seed instead of leaving it
to chance. After the fix, verified on frozen HOLDOUT data (companies never
used for fitting): G's litigation/fraud cluster correctly isolates at
severity 0.75 (37 pure claims: lawsuit/securities/million/class-action), the
two routine clusters correctly stay at 0.25.

**Current cluster labels** (`evidence_clusters.json`, committed artifact):

| Pillar | Cluster | n (tune) | severity | top terms |
|---|---|---|---|---|
| E | 0 | 7 | 0.75 | violation, environmental, pollution, federal |
| E | 1 | 68 | 0.25 | cdp, disclosure, climate |
| E | 2 | 132 | 0.5 | net zero, target, sbti |
| S | 0 | 22 | 1.0 | rights, bhrrc, human rights, incident |
| S | 1 | 24 | 0.5 | labor, dispute, union |
| S | 2 | 8 | 0.25 | board, female, composition |
| G | 0 | 37 | 0.75 | lawsuit, filed, securities, million, class action |
| G | 1 | 223 | 0.25 | compliance, independent, board, certification |
| G | 2 | 89 | 0.25 | report, sustainability, published |

**Measured effect (A/B, clusters on vs off, `calibration/sweep_cluster_k.py`)**:

| Pillar | Off (tune) | On (tune) | Off (holdout) | On (holdout) |
|---|---|---|---|---|
| E | +0.299 | +0.322 | +0.162 | +0.187 |
| S | +0.210 | +0.206 | -0.099 | -0.105 |
| G | +0.141 | +0.143 | -0.008 | -0.016 |

**Honest verdict**: real, consistent improvement for E (~+0.02-0.03 Spearman,
holds on holdout). Flat-to-noise for S/G — the Spearman metric didn't reward
the G severity-discrimination fix, because G's OVERALL ranking signal is
close to zero regardless of how well-calibrated severity is within it. This
is the key finding: **clustering fixes "how much should this evidence count,"
it cannot fix "we don't have enough/good enough evidence to rank G at all."**
Those are different problems. One remaining known gap: a cluster/membership
mismatch can still misfire when a claim's wording overlaps a wrong-valence
cluster (e.g. a *positive* "zero tolerance on modern slavery" policy claim
matched the human-rights-INCIDENT cluster on holdout data, inheriting a high
severity meant for negative incidents — polarity is stored separately so this
is a magnitude, not direction, error, but it's a live gap, not yet fixed).

### Attempt 2: External enforcement-data severity yardstick (researched, not yet implemented)

**Idea**: for evidence with a monetary component (fines, settlements,
penalties), don't guess severity at all — measure it. Take the company's
fine, divide by its own revenue (so a $2M fine means something different to
Shell vs a $50M company), and percentile-rank that ratio against a large
external database of real fines. The percentile IS the severity, computed,
not guessed.

**What we found available** (see `research/ENFORCEMENT_DATA_SOURCES.md` for
full detail, this session's 4-agent deep research sweep): Violation Tracker US
(~600k records, have it), Brazil IBAMA (~1M+ records, free daily bulk CSV,
amounts + tax IDs), CMS GDPR tracker (3,202 EU fines), China IPE Blue Map
(3.5M records, batch search), OpenSanctions (196-country flag/debarment
layer, no amounts but usable for list-tier/duration/rarity-based severity),
Stanford FCPA/TRACE (916 global bribery settlements), and more.

**Known design risk, resolved but worth restating**: naively pooling ALL
countries' fines into one global distribution would make the biggest dataset
(e.g. Brazil's ~1M rows) dominate the "global" severity norm — the same
skew problem found earlier in the bcorp/upright peer-anchor analysis. Fix:
build ONE ruler per jurisdiction from its own data, and combine ruler CURVES
(not raw rows) with equal weight per jurisdiction in a global fallback —
so row count improves a jurisdiction's own ruler's precision, never its vote
share in the blend.

**Status: designed, not implemented.** No code has been written for this yet
— it was scoped as a follow-on to the clustering work, for after this
session's demo deadline.

### What we have NOT tried

- **TF-IDF + SVD (a formula-based, non-model synonym fix)**: discussed but
  not implemented. Would help TF-IDF recognize e.g. "strike" and "walkout" as
  related via co-occurrence statistics, without any trained model weights.
  Most likely to help S (thin corpus, high vocabulary variance) and least
  likely to help — S is exactly the pillar with too little data (54 claims)
  for co-occurrence statistics to be reliable.
- **Sentence-transformer embeddings** (a real trained model, e.g.
  `all-MiniLM-L6-v2`): would directly solve the synonym/leakage problems TF-IDF
  can't, but was explicitly ruled out by user preference for formula-based,
  auditable methods over model-based ones.
- **Weight refitting against ground truth** (matched-pair / ridge regression
  of `factor_registry.py`'s hand-set weights against the full bcorp/upright
  pool, not just this session's small demo corpus): the most likely lever for
  a real Spearman improvement, discussed at length this session, not yet
  started. This is a DIFFERENT lever than cluster severity — cluster severity
  calibrates "how bad is this KIND of evidence," weight refitting calibrates
  "how many points should this FACTOR swing the pillar score at full
  confidence." Both matter; only cluster severity has been touched so far.

## Where this leaves the +0.5 Spearman target

Not close, honestly, especially on holdout. The clustering work is real and
verified but was never going to single-handedly close this gap — it improves
evidence weighting, and G/S's core problem (measured earlier this session:
G's evidence hit-rate is near-zero for most factors) is an evidence-VOLUME and
weight-CALIBRATION problem that clustering doesn't touch. The two concrete
next levers, in likely-impact order: (1) weight refitting against the full
ground-truth pool, (2) the enforcement-data yardstick for monetary claims,
both discussed above but not yet built.

## Files referenced

- `research/evidence_quantification.md` — original problem framing, five
  candidate techniques evaluated (rule-based, LLM severity, clustering,
  embeddings, measurement models), concludes no universal converter exists.
- `research/ENFORCEMENT_DATA_SOURCES.md` — full verified survey of external
  enforcement/penalty databases for the severity-yardstick idea.
- `agentic_estimation/layer_2/evidence_clusters.py` — the shipped clustering
  implementation (build/label/replay CLI).
- `calibration/sweep_cluster_k.py` — the A/B measurement script (clusters
  on/off, k sweep) against frozen tune/holdout dumps.
- `agentic_estimation/layer_2/factor_registry.py` — the hand-set weights that
  would be the target of weight refitting.
