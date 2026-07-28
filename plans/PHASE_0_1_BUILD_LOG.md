# ESG Agentic Rebuild — Phase 0 & Phase 1 Build Log

Companion to the architecture plan (`all-data-metadata-linked-dijkstra.md`). This
documents what was actually built, what was tested, and every real issue found along
the way — not just a feature checklist.

## Why this rebuild

The original pipeline (`scoring_agent.py`) asked a single LLM call to freehand E/S/G
scores 0-100. Backtested against real ground truth (`bcorp_lookup`, 10,337 companies;
`upright_lookup`, 10,086 companies) via a new calibration harness:

| Pillar | Spearman ρ | pct-MAE |
|---|---|---|
| E | +0.246 | 27.8 |
| S | −0.154 | 35.8 |
| G | +0.045 | 31.1 |
| Total | −0.132 | 37.9 |

Negative/near-zero correlation on S, G, Total. Root cause: the LLM's own instruction
("if evidence is thin, score near 45-55") collapsed most predictions into a narrow
band regardless of real signal strength — e.g. "Spotlight" had the *highest* real
social score in the sample (39.45) but was predicted 45 (i.e. "no idea"). This is
what "LLM guessing" looks like when actually measured, and it's the problem this
rebuild targets: replace freehand scores with evidence extraction + a deterministic
formula, verified by independent estimators and adversarial critics.

---

## Phase 0 — Foundations

**Goal:** stand up the new infrastructure (evidence-claims schema, LangGraph
orchestration) with zero change to estimation logic, so the orchestration rebuild
could be validated in isolation before touching scoring.

### Built
- **`db_migrations/002_evidence_claims.sql`** + `CompanyEvidenceClaim` ORM model
  (`api/v1/models.py`) — a new table for Layer 2 to write typed, source-attributed
  claims into (`{pillar, factor, polarity, strength, confidence, source_id,
  produced_by, method}`), instead of a bare LLM score.
- **`agentic_estimation/graph.py`** — a LangGraph state graph wrapping the existing
  pipeline stages (signals → metadata → scoring → evaluator → metrics →
  explainability). One unified graph; a `dry_run` flag conditionally routes around
  the DB-persisting nodes via a conditional edge. Every node is a thin wrapper
  reusing `orchestrator.py`'s existing `_run_*` helpers unchanged — pure
  orchestration swap, no estimation-logic change.
- Added `langgraph` to `requirements.txt`.
- Extended `calibration_harness.py` with a `--via-graph` flag to route estimation
  through the new scaffold for regression testing.

### Issues found & fixed

**1. Schema-constraint design — verified, not just written.**
The `company_evidence_claims` table has a CHECK constraint: a claim with
`confidence > 0` must have either a `source_signal_id` or a `source_note` — i.e. "no
source, no claim" is enforced by the database, not just prompt wording. Tested
directly against the live DB with three cases: (a) confident claim with no
provenance → correctly rejected; (b) zero-confidence claim with no provenance →
correctly allowed (honest "we know nothing" case); (c) confident claim with a
`source_note` describing a peer-ratio fallback → correctly allowed. All three passed.

**2. Console-logging change silently broke file logging (found via user report).**
Added a console-mirroring feature to the calibration harness so DDG rate-limit
warnings would be visible live. The naive implementation called
`logging.getLogger(name)` directly to attach a console handler — but
`pipeline_logger.get_logger()` short-circuits ("if log.handlers: return log") when a
logger already has *any* handler. So attaching a console handler first meant the
*file* handler never got attached for `signal_agent`, `scoring_agent`,
`evaluator_agent`, `company_metadata` — silently killing file logging for those
modules for an entire run. Root-caused by checking `signal_agent.log`'s line count
before/after and finding zero growth despite a live run in progress. **Fix:** the
console-attaching function now calls `pipeline_logger.get_logger(name)` first
(guaranteeing the file handler exists), then layers the console handler on top.
Verified with a before/after line-count test showing both file and console received
the same log line.

**3. Suspected — then disproved — a graph-orchestration regression.**
After building the graph scaffold, ran the same 30-company B-Corp calibration sample
(seed=42) through it. Result was measurably worse than the original baseline (E ρ
+0.246 → +0.086, Total ρ −0.132 → −0.272), even after excluding one clear outlier.
Rather than assume this was a graph bug, ran a **third** test: the *original*
(non-graph) code path, fresh, same seed, no cache. That fresh rerun *also* diverged
substantially from the original baseline (E ρ +0.246 → +0.174, G ρ +0.045 → +0.248) —
by roughly the same magnitude as the graph run did. Since both the graph path and a
fresh run of the *unmodified* original path show similar swings, this confirms the
variance is **LLM non-determinism on a small (n=30), already-noisy sample** — not a
defect introduced by the graph rebuild. (One outlier was separately root-caused: a
company named "Spotlight" scored 0/0/0 because the pre-existing `evaluator_agent`
correctly noticed the gathered web signals didn't actually pertain to the target
company — likely a name-collision in search — and zeroed the score in response to "no
company-specific evidence." This is a pre-existing `evaluator_agent`/`signal_agent`
behavior, not a graph defect.)

---

## Phase 1 — Layer 1 (all data/metadata extraction)

**Goal:** consolidate all company data/metadata gathering into one layer, before any
pillar-specific scoring logic runs, per explicit direction to keep Layer 1 as the
single place all extraction happens.

### Built

**`company_metadata.py` (extended)**
- `classify_manufacturing_vs_services()` — deterministic keyword match over
  free-text industry strings (same pattern as the existing
  `wikirate_fetcher.WIKIRATE_KEYWORD_MAP`), explicitly NOT an LLM call. Returns
  `manufacturing` / `services` / `mixed` / `unknown` — "unknown" is a real, honest
  outcome when the industry text gives no signal, not a forced binary guess.
- Brand-name → legal-entity alias resolution (`_BRAND_ALIASES`) — tried before
  falling through to "no match," e.g. `"zara" → "Inditex"`.

**`peer_anchor_collector.py` (new)**
Pure SQL, no LLM call. Finds comparable companies by sector/country/size across
`bcorp_lookup` (10,337 rows), `upright_lookup` (10,086 rows), and real
(non-agentic-source) rows in `company_metric_values`. This is the mechanism that
keeps Tier-3 back-fill honest — every value it returns is a real statistic over real
companies, never an LLM guess.

**`governance_collector.py` (new)**
Fills the evidence category that was root-caused as the direct cause of G's flat
predictions — board composition, regulatory fines/violations, compliance
certifications, litigation records. Reuses `signal_agent`'s existing shared DDG
rate-limiter (`_ddg_fallback`) directly, so it adds zero new rate-limit exposure.

**`facility_extractor.py` (new)**
Factory/facility count, scale, locations. Two-tier source strategy: SEC EDGAR 10-K
"Item 2. Properties" section (free API, no key, highest reliability but US-listed
only) as the primary source, DDG web search as the fallback for everyone else.

**`ratio_estimator.py` (new)**
Deterministic (non-LLM) Tier-3 back-fill with a strict ordered fallback chain:
1. sector+country peer median (confidence ~0.3)
2. sector-only peer median (confidence ~0.2)
3. coarse size-bucket median (confidence ~0.15)
4. absent (confidence 0, no claim written — no fabricated value, ever)

`factory_workforce_share` and `female_employees_pct` (gender ratio) are explicitly
**optional enrichers** per direction: the fallback chain is attempted, but a miss at
every step leaves them absent rather than forcing a weaker estimate through — their
absence never blocks or degrades the rest of the estimate.

### Issues found & fixed

**4. "Zara" — and likely other well-known consumer brands — failed metadata
resolution entirely.**
Live-tested `company_metadata.py` against 15 companies spanning large/global and
small/private. Large companies resolved 10/10 via Wikidata (genuinely global — US,
Germany, Japan, India, Switzerland, South Korea all hit with real employee/revenue
data). Small/private companies resolved 0/5. One large-company miss stood out: Zara
failed completely, despite being a household name — because Wikidata tracks it under
its legal parent entity, "Inditex," not "Zara." **Fix:** added a small, curated
brand-alias table tried before giving up. Verified: Zara now resolves (country=Spain,
industry=retail+textile, correctly classified "mixed").

**5. Board-composition governance queries triggered Wikipedia false-rejects.**
The Governance Collector's board-composition query initially used generic wording
("board of directors independent members"). Tested against Nvidia and got an empty
result. Debug logging showed why: DDG's top result for generic board queries on
well-known companies is almost always the company's own Wikipedia infobox, which the
existing `reject_wikipedia` safety guard correctly discards (that guard exists
specifically to prevent silent Wikipedia bleed-through from corrupting `site:X`
queries elsewhere in the pipeline — weakening it was not an acceptable fix). **Fix:**
changed the query to target proxy-statement-specific phrasing ("proxy statement
independent directors," "audit committee") instead of generic wording, which
naturally avoids Wikipedia. Verified: same query now returns real proxy-statement
text instead of empty.

**6. Governance sources can return real, on-topic-looking, but actually irrelevant
text — not fixed, deliberately documented for Phase 2.**
Testing the Governance Collector against "Blackmores" (a vitamins/supplements
company, chosen specifically as a smaller/less-documented case) returned text that
passed every current filter (real, non-Wikipedia, long enough) but was about yoga
poses and bra fitting — DDG fell back to generic brand-website content when no
governance-specific page existed for this smaller company. This is a real gap: a
`source_id` pointing to a genuinely retrieved document is not the same as that
document being *relevant*. **Not fixed in Phase 1** — flagged explicitly in both code
comments and the architecture plan as a requirement for Phase 2: Extractor agents
must include an explicit relevance check ("does this text actually discuss X") as
part of extraction, not just "extract claims from this text."

**7. `find_peers()` cross-source sector matching silently returns zero for one whole
source — documented, not fixed.**
`bcorp_lookup.sasb_sector` uses a coarse vocabulary (`manufacturing` / `services` /
`apparel_retail` / `general`); `upright_lookup.industry` uses a different,
finer-grained vocabulary (`"Industrial Manufacturing and Services"`,
`"Automotive"`, etc.). Passing a bcorp-style sector string into `find_peers()`
therefore always returns 0 Upright peers, and vice versa, with no error or warning —
discovered when a `manufacturing`-sector query returned real bcorp peers but zero
Upright peers even though Upright clearly has manufacturing companies. **Not fixed**
— normalizing this would require a validated sector-mapping table, which was
deliberately not guessed at without real validation. Documented as a `KNOWN
LIMITATION` docstring in `peer_anchor_collector.py` and in the plan.

**8. Peer-median sample-size threshold was checking the wrong denominator — a real
bug, caught before it shipped.**
While verifying the Ratio Estimator's fallback chain against a real case (small
Spanish manufacturer), it returned `employee_count = 1.0` backed by "205 peers" —
implausible on its face for a median company. Investigation traced it to
`_find_real_metric_peers()`: the real (wikirate-sourced) `employee_count` data has
only 5 rows in the *entire database*, and those 5 values were
`[0, 0.84, 1, 1000, 402614]` — extremely skewed, with at least two clearly-bad rows
(a company reporting 0 or 0.84 employees). The `len(peers) >= 3` sample-size check
was counting the full *unioned* peer list across all three sources (bcorp + upright +
wikirate = 205 records), not how many of those records actually carried an
`employee_count` value (5). So a median computed from 5 noisy points was being
reported as if it had 205 points of support. **Fix:** added `peer_sample_size()` as
the correct per-field denominator, and a much higher trust floor
(`_MIN_PEERS_REAL_METRIC = 15`) specifically for the sparse, noisy wikirate-backed
real-metric path (vs. `_MIN_PEERS_BCORP_UPRIGHT = 3` for the much larger bcorp/
upright-backed paths). Verified: the same test case now correctly returns `absent`
for `employee_count` instead of a fabricated-looking 1.0.

**9. A stale row from an earlier manual DB test caused a confusing false-positive
during automated verification — caught and root-caused, not a real product bug.**
While writing an automated test for `save_ratio_estimates()`'s DB write path, an
unexpected row appeared in the query results (`method='peer_ratio_fallback'` with
`value=None`) that looked like it violated the function's own skip-absent-estimates
logic. Rather than assume the function was broken, traced it back to a leftover row
from an earlier, unrelated manual constraint test in the same session that hadn't
been cleaned up (it happened to reuse `produced_by='ratio_estimator'`). Re-ran the
test with a purpose-built `RatioEstimate` fixture (one non-absent, one absent) and
confirmed exactly one row is written and the absent one is correctly skipped, then
cleaned up test data.

### Verification

Every component above was tested against **real, live data** — real companies, real
DB rows, real SEC filings, real web search results — not mocked. Final Phase 1
check ran all five pieces together end-to-end against one company (Adidas):
correct manufacturing classification, 3/4 governance sources hit, 1/1 facility
source (correctly no SEC match — Adidas isn't US-listed), and the Ratio Estimator
correctly skipped back-fill for factors the Company Profiler had already resolved,
running the fallback chain only for genuinely-missing factors. No exceptions, no
fabricated values anywhere in the chain.

---

## Phase 1.5 — Governance sourcing improvement (before starting Phase 2)

Before building Phase 2's Extractor agents, user direction was to first improve
Governance Collector's evidence quality — G was the worst-performing pillar in the
baseline, and Phase 1 verification had already shown uneven hit rates (Nvidia 3-4/4,
Blackmores 2/4 with one irrelevant-content false-positive). Rather than build an
Extractor on top of weak evidence, added real structured sources first.

### Researched and verified real sources (not assumed)

- **SEC DEF 14A (proxy statement)** — the single richest structured governance
  source for a US-listed company: board composition, committee structure, executive
  compensation. Verified live against a real filing (Nvidia FY2026 proxy) before
  building against it.
- **SEC 10-K Item 3 "Legal Proceedings"** — mandated litigation disclosure, same
  document already fetched for facility data.
- **Wikidata `board_member_count`** — already being fetched by `company_metadata.py`
  for every company but never used downstream; a free, structured, global (not
  US-only) signal.
- Investigated and **rejected** two options: Violation Tracker (Good Jobs First) has
  no public API, would need scraping at the same reliability tier as existing DDG
  fallback — not worth a dedicated integration; SEC EDGAR full-text search only
  covers SEC's own filings, not fines from other regulators.

### Built
- **`sec_filings.py`** (new, shared module) — extracted CIK-resolution and
  document-fetch logic out of `facility_extractor.py` into a reusable module (both
  Facility and Governance collectors need the same SEC access pattern). Generalized
  the "always take the LAST occurrence of a section heading, never the first"
  extraction rule discovered in Phase 1 (the TOC-vs-real-section problem) into a
  shared `extract_section()` used by all SEC-based collectors.
- Refactored `facility_extractor.py` to use the shared module — **verified byte-
  identical output** against Nvidia before/after the refactor, confirming no
  regression from the extraction.
- Extended `governance_collector.py` with 3 new Tier-1 sources
  (`gov_board_sec`, `gov_litigation_sec`, `gov_board_count`) layered on top of the
  existing 4 DDG-based Tier-2 sources (not replacing them — Tier 1 only covers
  US-listed companies, so Tier 2 remains the only coverage for everyone else).

### Issues found & fixed

**10. 10-K Item 3 often contains only a cross-reference, not real litigation
detail — found, documented, not "fixed" (correctly out of scope).**
Verified against both Nvidia and Tesla: companies with *material* litigation
commonly write Item 3 as "see Note 13 of the financial statements" rather than
repeating the detail — standard 10-K practice to avoid duplicating disclosure. This
is real, non-fabricated text, but it's a pointer, not evidence. Following the
cross-reference into the financial-statement notes would require a much deeper
document crawl, deliberately left out of scope. Documented directly in the function
docstring so Layer 2's G Extractor is warned: a boilerplate cross-reference must be
treated as weak/no evidence, not as "confirmed no litigation."

**11. Nearly claimed credit for a fix that was actually DDG variance — caught by
re-testing before writing it down.**
Blackmores' `gov_board` result initially looked meaningfully improved between two
test runs (real board/audit-committee text vs. the earlier yoga-poses false
positive). Before documenting this as a fix, re-ran the *exact same, unmodified*
query a second time — confirming `gov_board` was consistently good but
`gov_compliance` (a source this session's changes never touched) showed real
run-to-run content variance (vitamin-B12 marketing copy vs. real controversy text
on separate calls). Concluded correctly: the apparent board-query improvement was
most likely DDG variance from the original Phase 1 test, not a result of any change
made this session — since the query wording for `gov_board` was never edited in
this round. Reported honestly rather than overclaiming.

**12. CLI crashed on Windows console encoding — found while testing Adidas.**
`governance_collector.py`'s CLI printer crashed with `UnicodeEncodeError` when
scraped web text contained characters outside `cp1252` (Windows' default console
encoding) — e.g. curly quotes/em-dashes common in web copy. Not a data problem, a
display bug. **Fix:** encode/decode through the actual stdout encoding with
`errors="replace"` before printing, so unsupported characters degrade to `?` instead
of crashing. Verified: same Adidas query that previously crashed now prints cleanly
(and separately surfaced a real, if partially garbled by DDG's own indexing,
trademark-dispute signal — Thom Browne vs. Adidas — a genuine data point, not a bug).

### Verification

Ran the improved collector against 4 real companies: Nvidia (6/7 sources, real DEF
14A + board count + a real China SAMR antitrust investigation surfaced by the
existing DDG fallback), Tesla (6/7 sources, real DEF 14A board addition + a real
California hazardous-waste settlement), Blackmores (2/7 — correctly zero Tier-1 SEC
hits since it's Australian-listed, not US; confirms the collector degrades honestly
rather than fabricating for non-US companies), Adidas (2/7 — same correct non-US
degradation, one genuine new signal: a real DOL $235,000 safety-violation
settlement). Net result: G now has real, specific, high-value evidence
(investigations, settlements, board changes) for US-listed companies that the
original collector never surfaced at all.

---

## What's next

Phase 2 (per the architecture plan): define the full E/S/G factor list with
hand-set starting weights, build the schema-forced Layer 2 Extractor agents (with
the relevance-check requirement from issue #6 above baked in), and the deterministic
Formula Estimator that replaces the freehand LLM score. Every phase is gated by
re-running `calibration_harness.py` against the documented baseline — no phase ships
if it regresses the harness score from the prior phase.
