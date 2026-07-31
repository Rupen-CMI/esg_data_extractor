# ESG Scoring — Route Comparison Benchmark

Tests 12 scoring approaches against real, independently-published ESG ratings
(B Corp + Upright), for each of the three pillars (Environmental, Social,
Governance). Every approach is checked on two separate, non-overlapping
groups of companies:

- **Exploration set** (336 companies for E/S/G) — used to try ideas.
- **Confirmation set** (57 companies for E/S/G) — untouched during
  exploration, used only to double-check a promising result isn't a fluke.

**ρ (Spearman correlation)**: −1.0 to +1.0. Measures whether the approach
ranks companies in the same order as their real rating. **0 = no better than
guessing. Negative = actively backwards.** Higher is better.

**Verdict**: `BETTER` / `WORSE` means the difference from the current method
(`base`) is statistically real (95% confidence range doesn't cross zero).
`undecided` means the result could just be noise from which companies
happened to be in that sample — not proven either way.

An approach only counts as a genuine, ship-worthy finding if it says
**BETTER on both** the exploration set and the confirmation set.

---

## Evidence sources used to build the corpus

Every company's evidence was gathered from the pipeline's full collector set
(one pass per company, then frozen so all 12 approaches score the exact same
evidence):

**News (real-time):**

| # | Source | What it provides |
|---|---|---|
| 1 | `news_api` | NewsAPI.org ESG keyword search — **disabled for this corpus** (user decision: poor yield; remains on in production) |
| 2 | `google_news_rss` | Google News headline search for the company |
| 3 | `reuters` | Reuters coverage (site-restricted search) |
| 4 | `bloomberg` | Bloomberg coverage (site-restricted search) |
| 5 | `financial_times` | Financial Times coverage (site-restricted search) |
| 6 | `esg_today` | ESG Today (ESG trade press) coverage |
| 7 | `greenbiz` | GreenBiz (sustainability trade press) coverage |
| 8 | `localized_esg` | Native-language ESG news in the company's home country (19 supported markets, e.g. German CSRD/Lieferkettengesetz terms, India BRSR) |

**Specialist ESG databases & registries:**

| # | Source | What it provides |
|---|---|---|
| 9 | `bhrrc` | Business & Human Rights Resource Centre — human-rights allegations |
| 10 | `sbti` | Science Based Targets initiative — emissions-target commitments |
| 11 | `cdp` | CDP — climate disclosure participation |
| 12 | `gri` | Global Reporting Initiative — reporting-standard usage |
| 13 | `net_zero` | Net-zero pledge search |
| 14 | `sustainability_report` | The company's own sustainability/ESG report (web search) |
| 15 | `controversies` | Violations / scandals / fines search |
| 16 | `wikipedia` | Company background summary |

**Structured collectors (separate from the news pool):**

| # | Source | What it provides |
|---|---|---|
| 17 | `country_governance` | The company's home-country corporate-governance regime — named governance codes, statutes and regulators (e.g. India's SEBI LODR, Germany's Corporate Governance Kodex / Aufsichtsrat, Japan's 会社法, Taiwan's 公司治理守則), queried in the country's own language via its Google News edition plus regulator/law-firm web search. **Added after this benchmark run** — 30 country profiles, cached per (country, industry). Fills the G-pillar gap for the ~46% of companies with no company-level governance evidence; note it is constant within a country, so it improves coverage and calibration rather than within-country ranking |
| 18 | `governance_collector` | SEC DEF 14A board-independence text, SEC 10-K legal proceedings, Wikidata board size, plus web searches for board/fines/compliance/litigation (7 sub-sources, tagged `gov_*`) |
| 19 | `facility_extractor` | SEC 10-K "Properties" section + web search for factory/facility footprint |
| 20 | `company_metadata` | Wikidata → GLEIF → OpenStreetMap chain: employees, revenue, industry, country, HQ |
| 21 | `climate_trace` | Climate TRACE satellite-derived emissions database — real facility emissions when the company owns tracked facilities, else country×sector intensity |

Plus two non-evidence inputs the formula uses: **World Bank country ESG
baselines** (the starting score per country) and **peer statistics** from the
B Corp / Upright databases (how similar companies score).

---

## What each of the 12 approaches means

The current method (`base`) blends two ingredients: a **rule-based formula**
(math using real disclosed numbers like emissions, board makeup, etc.) and an
**AI judgment** (the AI reads all the evidence and forms its own opinion),
weighted 70% formula / 30% AI. Every other approach changes that recipe one
way or another.

| Approach | In plain language |
|---|---|
| **base** | The current method: 70% formula + 30% AI judgment, blended together. |
| **no_holistic** / **formula_only** | Throw away the AI's opinion entirely. Score using only the rule-based formula. (Two names, same test.) |
| **holistic_only** | Throw away the formula entirely. Score using only the AI's own judgment. |
| **blend_60_40** | Same idea as base, but shift the mix to 60% formula / 40% AI. |
| **blend_50_50** | Shift the mix to an even 50% formula / 50% AI. |
| **no_peer_anchor** | Remove the "compare this company to similar companies" ingredient from the formula, keep everything else. |
| **baseline_only** | Ignore all company-specific evidence. Just use the country's average score as the answer. (Tests: is our evidence even helping, or would a dumb average do just as well?) |
| **peer_baseline** | Instead of starting from the country average, start from "how do similar companies score" and build from there. |
| **peer_baseline_blend** | A middle ground between the country average and the peer-comparison starting point, weighted by how much we trust the peer data. |
| **no_tier0** | Skip a data-cleaning step that normally filters out low-quality or contradictory evidence before scoring. |
| **no_freshness** | Ignore how recent a piece of evidence is — treat a 5-year-old news article the same as one from last week. |

---

## E (Environmental)

| Approach | Exploration ρ | Verdict | Confirmation ρ | Verdict |
|---|---|---|---|---|
| base | 0.308 | — | 0.201 | — |
| no_holistic / formula_only | 0.221 | WORSE | 0.091 | undecided |
| holistic_only | 0.323 | undecided | 0.310 | undecided |
| blend_60_40 | 0.317 | undecided | 0.224 | undecided |
| blend_50_50 | 0.323 | undecided | 0.241 | undecided |
| no_peer_anchor | 0.317 | undecided | 0.125 | undecided |
| baseline_only | 0.395 | BETTER | 0.274 | undecided |
| peer_baseline | 0.121 | WORSE | 0.182 | undecided |
| peer_baseline_blend | 0.207 | WORSE | 0.163 | undecided |
| no_tier0 | 0.302 | undecided | 0.183 | undecided |
| no_freshness | 0.288 | undecided | 0.264 | undecided |

**Verdict: no approach beats `base` on both sets. Current method stands.**

---

## S (Social)

| Approach | Exploration ρ | Verdict | Confirmation ρ | Verdict |
|---|---|---|---|---|
| base | 0.155 | — | −0.036 | — |
| no_holistic / formula_only | 0.040 | WORSE | −0.165 | undecided |
| **holistic_only** | **0.466** | **BETTER** | **0.560** | **BETTER** |
| blend_60_40 | 0.208 | BETTER | 0.045 | undecided |
| blend_50_50 | 0.274 | BETTER | 0.128 | undecided |
| no_peer_anchor | 0.221 | BETTER | 0.040 | undecided |
| baseline_only | 0.218 | BETTER | 0.055 | undecided |
| peer_baseline | 0.336 | BETTER | 0.203 | undecided |
| peer_baseline_blend | 0.253 | undecided | 0.127 | undecided |
| no_tier0 | 0.177 | undecided | −0.143 | undecided |
| no_freshness | 0.164 | undecided | −0.090 | undecided |

**Verdict: `holistic_only` (AI judgment alone, no formula) is confirmed
BETTER on both sets — the one genuine finding in this whole benchmark.**

---

## G (Governance)

| Approach | Exploration ρ | Verdict | Confirmation ρ | Verdict |
|---|---|---|---|---|
| base | 0.079 | — | 0.229 | — |
| no_holistic / formula_only | 0.133 | undecided | 0.124 | undecided |
| holistic_only | −0.082 | undecided | 0.263 | undecided |
| blend_60_40 | 0.055 | undecided | 0.280 | undecided |
| blend_50_50 | 0.034 | undecided | 0.269 | undecided |
| no_peer_anchor | 0.001 | undecided | 0.220 | undecided |
| baseline_only | 0.005 | undecided | 0.273 | undecided |
| peer_baseline | 0.189 | undecided | 0.095 | undecided |
| peer_baseline_blend | 0.140 | undecided | 0.163 | undecided |
| no_tier0 | 0.087 | undecided | 0.086 | undecided |
| no_freshness | 0.128 | undecided | 0.034 | undecided |

**Verdict: nothing has enough signal either way. Governance ground truth
sample is smallest (only B Corp companies have it), so no approach clears
the bar for a confirmed result yet.**

---
