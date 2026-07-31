# ESG ground-truth / evidence data sources — research findings

Compiled from four parallel deep-research passes (2026-07-30), triggered by a
measured skew problem: the pipeline's two ground-truth sources (`bcorp_lookup`,
10,337 companies; `upright_lookup`, 10,086 companies) are heavily concentrated
by country (bcorp top-5 = 61.3% of rows, HHI 0.12; upright top-5 = 58.4%, HHI
0.14) and, for bcorp, by size (92% small/medium, only 7.8% "well-known"/large).
Measured effect: peer-anchor tier fallback to loose/global comparison tiers
hits 41.6% of even Anglo-country companies and 50.9% of non-Anglo companies
(`calibration/abl_final_tune400.json` + `abl_final_holdout68.json`, tier field
on `pillars.*.peer_anchor`).

Goal of this research: find NEW free/low-cost structured ESG data sources to
diversify country + size coverage. Four agents were dispatched in parallel —
EU/UK, Asia, Africa/Middle East/Latin America, and free large-company datasets
— each told to be skeptical and distinguish "actually bulk-accessible" from
"technically public but PDF-only/paid/scraping-required."

**Bottom line across all four:** regulators and exchanges mandate ESG
disclosure almost everywhere, but almost nobody aggregates it into structured,
bulk-accessible data. The recurring trap is mistaking "index constituent list"
or "signatory directory" for an ESG *metrics* dataset — those exist in almost
every region researched and are NOT metric data.

---

## Tier 1 — Actually usable, free, worth building against now

### Wikirate — highest priority, already partially integrated
- **What**: crowdsourced/researcher-curated company ESG "metric answers" (GRI
  codes, KnowTheChain, Fashion Revolution Transparency Index, World
  Benchmarking Alliance projects). Real per-company, per-metric, per-year
  structured values with a source document link — not a single composite score.
- **Scale**: ~150,000 companies, 8M+ data points claimed. Best-populated
  projects (FTSE 100, Fashion Transparency Index's 200 large global brands,
  KnowTheChain) skew LARGE-CAP — directly fixes the bcorp size skew. Global
  spread, not US/UK-concentrated.
- **Access**: free, official Python client `wikirate4py` (PyPI, GitHub
  `wikirate/wikirate4py`, GPLv3). Needs a free API token (account
  registration). REST API also returns JSON/CSV/HTML/TXT directly
  (`wikirate.org/Use_the_API`). No documented rate limit found — confirm with
  `info@wikirate.org` before scaling.
- **License**: CC BY 4.0 / Digital Public Good (DPG-recognized June 2025).
  Actively maintained — NLnet grants through 2025-2026, client libs updated
  Dec 2025/Jan 2026.
- **Already in this codebase**: `api/v1/esg_data/fetchers/wikirate_fetcher.py`
  — but `fetch_wikirate(company_name)` is a REACTIVE per-name lookup, called
  only for companies already sourced from bcorp/upright. It never pulls
  Wikirate's own company lists (FTSE 100, Fashion Transparency Index,
  KnowTheChain) as a NEW company universe. This is the single highest-leverage
  fix identified — no new integration needed, just a different ingestion
  pattern. **This is what's being explored next (see task in progress).**
- **Known limitation flagged by EU/UK research**: European-specific angle
  exists — Frank Bold's assessment of 300 CEE/Southern-European companies'
  climate/environmental disclosure is hosted on Wikirate
  ("New Research on WikiRate: Assessing European Corporate Non-Financial
  Reporting", Wikirate Medium blog) — directly relevant to the bcorp/upright
  country skew.
- **Caveat**: coverage is crowdsourced/research-project-driven, so density
  varies enormously per metric — some metrics well-populated, others sparse.
  Not a uniform panel like bcorp/upright.

### Japan — EDINET (Financial Services Agency)
- Real public API + bulk XBRL download, free, no paid tier. Ready-made Python
  library: `pip install edinet-tools` (github.com/matthelmer/edinet-tools),
  parses 42 EDINET document types, entity resolution by ticker/EDINET
  code/法人番号.
- Coverage: ~11,000+ entities (per third-party indexing tool).
- **Caveat**: currently strong on governance/financial structure, thin on
  numeric E/S/G KPIs specifically — SSBJ sustainability taxonomy extension is
  voluntary from FY ending March 2026, mandatory later by market-cap tier
  ("2027 EDINET taxonomy"). Real ESG numeric tags are a 2026-2027 story, not
  available in bulk today. Narrative text blocks may contain sustainability
  content today, unmapped to typed fields.
- Language: Japanese-native tags/narrative.

### India — SEBI BRSR / BRSR Core (high ceiling, requires building the aggregator)
- BRSR Core = 49 KPIs, genuinely numeric (absolute + intensity GHG emissions,
  renewable energy %, water use, gender diversity). Mandatory for top-1,000
  listed companies by market cap (phasing in third-party assurance through
  FY2026-27).
- Filed as BOTH PDF and XBRL to NSE/BSE via a shared XBRL utility — structured
  data exists per-filing.
- **No central bulk aggregator, API, or downloadable dataset found** covering
  all 1,000 companies' XBRL in one place. NSE's own filing links resolve to
  individual company PDFs. Would require scraping NSE/BSE per-filing
  (tractable — 1,000 companies — but real engineering work: a scraper +
  XBRL/PDF parser, not a quick pull).
- Language: English (genuine advantage over China/Korea/Japan — no
  translation layer needed).
- **Flagged for follow-up**: NSE has an unofficial-but-usable JSON API for
  corporate filings used by open-source scrapers — not verified in this pass.

---

## Tier 2 — Worth pursuing but gated / narrow / needs verification

### Taiwan — TWSE ESG InfoHub / OpenAPI (directly relevant — fixes our own Foxconn/Formosa Plastics gap)
- TWSE ESG InfoHub dashboard shows real numeric KPIs: GHG emissions, water,
  waste (E); female executive ratio, occupational injury counts, salaries
  (S); independent director ratio, board attendance (G) — aggregated from
  statutory filings via the Market Observation Post System (MOPS).
- Coverage: all TWSE + TPEx listed companies — would directly cover Foxconn
  Technology and Formosa Plastics, the two companies in our corpus with no
  real World Bank-derived country baseline (see `_NO_WB_BASELINE_ISO3` /
  `_PEER_BASELINE_SETS` in `country_baseline_agent.py`).
- **Access unconfirmed**: the InfoHub dashboard reads as a human-facing
  query/compare tool. A `TWSEMCPServer` GitHub project references a TWSE
  OpenAPI wrapping "ESG data" among other TWSE/TPEx/TAIFEX feeds, but the
  exact endpoint/schema was not confirmed in this pass.
- **Action item**: short, focused verification spike — find and test the
  actual OpenAPI endpoint (likely `openapi.twse.com.tw` or similar) before
  committing engineering time.

### Germany — Deutscher Nachhaltigkeitskodex (DNK)
- Structured 20-criteria sustainability declarations (strategy, process
  management, environment, society), CSRD/VSME-compatible.
- Claimed 3,000+ companies / 5,000+ users on a new "DNK Platform" (2025) —
  treat with caution, only found via secondary summarization.
- REST API exists (`api.deutscher-nachhaltigkeitskodex.de/Help`,
  JSON/HTTP-POST) but **gated**: "each external application must be explicitly
  enabled for data exchange" — requires manual approval, not self-serve
  API-key signup. Real cost is admin lead-time, not money.

### UN Global Compact — Participant/Signatory Database
- Free, ~20,000+ participants, real density in Africa/MEA/LatAm (Nigeria,
  Kenya, South Africa, UAE, Saudi Arabia, Egypt, Brazil, Mexico, Colombia,
  Argentina) — exactly the underrepresented regions.
- **Shallow**: company/country/sector/join-date/status/COP-submitted-flag
  only. NOT scored metrics — Communications on Progress are free-text/PDF.
- No official bulk API; would need light scraping of the participant
  directory (no documented rate limit, but no bulk endpoint either).
- **Recommended use**: same tier as BHRRC in the existing pipeline — a weak
  evidence/coverage signal, not peer-anchor-quality ground truth. Would slot
  in as a new `ungc_lookup` table.

### LSEG/Refinitiv — free ESG Scores Finder
- Genuinely free tool, 16,000+ companies, 76 countries, 88% of global market
  cap, full Pillar + Theme (E/S/G) breakdown — best of the three major paid
  vendors' free offerings (vs. MSCI's letter-grade-only ~2,900 ACWI large-caps,
  Sustainalytics' single risk score only).
- **Terms cap it to non-commercial reference use** — written approval needed
  for anything beyond spot-checking/backtesting your own scores against it.
  Not usable as an ingested ground-truth source without their sign-off.

### IFC / World Bank Group project data
- IFC Projects Database: real project-level metadata (Environmental & Social
  Category A/B/C/FI risk rating, sector, country, linked ESRS/monitoring
  documents) — but ONLY for IFC's own investees, a few thousand entries, with
  disproportionately strong Africa/LatAm/MEA representation (IFC's mandate).
  No confirmed bulk API — browsable/searchable only.
- World Bank Enterprise Surveys (`microdata.worldbank.org`): genuinely free,
  bulk-downloadable (Stata/CSV), tens of thousands of firms across
  Africa/LatAm/MEA, with governance/environmental-compliance-adjacent survey
  questions. NOT an ESG score — raw firm-characteristic survey data.
  **Caveat**: many waves anonymize firm IDs — unclear if names are
  identifiable/matchable to this pipeline's name-based lookup pattern. Needs a
  feasibility check before any ingestion effort.

---

## Tier 3 — Free but stale, narrow, or fragile; use only as backtest anchors

- **Harvard Business School Impact-Weighted Accounts** (open dataset, email
  signup, no restrictions but citation): 13,000 firm-year observations, global
  large/mid-cap, but **Environmental-only (no S/G)** and stale (2010-2018).
- **Kaggle "alistairking" Public Company ESG Ratings**: 700+ mid/large-cap,
  real (not synthetic) E/S/G sub-scores + letter grades, multi-exchange.
  Undated ~2022-23 snapshot — verify license field before use.
- **S&P 500-only Kaggle sets** (Sustainalytics-sourced): real data, useful for
  backtesting, but US-large-cap only — doesn't help geographic spread.
- **EXPLICITLY FLAGGED AS FAKE**: `shriyashjagtap/esg-and-financial-performance-dataset`
  ("1,000 global companies") is self-described as **synthetic/simulated**.
  Do not use as ground truth despite looking like the best match on paper.
- **Datarade/SustainableHQ**: 7,000 companies, 70 countries, real global
  spread — but paid/commercial, not free.

---

## Confirmed dead ends — do not pursue without a materially different budget/approach

- **CSRD / EFRAG XBRL taxonomy / ESAP** (EU): the eventual right long-term
  answer (mandated free, public, machine-readable, EU-wide) but ESAP's public
  portal isn't scheduled to go live until **July 2027**, and CSRD/sustainability
  data specifically not until **January 2028**. Also: Omnibus I (finalized Feb
  2026) shrank CSRD scope to companies with >1,000 employees AND >€450M
  turnover only — won't help SME/mid-cap diversity even once live. No
  central registry exists today; would mean scraping ~27 different national
  "Officially Appointed Mechanisms," each with its own rules, mostly PDF.
- **UK Companies House / France's INPI RNE**: both are free, well-documented,
  bulk-accessible company REGISTRIES — but ESG content (SECR carbon
  disclosures / DPEF) is buried in unstructured PDF/iXBRL prose sections, not
  exposed as queryable fields. Would require document parsing, not a clean
  pull. Useful for entity resolution/metadata enrichment only.
- **GRI Sustainability Disclosure Database**: **confirmed dead** —
  `database.globalreporting.org` returns DNS failure; the product stopped
  being populated in Dec 2020, fully offline ~April 2021. Even alive, it was
  a report-existence index, never actual disclosed metric values. The often-
  cited "62% of global market cap, 107 jurisdictions" stat is from KPMG's 2024
  Survey of Sustainability Reporting (N100/G250 largest companies only), NOT
  from GRI's own database — don't conflate the two.
  - Successor-in-spirit: SustainabilityReports.com (336,000+ reports, 52,000+
    companies, 182 countries) — free search (3/day), $10-100/yr for more —
    but report-existence only, no extracted metrics.
- **South Africa (JSE), Nigeria (NGX), Kenya (NSE)**: King IV / ESG disclosure
  guidance exists as a compliance REQUIREMENT; no exchange or third party
  aggregates the disclosed content into structured data. Only real path is
  paid commercial vendors (MSCI/Sustainalytics/Refinitiv) who manually extract
  from individual integrated reports, or building the pipeline's own
  LLM-extraction against those reports directly (same machinery this pipeline
  already runs, not a new ground-truth source).
- **Middle East / GCC** (ADX, DFM, Tadawul, Qatar Stock Exchange): same
  pattern — disclosure guidance exists, zero aggregated data anywhere. "Gulf
  Sustainability Assessment System" does not correspond to a company-ESG
  initiative (closest real acronym, GSAS, is a green-building certification
  system, unrelated). Confirmed genuine dead end for free/structured data.
- **China (SSE/SZSE/CNINFO)**: mandatory guidelines from May 2024, but only
  for index-subset companies (34.7% of market as of June 2025), Chinese-only
  PDFs, CNINFO is a filing repository not a structured database. Highest
  effort / lowest near-term yield of everything researched.
- **South Korea (KRX ESG portal)**: real structured per-company data (ratings,
  compliance flags) but per-company lookup UI only, no bulk export — would
  need to enumerate ISU codes and scrape (~200 companies with reports as of
  2024, small N). KCGS-sourced data explicitly restricted to
  non-commercial/internal use.
- **Singapore (SGX ESGenome)**: appears to be a company-side reporting SaaS
  tool, not a public dataset — unconfirmed whether any public
  browsing/export exists. Small universe size (SGX) limits value even if
  accessible.
- **Latin America beyond B Corp** (B3's ISE, Mexico's BMV, Chile's Bolsa de
  Santiago, Colombia's BVC): ALL confirmed index-constituent-list-only
  (10-40 large caps each) — underlying RobecoSAM/S&P Global CSA scores are
  proprietary. Sistema B (B Corp's LatAm arm) redirects to the same
  bcorporation.net directory already in use — no incremental data.
  B3's "ESG Workspace" (`iseb3.com.br/esg-workspace`) requires registration to
  see what's behind the login — unverified, flagged as a possible follow-up
  but not confirmed valuable.
- **CDP full scores**: paywalled. Free tier is only the A-List (877 of
  ~20,000 scored companies) and disclosed/not-disclosed status — this
  pipeline's existing use (participation signal only) is already the correct
  free ceiling.
- **SEC EDGAR bulk data**: genuinely free, keyless bulk XBRL
  (`companyfacts.zip`, Financial Statement Data Sets) — but GAAP financial
  tags only, no ESG-specific taxonomy. SEC's 2024 climate disclosure rule has
  been STAYED since April 2024 and the SEC is now proposing outright
  rescission (Federal Register, June 3 2026) — no structured climate tags
  exist or are coming. No shortcut over the pipeline's existing per-filing
  NLP extraction.
- **Climate TRACE ecosystem**: confirmed emissions-only; coalition partners
  are all climate/remote-sensing orgs, no sibling S/G dataset exists.
- **WRDS-hosted academic access** (MSCI/Sustainalytics/Refinitiv/Trucost/
  RepRisk via university subscription): sold only to subscribing
  universities — no individual/small-business path at any price.
- **Paid vendor real pricing** (for context, not action): MSCI ~$5K-2M/yr
  (SMB avg ~$121K/yr per Spendhound); LSEG Workspace $150K-400K/yr (up to
  $1M+); Sustainalytics fully quote-only, no public figures found anywhere;
  Clarity AI (~30,000 companies, 400+ metrics) also quote-only/no free tier.

---

## Recommended action order (as of 2026-07-30)

1. **Wikirate expansion** — pull FTSE 100 / Fashion Transparency Index /
   KnowTheChain company lists as NEW corpus seeds, not just reactive lookups
   for existing bcorp/upright names. Free, already-integrated, directly fixes
   both size and country skew. *(Currently being scoped — see next research
   pass on the Wikirate SDK/API surface.)*
2. **Taiwan TWSE OpenAPI verification spike** — short, targeted; would close
   a gap for companies already in our corpus (Foxconn, Formosa Plastics).
3. **UN Global Compact signatory scrape** — cheap, adds Africa/MEA/LatAm
   coverage as a weak evidence signal (same tier as BHRRC today).
4. **India SEBI BRSR scraper** — highest ceiling, but a real build (NSE/BSE
   per-filing scraping + XBRL/PDF parsing for ~1,000 companies). Scope as its
   own project if pursued.
5. Everything in "Tier 3" and "Confirmed dead ends" — do not build against
   without a materially different budget or a specific new lead surfacing.
