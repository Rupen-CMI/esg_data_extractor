# Worldwide enforcement / penalty data sources — verified survey

Compiled 2026-07-31 from 4 parallel deep-research agents (EU/UK, Asia-Pacific,
global NGO/incident trackers, LatAm/Africa/fin-crime). Every source below was
live-verified by direct fetch on 2026-07-31 unless marked otherwise — record
counts were read off the live sites, not repeated from marketing copy.

## Why we need this (the two use cases)

- **(a) Severity yardstick**: distributions of fine/penalty amounts (normalized
  by company revenue) so any new fine can be percentile-ranked deterministically
  instead of an LLM guessing claim strength. See
  `evidence_quantification.md` — this replaces the guessed `strength` on
  penalty-bearing claims with a computed number.
- **(b) Company lookup**: search a company name → its violations/fines
  worldwide, as a structured evidence source alongside news/RSS.

Already in use / previously known: **Violation Tracker US** (~600k records,
all company sizes, amounts) and **Violation Tracker Global** (~50k records,
45 countries, large multinationals only, weak in env/labor for EU).

## The worldwide map after this survey

| Region | Best source | Scale | Amounts? | Access |
|---|---|---|---|---|
| US | Violation Tracker US | ~600k | yes | known |
| UK | **Violation Tracker UK** | 130k+, 75 regulators + 100 local authorities, SME coverage | yes | search free; bulk via Good Jobs First arrangement |
| EU privacy | **CMS GDPR Enforcement Tracker** | 3,202 fines, €6.31B, hourly updates | yes | free table, low-effort scrape (no official bulk) |
| EU financial | ESMA annual-report Excel annexes | ~970 sanctions/yr, 30 EEA states | yes | free annexes; portal is CAPTCHA-gated |
| EU antitrust | EC competition portal + cartel stats PDF | all EC cases | yes (in decisions) | search only; overlaps VT Global |
| China | **IPE Blue Map** | **3,505,731 records** (live-verified counter), facility-level, all provinces, SMEs included | in Chinese decision text, not a field | English UI + batch company search free; bulk = negotiate with IPE or scrape |
| China (structured) | CnOpenData env-penalty dataset | 2000–2025, nationwide, `Fine Amount` is a real column | yes | **paid** |
| India | Watchout Investors (Prime Database) | ~408k entities, 35 regulators | partial | free lookup, no bulk; pair with SEBI PDF orders |
| Japan | RegBase (regbase.jp/en) | ~10k enforcement actions, 9 agencies | partial (surcharge orders) | free English search, no API |
| Brazil | **IBAMA autos de infração** | ~1M+ records, 1980–present, daily updates | **yes (`VAL_AUTO_INFRACAO`) + CNPJ tax IDs** | **free bulk CSV, no registration** |
| Chile | SMA SNIFA | sanctions registry + open-data bulk (2014+) | yes (UTA units) | free; Google-Drive-hosted files |
| Peru | OEFA RUIAS | env infractions by sector (mining/hydrocarbons heavy) | yes (UIT units) | free CSV; annual update lag |
| Australia/NZ | WorkSafe NZ + NSW EPA registers | 608 NZ prosecutions w/ sentencing detail; NSW penalty notices | yes | free, trivial scrape |
| Global screen | **OpenSanctions** (regulatory + debarment collections) | 396k + 573k entities, 196 countries, daily | mostly no | free bulk **CC-BY-NC**; commercial license needed for production |
| Global bribery | Stanford FCPA Clearinghouse (+ donated TRACE data) | 916 global cases | yes (settlements) | free account; no bulk; frozen at mid-2025 during restructuring |
| Global accidents | France ARIA (full base on data.gouv.fr) + eMARS | ~60k / ~1.3k structured accidents (deaths, injuries) | n/a | free bulk — but **operators anonymized: severity reference only** |
| Global sanctions | OFAC + FinCEN | penalty amounts incl. foreign companies | yes | free; FinCEN easiest via OpenSanctions structured mirror; check VT overlap first |
| Mining incidents | WISE tailings-dam chronology | 300+ failures w/ parent company, volume, deaths | volumes/deaths | free static HTML table |
| US spills | NOAA IncidentNews CSV | thousands, `max_ptl_release_gallons` | volumes | free CSV; no company field |

## High-signal binary flags (no amounts, high severity)

- **Brazil "lista suja"** — 613 employers on the slave-labor registry (Apr 2026
  update), names + CNPJ, free XLSX. The strongest single S-pillar negative flag
  available anywhere.
- **Argentina REPSAL** — daily-updated labor-sanctions registry (child labor,
  trafficking, unregistered work). Lookup flag; direct fetch was flaky
  (possible geo-block) — retry or proxy.
- **World Bank + MDB cross-debarments** — via OpenSanctions debarment
  collection (World Bank 2,772 + AfDB/ADB/IADB/EBRD).
- **US DOL UFLPA Entity List** — named forced-labor companies.
- **BHRRC Lawsuits Database** — 342 lawsuit profiles, downloadable (the main
  20k-company allegations tracker is narrative-only, scrape-required).

## Confirmed dead ends (do not re-investigate)

- **Mexico**: PROFEPA has no case-level public DB (page is descriptive text);
  datos.gob.mx had broken SSL. COFECE = amounts only inside PDFs.
- **Germany**: no national per-company enforcement registry exists (Länder
  level, unpublished). Umweltbundesamt publishes no register.
- **Korea**: no aggregator; KFTC/FSS/MOE sanction data is Korean-only,
  press-release or domestic-portal shaped.
- **SE Asia** ex-SG/MY: Indonesia/Thailand/Vietnam/Philippines = press releases
  only (PH central DB exists only as a proposed bill).
- **Kenya / Egypt / Gulf**: press releases or name-only lists; Egypt FRA's new
  violators list has names, no amounts.
- **Nordics**: no public per-company fine registers for labor; GDPR fines
  already in CMS tracker.
- **ITOPF**: raw global tanker-spill DB explicitly not released (aggregates
  only). **eMARS**: names deliberately anonymized. **ILO / Global Slavery
  Index**: country-level only, no company data. **ORX**: member banks only.
- **RepRisk**: 330k+ companies with risk incidents — exactly our use case —
  but enterprise-only unpublished pricing. **Corlytics**: same.
- **OpenCorporates**: registry/entity data only, no enforcement content
  (useful for entity resolution, not violations).
- **UK EA bulk CSV caveat**: the one-click enforcement CSV (8,768 rows,
  verified by download) has **no fine-amount column** — amounts require the
  companion prosecutions/undertakings datasets or VT UK.

## License watch-list (resolve before production use)

- **OpenSanctions**: CC-BY-NC — free evaluation, paid license for commercial.
- **EJAtlas**: CC BY-NC-SA + bulk only on approved request.
- **Violation Tracker UK bulk**: via Good Jobs First data-licensing arrangement
  (same channel as VT US/Global).
- **CnOpenData**: paid, price on registration.
- **Finbold bank-fines reports**: explicitly free for commercial reuse with
  attribution (yardstick anchor points only).

## Recommended ingestion order

1. **Brazil IBAMA** — the only Tier-1-for-both-use-cases source in the survey:
   free daily bulk, fine amounts, tax IDs (revenue-join ready), 45 years of
   history, all company sizes. Build the first non-US severity distribution
   from this.
2. **CMS GDPR tracker scrape** — 3,202 clean EU fines, one afternoon of work.
3. **OpenSanctions bulk (evaluation)** — instant 196-country company-screening
   layer; start the commercial-license conversation in parallel.
4. **IPE Blue Map batch search** — wire company-name batch lookup into the
   evidence fetcher for any company with China operations; discuss bulk/data
   agreement with IPE for the severity table.
5. **Violation Tracker UK bulk** — one licensing conversation covers the
   UK gap end to end.
6. **Chile SNIFA + Peru RUIAS CSVs** — cheap LatAm severity distributions.
7. **ARIA + WorkSafe NZ + WISE + NOAA** — free severity-reference distributions
   (accident deaths/volumes) feeding the facts-based rubric thresholds.

## What this does and does not solve

Solves: penalty-bearing evidence (fines, sanctions, settlements) can now be
severity-ranked against real distributions on every continent except Africa,
and company lookup coverage now spans US/UK/EU/China/India/Japan/Brazil/LatAm
plus a 196-country screening layer.

Does not solve: non-penalty severity (allegations, spills without fines yet)
still needs the facts-based ordinal rubric; Korea/SE-Asia/Africa remain
press-release territory; narrative sources (BHRRC) remain evidence-fetch, not
yardstick, inputs.
