# Wikirate SDK exploration — what data/metrics are actually available

Live-verified against the real Wikirate API (`wikirate4py==2.0.8`, already
installed; key already configured in `.env` as `WIKIRATE_API_KEY`) on
2026-07-30. Everything below was pulled from live API calls, not documentation
alone — every claim was reproduced against real responses.

## TL;DR

Wikirate is NOT just a per-company reactive lookup source (which is all this
codebase currently uses it for, via `fetch_wikirate(company_name)` in
`api/v1/esg_data/fetchers/wikirate_fetcher.py`). It's a genuinely bulk-
queryable database: `get_answers(identifier=<metric_id>, country=<country>)`
pulls every company's value for one metric, filtered by country, in real
tonnes/percentages/booleans with attached year and source URL. This is the
mechanism that can directly diversify country + size coverage without new
scraping infrastructure.

## What's already installed / configured

- `wikirate4py==2.0.8` is in `venv` already (`pip show` confirms).
- `WIKIRATE_API_KEY` is already set in `.env` and works (live calls succeeded).
- Existing integration (`wikirate_fetcher.py`) only calls per-company lookups
  for names already sourced from bcorp/upright — never pulls Wikirate's own
  company/metric lists as new corpus seeds. This is the gap to close.

## The SDK object model

`wikirate4py.API` — 52 public methods. The ones that matter for bulk pulling:

| Method | What it does |
|---|---|
| `get_companies(country=, company_category=, company_group=, company_identifier=)` | List companies, filterable by country/sector/named group (e.g. a company group like "FTSE 100") |
| `get_metrics(designer=, topic=, metric_keyword=, metric_type=, value_type=)` | List/search metrics (the "questions" — e.g. "Scope 1 emissions") |
| `get_metric(identifier=)` or `get_metric(metric_name=, metric_designer=)` | Full detail on one metric: unit, value_type, methodology, formula |
| `get_answers(identifier=<metric_id>, country=, company=, company_group=, year=, value_from=, value_to=, ...)` | **The bulk-pull method.** Every company's answer to ONE metric, filtered any way you like |
| `get_answers(identifier=<company_id>)` | All metrics/years for ONE company (what the current fetcher effectively does, one at a time) |
| `get_projects()`, `get_datasets()` | List curated research projects/datasets (e.g. "Fashion Checker Brand Research 2026") |
| `search_by_name(entity_type, name)` | **Has a real bug for non-Company types — see below** |

### Data model — what fields you actually get

`CompanyItem`/`Company`: `name`, `headquarters` (real country string, e.g.
`['Japan']`), `aliases`, `wikipedia_url`, `lei`, `isin`, `sec_cik`,
`open_corporates`, `uk_company_number`, `australian_business_number`. **This
overlaps directly with the entity-resolution keys `company_metadata.py`
already uses (LEI via GLEIF)** — meaning cross-referencing by LEI/ISIN instead
of fuzzy name matching is possible.

`AnswerItem` (the actual data point): `metric` (full name incl. designer),
`company`, `value` (raw string/number), `year`, `sources` (list of source
citations), `url` (public, citable, e.g.
`wikirate.org/Fashion_Revolution+1_1_Governance+Nike_Inc+2024`).

`MetricItem`/`Metric`: `designer` (the organization that defined it — GRI,
World Benchmarking Alliance, SBTi, etc.), `name`, `question`, `value_type`
(Number/Text/Boolean), `unit`, `methodology`, `topics`, `answers` (count).

## Real data pulled live

### Example 1 — Nike Inc. (id 5800), all metrics, 1000 answers returned (page-limited, real total likely higher)

Metric **designers** (source frameworks) actually present for one company:

| Designer | # answers for Nike |
|---|---|
| World Benchmarking Alliance | 252 |
| Fashion Revolution | 249 |
| Center for Political Accountability | 231 |
| US Securities and Exchange Commission | 55 |
| GreenDex | 43 |
| Apparel Research Group | 38 |
| Global Reporting Initiative | 12 |
| Net Zero Tracker | 13 |
| Science Based Targets Initiative (SBTi) | 2 |
| UN Global Compact | 1 |
| ...(29 distinct designers total for one company) | |

This is much broader than "just Fashion Revolution" — real governance
(Center for Political Accountability = political spending/lobbying
disclosure), SEC filing data, GRI-coded metrics, SBTi commitments, UN Global
Compact — all cross-referenced to ONE company record.

Sample real answers (multi-year, quantitative, qualitative):
```
Fashion Revolution+1.2 Traceability          2024/2025   5.777... (numeric score)
Fashion Revolution+1.1 Own Operations Policies  2020-2023  "Animal Welfare, Annual Leave..." (checklist text)
Fashion Revolution+1.1 Governance            2024/2025   0.0
```

### Example 2 — GRI 305-1-a (Scope 1 GHG emissions), designed by World
Benchmarking Alliance/GRI, metric id 826615

Pulled 100 real answers with NO country filter, then filtered by country
directly:

**Unfiltered sample** (proves genuinely global spread on ONE metric):
Honda Motor (Japan), Sinopec (China), Petrobras (Brazil), Chevron (US),
Hyundai Motor (Korea), PT Mitra Adiperkasa (**Indonesia**), Volkswagen
(Germany), Adidas (Germany), Tesco (UK), Astellas Pharma (Japan) — all with
real tonnage values (e.g. Sinopec 139,090,000 t, Petrobras 46,800,000 t,
Honda 997,000 t) and multi-year history (Honda: 2024=1,070,000 → 2025=997,000).

**`get_answers(identifier=826615, country='Japan')`** returned 20 real
companies: Honda, Astellas Pharma, Canon, Marubeni, Hitachi, Komatsu,
Seven & I Holdings, Takeda, Ricoh — genuine large Japanese companies with real
values.

**`get_answers(identifier=826615, country='India')`** returned 20 real
companies: Hindustan Construction, Arvind Limited, Coal India Limited,
Oil & Natural Gas (ONGC), Vedanta Limited, Reliance Industries, Aditya Birla
Fashion & Retail — genuine large Indian companies (exactly the "well-known"
tier the bcorp pool is starved of, per the earlier skew analysis — bcorp is
only 7.8% large companies) with real tonnage figures (Reliance: 43,960,000t;
Vedanta: 59,486,747t).

**This confirms the country filter works exactly as documented** — country
diversification is a query parameter, not something requiring new scraping.

### Example 3 — Governance-adjacent metrics (our weakest pillar)

`get_metrics(metric_keyword='anti-corruption')` returned real results:
- "Anti-bribery and anti-corruption fundamentals" (World Benchmarking Alliance)
- "Anti-corruption and bribery policy reflected in management procedures" (Apparel Research Group)
- "CSI.17.A Anti-bribery and Anti-Corruption Policy" (World Benchmarking Alliance)
- "Anti-Corruption Disclosure Rate" (Apparel Research Group, x2 variants)

`get_metrics(metric_keyword='women on board')` returned 5 designer variants of
the same underlying question: 2020WoB, Core, Commons, **PODER** (a Latin
America-focused corporate-accountability research group), GreenDex.

`metric_keyword='board independence'` and `'whistleblower'` returned 0 exact
hits — likely a phrasing mismatch (Wikirate's keyword search appears to be
fairly literal), not proof the concept is absent; worth trying synonyms
("independent director", "whistle-blowing") before concluding it's missing.

## A real bug found in the SDK

`search_by_name(entity_type, name)` is documented to search by name for six
entity types (Company, Metric, Topic, CompanyGroup, ResearchGroup, Project).
Live-tested: for `Metric`, it returns **identical results regardless of the
search string** ("board independence", "anti-corruption", "whistleblower" all
returned the same 5 items). Root cause, confirmed by reading the SDK source
(`wikirate4py/api.py`):

- `search_by_name(Metric, name)` calls `self.get_metrics(name=name, **kwargs)`
- but `get_metrics()`'s server-side filter whitelist is
  `('bookmark', 'topic', 'topic_framework', 'designer', 'published',
  'metric_type', 'value_type', 'metric_keyword', 'research_policy',
  'dataset')` — **`name` is not in that list**, so the filter is silently
  dropped and the endpoint returns its default unfiltered listing every time.

**Workaround** (verified working): call `api.get_metrics(metric_keyword=term)`
directly instead of `search_by_name(Metric, term)`. This returned correct,
query-specific results in every test. `search_by_name` for `Company` works
correctly (verified: searching "Nike" returned Nike Inc., NIKE UK Limited,
etc. — real, relevant matches) — the bug appears specific to entity types
whose underlying `get_*` method doesn't accept a `name` filter param.

## Pagination

`wikirate4py.Cursor` handles this: max 200 items/page
(`if per_page > 200: self.per_page = 200`), standard offset-based pagination
(`has_next()` / `next()`). Fine for a bulk pull loop — no rate-limit headers
were hit in this exploration (single-session, moderate call volume), but per
earlier research findings, no documented rate limit was found in Wikirate's
own docs either — worth confirming with `info@wikirate.org` before a
large-scale production pull, exactly as the earlier research flagged.

## What this means concretely for the pipeline

1. **The size-skew fix is real and immediately actionable**: `get_answers`
   filtered by metric + country pulled genuine large-cap companies (Reliance,
   Vedanta, ONGC, Coal India for India; Honda, Hyundai, Sinopec, Petrobras
   elsewhere) that bcorp's pool almost entirely lacks (92% small/medium).
2. **The country-skew fix is real**: direct `country=` filtering on
   `get_answers`, no scraping, no new infrastructure — just different calls
   to the SDK already installed.
3. **Governance coverage exists** but needs the right keyword phrasing per
   pull — "anti-corruption", "bribery", "women on board" all returned real
   designer-attributed metrics; this is worth building out as a proper
   G-pillar metric list rather than assuming Wikirate is E/S-only.
4. **Entity resolution is a bonus**: Wikirate companies carry LEI/ISIN/SEC
   CIK — the same keys `company_metadata.py` already resolves via GLEIF —
   so matching Wikirate records to existing corpus companies can go through a
   real identifier, not just fuzzy name matching (though `_best_company_match`
   in the existing fetcher already does fuzzy matching as a fallback).

## What's NOT yet verified (honest gaps in this exploration)

- Total company count / total answer count in the whole database — only
  spot-checked individual metrics/companies, didn't attempt a full census.
- Real-world rate limits under sustained bulk-pull load (this session's calls
  were light/interactive).
- Whether `sources` (citations) are populated consistently — one sample
  answer had an empty `sources` list; unclear if that's typical or an outlier.
- Coverage DENSITY per country/metric combination — confirmed data EXISTS for
  Japan/India/China/Brazil/Indonesia on Scope 1 emissions specifically, but
  didn't measure how many companies per country per metric, which is the real
  question for whether this meaningfully thickens peer-anchor tiers.
- Whether Wikirate exposes a `Dataset` (the `Dataset`/`DatasetItem` classes
  exist in the SDK) that's a pre-curated "download this whole panel" object —
  `get_datasets()` wasn't explored in this pass.

## Suggested next step

Before building an ingestion pipeline: pick 2-3 concrete metrics that map to
existing `factor_registry.py` factors (Scope 1/2/3 emissions map directly;
"Anti-bribery and anti-corruption fundamentals" maps to
`anti_corruption_policy`), and measure real answer DENSITY per country for
each — that's the number that tells us whether this meaningfully thickens the
peer-anchor tiers for underrepresented countries, versus adding a thin
sprinkle of extra data.
