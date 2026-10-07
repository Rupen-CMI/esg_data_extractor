"""
public_records_collector.py — free, key-free, per-company public-records
evidence, live-verified 2026-09-22 (see research/ESG_SOURCES_READY_TO_BUILD.md).

Distinct from enforcement_collector.py (EPA ECHO / SEC / World Bank -- ONE
curated enforcement-only set, framed around "a regulator only creates a
record when something went wrong"). The sources here are more varied in
shape -- consumer-safety recalls, federal contract awards, per-facility
pollutant releases, sanctions, and a foreign structured ESG-KPI filing --
so they get their own module rather than stretching that one's identity.

  CPSC recalls        S  US Consumer Product Safety Commission recalls --
                          manufacturer/importer/distributor name, product,
                          hazard, date. Adjudicated fact, negative-only.
  USAspending          G  US federal contract/award history by recipient --
                          both an accountability-exposure signal (does a
                          company do business with the US government) and
                          a revenue/scale proxy. NEUTRAL existence flag, not
                          negative-polarity -- see _usaspending_signal.
  OFAC SDN             G  US Treasury sanctions list (Specially Designated
                          Nationals) -- same evidence class as World Bank
                          debarment (enforcement_collector.py), the single
                          hardest free G-pillar red flag after debarment.
  EPA Envirofacts TRI   E  Toxics Release Inventory -- per-facility chemical
                          release data, DISTINCT from EPA ECHO's enforcement
                          CASES/facility-compliance summary (enforcement_
                          collector.py) -- this is raw release volume, not
                          a violation or penalty.
  Taiwan TWSE ESG KPIs  E/S/G  Per-company GHG/energy/water/board-diversity/
                          pay KPIs for Taiwan-listed companies -- the one
                          source in this module that is POSITIVE-capable
                          structured disclosure, not an enforcement fact.
                          Kept here (not report_collector.py) because it is
                          a clean JSON API, not a PDF to parse.
  Cornell ILR Labor      S  Strikes/labor actions with an explicit `Employer`
    Action Tracker           field -- S-pillar signal independent of company
                          self-disclosure, same rationale as this pipeline's
                          existing negative-evidence sources. A strike is
                          not itself proof of misconduct, but it is real,
                          independently-reported labor unrest -- adjacent to
                          BHRRC's existing role, distinct evidence tag.
  UK Modern Slavery       S  ~33-34k UK organizations' modern-slavery
    Registry                  statements -- POSITIVE-capable (a real filed
                          statement, low-weight self-disclosure) but also
                          carries a genuine negative angle when the
                          statement's own risk-assessment fields (NoRisks,
                          RiskN) show gaps -- same polarity-aware handling
                          as UK MSA's "overdue" framing in the source doc.
  NLRB (labordata)        S  US unfair-labor-practice case data, 1979-
                          present, nightly-refreshed bulk SQLite --
                          case-level detail (allegations) that Cornell's
                          strike tracker doesn't carry; kept as a distinct
                          evidence tag in the same labor-relations bucket.
  ICIJ Offshore Leaks     G  810k+ named entities from the Panama/Pandora/
                          Paradise/Bahamas Leaks -- appearing in a leak is
                          NOT itself proof of wrongdoing (many are entirely
                          legal offshore structures); this is a
                          transparency/complexity flag, not an accusation.
  SAM.gov Exclusions      G  US federal debarment/exclusion (keyed API,
                          free signup, see SAM_GOV_API_KEY in .env) --
                          the US-federal-specific sibling of World Bank
                          debarment (enforcement_collector.py) and OFAC
                          SDN above: catches companies banned by an
                          individual US agency (EPA, DOJ, etc.) that may
                          not appear on either of those two lists.
  ESGsource Greenwashing  G  Regulatory/advertising-standards/litigation
    Enforcement Tracker      outcomes for greenwashing accusations --
                          company-scoped (unlike ESGsource's other 3
                          datasets, which are country/regulation-level
                          context, not company evidence -- deliberately
                          NOT pulled here, see _MAX_CHARS note below).
                          Real adjudicated outcomes (fine/ad-banned/
                          court-injunction) alongside disputed/dismissed
                          ones -- outcome_class carries the actual
                          polarity signal, not mere existence in the
                          tracker (a "court-dismissed"/"not-upheld" row
                          is not itself negative evidence).
  NY spill registry       E  New York State DEC's chemical/oil spill
                          incident database -- per-incident material,
                          quantity, location, dates. Negative-polarity
                          (a spill is a real environmental incident),
                          NY-facility-scoped only.
  Sweden SMP              E  "Utslapp i siffror" (Swedish EPA) national
    ("Utslapp i siffror")    PRTR-style facility register -- per-facility
                          industry classification + environmental
                          management system status. FOOTPRINT/existence
                          signal like Envirofacts TRI above, not itself
                          a violation -- Sweden-facility-scoped only.
  Crisil ESG Ratings      E/S/G  A genuine external third-party ESG
    (India)                   rating (0-100 composite + Leadership/
                          Strong/Adequate/Below-average band) for
                          ~944 SEBI-listed Indian companies -- DIFFERENT
                          IN KIND from every other source in this module:
                          those are all raw facts (recalls, sanctions,
                          awards); this is another rater's *opinion*,
                          fed as evidence text for the extractor to
                          weigh, not as ground truth (bcorp_lookup/
                          upright_lookup remain the only calibration
                          answer-key -- see esg-truth-source-decision
                          memory). India-listed-company-scoped only.
                          Parsed from a MANUALLY SAVED snapshot, not a
                          live fetch -- crisilesg.com/robots.txt
                          explicitly disallows anthropic-ai/claudebot/
                          Claude-Web by name (no citation/search
                          exception, unlike KnowESG's robots.txt) --
                          see _load_crisil_ratings docstring.
  NBIM/Norges Bank        G  Norway's sovereign wealth fund's own
    exclusions (via            investment-exclusion list, via
    OpenSanctions)            OpenSanctions' small per-topic export (NOT
                          its 2.6GB full bulk dataset, see below).
                          Same evidence class as OFAC SDN/World Bank
                          debarment -- conduct-based exclusion is a real
                          adjudicated negative signal. CC-BY-NC,
                          cleared for this non-commercial in-house
                          pipeline (see esg-app-non-commercial-inhouse
                          memory).
  Australia NGER          E  National Greenhouse and Energy Reporting --
    (emissions/energy)        mandatory annual Scope 1/2 emissions +
                          net energy consumption for every registered
                          Australian corporation. FOOTPRINT signal like
                          Envirofacts TRI/Sweden SMP above, not itself a
                          violation. Bulk CSV already downloaded to
                          external_data/ (manual one-time fetch, not
                          fetched by this module itself) -- see
                          _load_nger_corporations docstring.
  Banking on Climate      E  Per-company fossil-fuel financing received
    Chaos (BOCC)              from the world's 65 biggest banks, 2021-
                          2025, by year. NEUTRAL existence/exposure
                          signal, same polarity caution as USAspending/
                          Envirofacts TRI above -- receiving fossil-
                          fuel financing is not itself a violation, but
                          is a real accountability-exposure/scale
                          signal the extractor should weigh, not treat
                          as an accusation. The site's own dropdown UI
                          is JS-rendered (no company data in raw HTML),
                          but the real master CSV it fetches
                          client-side is a plain static file -- found
                          by reading the theme's own JS, not the
                          rendered page -- see _load_bocc_financing
                          docstring.
  Trase.earth             E  Per-company agricultural supply-chain
    (deforestation             deforestation exposure -- 38 country x
    exposure)                 commodity datasets (soy, palm oil, beef,
                          cocoa, coffee, etc across 11 producer
                          countries), covering exporters/importers/
                          mills/refineries. FOOTPRINT/exposure signal
                          like Envirofacts TRI/NGER above, not itself a
                          violation -- only fires for companies in
                          commodity trading/food supply chains, not a
                          general-purpose source. Queried against a
                          pre-built SQLite index (external_data/trase/
                          trase_index.db, ~113k distinct entities,
                          24.7MB), NOT the 38 raw CSVs (~7.5M rows,
                          2.9GB) directly -- see _trase_signal
                          docstring and _build_trase_index.py for why.
  WARN Act notices        S  US state-level mass-layoff/plant-closure
    (17 US states)            notices (the WARN Act), per state --
                          company name, layoff date, employee count.
                          Negative-polarity, adjacent to Cornell's
                          labor-action tracker above but a DIFFERENT
                          evidence class (a mandated legal notice of a
                          closure/layoff, not a strike). Only 17 of 41
                          supported states' scrapers confirmed actually
                          working live as of 2026-10-06 -- the other
                          24 fail for real, varied reasons (site HTML/
                          JSON structure changed, missing dependency,
                          requires a paid third-party API key, requires
                          Xvfb which doesn't run on Windows, DNS/
                          network issues) -- see WARN_WORKING_STATES
                          and research/ESG_SOURCES_READY_TO_BUILD.md
                          for the full per-state breakdown. Each
                          state's CSV has a GENUINELY DIFFERENT column
                          schema (not just naming -- Maryland's file
                          has no header row at all) -- see
                          _load_warn_notices docstring for the regex +
                          override approach, same reasoning as Trase's
                          38 differently-shaped datasets above.
  Open Sustainability     E  Structured per-company GHG emissions
    Index                     (Scope 1/2/3, by year), climate targets,
                          and commitment status for ~588+ companies.
                          POSITIVE-capable disclosed-data source, same
                          class as Taiwan TWSE above (a real KPI
                          filing, not an enforcement fact) -- but also
                          exposes `commitment.status: "Removed"` cases
                          (an abandoned/expired climate commitment),
                          which IS a real negative signal the extractor
                          should weigh. The site's own docs page
                          (opensustainabilityindex.org/api) is a
                          JS-rendered Next.js app with no API host in
                          its raw HTML -- found via a one-time
                          Playwright trace to `api.opensustainability
                          index.org/openapi.json` (the real OpenAPI
                          spec). **No signup needed**: the
                          `api-key=demo` DEFAULT VALUE shown in their
                          own docs UI works as a genuine, fully
                          functional free-tier key (confirmed live
                          2026-10-06, not a placeholder) -- reverses the
                          source doc's "gated request-a-key flow"
                          finding. Company name resolved to the
                          platform's own slug via /v1/search first
                          (slugs are custom, e.g. "microsoft" not
                          "microsoft-corporation"), then queried via
                          /v1/companies/{slug} -- see _osi_signal
                          docstring.
  EDINET (Japan)          E/S/G  Japan's mandatory corporate-disclosure
    securities reports        system -- annual Securities Reports
                          (doc type 120) carry real, substantive
                          climate/governance/human-capital narrative
                          sections (Japan's ISSB-aligned sustainability
                          disclosure requirement), not just financials.
                          Confirmed live 2026-10-06 against a real
                          Toyota filing: a genuine multi-paragraph
                          climate strategy section (carbon-neutrality/
                          circular-economy/nature-positive targets
                          across 10 regions), real corporate-governance
                          philosophy text, plus structured numeric
                          fields (ROE, employee count, average number
                          of temporary workers, total shareholder
                          return). KEY FOUND VIA THE THIRD-PARTY
                          `edinet-tools` PyPI package (matthelmer/
                          edinet-tools), NOT hand-rolled XBRL parsing --
                          it solves the exact company-name-to-EDINET-
                          code mapping problem that blocked NSE/India
                          above (entity() does offline fuzzy name
                          search against a bundled FSA registry
                          snapshot, no API key needed for THAT step;
                          only fetching documents needs the key).
                          Text blocks are in JAPANESE, surfaced as-is
                          (same "trust the downstream LLM to read
                          non-English text" precedent as Taiwan TWSE
                          above) -- no translation step. Requires
                          JAPAN_EDINET_KEY in .env (free signup via
                          api.edinet-fsa.go.jp, see _edinet_signal
                          docstring); gracefully skips if unset, same
                          discipline as SAM.gov's unset-key case.

DELIBERATELY NOT PULLED: OpenSanctions' full "default" bulk dataset (the
broadest cross-country sanctions/PEP aggregator, 396k+ entities). Its own
bulk export is ~2.6GB uncompressed -- an order of magnitude larger than
every other source in this module (OFAC 5.7MB, Cornell 5.3MB, UK MSA
~50MB/3yr, ICIJ's usable entities file ~199MB, NLRB's extracted SQLite
~1GB -- NLRB's zip download is only ~176MB, the on-disk unzipped file is
the real cost, verified live 2026-09-23) -- for a marginal coverage
gain over sources already wired here: OFAC (US) + World Bank debarment
(enforcement_collector.py) + NBIM (via OpenSanctions already, small)
cover the highest-severity red flags already. The genuinely new content
(EU/UK/UN sanctions, PEP data) either overlaps heavily with OFAC in
practice, or (PEP data specifically) is about INDIVIDUALS and would need
a company->owner->PEP linking step this pipeline doesn't have. Revisit
only if a real gap is observed (a company that should have been flagged
and wasn't) -- and prefer OpenSanctions' smaller per-topic exports over
the full bundle if/when that happens.

DELIBERATELY NOT PULLED (2026-09-24): South Africa's NECER report
(dffe.gov.za) -- real, valuable data confirmed live (a genuine annual PDF
naming companies in "S v <company>" prosecution cases: Eskom Holdings,
Northam Platinum, Foskor, Vantage Goldfields all confirmed present in the
2024-25 report). NOT built because the file-serving path returns a real
SSL certificate chain error (`SSLCertVerificationError: unable to get
local issuer certificate`) specific to that URL -- reproducible across 3
retries, while a bare hostname handshake to the same host succeeds
(likely a load-balanced/CDN setup routing the file path to a backend with
an incomplete cert chain). `curl` tolerates this (its OS-level cert store
handles it differently); Python's `requests` correctly rejects it.
Deliberately NOT worked around by disabling certificate verification for
this one host -- that would be a real security regression for a
general-purpose HTTP helper, not a narrow fix. Revisit if their SSL
config is ever corrected server-side.

DELIBERATELY NOT PULLED (2026-09-24): India BRSR/corporate-announcement
data via NSE's own API (`nseindia.com/api/corporate-announcements`).
Fully confirmed reachable (cookie-priming + Referer-header trick works,
verified live) -- but NSE's API requires the bare NSE ticker symbol
(e.g. "INFY", "HDFCBANK"), not a company name, and this pipeline's
existing ticker resolver (yfinance_fetcher._find_ticker, reused
elsewhere for the public-company-uplift critic) does not reliably
produce one: verified live that it returns a US ADR ticker for HDFC
Bank ("HDB", not NSE's "HDFCBANK") and ".NS"-suffixed symbols for
others (e.g. "TATASTEEL.NS") that NSE's own API silently returns zero
results for -- confirmed live: `symbol=INFOSYS` (company name) and
`symbol=TATASTEEL.NS` both return `[]`, only the bare NSE symbol
("INFY", "TATASTEEL") returns real data. This is a real, unsolved
company-name-to-bare-NSE-ticker mapping gap, same structural class as
A3/A5's company->geolocation gap and ICIJ's company->ownership-graph
gap -- not a header/auth problem, and not safe to guess at (a wrong
ticker returns a clean empty result, not an error, so a bad guess would
silently look like "no BRSR filing" rather than fail loudly). Revisit
only with a real NSE-specific symbol-lookup source, not by reusing
yfinance's resolver as-is.

ENTITY MATCHING follows enforcement_collector.py's discipline exactly:
every source's raw result set gets the strict all-token match
(evidence_filters._company_tokens + word-boundary regex, imported from
enforcement_collector.py rather than re-implemented) before a hit is
trusted -- EXCEPT USAspending, which uses a deliberately more lenient
local matcher (_neutral_name_match) since its signal is neutral, not an
accusation (see _usaspending_signal's docstring for why).

CPSC's `RecallFirmName` param does NOT filter server-side -- verified
live 2026-09-22 that a query for "Amazon" and a query with no name filter
at all return the exact identical ~10,016-row full corpus (same trap
enforcement_collector.py's ECHO docstring documents for `p_name`).
USAspending's `recipient_search_text` filter DOES work server-side, but
its `award_type_codes` filter is a REQUIRED field the API 422s without --
the source doc's original snippet omitted it. OFAC's and Cornell's bulk
files are fetched whole (no filter param exists) and matched client-side,
same as CPSC ends up being despite its API accepting a now-known-useless
parameter.

RATE LIMITING follows the same "public-sector endpoint, no published
limit -> assume nothing, pace conservatively" rule as enforcement_
collector.py and its cross-cutting Tier S/A ban-avoidance notes (see
research/ESG_SOURCE_ANALYSIS.md Part 3/4): each host gets its own
_RateLimiter so one host's pacing never blocks another's budget, matching
the ECHO/World Bank split already established.
"""

import csv
import io
import json
import os
import re
import sqlite3
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger, log_header
from agentic_estimation.layer_1.enforcement_collector import _strict_name_match, _is_recent, _MAX_CHARS
from agentic_estimation.layer_1.evidence_filters import _company_tokens
# signal_agent.py already calls load_dotenv() at import time -- SAM_GOV_API_KEY
# is readable via os.getenv by the time this module's os.getenv call below runs.
from agentic_estimation.layer_1.signal_agent import RateLimitTripped, _RateLimiter, _get

log = get_logger("public_records")

_TIMEOUT = 45

# Each host paced independently -- see module docstring. All four are
# government infra with no published rate limit; kept at least as
# conservative as enforcement_collector.py's WB limiter (1.5s+1.0) since
# none of these have been probed live long enough to justify going faster.
_CPSC_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_USASPENDING_LIMITER = _RateLimiter(min_gap=2.0, jitter=1.0)
_OFAC_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_ENVIROFACTS_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_TWSE_LIMITER = _RateLimiter(min_gap=1.0, jitter=0.5)
_CORNELL_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_UKMSA_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_NLRB_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_ICIJ_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_SAMGOV_LIMITER = _RateLimiter(min_gap=2.0, jitter=1.0)
_ESGSOURCE_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_NYSPILL_LIMITER = _RateLimiter(min_gap=1.0, jitter=0.5)
_SWEDEN_SMP_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_NBIM_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_BOCC_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_OSI_LIMITER = _RateLimiter(min_gap=1.5, jitter=1.0)
_EDINET_LIMITER = _RateLimiter(min_gap=2.0, jitter=1.0)

_CPSC_URL = "https://www.saferproducts.gov/RestWebServices/Recall"
_USASPENDING_URL = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
# Live-verified 2026-09-22 (research/ESG_SOURCES_READY_TO_BUILD.md Tier A ->
# A14) -- confirmed HTTP 200, real Content-Disposition/Last-Modified headers.
_OFAC_SDN_URL = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"
_ENVIROFACTS_TRI_URL = "https://data.epa.gov/efservice/TRI_FACILITY/FACILITY_NAME/CONTAINING/{name}/JSON"
_TWSE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap46_L_{n}"
# TWSE datasets t187ap46_L_1..21 -- one call per dataset, each returns EVERY
# company for that KPI (GHG, energy, water, board diversity, pay, etc), not
# one company per call. Fetched once per process and filtered locally, same
# "corpus-wide feed, cache it" pattern as enforcement_collector.py's SEC RSS
# and World Bank debarment list -- a per-company live query would refetch
# the same ~21 full-corpus datasets for every company in a run.
_TWSE_DATASET_COUNT = 21
# Live-verified 2026-09-22 -- returns a dict keyed by numeric row ID (NOT a
# list), each value a full labor-action record with an explicit "Employer"
# field. Whole-corpus JSON, no pagination, no key.
_CORNELL_URL = "https://striketracker.ilr.cornell.edu/labor_actions.json"
# Live-verified 2026-09-23 -- real bulk per-year CSV, on a SEPARATE
# "downloads." subdomain from the main service host (the source doc's
# guessed path on the main host 404s). One file per statement year;
# pulling the 3 most recent years covers "currently active" disclosure
# without re-fetching the full historical archive back to 2016.
_UKMSA_URL = "https://downloads.modern-slavery-statement-registry.service.gov.uk/publicdownloads/StatementSummaries{year}.csv"
_UKMSA_YEARS = (2024, 2025, 2026)
# Live-verified 2026-09-23 -- the doc's guessed "main" branch raw path is
# wrong; the real asset lives under the "nightly" release tag (nightly-
# refreshed, hence the tag name, not a branch).
_NLRB_URL = "https://github.com/labordata/nlrb-data/releases/download/nightly/nlrb.db.zip"
# Live-verified 2026-09-23 -- found by fetching the real database page
# (offshoreleaks.icij.org/pages/database) and reading its actual download
# link, NOT via the GitHub Releases API the source doc assumed (ICIJ's
# offshoreleaks-data-packages repo has no tagged Release at all).
_ICIJ_URL = "https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.LATEST.zip"
# Live-verified 2026-09-23 with a real key (SAM_GOV_API_KEY in .env,
# expires ~89 days from creation -- see fetch_public_records_signals'
# graceful skip when unset/expired). Real params confirmed against
# open.gsa.gov/api/exclusions-api/'s actual docs page (NOT the source
# doc's guess): `exclusionName` filters server-side (verified: a query
# for a real firm returned 2 rows, not the full 168,661-row corpus), and
# `classification=Firm` restricts to companies (excludes individuals).
_SAMGOV_URL = "https://api.sam.gov/entity-information/v4/exclusions"
# Live-verified 2026-09-23 -- ESGsource's OTHER 3 datasets (disclosure
# mandates, regulatory timeline events, credential cost index) are
# country/regulation-level context, not company-scoped evidence -- see
# module docstring. Only this one has a real `company` field per row.
_ESGSOURCE_GREENWASH_URL = "https://esgsource.com/data/greenwashing-enforcement-tracker/greenwashing-enforcement-tracker.json"
# Live-verified 2026-09-23 -- Socrata SODA API, standard query params
# ($limit/$where/$q) work server-side (unlike CPSC/Sweden SMP below).
# Dataset "Spill Incidents" (u44d-k5fk), updated same-day. NY-scoped only
# (source doc's own caveat: CA OSPR would be a separate, un-tested dataset).
_NYSPILL_URL = "https://data.ny.gov/resource/u44d-k5fk.json"
# Live-verified 2026-09-23 -- `searchText` is NOT a real server-side name
# filter (same trap as CPSC's RecallFirmName): a query for "Volvo" and the
# bare unfiltered call return DIFFERENT but NOT name-matched result sets
# (zero literal "Volvo" hits in the "filtered" response) -- the param
# does *something* server-side (likely an unrelated cutoff/pagination
# artifact) but not company matching. Fetched whole (861 facilities, one
# call, ~1.1MB) and matched client-side, same pattern as _load_cpsc_recalls.
_SWEDEN_SMP_URL = "https://utslappisiffror.naturvardsverket.se/api/combinedsearch/anlaggning"
# Live-verified 2026-09-23 -- OpenSanctions-hosted FTM JSON Lines export of
# Norway's sovereign wealth fund's own exclusion list (debarment topic).
# Small (~400 rows), 307-redirects to a dated CDN artifact -- MUST follow
# redirects (same as A18's full bulk export, but this one is small enough
# to actually pull, unlike that 2.6GB dataset). CC-BY-NC -- cleared for
# this non-commercial in-house pipeline (see esg-app-non-commercial-
# inhouse memory).
_NBIM_URL = "https://data.opensanctions.org/datasets/latest/no_nbim_exclusions/entities.ftm.json"
# Live-verified 2026-09-28 -- the site's own "Bank & Client Profiles"
# dropdown UI is a client-side jQuery/select2 widget with no data in raw
# HTML (same JS-rendering problem Trase.earth and GFW's dashboard have).
# But WordPress + jQuery (not a Next.js/React SPA), so the real master
# CSV it fetches via jQuery.get() is a plain static file -- found by
# reading bocc-2026-charts.js's own fc_get_data()/jQuery.get() calls and
# the page's inline <script>'s filename construction, not by rendering
# the page. 19,239 rows: Bank, Company, Company_Parent, Company_Country_
# Code, year-by-year financing amounts (2021-2025), Total.
_BOCC_URL = "https://www.bankingonclimatechaos.org/wp-content/themes/bocc-2021/inc/bcc-data-2026/full-data.csv"
# Live-verified 2026-10-06 -- real server found via a one-time Playwright
# trace of opensustainabilityindex.org/api (a JS-rendered Next.js docs
# page with no API host in raw HTML) to its real openapi.json spec.
# api-key=demo is a genuine working free-tier key (confirmed with a
# real Microsoft Scope 1/2/3 emissions response), not a placeholder --
# no signup needed despite the source doc's "gated request-a-key" note.
_OSI_SEARCH_URL = "https://api.opensustainabilityindex.org/v1/search"
_OSI_COMPANY_URL = "https://api.opensustainabilityindex.org/v1/companies/{slug}"
_OSI_API_KEY = "demo"

_UA = {"User-Agent": "ESG-Signal-Agent/1.0 (research@example.com)"}
_SAM_GOV_API_KEY = os.getenv("SAM_GOV_API_KEY", "")
# signal_agent.py already calls load_dotenv() at import time -- same
# timing guarantee as SAM_GOV_API_KEY above.
_JAPAN_EDINET_KEY = os.getenv("JAPAN_EDINET_KEY", "")

_ofac_cache: Optional[list[dict]] = None
_ofac_lock = threading.Lock()
_twse_cache: Optional[list[dict]] = None
_twse_lock = threading.Lock()
_cpsc_cache: Optional[list[dict]] = None
_cpsc_lock = threading.Lock()
_cornell_cache: Optional[list[dict]] = None
_cornell_lock = threading.Lock()
_ukmsa_cache: Optional[list[dict]] = None
_ukmsa_lock = threading.Lock()
_nlrb_lock = threading.Lock()
_icij_cache: Optional[list[dict]] = None
_icij_lock = threading.Lock()
# Small (34 rows) and free/no-key -- in-memory-only, no disk cache needed
# unlike OFAC/Cornell/UK MSA/NLRB/ICIJ (all much larger corpora).
_esgsource_greenwash_cache: Optional[list[dict]] = None
_esgsource_greenwash_lock = threading.Lock()
# Small (~860 facilities) and free/no-key -- in-memory-only, same rationale
# as ESGsource's greenwashing cache above.
_sweden_smp_cache: Optional[list[dict]] = None
_sweden_smp_lock = threading.Lock()
# Parsed once per process from the on-disk snapshot (never re-fetched --
# see _CRISIL_DIR).
_crisil_cache: Optional[list[dict]] = None
_crisil_lock = threading.Lock()
_nbim_cache: Optional[list[dict]] = None
_nbim_lock = threading.Lock()
_nger_cache: Optional[list[dict]] = None
_nger_lock = threading.Lock()
_bocc_cache: Optional[list[dict]] = None
_bocc_lock = threading.Lock()

# Repo-root-relative, matching country_baseline_agent.py / ilo_ratification.py's
# convention for static bulk files -- see those modules' _EXCEL_PATH/_CACHE_PATH.
# external_data/ (not raw_esg_data/) per project convention: this directory
# holds pipeline-fetched bulk cache, not manually-curated input.
_EXTERNAL_DATA_DIR = Path(__file__).parent.parent.parent / "external_data"
_OFAC_CACHE_PATH = _EXTERNAL_DATA_DIR / "ofac_sdn.csv"
_CORNELL_CACHE_PATH = _EXTERNAL_DATA_DIR / "cornell_labor_actions.json"
_UKMSA_CACHE_PATH = _EXTERNAL_DATA_DIR / "uk_msa_{year}.csv"
# NLRB's real payload is the SQLite file itself, not the zip -- kept
# unzipped on disk so every process opens it directly with no re-extract
# cost. Zip download is ~176MB; the UNZIPPED file is ~1GB on disk
# (verified live 2026-09-23) -- the zip's Content-Length is not the real
# storage cost, see module docstring's size note.
_NLRB_DB_PATH = _EXTERNAL_DATA_DIR / "nlrb.db"
# ICIJ's zip contains 6 files (entities/officers/addresses/intermediaries/
# others/relationships); only nodes-entities.csv is kept -- see module
# docstring, this is a name-match existence flag, not a full graph
# traversal, so the officer/address/relationship files (~420MB combined,
# uncompressed) would be pure dead weight on disk for no signal this
# module actually uses.
_ICIJ_ENTITIES_PATH = _EXTERNAL_DATA_DIR / "icij_offshoreleaks_entities.csv"
_BOCC_CACHE_PATH = _EXTERNAL_DATA_DIR / "bocc_full_data.csv"
# Manually saved snapshot (see _load_crisil_ratings docstring) -- one HTML
# file per alphabetical tab, saved 2026-09-23 by the user via a real
# browser (crisilesg.com/robots.txt blocks anthropic-ai/claudebot/
# Claude-Web by name, so this module never fetches it live).
_CRISIL_DIR = _EXTERNAL_DATA_DIR / "crisil_esg_ratings"
_CRISIL_TAB_FILES = ("a_d.html", "e_h.html", "i_l.html", "m_p.html", "q_t.html", "u_x.html", "y_z.html")
# Already downloaded 2026-09-23 (fetch_nger.py, deleted after use) --
# real files, live-verified against a fresh CSV row spot-check. Same
# "external_data/ = pipeline-usable bulk cache" role as OFAC/Cornell.
_NGER_CSV_PATH = _EXTERNAL_DATA_DIR / "nger_controlling_corporations_2024_25.csv"
# Pre-built by _build_trase_index.py (one-time, manual, NOT run by this
# module) from 38 CSVs unzipped from trase.earth's own per-dataset
# downloads -- see that script's docstring for why an aggregated index
# rather than an in-memory cache.
_TRASE_DB_PATH = _EXTERNAL_DATA_DIR / "trase" / "trase_index.db"
_WARN_DIR = _EXTERNAL_DATA_DIR / "warn"
# Confirmed by actually running `python -m warn.cli <state>` for all 41
# warn-scraper-supported states live, 2026-10-06. Only these 17 produced
# real output (some with 0 rows for the current period, which is a
# legitimate "nothing currently filed" result, not a failure) --
# see research/ESG_SOURCES_READY_TO_BUILD.md for the full per-state
# failure breakdown (site structure changes, missing pyquery/Zyte
# dependencies, Xvfb/Windows incompatibility, DNS issues in this
# environment). This list is a snapshot, not auto-detected -- a state
# scraper that starts working later needs a manual re-run + this list
# updated, same "needs periodic manual refresh" discipline as Crisil's
# saved snapshot above.
WARN_WORKING_STATES = (
    "ak", "al", "az", "de", "il", "ks", "md", "me", "mt",
    "nj", "ny", "ok", "sc", "sd", "tn", "vt", "wa",
)
# Company-identifying column per state where a generic regex guess
# would be wrong or ambiguous -- verified live 2026-10-06 against each
# state's actual header row. States not listed here are handled by
# _WARN_COMPANY_FIELD_RE below.
_WARN_COMPANY_FIELD_OVERRIDE = {
    "il": "Location Name",          # 33-column schema; "Doing Business As Name" is usually empty
    "ny": "Business Legal Name",    # no "company"/"employer" substring in any column name
    "mt": "Name of Company",        # would match the regex anyway, listed for clarity
}
# Maryland's CSV has NO header row at all -- positional columns only,
# verified live: date, NAICS code, COMPANY NAME, address, county,
# employee count, date, layoff type. Index 2 (0-based) is the company.
_WARN_MD_COMPANY_COL_INDEX = 2
_WARN_COMPANY_FIELD_RE = re.compile(r"company|employer|business.*name", re.IGNORECASE)


def _load_cpsc_recalls() -> list[dict]:
    """CPSC recall corpus -- fetched once per process, then filtered
    locally, same pattern as _load_ofac_sdn/_load_twse_esg.

    NOT a per-company live query: verified live 2026-09-22 that
    `RecallFirmName` is SILENTLY IGNORED by this endpoint -- a query for
    "Amazon" and a query with no filter at all both returned the exact
    same ~10,016-row full corpus (same trap enforcement_collector.py's
    ECHO docstring documents for `p_name`: accepted, not applied). Worse,
    a firm name containing a hyphen (e.g. "Char-Broil") tripped the
    endpoint's own Akamai WAF into a 403 -- so passing untrusted company
    names as a query param here is not just useless, it is a live risk
    of getting the whole endpoint blocked. Fetching the unfiltered corpus
    ONCE with no name param, then matching client-side, avoids both
    problems."""
    global _cpsc_cache
    with _cpsc_lock:
        if _cpsc_cache is not None:
            return _cpsc_cache
        _cpsc_cache = []
        _CPSC_LIMITER.wait()
        r = _get(_CPSC_URL, params={"format": "json"}, timeout=_TIMEOUT)
        if not r:
            log.warning("CPSC recall corpus unavailable")
            return _cpsc_cache
        try:
            data = r.json() or []
        except (ValueError, AttributeError):
            log.warning("CPSC recall corpus parse failed")
            return _cpsc_cache
        if isinstance(data, list):
            _cpsc_cache = data
        log.info("loaded CPSC recall corpus: %d recalls", len(_cpsc_cache))
        return _cpsc_cache


def _cpsc_signal(company: str) -> Optional[str]:
    """CPSC recall history, matched client-side against the full corpus
    (see _load_cpsc_recalls) -- the strict all-token match here is load-
    bearing, not a safety net, since nothing upstream has filtered by
    name at all."""
    parts: list[str] = []
    for rec in _load_cpsc_recalls():
        # Deliberately excludes "Retailers" -- a store that SOLD a recalled
        # product did not cause the defect; attributing the recall to it
        # would be the same wrong-entity risk this module's matching
        # discipline exists to prevent.
        firm_names = " ".join(
            (f.get("Name") or "") for f in (rec.get("Manufacturers") or []) + (rec.get("Importers") or [])
            + (rec.get("Distributors") or [])
        )
        if not _strict_name_match(firm_names, company):
            continue
        date = (rec.get("RecallDate") or "").strip()
        if not _is_recent(date):
            continue
        title = (rec.get("Title") or "").strip()
        hazards = "; ".join(h.get("Name", "") for h in (rec.get("Hazards") or []) if h.get("Name"))
        bits = [b for b in (title, hazards, date) if b]
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] cpsc_recall → %d matched recall(s)", company, len(parts))
    return f"CPSC product recall: {body} <https://www.saferproducts.gov/>"


def _neutral_name_match(record_name: str, company: str) -> bool:
    """A more lenient sibling of enforcement_collector._strict_name_match,
    used ONLY where a false positive is genuinely low-cost -- currently
    USAspending (a neutral existence signal, holding a federal contract
    is not an accusation) and ESGsource's greenwashing tracker (a small,
    editorially-curated company list, not a noisy free-text search over
    a huge government corpus, so substring-bleed risk is much lower than
    CPSC/ECHO/World Bank's raw record sets).

    _strict_name_match rejects a single token under 10 chars (e.g.
    "Boeing", "Dow", "Shein") as "not identifying enough", which is the
    right call for accusation-grade sources (debarment, sanctions,
    recalls, pollution) where a false positive wrongly accuses a company
    of something. Verified live 2026-09-22/23: this silently dropped a
    real, unambiguous "THE BOEING COMPANY" USAspending hit, and would
    equally drop "Shein" against ESGsource's own curated "Shein" row.
    Both sources' false-negative cost (real lost data for exactly the
    well-known, short-named companies most likely to appear) outweighs
    their false-positive risk, unlike the accusation-grade sources. Still
    requires every token to appear as a whole word (word-boundary
    match), just without the length/count floor."""
    toks = _company_tokens(company)
    if not toks:
        return False
    low = (record_name or "").lower()
    return all(re.search(rf"\b{re.escape(t)}\b", low) for t in toks)


def _usaspending_signal(company: str) -> Optional[str]:
    """US federal award history. NEUTRAL existence signal, not negative --
    unlike every other source in this module, holding federal contracts is
    not itself an ESG red flag. Feed as accountability-exposure/scale
    context and let claim_validators.py / the extractor assign polarity,
    same caution enforcement_collector.py documents for peer_anchor/
    dataset_lookup claims elsewhere in this pipeline.

    POST, not GET -- USAspending's search endpoint takes its filters as a
    JSON body, so this bypasses signal_agent._get (GET-only) the same way
    enforcement_collector._worldbank_signal bypasses it for a custom
    apikey header. Same try/except/RateLimitTripped-reraise discipline.

    award_type_codes is REQUIRED -- verified live 2026-09-22 that omitting
    it (as the source doc's snippet did) 422s with "Missing value:
    'filters|award_type_codes' is a required field". A/B/C/D are
    USAspending's own contract-award type codes (definitive/BPA
    call/delivery order/purchase order) -- the "does this company hold US
    federal CONTRACTS" question this signal exists to answer, as opposed
    to grants/loans/direct-payment assistance codes which are a different
    question this signal isn't trying to answer."""
    _USASPENDING_LIMITER.wait()
    try:
        import requests
        resp = requests.post(
            _USASPENDING_URL,
            json={
                "filters": {"recipient_search_text": [company[:100]], "award_type_codes": ["A", "B", "C", "D"]},
                "fields": ["Award ID", "Recipient Name", "Award Amount", "Awarding Agency", "Start Date"],
                "page": 1, "limit": 20,
            },
            headers=_UA, timeout=_TIMEOUT,
        )
        r = resp if resp.status_code == 200 else None
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.warning("[%s] USAspending fetch failed: %s", company, exc)
        r = None
    if not r:
        return None
    try:
        results = (r.json() or {}).get("results") or []
    except (ValueError, AttributeError):
        return None

    parts: list[str] = []
    for award in results:
        recipient = (award.get("Recipient Name") or "").strip()
        if not _neutral_name_match(recipient, company):
            continue
        agency = (award.get("Awarding Agency") or "").strip()
        amount = award.get("Award Amount")
        date = (award.get("Start Date") or "").strip()
        bits = [b for b in (recipient, agency, date) if b]
        if amount:
            bits.append(f"${amount:,.0f}" if isinstance(amount, (int, float)) else str(amount))
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] usaspending → %d matched award(s)", company, len(parts))
    return f"US federal awards (neutral, not a violation): {body} <https://www.usaspending.gov/>"


def _load_ofac_sdn() -> list[dict]:
    """OFAC SDN bulk CSV -- ~1000s of rows, fetched once per process (same
    "corpus-wide list, cache it" pattern as enforcement_collector.py's
    World Bank debarment cache). Persisted to external_data/ so a re-run
    within the same day doesn't re-fetch; refresh is manual (delete the
    file) since OFAC updates are irregular, not scheduled."""
    global _ofac_cache
    with _ofac_lock:
        if _ofac_cache is not None:
            return _ofac_cache
        _ofac_cache = []

        raw_text: Optional[str] = None
        if _OFAC_CACHE_PATH.exists():
            raw_text = _OFAC_CACHE_PATH.read_text(encoding="utf-8", errors="replace")
        else:
            _OFAC_LIMITER.wait()
            r = _get(_OFAC_SDN_URL, timeout=_TIMEOUT, cache=False)
            if r:
                raw_text = r.text
                try:
                    _EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    _OFAC_CACHE_PATH.write_text(raw_text, encoding="utf-8")
                except OSError as exc:
                    log.warning("could not persist OFAC SDN cache: %s", exc)

        if not raw_text:
            log.warning("OFAC SDN list unavailable")
            return _ofac_cache

        # OFAC's SDN.CSV has no header row -- fixed column order per their
        # own published layout. Column 1 (0-indexed) is the entity name.
        try:
            reader = csv.reader(io.StringIO(raw_text))
            rows = [row for row in reader if len(row) > 1 and row[1].strip()]
            _ofac_cache = [{"name": row[1].strip(), "type": (row[2].strip() if len(row) > 2 else ""),
                            "program": (row[3].strip() if len(row) > 3 else "")} for row in rows]
            log.info("loaded OFAC SDN list: %d entries", len(_ofac_cache))
        except csv.Error as exc:
            log.warning("OFAC SDN parse failed: %s", exc)
        return _ofac_cache


def _ofac_signal(company: str) -> Optional[str]:
    """US Treasury sanctions (SDN) hit -- same evidence class and matching
    discipline as enforcement_collector.py's World Bank debarment (the
    strict all-token match is load-bearing here too: a free-text bulk
    list of this size WILL substring-collide on short names)."""
    parts: list[str] = []
    for row in _load_ofac_sdn():
        if not _strict_name_match(row["name"], company):
            continue
        parts.append(f"{row['name']}" + (f"; program: {row['program']}" if row.get("program") else ""))

    if not parts:
        return None
    log.info("[%s] ofac_sdn → %d matched entr(y/ies)", company, len(parts))
    return ("OFAC sanctions (SDN) list: " + " | ".join(parts)[:_MAX_CHARS] +
            " <https://sanctionslistservice.ofac.treas.gov/>")


def _envirofacts_signal(company: str) -> Optional[str]:
    """EPA Envirofacts TRI -- per-facility chemical release volume.
    DISTINCT from enforcement_collector.py's ECHO facilities signal: TRI is
    raw disclosed release data (not a violation/compliance judgment), so a
    hit here is evidence of environmental FOOTPRINT, not necessarily
    wrongdoing -- leave polarity to the extractor, same caution as the
    USAspending signal above."""
    _ENVIROFACTS_LIMITER.wait()
    url = _ENVIROFACTS_TRI_URL.format(name=company[:40])
    r = _get(url, timeout=_TIMEOUT)
    if not r:
        return None
    try:
        rows = r.json() or []
    except (ValueError, AttributeError):
        return None
    if not isinstance(rows, list):
        return None

    parts: list[str] = []
    for row in rows:
        # Envirofacts returns lowercase field names (verified live
        # 2026-09-22: "facility_name", not "FACILITY_NAME" as the URL
        # path segment's casing might suggest) -- using the uppercase
        # key silently matched nothing on every real row.
        fac_name = (row.get("facility_name") or "").strip()
        if not _strict_name_match(fac_name, company):
            continue
        city = (row.get("city_name") or "").strip()
        state = (row.get("state_abbr") or "").strip()
        bits = [b for b in (fac_name, city, state) if b]
        if bits:
            parts.append(", ".join(bits))

    if not parts:
        return None
    # De-dup: TRI returns one row per chemical/year per facility, so the
    # same facility name can repeat many times for one company.
    parts = list(dict.fromkeys(parts))
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] envirofacts_tri → %d distinct facility match(es)", company, len(parts))
    return f"EPA TRI-registered facilities (footprint, not a violation): {body} <https://enviro.epa.gov/>"


def _load_twse_esg() -> list[dict]:
    """Taiwan TWSE ESG KPI datasets t187ap46_L_1..21 -- corpus-wide, fetched
    once per process (same caching rationale as _load_ofac_sdn/World Bank).
    Each row already carries a Chinese company name field; matching is
    done by substring against that field since TWSE has no English-name
    column in the raw JSON."""
    global _twse_cache
    with _twse_lock:
        if _twse_cache is not None:
            return _twse_cache
        _twse_cache = []
        rows: list[dict] = []
        for n in range(1, _TWSE_DATASET_COUNT + 1):
            _TWSE_LIMITER.wait()
            r = _get(_TWSE_URL.format(n=n), timeout=_TIMEOUT)
            if not r:
                continue
            try:
                dataset = r.json() or []
            except (ValueError, AttributeError):
                continue
            if isinstance(dataset, list):
                rows.extend(dataset)
        _twse_cache = rows
        log.info("loaded Taiwan TWSE ESG datasets: %d total rows across %d dataset(s)",
                  len(rows), _TWSE_DATASET_COUNT)
        return _twse_cache


# TWSE's company-name field varies by dataset (公司名稱/公司簡稱) -- try both
# rather than assuming one dataset shape holds for all 21.
_TWSE_NAME_FIELDS = ("公司名稱", "公司簡稱", "CompanyName")


def _twse_signal(company: str) -> Optional[str]:
    """Only fires for companies whose TWSE-listed name substring-matches
    `company` -- since TWSE names are in Chinese, this will only ever hit
    for a company whose `company` argument is ALSO given in/containing
    Chinese characters (e.g. a metadata field carrying the local name).
    A plain English company name will simply never match, which is
    correct (no false positive risk) rather than a bug to fix here."""
    parts: list[str] = []
    for row in _load_twse_esg():
        name = next((row.get(f) for f in _TWSE_NAME_FIELDS if row.get(f)), None)
        if not name or company[:40].strip().lower() not in str(name).lower():
            continue
        kvs = [f"{k}={v}" for k, v in row.items() if v not in (None, "", "-") and k not in _TWSE_NAME_FIELDS]
        if kvs:
            parts.append(f"{name}: " + ", ".join(kvs[:8]))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] twse_esg → %d matched row(s)", company, len(parts))
    return f"Taiwan TWSE disclosed ESG KPIs: {body} <https://openapi.twse.com.tw/>"


def _load_cornell_labor_actions() -> list[dict]:
    """Cornell ILR strike tracker -- whole-corpus JSON, fetched once per
    process (same pattern as OFAC/CPSC/TWSE above), persisted to
    external_data/ so a same-day re-run doesn't re-fetch ~5,000 records.
    The live endpoint returns a dict keyed by numeric ID; normalized to a
    plain list of the values here since callers only ever iterate."""
    global _cornell_cache
    with _cornell_lock:
        if _cornell_cache is not None:
            return _cornell_cache
        _cornell_cache = []

        raw: Optional[dict] = None
        if _CORNELL_CACHE_PATH.exists():
            try:
                raw = json.loads(_CORNELL_CACHE_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                log.warning("Cornell labor-action cache unreadable: %s", exc)
        if raw is None:
            _CORNELL_LIMITER.wait()
            r = _get(_CORNELL_URL, timeout=60, cache=False)
            if r:
                try:
                    raw = r.json()
                except (ValueError, AttributeError):
                    raw = None
                if raw is not None:
                    try:
                        _EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                        _CORNELL_CACHE_PATH.write_text(json.dumps(raw), encoding="utf-8")
                    except OSError as exc:
                        log.warning("could not persist Cornell labor-action cache: %s", exc)

        if not raw:
            log.warning("Cornell labor-action tracker unavailable")
            return _cornell_cache
        _cornell_cache = list(raw.values()) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        log.info("loaded Cornell labor-action tracker: %d actions", len(_cornell_cache))
        return _cornell_cache


def _cornell_signal(company: str) -> Optional[str]:
    """Strike/labor-action hit, matched on the record's own `Employer`
    field. Not an accusation of wrongdoing the way debarment/sanctions
    are -- a strike is independently-reported labor unrest, not a
    company's own admission -- but it is negative-polarity by nature
    (workers striking is not neutral the way a federal contract is), so
    kept on the strict matcher rather than the lenient USAspending-only
    one."""
    parts: list[str] = []
    for action in _load_cornell_labor_actions():
        employer = (action.get("Employer") or "").strip()
        if not _strict_name_match(employer, company):
            continue
        action_type = (action.get("Action_type") or "").strip()
        start = (action.get("Start_date") or "").strip()
        end = (action.get("End_date") or "").strip()
        demands = ", ".join(action.get("Worker_demands") or [])
        bits = [b for b in (employer, action_type, start, end) if b]
        if demands:
            bits.append(f"demands: {demands}")
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] cornell_labor_action → %d matched action(s)", company, len(parts))
    return f"Labor action (Cornell ILR tracker): {body} <https://striketracker.ilr.cornell.edu/>"


def _load_ukmsa_statements() -> list[dict]:
    """UK Modern Slavery Registry -- one bulk CSV per statement year,
    fetched once per process (see _UKMSA_YEARS -- most recent 3 years
    only, not the full archive back to 2016). Persisted to external_data/
    per year, same "cache to disk, refresh is manual" rule as OFAC."""
    global _ukmsa_cache
    with _ukmsa_lock:
        if _ukmsa_cache is not None:
            return _ukmsa_cache
        _ukmsa_cache = []
        rows: list[dict] = []
        for year in _UKMSA_YEARS:
            cache_path = Path(str(_UKMSA_CACHE_PATH).format(year=year))
            raw_text: Optional[str] = None
            if cache_path.exists():
                raw_text = cache_path.read_text(encoding="utf-8-sig", errors="replace")
            else:
                _UKMSA_LIMITER.wait()
                r = _get(_UKMSA_URL.format(year=year), timeout=90, cache=False)
                if r:
                    raw_text = r.text
                    try:
                        _EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                        cache_path.write_text(raw_text, encoding="utf-8")
                    except OSError as exc:
                        log.warning("could not persist UK MSA %s cache: %s", year, exc)
            if not raw_text:
                continue
            try:
                reader = csv.DictReader(io.StringIO(raw_text))
                rows.extend(reader)
            except csv.Error as exc:
                log.warning("UK MSA %s parse failed: %s", year, exc)
        _ukmsa_cache = rows
        log.info("loaded UK Modern Slavery Registry: %d statements across %d year(s)",
                  len(rows), len(_UKMSA_YEARS))
        return _ukmsa_cache


def _ukmsa_signal(company: str) -> Optional[str]:
    """UK Modern Slavery statement hit, matched on OrganisationName OR
    ParentName (a subsidiary's group statement should still count for the
    parent). POSITIVE-capable (a filed statement is disclosure, not an
    accusation) but carries a real negative angle via NoRisks/RiskN gaps
    -- the extractor, not this function, assigns final polarity, same
    caution as every neutral/mixed signal elsewhere in this module."""
    parts: list[str] = []
    for row in _load_ukmsa_statements():
        org = (row.get("OrganisationName") or "").strip()
        parent = (row.get("ParentName") or "").strip()
        # For a GroupSubmission='Yes' row, OrganisationName is a bare
        # internal numeric ID, not a real name (verified live 2026-09-23:
        # e.g. "10246724" with the real identity only in ParentName) --
        # matching against it is harmless (a numeric string won't collide
        # with a real company name) but DISPLAYING it as if it were one is
        # actively misleading, so it's excluded from `bits` when non-alpha.
        org_is_real_name = org and not org.isdigit()
        if not (_strict_name_match(org, company) or _strict_name_match(parent, company)):
            continue
        year = (row.get("StatementYear") or "").strip()
        no_risks = (row.get("NoRisks") or "").strip()
        bits = [b for b in ((org if org_is_real_name else None), year) if b]
        if parent and parent != org:
            bits.append(f"group parent: {parent}")
        if no_risks:
            bits.append(f"stated no risks identified: {no_risks}")
        url = (row.get("StatementURL") or "").strip()
        if url:
            bits.append(f"<{url}>")
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] uk_modern_slavery → %d matched statement(s)", company, len(parts))
    return f"UK Modern Slavery statement: {body}"


def _load_nlrb_db():
    """NLRB unfair-labor-practice case data -- downloads a ~176MB zip
    once, extracting to a ~1GB SQLite file (not re-extracted per process;
    kept unzipped on disk, see _NLRB_DB_PATH), then returns a live
    sqlite3 connection every call (cheap; the file itself is the cache,
    not an in-memory copy the way OFAC/Cornell/ICIJ hold their smaller
    corpora)."""
    with _nlrb_lock:
        if not _NLRB_DB_PATH.exists():
            _NLRB_LIMITER.wait()
            r = _get(_NLRB_URL, timeout=180, cache=False)
            if not r:
                log.warning("NLRB database unavailable")
                return None
            try:
                _EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extract("nlrb.db", path=str(_EXTERNAL_DATA_DIR))
                log.info("downloaded + extracted NLRB database to %s", _NLRB_DB_PATH)
            except (zipfile.BadZipFile, KeyError, OSError) as exc:
                log.warning("NLRB database extraction failed: %s", exc)
                return None
    try:
        return sqlite3.connect(str(_NLRB_DB_PATH))
    except sqlite3.Error as exc:
        log.warning("NLRB database open failed: %s", exc)
        return None


def _nlrb_signal(company: str) -> Optional[str]:
    """NLRB case hit, matched on participant.participant where
    type IN ('Employer', 'Charged Party', 'Charged Party / Respondent')
    -- the respondent/employer side of a case, never the union/petitioner
    side. Joined to allegation for what was alleged and docket for the
    most recent case activity date (for the _MIN_YEAR recency filter,
    same cutoff enforcement_collector.py uses -- NLRB cases go back to
    1979, and ESG standing is a present-state question)."""
    conn = _load_nlrb_db()
    if conn is None:
        return None
    try:
        rows = conn.execute(
            "SELECT DISTINCT case_number, participant FROM participant "
            "WHERE type IN ('Employer', 'Charged Party', 'Charged Party / Respondent') "
            "AND participant LIKE ?", (f"%{company[:60]}%",),
        ).fetchall()
    except Exception as exc:
        log.warning("[%s] NLRB query failed: %s", company, exc)
        return None
    finally:
        conn.close()

    parts: list[str] = []
    seen_cases: set = set()
    for case_number, participant_name in rows:
        if not _strict_name_match(participant_name or "", company):
            continue
        if case_number in seen_cases:
            continue
        seen_cases.add(case_number)
        bits = [f"case {case_number}", (participant_name or "").strip()]
        parts.append("; ".join(b for b in bits if b))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] nlrb_case → %d matched case(s)", company, len(parts))
    return f"NLRB unfair-labor-practice case: {body} <https://www.nlrb.gov/>"


def _load_icij_entities() -> list[dict]:
    """ICIJ Offshore Leaks named entities -- downloads the ~72MB zip once,
    extracts and keeps ONLY nodes-entities.csv (~199MB uncompressed; the
    officers/addresses/intermediaries/relationships files are dropped --
    see module docstring, this module only needs a name-match existence
    flag, not the full ownership graph)."""
    global _icij_cache
    with _icij_lock:
        if _icij_cache is not None:
            return _icij_cache
        _icij_cache = []

        if not _ICIJ_ENTITIES_PATH.exists():
            _ICIJ_LIMITER.wait()
            r = _get(_ICIJ_URL, timeout=180, cache=False)
            if not r:
                log.warning("ICIJ Offshore Leaks data unavailable")
                return _icij_cache
            try:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    with z.open("nodes-entities.csv") as src:
                        _EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                        _ICIJ_ENTITIES_PATH.write_bytes(src.read())
                log.info("downloaded + extracted ICIJ entities to %s", _ICIJ_ENTITIES_PATH)
            except (zipfile.BadZipFile, KeyError, OSError) as exc:
                log.warning("ICIJ Offshore Leaks extraction failed: %s", exc)
                return _icij_cache

        try:
            with open(_ICIJ_ENTITIES_PATH, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                _icij_cache = list(reader)
        except (OSError, csv.Error) as exc:
            log.warning("ICIJ entities parse failed: %s", exc)
        log.info("loaded ICIJ Offshore Leaks entities: %d rows", len(_icij_cache))
        return _icij_cache


def _icij_signal(company: str) -> Optional[str]:
    """Named-entity hit in the offshore leaks. EXPLICITLY a transparency/
    complexity flag, not an accusation -- most entities in these leaks
    are entirely legal offshore structures (see enforcement_collector.py-
    style caution already applied to peer_anchor/dataset_lookup claims
    elsewhere: existence is not itself the polarity)."""
    parts: list[str] = []
    for row in _load_icij_entities():
        name = (row.get("name") or "").strip()
        former = (row.get("former_name") or "").strip()
        if not (_strict_name_match(name, company) or (former and _strict_name_match(former, company))):
            continue
        jurisdiction = (row.get("jurisdiction_description") or "").strip()
        status = (row.get("status") or "").strip()
        source = (row.get("sourceID") or "").strip()
        bits = [b for b in (name, jurisdiction, status, source) if b]
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] icij_offshoreleaks → %d matched entit(y/ies)", company, len(parts))
    return ("Named in ICIJ Offshore Leaks (transparency flag, not an accusation): "
            + body + " <https://offshoreleaks.icij.org/>")


def _samgov_signal(company: str) -> Optional[str]:
    """US federal debarment/exclusion -- same evidence class as World Bank
    debarment (enforcement_collector.py) and OFAC SDN above, but US-
    federal-agency-specific (EPA, DOJ, etc. can each independently
    exclude a firm). `exclusionName` and `classification=Firm` both
    filter server-side (verified live 2026-09-23, unlike CPSC's dead
    RecallFirmName param) so this is a genuine per-company query, not a
    fetch-once-corpus pattern like most of this module."""
    if not _SAM_GOV_API_KEY:
        log.debug("[%s] samgov_exclusion → skipped (SAM_GOV_API_KEY not set)", company)
        return None

    _SAMGOV_LIMITER.wait()
    r = _get(_SAMGOV_URL, params={
        "api_key": _SAM_GOV_API_KEY, "exclusionName": company[:60], "classification": "Firm",
    }, timeout=_TIMEOUT)
    if not r:
        return None
    try:
        entities = (r.json() or {}).get("excludedEntity") or []
    except (ValueError, AttributeError):
        return None

    parts: list[str] = []
    for entity in entities:
        ident = entity.get("exclusionIdentification") or {}
        entity_name = (ident.get("entityName") or "").strip()
        if not _strict_name_match(entity_name, company):
            continue
        details = entity.get("exclusionDetails") or {}
        agency = (details.get("excludingAgencyName") or "").strip()
        exclusion_type = (details.get("exclusionType") or "").strip()
        actions = (entity.get("exclusionActions") or {}).get("listOfActions") or []
        term_date = (actions[0].get("terminationDate") if actions else "") or ""
        comments = ((entity.get("exclusionOtherInformation") or {}).get("additionalComments") or "").strip()
        bits = [b for b in (entity_name, agency, exclusion_type, term_date, comments) if b]
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] samgov_exclusion → %d matched exclusion(s)", company, len(parts))
    return f"SAM.gov federal exclusion: {body} <https://sam.gov/content/exclusions>"


def _load_esgsource_greenwashing() -> list[dict]:
    """ESGsource's Greenwashing Enforcement Tracker -- small (34 rows),
    free/no-key, fetched once per process (in-memory only, no disk cache
    -- see module-level cache comment). The other 3 ESGsource datasets
    (disclosure mandates, regulatory timeline, credential cost index) are
    deliberately NOT pulled -- they're country/regulation-level context,
    not company-scoped evidence (see module docstring)."""
    global _esgsource_greenwash_cache
    with _esgsource_greenwash_lock:
        if _esgsource_greenwash_cache is not None:
            return _esgsource_greenwash_cache
        _esgsource_greenwash_cache = []
        _ESGSOURCE_LIMITER.wait()
        r = _get(_ESGSOURCE_GREENWASH_URL, timeout=_TIMEOUT)
        if not r:
            log.warning("ESGsource greenwashing tracker unavailable")
            return _esgsource_greenwash_cache
        try:
            data = r.json() or {}
            _esgsource_greenwash_cache = (data.get("tables") or {}).get("accusations", {}).get("rows") or []
        except (ValueError, AttributeError):
            log.warning("ESGsource greenwashing tracker parse failed")
        log.info("loaded ESGsource greenwashing tracker: %d accusations", len(_esgsource_greenwash_cache))
        return _esgsource_greenwash_cache


# outcome_class values that represent a genuine adjudicated/negative
# result -- "litigation-ongoing"/"investigation-ongoing"/"not-upheld"/
# "court-dismissed"/"no-action" are explicitly NOT included here (an
# accusation that was dismissed or never actioned is not itself negative
# evidence; existence in the tracker is not the polarity, the outcome is
# -- same discipline as ICIJ/Verra/BoCC's "existence flag, not an
# accusation" caution elsewhere in this pipeline).
_ESGSOURCE_NEGATIVE_OUTCOMES = {
    "fine", "settlement", "ad-banned", "court-injunction", "voluntary-change",
}


def _esgsource_greenwashing_signal(company: str) -> Optional[str]:
    """Greenwashing accusation hit, matched on the row's own `company`
    field. Only rows with a genuinely adjudicated/resolved-against-the-
    company outcome_class count as negative evidence (see
    _ESGSOURCE_NEGATIVE_OUTCOMES) -- an ongoing/dismissed/not-upheld
    accusation is reported too but flagged as such, left for the
    extractor to weigh rather than silently dropped."""
    parts: list[str] = []
    for row in _load_esgsource_greenwashing():
        name = (row.get("company") or "").strip()
        if not _neutral_name_match(name, company):
            continue
        outcome_class = (row.get("outcome_class") or "").strip()
        outcome_label = (row.get("outcome_label") or "").strip()
        claim_category = (row.get("claim_category") or "").strip()
        date = (row.get("date") or "").strip()
        penalty = row.get("penalty_amount")
        currency = (row.get("penalty_currency") or "").strip()
        is_negative = outcome_class in _ESGSOURCE_NEGATIVE_OUTCOMES
        bits = [b for b in (name, claim_category, outcome_label, date) if b]
        if penalty and currency:
            bits.append(f"{penalty:,.0f} {currency}")
        if not is_negative:
            bits.append("(accusation not upheld/resolved -- not treated as negative evidence)")
        url = (row.get("source_url") or "").strip()
        if url:
            bits.append(f"<{url}>")
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] esgsource_greenwashing → %d matched accusation(s)", company, len(parts))
    return f"Greenwashing enforcement tracker (ESGsource): {body}"


def _nyspill_signal(company: str) -> Optional[str]:
    """NY State DEC spill-incident hit. Socrata's $q full-text search and
    $where filters work server-side here (unlike CPSC/Sweden SMP's dead
    name params) -- a genuine per-company live query, no bulk-fetch
    needed. NY-facility-scoped only: a company with no NY operations will
    simply never appear, which is correct (no false-negative risk to
    worry about, just narrow geographic coverage)."""
    _NYSPILL_LIMITER.wait()
    r = _get(_NYSPILL_URL, params={"$q": company[:60], "$limit": 20}, timeout=_TIMEOUT)
    if not r:
        return None
    try:
        rows = r.json() or []
    except (ValueError, AttributeError):
        return None
    if not isinstance(rows, list):
        return None

    parts: list[str] = []
    for row in rows:
        facility = (row.get("program_facility_name") or "").strip()
        if not _strict_name_match(facility, company):
            continue
        date = (row.get("spill_date") or "").strip()[:10]
        material = (row.get("material_name") or "").strip()
        quantity = (row.get("quantity") or "").strip()
        units = (row.get("units") or "").strip()
        locality = (row.get("locality") or "").strip()
        bits = [b for b in (facility, material, locality, date) if b]
        if quantity and quantity != "0.00":
            bits.append(f"{quantity} {units}".strip())
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] ny_spill → %d matched incident(s)", company, len(parts))
    return f"NY State spill incident: {body} <https://data.ny.gov/resource/u44d-k5fk>"


def _load_sweden_smp() -> list[dict]:
    """Swedish EPA facility register -- fetched once per process, then
    filtered locally. NOT a per-company live query: verified live
    2026-09-23 that `searchText` is not a real server-side name filter
    (same trap as _load_cpsc_recalls' RecallFirmName -- a query for
    "Volvo" returned a DIFFERENT but not name-matched result set, no
    literal "Volvo" hits at all). Small corpus (~860 facilities, ~1.1MB),
    in-memory only, no disk persistence needed."""
    global _sweden_smp_cache
    with _sweden_smp_lock:
        if _sweden_smp_cache is not None:
            return _sweden_smp_cache
        _sweden_smp_cache = []
        _SWEDEN_SMP_LIMITER.wait()
        r = _get(_SWEDEN_SMP_URL, timeout=_TIMEOUT)
        if not r:
            log.warning("Sweden SMP facility register unavailable")
            return _sweden_smp_cache
        try:
            data = r.json() or []
        except (ValueError, AttributeError):
            log.warning("Sweden SMP facility register parse failed")
            return _sweden_smp_cache
        if isinstance(data, list):
            _sweden_smp_cache = data
        log.info("loaded Sweden SMP facility register: %d facilities", len(_sweden_smp_cache))
        return _sweden_smp_cache


def _sweden_smp_signal(company: str) -> Optional[str]:
    """Swedish PRTR-style facility hit, matched on the record's own
    `huvudman` (operator/principal) field. FOOTPRINT/existence signal
    like _envirofacts_signal above -- appearing in this register reflects
    industrial-emissions-permit scale, not a violation; polarity left to
    the extractor. Sweden-facility-scoped only, same narrow-coverage-not-
    false-negative caveat as the NY spill signal above.

    Uses _neutral_name_match, not _strict_name_match: verified live
    2026-09-23 that a real "AB Sandvik Coromant" row was being silently
    dropped for a company argument of "Sandvik" (single 7-char token,
    below _strict_name_match's accusation-grade floor) -- same false-
    negative class as USAspending's "Boeing" issue. This corpus is a
    curated ~860-facility national register, not a noisy free-text
    search over a huge government corpus, so substring-bleed risk is low
    (same reasoning as _neutral_name_match's other use sites)."""
    parts: list[str] = []
    for row in _load_sweden_smp():
        operator = (row.get("huvudman") or "").strip()
        if not _neutral_name_match(operator, company):
            continue
        facility = (row.get("namn") or "").strip()
        industry = (row.get("huvudBranschNamn") or "").strip()
        municipality = (row.get("kommun") or "").strip()
        env_mgmt = (row.get("miljoledningssystem") or "").strip()
        bits = [b for b in (operator, facility, industry, municipality) if b]
        if env_mgmt:
            bits.append(f"environmental mgmt system: {env_mgmt}")
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    parts = list(dict.fromkeys(parts))
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] sweden_smp → %d matched facilit(y/ies)", company, len(parts))
    return f"Swedish EPA facility register (footprint, not a violation): {body} <https://utslappisiffror.naturvardsverket.se/>"


_CRISIL_ROW_RE = re.compile(r"<tr>\s*(<td.*?)</tr>", re.DOTALL)
_CRISIL_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)
_CRISIL_TAG_RE = re.compile(r"<[^>]+>")


def _load_crisil_ratings() -> list[dict]:
    """Crisil ESG Ratings (India) -- parsed once per process from a
    MANUALLY SAVED on-disk HTML snapshot, never fetched live.

    crisilesg.com/robots.txt explicitly disallows anthropic-ai/claudebot/
    Claude-Web BY NAME, with no citation/search carve-out (unlike
    KnowESG's robots.txt, which blocks the training crawler anthropic-ai
    but explicitly allows ClaudeBot for search/citation use) -- verified
    live 2026-09-23. Rather than work around a block the site operator
    wrote specifically naming Claude, this module reads a snapshot the
    user saved themselves through their own browser (external_data/
    crisil_esg_ratings/<tab>.html, one file per the site's own
    alphabetical tab split: a_d/e_h/i_l/m_p/q_t/u_x/y_z) -- same
    "external_data/ = pipeline-usable bulk cache" role as OFAC/Cornell/
    NLRB, just populated by a human click instead of an HTTP fetch.
    Re-running this collector never re-fetches anything; refreshing the
    rating data requires the user to manually re-save the page (ratings
    are periodically re-issued, see each row's own date field).

    Real table shape confirmed live 2026-09-23: each <tr> has exactly 5
    <td> cells (issuer name, sector, "Crisil ESG <score>", date, category
    band) -- parsed with a plain regex rather than adding a new
    BeautifulSoup/lxml dependency this codebase doesn't otherwise use,
    since the row shape is simple and fixed."""
    global _crisil_cache
    with _crisil_lock:
        if _crisil_cache is not None:
            return _crisil_cache
        _crisil_cache = []
        if not _CRISIL_DIR.exists():
            log.debug("Crisil ESG ratings snapshot not found at %s -- skipped", _CRISIL_DIR)
            return _crisil_cache

        rows: list[dict] = []
        for fname in _CRISIL_TAB_FILES:
            path = _CRISIL_DIR / fname
            if not path.exists():
                continue
            try:
                html = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("Crisil snapshot %s unreadable: %s", fname, exc)
                continue
            for row in _CRISIL_ROW_RE.findall(html):
                cells = [_CRISIL_TAG_RE.sub("", c).replace("&amp;", "&").strip()
                         for c in _CRISIL_CELL_RE.findall(row)]
                if len(cells) != 5 or not cells[2].startswith("Crisil ESG"):
                    continue
                name, sector, rating, date, category = cells
                score_match = re.search(r"(\d+)", rating)
                rows.append({
                    "name": name, "sector": sector,
                    "score": int(score_match.group(1)) if score_match else None,
                    "date": date, "category": category,
                })
        _crisil_cache = rows
        log.info("loaded Crisil ESG ratings snapshot: %d rows across %d tab file(s)",
                  len(rows), len(_CRISIL_TAB_FILES))
        return _crisil_cache


def _crisil_signal(company: str) -> Optional[str]:
    """Crisil ESG rating hit, matched on the issuer name. Same
    lenient _neutral_name_match as USAspending/Sweden SMP/ESGsource --
    this is a curated ~944-row named-issuer list, not a noisy free-text
    government corpus, so short-name false positives are unlikely and
    the false-negative cost (dropping a real short-named issuer) is not
    worth the accusation-grade floor. A company can appear more than
    once across re-rating history (same issuer, different date/score) --
    kept as-is (both shown) rather than de-duplicated to "latest only",
    since the extractor benefits from seeing rating trend/history, not
    just a point value."""
    parts: list[str] = []
    for row in _load_crisil_ratings():
        if not _neutral_name_match(row["name"], company):
            continue
        bits = [row["name"], row["sector"], f"Crisil ESG {row['score']}" if row["score"] is not None else "",
                row["category"], row["date"]]
        bits = [b for b in bits if b]
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] crisil_esg_rating → %d matched rating(s)", company, len(parts))
    return f"Crisil ESG Rating (third-party rater opinion, India): {body}"


def _load_nbim_exclusions() -> list[dict]:
    """NBIM/Norges Bank exclusion list via OpenSanctions -- small (~400
    rows) FTM JSON Lines, fetched once per process, in-memory only (same
    rationale as ESGsource/Sweden SMP's small in-memory-only caches).
    `requests`/`_get` follow the 307 redirect to the dated CDN artifact
    automatically -- no special handling needed, unlike A18's full bulk
    export which is deliberately NOT pulled here (2.6GB, see module
    docstring)."""
    global _nbim_cache
    with _nbim_lock:
        if _nbim_cache is not None:
            return _nbim_cache
        _nbim_cache = []
        _NBIM_LIMITER.wait()
        r = _get(_NBIM_URL, timeout=60)
        if not r:
            log.warning("NBIM exclusion list unavailable")
            return _nbim_cache
        rows = []
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (ValueError, TypeError):
                continue
        _nbim_cache = rows
        log.info("loaded NBIM exclusion list: %d entities", len(_nbim_cache))
        return _nbim_cache


def _nbim_signal(company: str) -> Optional[str]:
    """NBIM (Norway's sovereign wealth fund) exclusion hit, matched on
    the FTM entity's `caption` field. Same evidence class as OFAC SDN/
    World Bank debarment above -- NBIM excludes companies from its
    investment universe for conduct-based reasons (the `properties.
    topics` field carries e.g. "debarment"), a real adjudicated
    negative signal, not a neutral existence flag."""
    parts: list[str] = []
    for entity in _load_nbim_exclusions():
        caption = (entity.get("caption") or "").strip()
        if not _strict_name_match(caption, company):
            continue
        topics = ", ".join((entity.get("properties") or {}).get("topics") or [])
        first_seen = (entity.get("first_seen") or "")[:10]
        bits = [b for b in (caption, topics, first_seen) if b]
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] nbim_exclusion → %d matched entit(y/ies)", company, len(parts))
    return f"NBIM (Norges Bank) investment exclusion: {body} <https://www.nbim.no/>"


def _load_nger_corporations() -> list[dict]:
    """Australia NGER controlling-corporations bulk CSV -- already
    downloaded to external_data/ (see _NGER_CSV_PATH), read once per
    process and kept in memory. NOT re-fetched here -- unlike OFAC/
    Cornell/etc. above, this file was a one-time manual pull (see
    module docstring's NGER note) rather than something this module
    fetches itself; a missing file is treated as "not yet downloaded",
    logged and skipped, same graceful-skip discipline as SAM.gov's
    unset-key case."""
    global _nger_cache
    with _nger_lock:
        if _nger_cache is not None:
            return _nger_cache
        _nger_cache = []
        if not _NGER_CSV_PATH.exists():
            log.debug("NGER corporations CSV not found at %s -- skipped", _NGER_CSV_PATH)
            return _nger_cache
        try:
            with open(_NGER_CSV_PATH, encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                _nger_cache = list(reader)
        except (OSError, csv.Error) as exc:
            log.warning("NGER corporations CSV parse failed: %s", exc)
        log.info("loaded Australia NGER corporations: %d rows", len(_nger_cache))
        return _nger_cache


def _nger_signal(company: str) -> Optional[str]:
    """Australia NGER (National Greenhouse and Energy Reporting) hit,
    matched on the record's own Organisation name field. FOOTPRINT
    signal like Envirofacts TRI/Sweden SMP above -- mandatory annual
    disclosure of real Scope 1/2 emissions + net energy consumed, not
    itself a violation."""
    parts: list[str] = []
    for row in _load_nger_corporations():
        org = (row.get("Organisation name") or "").strip()
        if not _strict_name_match(org, company):
            continue
        scope1 = (row.get("Total scope 1 emissions (t CO2-e)") or "").strip()
        scope2 = (row.get("Total scope 2 emissions (t CO2-e)") or "").strip()
        energy = (row.get("Net energy consumed (GJ)") or "").strip()
        bits = [org]
        if scope1:
            bits.append(f"Scope 1: {scope1} t CO2-e")
        if scope2:
            bits.append(f"Scope 2: {scope2} t CO2-e")
        if energy:
            bits.append(f"net energy: {energy} GJ")
        parts.append("; ".join(bits))

    if not parts:
        return None
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] nger_emissions → %d matched corporation(s)", company, len(parts))
    return f"Australia NGER disclosed emissions (footprint, not a violation): {body} <https://cer.gov.au/>"


def _load_bocc_financing() -> list[dict]:
    """Banking on Climate Chaos (BOCC) master financing CSV -- whole-
    corpus, fetched once per process (same "corpus-wide list, cache it"
    pattern as _load_ofac_sdn), persisted to external_data/ so a same-
    day re-run doesn't re-fetch ~2MB/19k rows. Refresh is manual (delete
    the file) since BOCC publishes a new year's data roughly annually,
    not on a schedule this module tracks."""
    global _bocc_cache
    with _bocc_lock:
        if _bocc_cache is not None:
            return _bocc_cache
        _bocc_cache = []

        raw_text: Optional[str] = None
        if _BOCC_CACHE_PATH.exists():
            raw_text = _BOCC_CACHE_PATH.read_text(encoding="utf-8", errors="replace")
        else:
            _BOCC_LIMITER.wait()
            r = _get(_BOCC_URL, timeout=60, cache=False)
            if r:
                raw_text = r.text
                try:
                    _EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
                    _BOCC_CACHE_PATH.write_text(raw_text, encoding="utf-8")
                except OSError as exc:
                    log.warning("could not persist BOCC financing cache: %s", exc)

        if not raw_text:
            log.warning("BOCC financing data unavailable")
            return _bocc_cache
        try:
            reader = csv.DictReader(io.StringIO(raw_text))
            _bocc_cache = list(reader)
        except csv.Error as exc:
            log.warning("BOCC financing data parse failed: %s", exc)
        log.info("loaded Banking on Climate Chaos financing data: %d rows", len(_bocc_cache))
        return _bocc_cache


def _bocc_signal(company: str) -> Optional[str]:
    """Banking on Climate Chaos hit, matched on the record's own
    Company field. NEUTRAL existence/exposure signal, same polarity
    caution as _usaspending_signal/_envirofacts_signal above --
    receiving fossil-fuel financing is not itself a violation. Uses
    _neutral_name_match, not _strict_name_match: this is a curated,
    name-identified per-company dataset (not a noisy free-text
    government search corpus), so the same false-negative risk for
    short company names (USAspending's "Boeing", Sweden SMP's
    "Sandvik") applies here too."""
    parts: list[str] = []
    for row in _load_bocc_financing():
        name = (row.get("Company") or "").strip()
        parent = (row.get("Company_Parent") or "").strip()
        if not (_neutral_name_match(name, company) or (parent and parent != name and _neutral_name_match(parent, company))):
            continue
        bank = (row.get("Bank") or "").strip()
        total = (row.get("Total") or "").strip()
        bits = [b for b in (name, bank) if b]
        if parent and parent != name:
            bits.append(f"parent: {parent}")
        if total:
            bits.append(f"${total} total 2021-2025")
        if bits:
            parts.append("; ".join(bits))

    if not parts:
        return None
    parts = list(dict.fromkeys(parts))
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] bocc_financing → %d matched record(s)", company, len(parts))
    return f"Fossil-fuel financing received (neutral, not a violation): {body} <https://www.bankingonclimatechaos.org/>"


def _trase_signal(company: str) -> Optional[str]:
    """Trase.earth agricultural supply-chain deforestation-exposure hit.
    Queried against the pre-built SQLite index (_TRASE_DB_PATH, see
    _build_trase_index.py) -- a SQL LIKE pre-filter narrows the
    candidate set cheaply (index-assisted), then _neutral_name_match
    does the real precision check, same two-step pattern as
    _nlrb_signal above. Uses _neutral_name_match, not _strict_name_match:
    commodity traders/exporters often have short real names (Cargill,
    Bunge, ADM) that _strict_name_match's accusation-grade floor would
    drop -- same class of fix as USAspending/Sweden SMP/BOCC. FOOTPRINT/
    exposure signal, not an accusation -- appearing as an exporter in a
    commodity's supply chain is not itself a violation; the
    exposure_flag_count/commitment_flags fields are what the extractor
    should weigh for polarity, not mere appearance in the dataset.
    Returns None (file missing) if the one-time index hasn't been
    built yet -- same graceful-skip discipline as SAM.gov's unset-key
    case."""
    if not _TRASE_DB_PATH.exists():
        log.debug("[%s] trase_deforestation → skipped (index not built, see _build_trase_index.py)", company)
        return None

    try:
        conn = sqlite3.connect(str(_TRASE_DB_PATH))
    except sqlite3.Error as exc:
        log.warning("Trase index open failed: %s", exc)
        return None
    try:
        rows = conn.execute(
            "SELECT entity_name, role, dataset, record_count, year_min, year_max, "
            "countries, exposure_flag_count, sample_exposure_detail, commitment_flags "
            "FROM entities WHERE entity_name LIKE ?", (f"%{company[:60]}%",),
        ).fetchall()
    except sqlite3.Error as exc:
        log.warning("[%s] Trase index query failed: %s", company, exc)
        return None
    finally:
        conn.close()

    parts: list[str] = []
    for (entity_name, role, dataset, record_count, year_min, year_max,
         countries, exposure_flag_count, sample_exposure, commitment_flags) in rows:
        if not _neutral_name_match(entity_name, company):
            continue
        years = f"{year_min}-{year_max}" if year_min and year_max else (year_min or year_max or "")
        bits = [entity_name, role.replace("_", " "), dataset.rsplit("_v", 1)[0].replace("_", " ")]
        if years:
            bits.append(years)
        if countries:
            bits.append(countries)
        if record_count:
            bits.append(f"{record_count} record(s)")
        if exposure_flag_count:
            bits.append(f"{exposure_flag_count} with deforestation-exposure data")
        if sample_exposure:
            bits.append(sample_exposure)
        if commitment_flags:
            bits.append(commitment_flags)
        parts.append("; ".join(b for b in bits if b))

    if not parts:
        return None
    parts = list(dict.fromkeys(parts))[:10]
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] trase_deforestation → %d matched entit(y/ies)", company, len(parts))
    return f"Agricultural supply-chain deforestation exposure (Trase.earth, footprint not a violation): {body} <https://trase.earth/>"


_warn_cache: dict[str, Optional[list[dict]]] = {}
_warn_lock = threading.Lock()

# Verified live 2026-10-06: a bare "notice" match (without "date") picks
# up boolean/non-date columns on some states (Illinois's "WARN Notice"
# is a True/False flag, not a date) -- requiring "date" literally fixes
# this without needing a per-state override.
_WARN_DATE_FIELD_RE = re.compile(r"date", re.IGNORECASE)
_WARN_EMPLOYEE_FIELD_RE = re.compile(r"employ|worker|affected|jobs", re.IGNORECASE)


def _load_warn_state(state: str) -> Optional[list[dict]]:
    """One state's already-downloaded WARN Act CSV, read once per
    process and kept in memory -- same "manual one-time pull, graceful
    skip if missing" pattern as _load_nger_corporations, not a live
    fetch (running warn-scraper live, per company, per state, on every
    pipeline call would be both slow -- real HTTP/Selenium work per
    state -- and would re-trigger the real, confirmed-live failures in
    the other 24 states every single time).

    Maryland's CSV has no header row (see _WARN_MD_COMPANY_COL_INDEX) --
    handled with csv.reader + positional indexing instead of
    DictReader, the only state needing that distinct code path."""
    with _warn_lock:
        if state in _warn_cache:
            return _warn_cache[state]
        path = _WARN_DIR / f"{state}.csv"
        if not path.exists():
            log.debug("WARN %s CSV not found at %s -- skipped", state, path)
            _warn_cache[state] = None
            return None
        try:
            if state == "md":
                with open(path, encoding="utf-8", errors="replace", newline="") as f:
                    rows = [{"_company": row[_WARN_MD_COMPANY_COL_INDEX], "_raw": row}
                            for row in csv.reader(f) if len(row) > _WARN_MD_COMPANY_COL_INDEX]
            else:
                with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    company_field = _WARN_COMPANY_FIELD_OVERRIDE.get(state)
                    if not company_field:
                        company_field = next((h for h in fieldnames if _WARN_COMPANY_FIELD_RE.search(h)), None)
                    if not company_field:
                        log.warning("WARN %s: no company-identifying field found in %s", state, fieldnames)
                        _warn_cache[state] = None
                        return None
                    date_field = next((h for h in fieldnames if _WARN_DATE_FIELD_RE.search(h)), None)
                    employee_field = next((h for h in fieldnames if _WARN_EMPLOYEE_FIELD_RE.search(h)), None)
                    rows = []
                    for row in reader:
                        rows.append({
                            "_company": row.get(company_field, ""),
                            "_date": row.get(date_field, "") if date_field else "",
                            "_employees": row.get(employee_field, "") if employee_field else "",
                        })
        except (OSError, csv.Error, IndexError) as exc:
            log.warning("WARN %s CSV parse failed: %s", state, exc)
            _warn_cache[state] = None
            return None
        _warn_cache[state] = rows
        log.info("loaded WARN %s: %d rows", state, len(rows))
        return rows


def _warn_signal(company: str) -> Optional[str]:
    """WARN Act mass-layoff/closure notice hit, matched on each state's
    own company-identifying field (see _load_warn_state). Only queries
    the WARN_WORKING_STATES snapshot -- see module docstring for why
    the other 24 states aren't included. Negative-polarity: a WARN
    notice is a real, legally-mandated filing for an actual closure/
    layoff, not a neutral existence flag."""
    parts: list[str] = []
    for state in WARN_WORKING_STATES:
        rows = _load_warn_state(state)
        if not rows:
            continue
        for row in rows:
            name = (row.get("_company") or "").strip()
            if not name or not _strict_name_match(name, company):
                continue
            if state == "md":
                raw = row.get("_raw") or []
                bits = [name, f"state: {state.upper()}"] + [b for b in raw if b and b != name]
            else:
                date = (row.get("_date") or "").strip()
                employees = (row.get("_employees") or "").strip()
                bits = [name, f"state: {state.upper()}"]
                if date:
                    bits.append(date)
                if employees:
                    bits.append(f"{employees} employees affected")
            parts.append("; ".join(bits))

    if not parts:
        return None
    parts = list(dict.fromkeys(parts))[:10]
    body = " | ".join(parts)[:_MAX_CHARS]
    log.info("[%s] warn_notice → %d matched notice(s)", company, len(parts))
    return f"WARN Act mass-layoff/closure notice: {body}"


def _osi_signal(company: str) -> Optional[str]:
    """Open Sustainability Index hit -- a genuine per-company query, not
    a fetch-once-corpus pattern (the platform's own /v1/search resolves
    a company name to its real slug server-side; no client-side name
    matching needed, unlike most of this module). POSITIVE-capable
    (disclosed Scope 1/2/3 emissions, climate targets) but a removed/
    expired commitment IS a real negative signal -- flagged explicitly,
    not silently dropped alongside the positive disclosure data.

    Both calls use api-key=demo -- verified live 2026-10-06 that this
    documented DEFAULT VALUE is a genuine working free-tier key, not a
    placeholder (see module docstring)."""
    _OSI_LIMITER.wait()
    r = _get(_OSI_SEARCH_URL, params={"query": company[:60], "api-key": _OSI_API_KEY}, timeout=_TIMEOUT)
    if not r:
        return None
    try:
        search_data = (r.json() or {}).get("data") or []
    except (ValueError, AttributeError):
        return None
    slug = next((item.get("slug") for item in search_data if item.get("type") == "company"), None)
    if not slug:
        return None

    _OSI_LIMITER.wait()
    r2 = _get(_OSI_COMPANY_URL.format(slug=slug), params={"api-key": _OSI_API_KEY}, timeout=_TIMEOUT)
    if not r2:
        return None
    try:
        payload = r2.json() or {}
    except (ValueError, AttributeError):
        return None
    if payload.get("error"):
        return None
    data = payload.get("data") or {}
    if not data:
        return None

    bits = [data.get("company_name") or company, data.get("industry") or ""]
    emissions = data.get("emissions") or []
    if emissions:
        latest = max(emissions, key=lambda e: e.get("year") or 0)
        year = latest.get("year")
        s1, s2, s3 = latest.get("scope_1"), latest.get("scope_2"), latest.get("total_scope_3")
        emis_bits = [b for b in (
            f"Scope 1: {s1}" if s1 is not None else None,
            f"Scope 2: {s2}" if s2 is not None else None,
            f"Scope 3: {s3}" if s3 is not None else None,
        ) if b]
        if emis_bits:
            bits.append(f"{year} emissions -- " + ", ".join(emis_bits))

    removed_commitments = [c for c in (data.get("commitment") or []) if c.get("status") == "Removed"]
    for c in removed_commitments[:2]:
        reason = c.get("reason_for_commitment_extension_or_removal") or ""
        bits.append(f"REMOVED commitment ({c.get('commitment_type', '')}): {reason}".strip())

    bits = [b for b in bits if b]
    if not bits:
        return None
    body = "; ".join(bits)[:_MAX_CHARS]
    log.info("[%s] open_sustainability_index → matched (slug=%s)", company, slug)
    return f"Open Sustainability Index disclosed emissions/targets: {body} <https://opensustainabilityindex.org/>"


# Text-block keys that carry real E/S/G narrative in a Japanese Securities
# Report (doc type 120) -- verified live 2026-10-06 against a real Toyota
# filing (substantive multi-paragraph content, not boilerplate/empty).
# Many more text_blocks exist on the parsed report (pure financial-
# statement notes, share-structure detail, etc.) -- deliberately NOT
# pulled here, same "evidence relevance over completeness" discipline
# report_collector.py applies to PDF sustainability reports.
_EDINET_ESG_TEXT_BLOCKS = (
    ("GovernanceClimateChangeTextBlock", "E"),
    ("StrategyClimateChangeTextBlock", "E"),
    ("MetricsAndTargetsClimateChangeTextBlock", "E"),
    ("RiskManagementClimateChangeTextBlock", "E"),
    ("GovernanceHumanCapitalTextBlock", "S"),
    ("RiskManagementHumanCapitalTextBlock", "S"),
    ("PolicyOnDevelopmentOfHumanResourcesAndInternalEnvironmentStrategyTextBlock", "S"),
    ("OverviewOfCorporateGovernanceTextBlock", "G"),
    ("OutsideDirectorsAndOutsideCorporateAuditorsTextBlock", "G"),
    ("RemunerationForDirectorsAndOtherOfficersTextBlock", "G"),
)
# Per-chunk cap, not the module's usual _MAX_CHARS -- this source emits
# MULTIPLE evidence chunks (one per pillar-relevant text block found),
# each independently truncated, rather than one combined blob, so a long
# governance section doesn't crowd out a short climate one.
_EDINET_BLOCK_CHARS = 700


def _edinet_signal(company: str) -> Optional[str]:
    """Japan EDINET securities-report hit via the third-party
    `edinet-tools` package (see module docstring for why -- it solves
    the exact company-name-to-entity-code mapping problem that blocked
    NSE/India). Entity resolution (edinet_tools.entity(name)) works
    OFFLINE against a bundled FSA registry snapshot, no key needed for
    that step -- only fetching/parsing documents needs JAPAN_EDINET_KEY.

    Only queries the most recent annual Securities Report (doc type
    120, within the last 400 days -- Japan's annual filing cadence)
    -- NOT every document type EDINET has (large-shareholding notices,
    tender offers, etc. are a different evidence class this module
    doesn't currently model). Surfaces REAL JAPANESE TEXT as-is, same
    no-translation precedent as Taiwan TWSE's Chinese-only KPI data
    above -- trust the downstream LLM extractor to read it directly."""
    if not _JAPAN_EDINET_KEY:
        log.debug("[%s] edinet_securities_report → skipped (JAPAN_EDINET_KEY not set)", company)
        return None

    import edinet_tools
    os.environ.setdefault("EDINET_API_KEY", _JAPAN_EDINET_KEY)

    try:
        entity = edinet_tools.entity(company[:60])
    except Exception as exc:
        log.warning("[%s] EDINET entity lookup failed: %s", company, exc)
        return None
    if not entity:
        return None

    _EDINET_LIMITER.wait()
    try:
        docs = entity.documents(days=400, doc_type="120")
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.warning("[%s] EDINET document search failed: %s", company, exc)
        return None
    if not docs:
        return None

    _EDINET_LIMITER.wait()
    try:
        report = docs[0].parse()
    except RateLimitTripped:
        raise
    except Exception as exc:
        log.warning("[%s] EDINET report parse failed: %s", company, exc)
        return None

    parts: list[str] = []
    text_blocks = getattr(report, "text_blocks", None) or {}
    for block_name, pillar in _EDINET_ESG_TEXT_BLOCKS:
        text = (text_blocks.get(block_name) or "").strip()
        if text:
            parts.append(f"[{pillar}] {block_name}: {text[:_EDINET_BLOCK_CHARS]}")

    if not parts:
        return None
    body = " | ".join(parts)
    log.info("[%s] edinet_securities_report → %d matched text block(s)", company, len(parts))
    return f"Japan EDINET Securities Report (filer: {entity.name}): {body} <https://disclosure2.edinet-fsa.go.jp/>"


def fetch_public_records_signals(company: str) -> dict[str, str]:
    """Public-records evidence for one company from all sources in this
    module. Returns {signal_name: text}; {} when the company appears in
    none of them -- same "absence is not evidence of good conduct" rule
    as enforcement_collector.fetch_enforcement_signals."""
    log_header(log, "PublicRecords", company=company, sources=21)
    signals: dict[str, str] = {}
    for name, fn in (("cpsc_recall", _cpsc_signal),
                     ("usaspending", _usaspending_signal),
                     ("ofac_sdn", _ofac_signal),
                     ("envirofacts_tri", _envirofacts_signal),
                     ("twse_esg", _twse_signal),
                     ("cornell_labor_action", _cornell_signal),
                     ("uk_modern_slavery", _ukmsa_signal),
                     ("nlrb_case", _nlrb_signal),
                     ("icij_offshoreleaks", _icij_signal),
                     ("samgov_exclusion", _samgov_signal),
                     ("esgsource_greenwashing", _esgsource_greenwashing_signal),
                     ("ny_spill", _nyspill_signal),
                     ("sweden_smp", _sweden_smp_signal),
                     ("crisil_esg_rating", _crisil_signal),
                     ("nbim_exclusion", _nbim_signal),
                     ("nger_emissions", _nger_signal),
                     ("bocc_financing", _bocc_signal),
                     ("trase_deforestation", _trase_signal),
                     ("warn_notice", _warn_signal),
                     ("open_sustainability_index", _osi_signal),
                     ("edinet_securities_report", _edinet_signal)):
        try:
            got = fn(company)
        except RateLimitTripped:
            raise
        except Exception as exc:
            log.warning("[%s] %s → exception: %s", company, name, exc)
            continue
        if got:
            signals[name] = got
    log.info("[%s] public_records done — %d/21 sources hit", company, len(signals))
    return signals


if __name__ == "__main__":  # manual probe
    import sys
    target = " ".join(sys.argv[1:]) or "Boeing"
    out = fetch_public_records_signals(target)
    if not out:
        print(f"(no public-records evidence for {target!r})")
    for k, v in out.items():
        print(f"\n=== {k} ===\n{v[:700]}")
