-- ============================================================
-- Migration 003: Climate TRACE emissions data cache (v7 API)
--
-- Two tables, harvested periodically by climate_trace_harvester.py, NOT
-- fetched live per company-estimation request. Storing our own copy
-- protects the pipeline against Climate TRACE downtime/rate-limits and
-- avoids re-paying the ~14,500-call owner-emissions harvest per request.
--
--   climate_trace_owners            -- name -> id lookup (~14,500 rows,
--                                        refreshed via a-z enumeration of
--                                        GET /v7/owners?name=<letter>)
--   climate_trace_owner_emissions   -- per-facility real emissions for a
--                                        known owner (GET /v7/sources?ownerIds=)
--                                        -- direct E-pillar evidence when a
--                                        company matches a tracked owner
--   climate_trace_country_emissions -- country totals AND country+sector
--                                        breakdown (GET /v7/sources/emissions
--                                        ?gadmId=&year=). sector/subsector
--                                        NULL = that country's grand total.
--
-- NO separate global/worldwide sector-totals table. VERIFIED live (2024,
-- 'power' sector): summing that sector's emissions_quantity across all 252
-- countries in climate_trace_country_emissions equals the API's own
-- no-gadmId global total to within float rounding (16,014,772,002.138445
-- vs 16,014,772,002.138453). So "sector-wise worldwide overall emissions"
-- is a derived aggregation over this one table, not separately harvested
-- data -- see the ct_global_sector_emissions view below.
--
-- Run once in NeonDB (psql or the web SQL editor)
-- ============================================================

CREATE TABLE IF NOT EXISTS climate_trace_owners (
    owner_id        VARCHAR(20) PRIMARY KEY,      -- Climate TRACE owner id, e.g. 'E100000000687'
    name            TEXT NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ct_owners_name ON climate_trace_owners USING gin (to_tsvector('simple', name));

CREATE TABLE IF NOT EXISTS climate_trace_owner_emissions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id            VARCHAR(20) NOT NULL REFERENCES climate_trace_owners(owner_id) ON DELETE CASCADE,
    source_id           BIGINT NOT NULL,           -- Climate TRACE source/asset id
    source_name         TEXT,                       -- e.g. 'ArcelorMittal Gent steel plant'
    country_iso3        VARCHAR(3),
    sector              VARCHAR(60),
    subsector           VARCHAR(60),
    -- asset_type is TEXT, not VARCHAR(60): some subsectors (e.g. pulp/paper)
    -- return long descriptive strings, not short codes -- found live during
    -- the full owner-emissions harvest ("Pulp misc. (known types include
    -- chemical wood pulp, pulp from fibres other than wood)", 85 chars),
    -- which crashed a VARCHAR(60) column. activity_units/capacity_units
    -- widened to TEXT too as a precaution against the same class of surprise.
    asset_type          TEXT,
    gas                 VARCHAR(20) NOT NULL DEFAULT 'co2e_100yr',
    emissions_quantity  DOUBLE PRECISION,          -- in the gas's native unit (tonnes)
    activity            DOUBLE PRECISION,
    activity_units      TEXT,
    capacity            DOUBLE PRECISION,
    capacity_units      TEXT,
    year                INTEGER NOT NULL,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_ct_owner_emissions UNIQUE (owner_id, source_id, gas, year)
);
CREATE INDEX IF NOT EXISTS idx_ct_owner_emissions_owner ON climate_trace_owner_emissions(owner_id);
CREATE INDEX IF NOT EXISTS idx_ct_owner_emissions_sector ON climate_trace_owner_emissions(sector, subsector);
CREATE INDEX IF NOT EXISTS idx_ct_owner_emissions_country ON climate_trace_owner_emissions(country_iso3);

CREATE TABLE IF NOT EXISTS climate_trace_country_emissions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_iso3        VARCHAR(3) NOT NULL,
    -- NULL sector = this country's grand total (from the `totals` block);
    -- non-null = one row per sector (from the `sectors` block). subsector
    -- follows the same NULL-means-parent-total pattern one level deeper.
    sector              VARCHAR(60),
    subsector           VARCHAR(60),
    gas                 VARCHAR(20) NOT NULL DEFAULT 'co2e_100yr',
    emissions_quantity  DOUBLE PRECISION NOT NULL,
    percentage_of_total DOUBLE PRECISION,          -- as reported by the API (% of the country total)
    year                INTEGER NOT NULL,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_ct_country_emissions
        UNIQUE NULLS NOT DISTINCT (country_iso3, sector, subsector, gas, year)
);
CREATE INDEX IF NOT EXISTS idx_ct_country_emissions_lookup ON climate_trace_country_emissions(country_iso3, sector, subsector, year);

-- Drop the earlier (unused, never-harvested) separate global-sector table
-- from the prior version of this migration -- superseded by the view below.
DROP TABLE IF EXISTS climate_trace_sector_emissions;

-- Worldwide sector totals, derived from climate_trace_country_emissions
-- rather than separately harvested (see verification note above).
CREATE OR REPLACE VIEW climate_trace_global_sector_emissions AS
SELECT
    sector,
    subsector,
    gas,
    year,
    SUM(emissions_quantity) AS emissions_quantity
FROM climate_trace_country_emissions
GROUP BY sector, subsector, gas, year;

-- Widen free-text columns on an already-created table (found live during
-- the full owner-emissions harvest -- see asset_type comment above).
-- Safe/idempotent to re-run: ALTER COLUMN ... TYPE TEXT is a no-op if the
-- column is already TEXT.
ALTER TABLE climate_trace_owner_emissions ALTER COLUMN asset_type TYPE TEXT;
ALTER TABLE climate_trace_owner_emissions ALTER COLUMN activity_units TYPE TEXT;
ALTER TABLE climate_trace_owner_emissions ALTER COLUMN capacity_units TYPE TEXT;
