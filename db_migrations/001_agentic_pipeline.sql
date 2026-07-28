-- ============================================================
-- Migration 001: Agentic ESG pipeline tables & metric seeds
-- Run once in NeonDB (psql or the web SQL editor)
-- ============================================================

-- 1. Add reasoning column to company_metric_values (idempotent)
ALTER TABLE company_metric_values
    ADD COLUMN IF NOT EXISTS reasoning TEXT;

-- 2. Company ESG signal store (one row per company per source)
CREATE TABLE IF NOT EXISTS company_esg_signals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id  UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    source      VARCHAR(60) NOT NULL,
    signal_text TEXT NOT NULL,
    gathered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_company_signal_source UNIQUE (company_id, source)
);
CREATE INDEX IF NOT EXISTS idx_esg_signals_company ON company_esg_signals(company_id);

-- 3. Country ESG baseline (World Bank, pre-computed by country_baseline_agent)
CREATE TABLE IF NOT EXISTS country_esg_baseline (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country         VARCHAR(100) UNIQUE NOT NULL,
    year            INTEGER NOT NULL,
    e_score         FLOAT NOT NULL,
    s_score         FLOAT NOT NULL,
    g_score         FLOAT NOT NULL,
    indicator_count INTEGER NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Seed the three ESG pillar metric definitions
--    These are universal (not sector-specific) and used by scoring_agent.
INSERT INTO esg_metric_definitions (key, name, description, unit, category, is_universal, source_framework)
VALUES
    ('esg_e_score',
     'Environmental Score',
     'Agentic ESG Environmental pillar score (0-100, country-baseline-adjusted). '
     'Covers climate/emissions posture, energy use, resource management, certifications.',
     'score',
     'E',
     true,
     'agentic_v1'),
    ('esg_s_score',
     'Social Score',
     'Agentic ESG Social pillar score (0-100, country-baseline-adjusted). '
     'Covers labor practices, human rights, supply chain, community engagement.',
     'score',
     'S',
     true,
     'agentic_v1'),
    ('esg_g_score',
     'Governance Score',
     'Agentic ESG Governance pillar score (0-100, country-baseline-adjusted). '
     'Covers board structure, transparency, disclosure, anti-corruption.',
     'score',
     'G',
     true,
     'agentic_v1')
ON CONFLICT (key) DO NOTHING;
