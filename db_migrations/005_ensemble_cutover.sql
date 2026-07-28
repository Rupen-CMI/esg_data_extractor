-- ============================================================
-- Migration 005: Ensemble scorer cutover (Phase 6)
-- Run once in NeonDB (psql or the web SQL editor)
--
-- Adds the columns needed to persist the new pipeline's uncertainty
-- output (Confidence Gate range + Phase 4 verify verdict) instead of
-- discarding it after a point score is chosen -- see
-- agentic_estimation/layer_3/confidence_gate.py and
-- agentic_estimation/layer_4/estimate_verifier.py. All columns are
-- nullable, no backfill -- existing rows (agentic_scoring_v1,
-- agentic_evaluator_v1, agentic_metrics_v1, etc.) are unaffected and
-- simply have NULL in the new columns.
--
-- Also (attempts to) seed the esg_summary metric definition -- CORRECTION,
-- verified when this migration was applied (2026-07-21): esg_summary
-- ALREADY EXISTED (category='G', 76 existing company_metric_values rows
-- already reference it) even though no .sql migration in this repo shows
-- it being created -- it must have been seeded out-of-band at some point.
-- The INSERT below is a no-op via ON CONFLICT DO NOTHING; kept so this
-- migration is still correct/idempotent if ever run against a fresh DB
-- that genuinely lacks the definition.
-- ============================================================

-- 1. Uncertainty/verdict columns on company_metric_values (idempotent)
ALTER TABLE company_metric_values
    ADD COLUMN IF NOT EXISTS low_value        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS high_value        DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS confidence_label TEXT,   -- 'high' | 'medium' | 'low'
    ADD COLUMN IF NOT EXISTS verdict          TEXT,   -- 'skipped' | 'passed' | 'passed_after_retry' | 'refuted'
    ADD COLUMN IF NOT EXISTS needs_review     BOOLEAN;

-- 2. Seed the esg_summary metric definition (previously never seeded --
--    explainability_agent.py's _upsert_summary raises if this key is
--    missing; this was a latent gap on the old path too, just never
--    exercised because production traffic mostly hit gap-fill mode).
INSERT INTO esg_metric_definitions (key, name, description, unit, category, is_universal, source_framework)
VALUES
    ('esg_summary',
     'ESG Summary',
     'Plain-English summary of a company''s ESG estimate and evidence trail. '
     'Not a score -- value is always the literal string "summary"; the '
     'content lives in reasoning.',
     'text',
     'E',   -- category is required NOT NULL; summary isn't pillar-specific,
            -- so 'E' is an arbitrary placeholder (mirrors no better option
            -- existing in the schema -- category has no 'none'/'summary' value).
     true,
     'agentic_v1')
ON CONFLICT (key) DO NOTHING;
