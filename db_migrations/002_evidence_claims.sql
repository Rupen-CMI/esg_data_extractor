-- ============================================================
-- Migration 002: Evidence-claims table for the rebuilt agentic
-- estimation pipeline (see plan: all-data-metadata-linked-dijkstra.md)
--
-- Layer 2 Extractor agents (E/S/G) emit typed, source-attributed
-- claims here instead of freehand scores. A claim with confidence > 0
-- MUST reference a real source_id (signal row or collector row) —
-- enforced by the NOT NULL FK below, not just prompt wording.
-- Run once in NeonDB (psql or the web SQL editor)
-- ============================================================

CREATE TABLE IF NOT EXISTS company_evidence_claims (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id      UUID NOT NULL REFERENCES companies(id) ON DELETE CASCADE,

    -- What the claim is about
    pillar          VARCHAR(1) NOT NULL CHECK (pillar IN ('E', 'S', 'G')),
    factor          VARCHAR(100) NOT NULL,           -- e.g. 'net_zero_pledge', 'board_independence'

    -- The claim itself (Layer 2 Extractor output shape)
    polarity        SMALLINT NOT NULL CHECK (polarity IN (-1, 0, 1)),
    strength        FLOAT NOT NULL CHECK (strength >= 0 AND strength <= 1),
    confidence      FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    value           FLOAT,                            -- optional numeric value if the factor is a quantity
    reasoning       TEXT,                              -- short justification, human-auditable

    -- Attribution — a claim with confidence > 0 must point to real evidence.
    -- source_signal_id references the raw text/data the claim was extracted from
    -- (company_esg_signals row, or a Layer-1 collector's own persisted evidence —
    -- new collectors added in Phase 1 should persist to company_esg_signals with
    -- their own `source` tag so this FK stays the single evidence-provenance path).
    source_signal_id UUID REFERENCES company_esg_signals(id) ON DELETE SET NULL,
    source_note      TEXT,                             -- e.g. peer-ratio fallback description when no signal applies

    -- Provenance / debuggability
    produced_by     VARCHAR(60) NOT NULL,              -- agent name, e.g. 'e_extractor', 'ratio_estimator'
    method          VARCHAR(30) NOT NULL DEFAULT 'extracted'
                        CHECK (method IN ('extracted', 'peer_ratio_fallback', 'coarse_bucket')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- A confident claim (confidence > 0) must be attributable to SOMETHING —
    -- either a real signal row or a documented fallback method/note. This is
    -- the schema-level enforcement of the "no source, no claim" rule.
    CONSTRAINT chk_confident_claim_has_provenance
        CHECK (confidence = 0 OR source_signal_id IS NOT NULL OR source_note IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_evidence_claims_company ON company_evidence_claims(company_id);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_company_pillar ON company_evidence_claims(company_id, pillar);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_factor ON company_evidence_claims(factor);
