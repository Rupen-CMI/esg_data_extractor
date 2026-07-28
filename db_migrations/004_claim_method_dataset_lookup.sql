-- ============================================================
-- Migration 004: allow 'dataset_lookup' as a company_evidence_claims.method
-- value, for deterministic Climate-TRACE-derived claims (Phase 2,
-- climate_trace_anchor.py). Kept distinct from 'extracted' (LLM-tagged
-- evidence) so Phase 5 weight-fitting can tell dataset-grounded claims
-- apart from LLM extractions -- they carry categorically different trust.
-- Idempotent: safe to re-run.
-- ============================================================

ALTER TABLE company_evidence_claims
    DROP CONSTRAINT IF EXISTS company_evidence_claims_method_check;

ALTER TABLE company_evidence_claims
    ADD CONSTRAINT company_evidence_claims_method_check
        CHECK (method IN ('extracted', 'peer_ratio_fallback', 'coarse_bucket', 'dataset_lookup'));
