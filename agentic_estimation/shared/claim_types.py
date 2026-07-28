"""
claim_types.py — shared claim dataclass for Phase 2 (see PHASE_2_PLAN.md).

Both the deterministic Climate TRACE anchor (climate_trace_anchor.py) and the
LLM-based pillar extractors (pillar_extractors.py) produce the same claim
shape, consumed downstream by formula_estimator.py. Kept in its own module so
neither producer depends on the other.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedClaim:
    factor: str             # must be a key in factor_registry.FACTORS
    pillar: str              # 'E' | 'S' | 'G'
    polarity: int             # -1 | 0 | 1 (0 = magnitude-only, e.g. a raw value claim)
    strength: float           # 0-1
    confidence: float         # 0-1
    value: Optional[float]    # raw disclosed/measured value, when the factor is a quantity
    source_tag: str           # signal source tag (e.g. 'wikipedia') or a dataset tag
                              # (e.g. 'climate_trace_owner_match') for dataset_lookup claims
    reasoning: str
    method: str = "extracted"  # 'extracted' | 'dataset_lookup' (peer_ratio_fallback/
                                # coarse_bucket are produced by ratio_estimator.py, not here)
