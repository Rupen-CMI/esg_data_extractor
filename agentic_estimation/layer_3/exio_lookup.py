"""
exio_lookup.py — EXIOBASE sector structural intensity as an E-pillar prior
rung for the low-evidence ladder (see research/LOW_EVIDENCE_LADDER_PLAN.md).

WHAT THIS IS: `exio_sector_intensity` (150 EXIOBASE sectors) carries real
physical intensity data -- co2e/water/material/land per M EUR of output,
already converted to a 0-100 PERCENTILE within EXIOBASE's own 150-sector
distribution (co2e_pct=100 -> the single most emissions-intensive sector in
EXIOBASE). This is structural: "how dirty is this INDUSTRY, physically",
independent of any single company's disclosed behavior -- exactly the signal
a company with zero fetched evidence should fall back to, rather than a flat
country baseline that ignores what the company actually does.

Confirmed 2026-08-18: this table has ZERO references in formula_estimator.py
or graph.py -- it was built and validated (E-pillar Spearman +0.44 to +0.47
on real holdout, see research/SCORING_PLAN.md) but never wired into anything
that runs. This module is the wiring.

MATCHING: no direct company-industry -> EXIO-sector join exists (the only
existing join, market_exio_sector, keys off our OWN markets table, not off a
free-text company/upright industry string). Reuses sector_matcher.py's
TF-IDF fuzzy matcher (same tool peer_anchor.py already uses to match a
company's industry string against upright's 30 labels) against
exio_sector_intensity's 150 sector labels directly -- same confidence-floor,
explicit-no-match discipline, no new matching logic invented.

SCOPE: E pillar only. water_pct/material_pct/land_pct exist but are not
(yet) mapped to S/G -- EXIOBASE is a physical-flow database, it has no
governance or labor content, and mapping water/material into S would need
its own validation pass (out of scope for this rung, see plan section 7).

Percentile POLARITY: exio's *_pct columns are "how intense", not "how good"
-- co2e_pct=100 is the WORST (most emissions-intensive) sector, so it must be
INVERTED (100 - pct) before use as a peer_anchor-style "higher percentile is
better" vote, to stay consistent with every other rung in the ladder
(peer_anchor_vote, upright_industry_prior all return higher=better).
"""

from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("exio_lookup")

# Cached once per process -- 150 rows, small, fixed, read-only.
_exio_cache: Optional[list[tuple]] = None
_exio_labels_cache: Optional[list[str]] = None


@dataclass
class ExioVote:
    percentile: float    # 0-100, higher = better (already inverted from raw co2e_pct)
    confidence: float
    sector_matched: str
    similarity: float
    tier: str             # 'exio_structural'
    basis: str


def _load_exio_rows() -> list[tuple]:
    global _exio_cache, _exio_labels_cache
    if _exio_cache is None:
        from agentic_estimation.layer_1.peer_anchor_collector import _db_conn
        conn = _db_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT sector, co2e_pct FROM exio_sector_intensity WHERE co2e_pct IS NOT NULL"
            )
            _exio_cache = cur.fetchall()
        finally:
            conn.close()
        _exio_labels_cache = [r[0] for r in _exio_cache]
        log.info("loaded %d exio_sector_intensity rows with usable co2e_pct", len(_exio_cache))
    return _exio_cache


def exio_e_vote(sector: Optional[str]) -> Optional[ExioVote]:
    """Fuzzy-match `sector` (company/upright industry free text) against
    exio_sector_intensity's 150 sector labels; returns an E-pillar vote from
    the matched sector's co2e_pct (inverted to higher-is-better), or None on
    no usable match (empty sector, no token overlap, or below the fuzzy
    matcher's confidence floor -- caller falls through to the next rung)."""
    if not sector:
        return None
    from agentic_estimation.layer_1.sector_matcher import best_sector_match

    rows = _load_exio_rows()
    if not rows:
        return None
    labels = _exio_labels_cache
    match = best_sector_match(sector, labels)
    if not match or not match.matched:
        return None

    co2e_pct = next(v for lbl, v in rows if lbl == match.label)
    percentile = 100.0 - co2e_pct  # invert: lower structural emissions intensity = higher (better) percentile

    return ExioVote(
        percentile=percentile, confidence=0.35, sector_matched=match.label,
        similarity=match.similarity, tier="exio_structural",
        basis=(f"exio sector {match.label!r} (fuzzy-matched from {sector!r}, sim={match.similarity:.2f}): "
               f"co2e_pct={co2e_pct:.1f} -> inverted percentile {percentile:.1f}"),
    )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Dairy processing"
    v = exio_e_vote(q)
    print(v)
