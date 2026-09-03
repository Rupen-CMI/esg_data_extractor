"""
exio_lookup.py — EXIOBASE sector structural intensity as an industry-median
prior for both the low-evidence ladder (see research/LOW_EVIDENCE_LADDER_PLAN.md)
and, as of 2026-08-27, the high-evidence formula path (formula_estimator.py).

WHAT THIS IS: `exio_sector_intensity` (150 EXIOBASE sectors) carries real
physical/structural data per M EUR of output -- co2e/water/material/land
intensity (E) and employees_per_meur/lowskill_share/female_share (S),
independent of any single company's disclosed behavior. This is "what does
this INDUSTRY structurally look like", not "what has this company disclosed"
-- a genuinely different signal from peer_anchor's bcorp/upright peer medians.

Confirmed 2026-08-18: this table had ZERO references in formula_estimator.py
or graph.py -- it was built and validated (E-pillar Spearman +0.44 to +0.47
on real holdout, see research/SCORING_PLAN.md) but never wired into anything
that runs. exio_e_vote is that wiring, used by evidence_ladder.py's
exio_structural rung.

MATCHING: no direct company-industry -> EXIO-sector join exists (the only
existing join, market_exio_sector, keys off our OWN markets table, not off a
free-text company/upright industry string). Reuses sector_matcher.py's
TF-IDF fuzzy matcher (same tool peer_anchor.py already uses to match a
company's industry string against upright's 30 labels) against
exio_sector_intensity's 150 sector labels directly -- same confidence-floor,
explicit-no-match discipline, no new matching logic invented.

E-PILLAR POLARITY: exio's *_pct columns are precomputed percentiles of "how
intense", not "how good" -- co2e_pct=100 is the WORST (most emissions-
intensive) sector, so it must be INVERTED (100 - pct) before use as a
peer_anchor-style "higher percentile is better" vote.

S-PILLAR (added 2026-08-27): employees_per_meur, lowskill_share, female_share
have NO precomputed _pct column -- percentiles are computed here, once, over
the live 150-sector distribution (same discipline as the E columns' existing
_pct values, just derived at load time instead of pre-stored). Combines all
three into one S vote (equal-weighted mean of per-column percentiles):
  - female_share: higher = better (gender diversity is an established ESG-S
    signal) -- used as-is, no inversion.
  - employees_per_meur, lowskill_share: NO established good/bad polarity on
    their own (labor intensity and low-skill-labor share are workforce
    COMPOSITION facts, not violations) -- included per explicit instruction,
    NOT inverted (higher labor intensity / higher low-skill share both used
    as "higher percentile" as-is). UNVALIDATED: unlike co2e_pct's measured
    +0.615 E rho, none of the three S columns have a backtested Spearman
    against Upright S truth yet -- follow-up: run the equivalent of
    calibration/shoot_out_rungs.py for this S vote before trusting its
    weight/confidence the way exio_structural's E vote is trusted.

G-PILLAR (added 2026-08-27): EXIOBASE has literally zero governance columns
(confirmed via information_schema query) -- there is no physical-flow proxy
for board composition, audit quality, exec pay, etc. Per explicit direction,
G is NOT a fourth independent EXIOBASE signal -- it is a structural ESTIMATE
derived from the SAME matched sector's own E and S percentiles (mean of the
two), on the reasoning that a sector's overall structural
intensity/composition profile is the only industry-level signal EXIOBASE can
offer G at all. This is explicitly weaker/more speculative than E or S
(confidence set lower) -- it is a proxy of a proxy, not a measured column.
"""

from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("exio_lookup")

# Cached once per process -- 150 rows, small, fixed, read-only.
_exio_cache: Optional[list[dict]] = None
_exio_labels_cache: Optional[list[str]] = None


@dataclass
class ExioVote:
    percentile: float    # 0-100, higher = better (already inverted where needed)
    confidence: float
    sector_matched: str
    similarity: float
    tier: str             # 'exio_structural'
    basis: str


def _percentile_rank(values: list[float], v: float) -> float:
    """% of values <= v, 0-100. Ties counted as 'at or below' (inclusive) --
    same convention as SQL PERCENT_RANK's simpler cousin; fine for a 150-row,
    read-once, no-tiebreak-sensitive use like this."""
    if not values:
        return 50.0
    n_le = sum(1 for x in values if x <= v)
    return 100.0 * n_le / len(values)


def _load_exio_rows() -> list[dict]:
    global _exio_cache, _exio_labels_cache
    if _exio_cache is None:
        from agentic_estimation.layer_1.peer_anchor_collector import _db_conn
        conn = _db_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT sector, co2e_pct, employees_per_meur, lowskill_share, female_share "
                "FROM exio_sector_intensity WHERE co2e_pct IS NOT NULL"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        emp_vals = [r[2] for r in rows if r[2] is not None]
        lowskill_vals = [r[3] for r in rows if r[3] is not None]
        female_vals = [r[4] for r in rows if r[4] is not None]

        table = []
        for sector, co2e_pct, emp, lowskill, female in rows:
            table.append({
                "sector": sector,
                "co2e_pct": co2e_pct,
                "employees_pct": _percentile_rank(emp_vals, emp) if emp is not None else None,
                "lowskill_pct": _percentile_rank(lowskill_vals, lowskill) if lowskill is not None else None,
                "female_pct": _percentile_rank(female_vals, female) if female is not None else None,
            })
        _exio_cache = table
        _exio_labels_cache = [r["sector"] for r in table]
        log.info("loaded %d exio_sector_intensity rows with usable co2e_pct", len(table))
    return _exio_cache


def _match_sector(sector: Optional[str]) -> Optional[tuple]:
    """Shared fuzzy-match step for all three pillar votes. Returns
    (row_dict, SectorMatch) or None on no usable match."""
    if not sector:
        return None
    from agentic_estimation.layer_1.sector_matcher import best_sector_match

    rows = _load_exio_rows()
    if not rows:
        return None
    match = best_sector_match(sector, _exio_labels_cache)
    if not match or not match.matched:
        return None
    row = next(r for r in rows if r["sector"] == match.label)
    return row, match


def exio_e_vote(sector: Optional[str]) -> Optional[ExioVote]:
    """Fuzzy-match `sector` against exio_sector_intensity's 150 sector
    labels; returns an E-pillar vote from the matched sector's co2e_pct
    (inverted to higher-is-better), or None on no usable match (empty
    sector, no token overlap, or below the fuzzy matcher's confidence
    floor -- caller falls through to the next rung)."""
    hit = _match_sector(sector)
    if hit is None:
        return None
    row, match = hit

    percentile = 100.0 - row["co2e_pct"]  # invert: lower structural emissions intensity = higher (better) percentile

    return ExioVote(
        percentile=percentile, confidence=0.35, sector_matched=match.label,
        similarity=match.similarity, tier="exio_structural",
        basis=(f"exio sector {match.label!r} (fuzzy-matched from {sector!r}, sim={match.similarity:.2f}): "
               f"co2e_pct={row['co2e_pct']:.1f} -> inverted percentile {percentile:.1f}"),
    )


def exio_s_vote(sector: Optional[str]) -> Optional[ExioVote]:
    """Fuzzy-match `sector`, then combine employees_per_meur / lowskill_share
    / female_share (equal-weighted mean of each column's own percentile rank
    within the 150-sector distribution) into one S-pillar vote. UNVALIDATED
    against real S truth -- see module docstring. Lower confidence than
    exio_e_vote's measured 0.35 for that reason."""
    hit = _match_sector(sector)
    if hit is None:
        return None
    row, match = hit

    parts = [(name, row[key]) for name, key in
             (("employees_pct", "employees_pct"), ("lowskill_pct", "lowskill_pct"), ("female_pct", "female_pct"))
             if row[key] is not None]
    if not parts:
        return None
    percentile = sum(v for _, v in parts) / len(parts)

    return ExioVote(
        percentile=percentile, confidence=0.20, sector_matched=match.label,
        similarity=match.similarity, tier="exio_structural",
        basis=(f"exio sector {match.label!r} (fuzzy-matched from {sector!r}, sim={match.similarity:.2f}): "
               f"mean of {', '.join(f'{n}={v:.1f}' for n, v in parts)} -> percentile {percentile:.1f} "
               f"(unvalidated -- no backtested rho against S truth yet)"),
    )


def exio_g_vote(sector: Optional[str]) -> Optional[ExioVote]:
    """No EXIOBASE governance columns exist at all -- this is a structural
    ESTIMATE, the mean of the SAME matched sector's own E and S votes, not a
    fourth independent measured signal. Weaker than either E or S alone
    (lowest confidence of the three) -- see module docstring."""
    e = exio_e_vote(sector)
    s = exio_s_vote(sector)
    if e is None and s is None:
        return None
    parts = [v.percentile for v in (e, s) if v is not None]
    percentile = sum(parts) / len(parts)
    matched_label = (e or s).sector_matched
    similarity = (e or s).similarity

    return ExioVote(
        percentile=percentile, confidence=0.12, sector_matched=matched_label,
        similarity=similarity, tier="exio_structural",
        basis=(f"exio sector {matched_label!r}: no governance columns exist -- "
               f"G estimated as mean of this sector's own E/S exio percentiles "
               f"({', '.join(f'{v.percentile:.1f}' for v in (e, s) if v is not None)}) -> {percentile:.1f}"),
    )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Dairy processing"
    print("E:", exio_e_vote(q))
    print("S:", exio_s_vote(q))
    print("G:", exio_g_vote(q))
