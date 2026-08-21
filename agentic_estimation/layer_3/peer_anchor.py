"""
peer_anchor.py — real peer-company statistics as a formula input, for
companies with thin/no extracted evidence on a pillar.

Wires agentic_estimation/layer_1/peer_anchor_collector.py's find_peers() (built
in Phase 1, never previously consumed by the formula) into formula_estimator.py
as a genuine fourth signal alongside country baseline + extracted claims.

WHY THIS EXISTS: measured live in Phase 2 calibration -- when a pillar has zero
extracted claims, the formula falls back to the country baseline alone, and
every company sharing that country gets the IDENTICAL predicted score
regardless of how differently they actually perform (confirmed: up to ~47% of
a bcorp sample tied at one value, with real B-Corp scores for those same
companies ranging e.g. 3.0-36.9). A real peer-company median is a genuine
statistic about comparable companies, not a guess -- it differentiates
thin-evidence companies from each other instead of collapsing them together.

SCALE PROBLEM (measured live): bcorp_lookup.impact_area_environment ranges
0-78.2 (avg ~16.8), impact_area_governance ranges 4-25 (avg ~16.5) -- neither
is 0-100, and they're on DIFFERENT scales from each other and from this
formula's 0-100 output. A raw peer median cannot be added directly. Every peer
median is converted to its PERCENTILE within the full bcorp_lookup column
distribution first (a real, computed rank, not a fabricated absolute value) --
that percentile (naturally 0-100) is what enters the formula.

GROUND-TRUTH LEAKAGE GUARD: exclude_name is passed on every find_peers() call.
Backtest companies scored against bcorp_lookup ARE bcorp_lookup rows -- without
this, a company would see its own real score in its own peer group.

UPRIGHT FALLBACK TIER (added after code-review-graph analysis of a separate
ESG product's embedding-based sector matching): find_peers() unions bcorp +
upright, but matches `sector` with SQL `=` against BOTH bcorp_lookup.sasb_sector
(4 coarse values) AND upright_lookup.industry (30 real fine-grained values,
e.g. "Automotive", "Food and Beverage") using the SAME literal string --
confirmed live that a real sector string almost never exactly matches
upright's vocabulary, so its 10,086-company peer pool was effectively
unreachable. sector_matcher.py's fuzzy TF-IDF match (no embeddings --
sentence-transformers/torch aren't installed in this environment) finds the
closest upright industry label when one exists.

Upright has no e_score/s_score/g_score columns, but upright_pillar_proxy.py
derives real E/S percentile proxies from Upright's raw impact sub-components
(e1-e5 for E, h1-h5+s1-s5 for S -- see that module's docstring for the full
percentile-averaging method). For E and S, this tier uses the peer group's
mean pillar proxy -- a genuine per-pillar signal, not a shared one. G has NO
upright analog at all (no governance-related raw columns exist), so the G
vote still falls back to the peer group's mean net_impact_ratio_percentile
(the single overall-impact score) as a SHARED, lower-confidence proxy --
exactly the original design, unchanged for G only.

CROSSWALK TIER (highest-confidence sector match, tried FIRST): sector_crosswalk.py
hand-maps all 30 upright industry labels onto bcorp's 22 industry_category
values -- built once, offline, by reading both label sets side by side, NOT
derived from string similarity. When `sector` matches either vocabulary
exactly, this tier pools BOTH bcorp peers (industry_category = the canonical
category) AND upright peers (industry IN every upright label mapped to that
category) as one combined peer group -- e.g. querying "Automotive" pulls
bcorp's "Manufactured Goods" peers AND upright's "Automotive" peers together,
since the crosswalk says they're the same real-world sector. This is exact-
match, not fuzzy, so it out-ranks the bcorp-fuzzy and upright-fuzzy tiers
below; those remain as fallback for sector strings the crosswalk doesn't
recognize (e.g. a company's own free-text industry field that matches
neither vocabulary's labels).
"""

import bisect
from dataclasses import dataclass
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("peer_anchor")

_MIN_PEERS_SECTOR_COUNTRY = 5   # tier 1 confidence floor
_MIN_PEERS_SECTOR_ONLY = 8      # tier 2 confidence floor (looser matching needs more support)
_MIN_PEERS_UPRIGHT_FUZZY = 8    # tier 3 (upright, fuzzy-matched sector) confidence floor
_MIN_PEERS_CROSSWALK = 5        # crosswalk tier (exact match, combined bcorp+upright pool)

_BCORP_PILLAR_FIELD = {"E": "e_score", "S": "s_score", "G": "g_score"}

# Cached once per process: sorted list of every real bcorp_lookup value for
# each pillar field, used to compute a peer median's percentile rank.
_distribution_cache: dict[str, list[float]] = {}

# upright_lookup's 30 distinct industry labels, cached once per process --
# small, fixed vocabulary, cheap to keep in memory for repeated fuzzy matches.
_upright_labels_cache: Optional[list[str]] = None


def _upright_industry_labels() -> list[str]:
    global _upright_labels_cache
    if _upright_labels_cache is None:
        from agentic_estimation.layer_1.peer_anchor_collector import _db_conn
        conn = _db_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT industry FROM upright_lookup WHERE industry IS NOT NULL")
            _upright_labels_cache = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()
        log.info("loaded %d distinct upright industry labels", len(_upright_labels_cache))
    return _upright_labels_cache


@dataclass
class PeerAnchorVote:
    pillar: str
    percentile: Optional[float]   # 0-100, or None if abstained
    confidence: float             # 0.0 when abstained
    n_peers: int
    tier: str                     # 'sector_country' | 'sector_only' | 'abstain'
    basis: str                    # audit line


# ── Tier suppression ─────────────────────────────────────────────────────────
# Two tiers were measured to be INVERTED -- their percentile anti-correlates
# with real ground truth, so using them actively pushes the score the wrong
# way. Measured on 669 unique companies pooled across every calibration dump,
# split into two independent halves by company-name hash:
#
#     tier               pillar   split A         split B
#     sector_only        E        -0.448 (n=77)   -0.204 (n=39)
#     crosswalk_global   E        -0.570 (n=62)   -0.153 (n=18)
#     crosswalk_global   S        -0.227 (n=62)   -0.352 (n=18)
#
# LIKELY CAUSE: both drop the country dimension, so they re-assert a global
# sector median against a formula that already carries a country baseline --
# double-counting sector while ignoring geography. The '..._country' variants
# of the same tiers are strongly POSITIVE (sector_country: +0.488/+0.405 on E),
# which is what points at the missing country conditioning rather than at the
# sector matching itself.
#
# Effect of suppressing them, same two splits, anchor-only Spearman:
#     E  +0.032/+0.069  ->  +0.439/+0.405   (coverage 78% -> 47%)
#     S  +0.108/+0.115  ->  +0.259/+0.312
#
# G IS DELIBERATELY EXCLUDED. G has no inverted tier -- all-tiers scores
# +0.267/+0.297 there and filtering only costs coverage, so G keeps every tier.
#
# NOT suppressed, despite also being global: bcorp_fuzzy_global measured
# STABLE POSITIVE (E +0.601/+0.550, G +0.273/+0.536). "Global" is not itself
# the problem, so the suppression list names the two specific tiers rather
# than pattern-matching on the name.
_INVERTED_TIERS: dict[str, frozenset] = {
    "E": frozenset({"sector_only", "crosswalk_global"}),
    "S": frozenset({"sector_only", "crosswalk_global"}),
    "G": frozenset(),
}


def _suppressed(pillar: str, tier: str) -> bool:
    """True if this tier's vote is known to anti-correlate for this pillar."""
    return tier in _INVERTED_TIERS.get(pillar, frozenset())


def _abstain(pillar: str, tier: str, n_peers: int, why: str) -> PeerAnchorVote:
    """An abstain that RECORDS which tier would have fired, so the audit trail
    shows a deliberate suppression rather than a missing peer group."""
    return PeerAnchorVote(
        pillar=pillar, percentile=None, confidence=0.0, n_peers=n_peers,
        tier="abstain", basis=f"suppressed {tier} vote: {why}",
    )


def _load_distribution(field: str) -> list[float]:
    if field not in _distribution_cache:
        from agentic_estimation.layer_1.peer_anchor_collector import _db_conn
        col = {"e_score": "impact_area_environment", "s_score": None, "g_score": "impact_area_governance"}[field]
        conn = _db_conn()
        try:
            cur = conn.cursor()
            if field == "s_score":
                # s_score isn't a single bcorp column -- it's the collector's
                # own mean(workers, community, customers); recompute the same
                # way here so the distribution matches what peer_median(s_score)
                # is actually measuring.
                cur.execute(
                    "SELECT impact_area_workers, impact_area_community, impact_area_customers "
                    "FROM bcorp_lookup WHERE impact_area_workers IS NOT NULL "
                    "OR impact_area_community IS NOT NULL OR impact_area_customers IS NOT NULL"
                )
                vals = []
                for w, c, cu in cur.fetchall():
                    parts = [v for v in (w, c, cu) if v is not None]
                    if parts:
                        vals.append(sum(parts) / len(parts))
            else:
                cur.execute(f"SELECT {col} FROM bcorp_lookup WHERE {col} IS NOT NULL")
                vals = [float(r[0]) for r in cur.fetchall()]
        finally:
            conn.close()
        _distribution_cache[field] = sorted(vals)
        log.info("loaded bcorp %s distribution: %d real values", field, len(vals))
    return _distribution_cache[field]


def _percentile_rank(value: float, sorted_dist: list[float]) -> float:
    """Midrank percentile: (count_below + 0.5*count_equal) / N * 100."""
    n = len(sorted_dist)
    if n == 0:
        return 50.0
    lo = bisect.bisect_left(sorted_dist, value)
    hi = bisect.bisect_right(sorted_dist, value)
    return (lo + 0.5 * (hi - lo)) / n * 100.0


def _candidate_sectors(sector: Optional[str]) -> list[str]:
    """Sector strings to try, in order, against bcorp_lookup's own coarse
    vocabulary (apparel_retail/general/manufacturing/services -- confirmed
    live to be ONLY these 4 values). Real metadata/industry strings ("Beverages",
    "Semiconductors", "Insurance") never match that vocabulary directly --
    found live: "Beverages" against bcorp returns 0 peers even though 0.8-confidence
    real peers exist once coarsened via classify_manufacturing_vs_services()
    (already built in company_metadata.py, reused here rather than
    reimplemented). Raw string tried first (matches on the rare case metadata
    already used bcorp's own vocabulary); coarse bucket tried second."""
    if not sector:
        return []
    from agentic_estimation.layer_1.company_metadata import classify_manufacturing_vs_services

    candidates = [sector]
    coarse = classify_manufacturing_vs_services(sector)
    if coarse["classification"] in ("manufacturing", "services") and coarse["classification"] != sector:
        candidates.append(coarse["classification"])
    return candidates


def peer_anchor_vote(pillar: str, company_name: str, sector: Optional[str],
                      country: Optional[str], truth_source: Optional[str] = None) -> PeerAnchorVote:
    """
    Real bcorp peer median for one pillar -> percentile-normalised 0-100 vote.
    Fallback chain mirrors ratio_estimator.py's discipline: sector+country ->
    sector-only -> abstain. No country-only tier (would just re-encode the
    country baseline the formula already has -- double counting). Each tier
    tries the raw sector string, then its coarse manufacturing/services bucket
    (see _candidate_sectors) -- first candidate clearing the sample floor wins.

    truth_source: 'bcorp' | 'upright' | None (default). When given, restricts
    which peer pool is used, to match the truth the caller is about to score
    against. Found live (2026-08-17): tiers 1-4 (sector+country, sector-only,
    crosswalk, bcorp-fuzzy) are ALL bcorp-sourced; only tier 5 is upright-
    sourced. Scoring UPRIGHT truth against a bcorp-sourced peer vote is a real
    cross-source contamination bug -- bcorp and upright disagree at Spearman
    -0.538 on industry ordering (see esg-truth-source-decision memory), so a
    bcorp peer's percentile is not a neutral estimate of an upright-scored
    company's standing. Measured: filtering to upright-only peers when
    scoring upright truth took E from rho=-0.239 to +0.001..+0.175 depending
    on remaining weight (see calibration/dump_frozen150_noLLM.json backtest).
    None (default) preserves today's exact behaviour -- no filtering, for
    backward compatibility with existing callers that don't know their truth
    source (e.g. live production scoring of a company with no ground truth
    at all, where BOTH pools are legitimately the best available estimate).
    """
    from agentic_estimation.layer_1.peer_anchor_collector import find_peers, peer_median, peer_sample_size

    field = _BCORP_PILLAR_FIELD[pillar]
    candidates = _candidate_sectors(sector)

    # Tiers 1-4 below are bcorp-sourced. Skip them entirely when scoring
    # against upright truth -- see truth_source docstring above.
    skip_bcorp_tiers = truth_source == "upright"

    for cand in candidates:
        if skip_bcorp_tiers:
            break
        if country:
            peers = find_peers(sector=cand, country=country, exclude_name=company_name)
            n = peer_sample_size(peers, field)
            if n >= _MIN_PEERS_SECTOR_COUNTRY:
                med = peer_median(peers, field)
                pctile = _percentile_rank(med, _load_distribution(field))
                return PeerAnchorVote(
                    pillar=pillar, percentile=pctile, confidence=0.5, n_peers=n, tier="sector_country",
                    basis=f"median {field}={med:.1f} of {n} bcorp peers ({cand}/{country}) -> pctile {pctile:.1f}",
                )

    if not skip_bcorp_tiers and not _suppressed(pillar, "sector_only"):
        for cand in candidates:
            peers = find_peers(sector=cand, country=None, exclude_name=company_name)
            n = peer_sample_size(peers, field)
            if n >= _MIN_PEERS_SECTOR_ONLY:
                med = peer_median(peers, field)
                pctile = _percentile_rank(med, _load_distribution(field))
                return PeerAnchorVote(
                    pillar=pillar, percentile=pctile, confidence=0.3, n_peers=n, tier="sector_only",
                    basis=f"median {field}={med:.1f} of {n} bcorp peers ({cand}, any country) -> pctile {pctile:.1f}",
                )

    # Tier 3 (crosswalk): sector is a RECOGNIZED label in either vocabulary
    # (bcorp's 22 industry_category values or upright's 30 industry values) --
    # exact, hand-built mapping, no similarity scoring, tried BEFORE any fuzzy
    # tier since it is the highest-confidence sector match available short of
    # an exact hit on the caller's own vocabulary (handled above). Bcorp-
    # sourced (see module docstring) -- skipped for upright truth.
    if not skip_bcorp_tiers:
        crosswalk_vote = _crosswalk_vote(pillar, sector, country, company_name, field)
        if crosswalk_vote is not None:
            return crosswalk_vote

    # Tier 4: bcorp, fuzzy-matched against industry_category (22 real values --
    # finer than sasb_sector's 4 coarse buckets, e.g. "Manufactured Goods",
    # "Energy", "Agriculture, forestry & fishing"). Still real bcorp per-pillar
    # E/S/G data (not a shared total-impact proxy like the upright tier below),
    # so this is tried BEFORE falling through to upright.
    if not skip_bcorp_tiers:
        bcorp_fuzzy_vote = _bcorp_category_fuzzy_vote(pillar, sector, country, company_name, field)
        if bcorp_fuzzy_vote is not None:
            return bcorp_fuzzy_vote

    # Tier 5: upright, fuzzy-matched sector. Last resort -- upright's
    # total-impact percentile is a weaker, shared-across-pillars signal (see
    # module docstring), lower confidence than any bcorp tier above. Skipped
    # when scoring bcorp truth, for the same source-matching reason.
    if truth_source != "bcorp":
        upright_vote = _upright_fuzzy_vote(pillar, sector, country, company_name)
        if upright_vote is not None:
            return upright_vote

    return PeerAnchorVote(pillar=pillar, percentile=None, confidence=0.0, n_peers=0, tier="abstain",
                           basis="no sector+country, sector-only, crosswalk, bcorp-fuzzy, or upright-fuzzy peer group met the sample-size floor")


def _crosswalk_vote(pillar: str, sector: Optional[str], country: Optional[str],
                     company_name: str, field: str) -> Optional[PeerAnchorVote]:
    """Exact-match sector tier via sector_crosswalk.py: `sector` is checked
    against BOTH bcorp's 22 industry_category values and upright's 30
    industry values; if it matches either, the crosswalk's canonical bcorp
    category identifies the bcorp peer group (industry_category = category).

    NOTE: this tier pools bcorp peers ONLY for the actual per-pillar median --
    upright_lookup carries no e_score/s_score/g_score (only a single
    net_impact_ratio_percentile, see module docstring), so an upright peer
    can never contribute a value for `field` and averaging it in would be
    combining incompatible scales. The crosswalk's real payoff for upright
    happens in _upright_fuzzy_vote below: upright_industries_for_bcorp_category
    lets that tier accept an EXACT crosswalk-mapped label with no fuzzy-match
    step, still reported under its own (lower, proxy-only) confidence.
    Returns None (not an abstain vote) if `sector` isn't a recognized label
    in either vocabulary -- caller falls through to the fuzzy tiers."""
    if not sector:
        return None
    from agentic_estimation.layer_1.peer_anchor_collector import find_peers, peer_median, peer_sample_size
    from agentic_estimation.layer_1.sector_crosswalk import crosswalk_sector

    canonical = crosswalk_sector(sector)
    if not canonical:
        return None

    attempts = [(None, "crosswalk_global", 0.4)]
    if country:
        attempts.insert(0, (country, "crosswalk_country", 0.55))
    # Drop the country-less attempt where it was measured to invert, so the
    # caller falls through to the fuzzy tiers (which are positive) instead of
    # returning a vote that points the wrong way.
    attempts = [a for a in attempts if not _suppressed(pillar, a[1])]

    for ctry, tier, base_conf in attempts:
        peers = find_peers(sector=canonical, country=ctry, exclude_name=company_name,
                            bcorp_sector_column="industry_category", include_upright=False)
        n = peer_sample_size(peers, field)
        if n >= _MIN_PEERS_CROSSWALK:
            med = peer_median(peers, field)
            pctile = _percentile_rank(med, _load_distribution(field))
            return PeerAnchorVote(
                pillar=pillar, percentile=pctile, confidence=base_conf, n_peers=n, tier=tier,
                basis=(f"median {field}={med:.1f} of {n} bcorp peers "
                       f"(crosswalked {sector!r}->{canonical!r}{'/' + ctry if ctry else ''}) -> pctile {pctile:.1f}"),
            )
    return None


def _bcorp_category_fuzzy_vote(pillar: str, sector: Optional[str], country: Optional[str],
                                company_name: str, field: str) -> Optional[PeerAnchorVote]:
    """Fuzzy-match `sector` against bcorp_lookup's 22 real industry_category
    values (see sector_matcher.py, peer_anchor_collector.bcorp_industry_category_labels).
    Unlike the upright fallback, this still yields real per-pillar bcorp
    E/S/G data -- a genuinely better-resolution version of the existing
    coarse sasb_sector tier, not a different data source. Returns None (not
    an abstain vote) on no fuzzy match or insufficient peers."""
    if not sector:
        return None
    from agentic_estimation.layer_1.peer_anchor_collector import (
        find_peers, peer_median, peer_sample_size, bcorp_industry_category_labels,
    )
    from agentic_estimation.layer_1.sector_matcher import best_sector_match

    match = best_sector_match(sector, bcorp_industry_category_labels())
    if not match or not match.matched:
        return None

    peers = find_peers(sector=match.label, country=country, exclude_name=company_name,
                        bcorp_sector_column="industry_category", include_upright=False) if country else []
    n = peer_sample_size(peers, field)
    tier = "bcorp_fuzzy_country"
    if n < _MIN_PEERS_SECTOR_COUNTRY:
        peers = find_peers(sector=match.label, country=None, exclude_name=company_name,
                            bcorp_sector_column="industry_category", include_upright=False)
        n = peer_sample_size(peers, field)
        tier = "bcorp_fuzzy_global"
    if n < _MIN_PEERS_SECTOR_ONLY:
        return None

    med = peer_median(peers, field)
    pctile = _percentile_rank(med, _load_distribution(field))
    conf = 0.45 if tier == "bcorp_fuzzy_country" else 0.35  # below the exact-match tiers (0.5/0.3), above upright-fuzzy (0.25)
    return PeerAnchorVote(
        pillar=pillar, percentile=pctile, confidence=conf, n_peers=n, tier=tier,
        basis=(f"median {field}={med:.1f} of {n} bcorp peers "
               f"(fuzzy-matched {sector!r}->{match.label!r} via industry_category, sim={match.similarity:.2f}) "
               f"-> pctile {pctile:.1f}"),
    )


def _upright_fuzzy_vote(pillar: str, sector: Optional[str], country: Optional[str],
                         company_name: str) -> Optional[PeerAnchorVote]:
    """Fuzzy-match `sector` against upright_lookup's 30 real industry labels
    (see sector_matcher.py) and, on a confident match, vote using that peer
    group's real data:
      - E/S: mean of upright_pillar_proxy.py's per-pillar percentile proxy
        across the peer group -- a genuine pillar-specific signal derived
        from Upright's raw impact sub-components (see that module).
      - G: median net_impact_ratio_percentile (already 0-100), the single
        overall-impact score, used as a SHARED vote -- unchanged from the
        original design, since no upright column set maps to governance.
    Returns None (not an abstain vote) on no fuzzy match or insufficient
    peers, so the caller's existing abstain path/message is used -- this
    function only ever adds a vote, never manufactures its own abstain
    reasoning."""
    if not sector:
        return None
    from agentic_estimation.layer_1.peer_anchor_collector import find_peers, peer_median, peer_sample_size
    from agentic_estimation.layer_1.sector_matcher import best_sector_match

    match = best_sector_match(sector, _upright_industry_labels())
    if not match or not match.matched:
        return None

    field = "net_impact_ratio_percentile"
    peers = find_peers(sector=match.label, country=country, exclude_name=company_name) if country else []
    n = peer_sample_size(peers, field)
    tier = "upright_fuzzy_country"
    if n < _MIN_PEERS_UPRIGHT_FUZZY:
        peers = find_peers(sector=match.label, country=None, exclude_name=company_name)
        n = peer_sample_size(peers, field)
        tier = "upright_fuzzy_global"
    if n < _MIN_PEERS_UPRIGHT_FUZZY:
        return None

    if pillar in ("E", "S"):
        from agentic_estimation.layer_1.upright_pillar_proxy import peer_group_pillar_proxy
        peer_names = [p.name for p in peers if p.source == "upright"]
        proxy_pctile = peer_group_pillar_proxy(peer_names, pillar)
        if proxy_pctile is None:
            return None  # matched peers exist but none carry usable E/S sub-components
        return PeerAnchorVote(
            pillar=pillar, percentile=proxy_pctile, confidence=0.3, n_peers=n, tier=tier,
            basis=(f"mean upright_{pillar.lower()}_proxy={proxy_pctile:.1f} of {n} upright peers "
                   f"(fuzzy-matched {sector!r}->{match.label!r}, sim={match.similarity:.2f}) "
                   f"-- derived per-pillar proxy from Upright's raw impact sub-components"),
        )

    pctile = peer_median(peers, field)  # already a 0-100 percentile, no rank conversion needed
    return PeerAnchorVote(
        pillar=pillar, percentile=pctile, confidence=0.25, n_peers=n, tier=tier,
        basis=(f"median net_impact_ratio_percentile={pctile:.1f} of {n} upright peers "
               f"(fuzzy-matched {sector!r}->{match.label!r}, sim={match.similarity:.2f}) "
               f"-- shared across pillars, no bcorp per-pillar data available (G has no upright proxy)"),
    )
