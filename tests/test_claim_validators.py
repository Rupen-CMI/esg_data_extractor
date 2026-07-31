"""
Tier-0 deterministic claim validators (agentic_estimation/layer_2/claim_validators.py).
Ported from this session's scratchpad verification script -- 14 checks,
covering all 5 rules: lexical relevance, numeric bounds, polarity
consistency, corroboration, known-failure shape.
"""
from agentic_estimation.shared.claim_types import ExtractedClaim
from agentic_estimation.layer_2.claim_validators import validate_claims


def test_yoga_page_claim_dropped():
    """Real source_tag, but cited text is topically irrelevant -> dropped."""
    claims = [ExtractedClaim(
        factor="labor_controversy", pillar="S", polarity=-1, strength=0.8, confidence=0.8,
        value=None, source_tag="controversies",
        reasoning="worker dispute reported", method="extracted",
    )]
    signals = {"controversies": "Come visit Lilikoi, Kauai's best new Restaurant and Bar serving Hawaiian cuisine."}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 0
    assert any(f.rule == "lexical_relevance" for f in flags)


def test_dataset_lookup_exempt_from_lexical_check():
    """dataset_lookup claims with synthetic source_tags are exempt -- kept
    even with no signal text (they were never LLM-read from gathered text)."""
    claims = [ExtractedClaim(
        factor="scope_1_emissions", pillar="E", polarity=0, strength=1.0, confidence=0.85,
        value=1000.0, source_tag="climate_trace_owner_match",
        reasoning="CT owner match", method="dataset_lookup",
    )]
    kept, flags = validate_claims(claims, signals={}, country=None)
    assert len(kept) == 1
    assert not any(f.rule == "lexical_relevance" for f in flags)


def test_negative_value_nulled():
    claims = [ExtractedClaim(
        factor="water_withdrawal", pillar="E", polarity=0, strength=1.0, confidence=0.6,
        value=-500.0, source_tag="sustainability_report",
        reasoning="water usage", method="extracted",
    )]
    signals = {"sustainability_report": "Our water withdrawal and stewardship program reduced usage."}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    assert kept[0].value is None
    assert any(f.rule == "numeric_bounds" for f in flags)


def test_pct_over_100_nulled():
    claims = [ExtractedClaim(
        factor="female_board_pct", pillar="S", polarity=0, strength=1.0, confidence=0.6,
        value=250.0, source_tag="gov_board",
        reasoning="board gender diversity women directors", method="extracted",
    )]
    signals = {"gov_board": "Board gender diversity: women directors make up a majority."}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    assert kept[0].value is None


def test_polarity_flip_confidence_halved(monkeypatch):
    """Inherently-negative factor claimed with polarity=+1 is suspicious --
    confidence capped, not dropped (it might be a legitimate edge case).
    Cluster-severity assignment (a separate, later step that also rescales
    confidence) is disabled so this asserts the polarity rule in isolation."""
    from agentic_estimation.layer_2 import evidence_clusters
    monkeypatch.setattr(evidence_clusters, "apply_cluster_severity", lambda claims: [])
    claims = [ExtractedClaim(
        factor="environmental_controversy", pillar="E", polarity=1, strength=0.5, confidence=0.8,
        value=None, source_tag="net_zero",
        reasoning="environmental violation pollution controversy", method="extracted",
    )]
    signals = {"net_zero": "The company faced an environmental violation and pollution controversy."}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    assert abs(kept[0].confidence - 0.4) < 1e-9
    assert any(f.rule == "polarity_consistency" for f in flags)


# Signal text kept >=200 chars in the corroboration tests below so rule (e)
# known-failure-shape does NOT also fire -- isolating rule (d) specifically.
_LONG_FINE_TEXT = ("The company received a fine and penalty in an SEC settlement enforcement "
                   "action following a multi-year regulatory investigation into its business "
                   "practices, disclosures, and internal financial controls across several "
                   "operating divisions and subsidiaries worldwide. ")
assert len(_LONG_FINE_TEXT) >= 200


def test_single_source_high_weight_negative_capped(monkeypatch):
    """Cluster severity disabled -- isolates the corroboration rule (a
    separate, later step that also rescales confidence once this claim's text
    matches a labeled cluster; see test_cluster_severity_* below)."""
    from agentic_estimation.layer_2 import evidence_clusters
    monkeypatch.setattr(evidence_clusters, "apply_cluster_severity", lambda claims: [])
    claims = [ExtractedClaim(
        factor="regulatory_fines", pillar="G", polarity=-1, strength=0.9, confidence=0.8,
        value=None, source_tag="controversies",
        reasoning="SEC fine penalty settlement enforcement", method="extracted",
    )]
    signals = {"controversies": _LONG_FINE_TEXT}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    assert abs(kept[0].confidence - 0.56) < 1e-9
    assert any(f.rule == "corroboration" for f in flags)


def test_two_source_negative_not_capped(monkeypatch):
    from agentic_estimation.layer_2 import evidence_clusters
    monkeypatch.setattr(evidence_clusters, "apply_cluster_severity", lambda claims: [])
    claims = [
        ExtractedClaim(factor="regulatory_fines", pillar="G", polarity=-1, strength=0.9, confidence=0.8,
                       value=None, source_tag="controversies", reasoning="SEC fine penalty settlement", method="extracted"),
        ExtractedClaim(factor="regulatory_fines", pillar="G", polarity=-1, strength=0.9, confidence=0.8,
                       value=None, source_tag="gov_fines", reasoning="regulator enforcement fine", method="extracted"),
    ]
    signals = {
        "controversies": _LONG_FINE_TEXT,
        "gov_fines": _LONG_FINE_TEXT.replace("SEC settlement", "regulator enforcement"),
    }
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert all(abs(c.confidence - 0.8) < 1e-9 for c in kept)


def test_duplicate_claims_same_source_all_capped(monkeypatch):
    """Regression: a single extraction call can return the same underlying
    fact as several near-duplicate claim objects (LLM rewording), all citing
    the SAME source_tag. Counting claim objects instead of distinct sources
    let this slip through uncapped in production -- live Cardinal Health run
    (2026-07-27): one google_news_rss article extracted 7x as
    labor_controversy, all sharing one source_tag, none capped, because
    len(negative)=7 short-circuited past the single-source check entirely.
    Real corroboration requires >=2 DISTINCT sources, not >=2 claim objects.
    Cluster severity disabled -- isolates the corroboration rule."""
    from agentic_estimation.layer_2 import evidence_clusters
    monkeypatch.setattr(evidence_clusters, "apply_cluster_severity", lambda claims: [])
    claims = [
        ExtractedClaim(factor="regulatory_fines", pillar="G", polarity=-1, strength=0.9, confidence=0.8,
                       value=None, source_tag="controversies", reasoning="SEC fine penalty settlement enforcement", method="extracted"),
        ExtractedClaim(factor="regulatory_fines", pillar="G", polarity=-1, strength=0.6, confidence=0.8,
                       value=None, source_tag="controversies", reasoning="regulator fine enforcement penalty", method="extracted"),
        ExtractedClaim(factor="regulatory_fines", pillar="G", polarity=-1, strength=0.7, confidence=0.8,
                       value=None, source_tag="controversies", reasoning="SEC settlement fine action", method="extracted"),
    ]
    signals = {"controversies": _LONG_FINE_TEXT}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 3
    assert all(abs(c.confidence - 0.56) < 1e-9 for c in kept), \
        "all 3 duplicate-source claims must be capped, not just the first"
    assert sum(1 for f in flags if f.rule == "corroboration") == 3


def test_known_failure_shape_short_text_high_confidence_capped():
    claims = [ExtractedClaim(
        factor="board_independence_pct", pillar="G", polarity=0, strength=1.0, confidence=0.9,
        value=None, source_tag="wikipedia",
        reasoning="board independent director", method="extracted",
    )]
    signals = {"wikipedia": "Board independent director."}  # < 200 chars
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    assert abs(kept[0].confidence - 0.5) < 1e-9
    assert any(f.rule == "known_failure_shape" for f in flags)


def test_known_failure_shape_long_text_not_capped():
    long_text = "Board independent director analysis. " * 10  # > 200 chars
    claims = [ExtractedClaim(
        factor="board_independence_pct", pillar="G", polarity=0, strength=1.0, confidence=0.9,
        value=None, source_tag="proxy",
        reasoning="board independent director", method="extracted",
    )]
    signals = {"proxy": long_text}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    assert abs(kept[0].confidence - 0.9) < 1e-9


def test_cluster_severity_rescales_boilerplate_strength():
    """A boilerplate ESG-report claim the extractor over-credited (strength
    0.8) is pulled down to its cluster's labeled impact magnitude (0.25 --
    'publishing an ESG report is routine'), with a cluster_severity flag
    recording the adjustment. Locks the evidence_clusters wiring in
    validate_claims. Requires the committed cluster artifacts."""
    claims = [ExtractedClaim(
        factor="esg_report_published", pillar="G", polarity=1, strength=0.8, confidence=0.9,
        value=None, source_tag="esg_report",
        reasoning="The company published its annual sustainability report describing its ESG program.",
        method="extracted",
    )]
    signals = {"esg_report": "The company published its annual sustainability report describing its ESG program."}
    kept, flags = validate_claims(claims, signals=signals, country=None)
    assert len(kept) == 1
    cluster_flags = [f for f in flags if f.rule == "cluster_severity"]
    assert cluster_flags, "expected the cluster_severity step to match this boilerplate claim"
    assert kept[0].strength == 0.25
    assert kept[0].confidence <= 0.9
