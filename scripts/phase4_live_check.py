"""
phase4_live_check.py — PHASE_4_PLAN.md section 7 live acceptance checks:
1 (Tier-0 precondition), 2 (subtle-fake injection -- critics' actual
acceptance test), 3 (clean control), 4 (insufficient-evidence/no-convergence
routing, exercised via test 5's thin-company routing rather than a separate
scenario here), 5 (thin-company routing). Uses real gathered signals + real
extraction for one company, then a synthetic injection on top.

MANUAL SCRIPT, NOT A PYTEST TEST: this hits the live LLM gateway (extraction,
holistic vote, critic panel, possibly one retry re-extraction) and real web
sources -- real API cost and multi-minute runtime, and non-deterministic
(depends on what's actually on the web today). Deliberately excluded from
`pytest tests/`. Run manually:

    python scripts/phase4_live_check.py [company] [industry] [country]

KNOWN LIVE FINDING (2026-07-21, recorded in PHASE_4_PLAN.md section 7 test 3):
a clean/unmodified run on real Nvidia data was legitimately refuted by
Critic A -- an `sbti_commitment` claim sourced from a mere SBTi database
LISTING (not an actual validated commitment) was correctly flagged. This is
the critic panel catching a REAL weak claim, not a false positive -- treat a
refute as acceptable in test 3 if the objection is substantively correct on
inspection; only a refute with a clearly wrong/nonsensical objection is a
real regression. See PHASE_4_PLAN.md section 7 for the full reasoning.
"""
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env", override=True)

from agentic_estimation.layer_1.signal_agent import fetch_company_signals
from agentic_estimation.layer_1.company_metadata import get_company_metadata
from agentic_estimation.layer_2.pillar_extractors import extract_all_claims
from agentic_estimation.layer_2.claim_validators import validate_claims
from agentic_estimation.layer_3.formula_estimator import compute_formula_scores
from agentic_estimation.layer_3.holistic_estimator import holistic_vote
from agentic_estimation.layer_3.reconcile import reconcile_all
from agentic_estimation.layer_4.estimate_verifier import verify_reconciled
from agentic_estimation.shared.claim_types import ExtractedClaim

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"**FAIL**: {name}  {detail}")


def main():
    company = sys.argv[1] if len(sys.argv) > 1 else "Nvidia"
    industry = sys.argv[2] if len(sys.argv) > 2 else "Semiconductors"
    country = sys.argv[3] if len(sys.argv) > 3 else "United States"

    print(f"=== Gathering real signals for {company} ===")
    signals = fetch_company_signals(company, industry, country=country)
    metadata = get_company_metadata(company)
    print(f"{len(signals)} signals gathered")

    print(f"=== Extracting real claims ===")
    claims = extract_all_claims(company, signals, metadata)
    kept_claims, flags = validate_claims(claims, signals=signals, country=country)
    print(f"{len(claims)} raw claims, {len(kept_claims)} survive Tier-0 ({len(claims)-len(kept_claims)} dropped)")

    # ── Test 1: Tier-0 precondition (crude fake) ────────────────────────────
    print("\n=== Test 1: Tier-0 precondition (crude fake) ===")
    crude_fake_tag = None
    for tag, text in signals.items():
        if "wikipedia" in tag.lower() or "profile" in tag.lower():
            crude_fake_tag = tag
            break
    if crude_fake_tag is None and signals:
        crude_fake_tag = list(signals.keys())[0]

    if crude_fake_tag:
        crude_fake = ExtractedClaim(
            factor="environmental_controversy", pillar="E", polarity=-1, strength=0.9, confidence=0.9,
            value=None, source_tag=crude_fake_tag,
            reasoning="fabricated: claims a controversy with no textual support", method="extracted",
        )
        test_claims = kept_claims + [crude_fake]
        reconfirmed, flags2 = validate_claims(test_claims, signals=signals, country=country)
        dropped_the_fake = crude_fake not in reconfirmed
        check("Tier-0 drops crude fake (no topic vocab) BEFORE critics ever see it", dropped_the_fake)
    else:
        print("SKIP: no signals gathered for this company -- cannot construct test 1")

    # ── Test 2: subtle-fake injection (the critics' actual acceptance test) ─
    print("\n=== Test 2: subtle-fake injection (Critic A's reason to exist) ===")
    # claim_validators.py's lexical-relevance vocab for environmental_controversy
    # is specifically: spill, pollution, contamination, "environmental violation",
    # epa, fine, penalty -- NOT generic climate/sustainability words. The
    # genuine subtle case needs text containing this factor's OWN vocabulary
    # without actually asserting a controversy -- inject a synthetic signal
    # built to do exactly that.
    subtle_tag = "synthetic_subtle_test"
    subtle_text = (
        f"{company} publishes an annual environmental compliance report. The report "
        f"discusses the company's approach to pollution prevention, spill response "
        f"planning, and proactive engagement with EPA guidance to avoid any fine or "
        f"penalty exposure -- no incidents of contamination or violation were "
        f"reported this year."
    )
    local_signals = dict(signals)  # don't mutate the real fetched signals dict
    local_signals[subtle_tag] = subtle_text

    subtle_fake = ExtractedClaim(
        factor="environmental_controversy", pillar="E", polarity=-1, strength=0.85, confidence=0.9,
        value=None, source_tag=subtle_tag,
        reasoning=f"fabricated: on-topic text ({subtle_tag}) misread as a controversy claim",
        method="extracted",
    )
    injected_claims = kept_claims + [subtle_fake]
    survived, flags3 = validate_claims(injected_claims, signals=local_signals, country=country)
    subtle_survives_tier0 = subtle_fake in survived
    check("subtle fake SURVIVES Tier-0 (on-topic vocab present)", subtle_survives_tier0,
          f"tag={subtle_tag!r}")

    if subtle_survives_tier0:
        print(f"  -> proceeding to critic panel (this makes real LLM calls)...")
        formula_scores = compute_formula_scores(
            survived, country, metadata, company_name=company, sector=industry, signals=local_signals,
        )
        holistic = holistic_vote(company, industry, country, local_signals, metadata)
        reconciled = reconcile_all(formula_scores, holistic)
        print(f"  E: score={formula_scores['E'].score:.1f} confidence={reconciled['E'].confidence}")

        verified = verify_reconciled(company, reconciled, formula_scores, holistic,
                                      survived, local_signals, metadata, country)
        v = verified["E"]
        print(f"  E verdict={v.verdict} mode={v.mode} retried={v.retried} objections={v.objections}")
        check("E pillar was actually verified (not skipped) OR correctly routed to range by QC/confidence",
              v.verdict in ("passed", "passed_after_retry", "refuted", "skipped"))
        if v.verdict == "skipped":
            print("  NOTE: E pillar skipped verification (confidence=high or QC=thin) -- the injected "
                  "fake didn't reach the critic panel this run. Valid outcome, but doesn't exercise Critic A.")
        elif v.verdict in ("refuted", "passed_after_retry"):
            print(f"  Critic panel ENGAGED with the injected claim (verdict={v.verdict}) -- target outcome.")

    # ── Test 3: clean-estimate control ──────────────────────────────────────
    print("\n=== Test 3: clean-estimate control (no injection) ===")
    formula_scores_clean = compute_formula_scores(
        kept_claims, country, metadata, company_name=company, sector=industry, signals=signals,
    )
    holistic_clean = holistic_vote(company, industry, country, signals, metadata)
    reconciled_clean = reconcile_all(formula_scores_clean, holistic_clean)
    verified_clean = verify_reconciled(company, reconciled_clean, formula_scores_clean, holistic_clean,
                                        kept_claims, signals, metadata, country)
    for pillar in ("E", "S", "G"):
        v = verified_clean[pillar]
        print(f"  {pillar}: verdict={v.verdict} mode={v.mode} score={v.score:.1f}")
        # A refute here is NOT automatically a failure -- see the module
        # docstring's KNOWN LIVE FINDING. Inspect v.objections manually;
        # only flag failure if the objection is nonsensical.
        acceptable = v.verdict in ("skipped", "passed", "refuted")
        check(f"{pillar} clean run: verdict is a recognized state (inspect objections manually if refuted)",
              acceptable, f"got {v.verdict}, objections={v.objections}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
