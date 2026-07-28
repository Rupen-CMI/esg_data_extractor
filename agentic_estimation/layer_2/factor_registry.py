"""
factor_registry.py — single source of truth for every scorable ESG factor.

Phase 2 of the rebuild (see PHASE_2_PLAN.md). Two kinds of factor:

  benchmark_band  — a numeric/pct metric with a fixed (value_for_100,
                     value_for_0) band, imported directly from CORE_METRICS
                     (metric_estimation_agent.py) so the numeric backbone is
                     never duplicated. Intensity-normalised by revenue where
                     the metric specifies it.
  event           — a qualitative/boolean signal (pledge, controversy, fine,
                     certification...) scored as polarity * strength, no
                     fixed value band.

Every factor carries a hand-set starting `weight` — pillar-score points of
swing at confidence=1, |delta|=1. These are documented starting points, not
fitted; Phase 5 regresses them against bcorp_lookup/upright_lookup ground
truth. Weights are chosen so a fully-evidenced pillar (several strong claims)
can plausibly swing roughly +/-30 points around the country baseline.

`direction == "neutral"` CORE_METRICS entries (employee_count, annual_revenue)
are context only, never scored -- excluded here.

CLI:
    python -m agentic_estimation.layer_2.factor_registry list [--pillar E|S|G]
"""

from dataclasses import dataclass
from typing import Optional

from agentic_estimation.layer_3.metric_estimation_agent import CORE_METRICS


@dataclass(frozen=True)
class Factor:
    key: str
    pillar: str              # 'E' | 'S' | 'G'
    weight: float            # pillar-score points of swing at c=1, |delta|=1
    delta_shape: str         # 'benchmark_band' | 'event'
    direction: str           # 'higher' | 'lower' -- meaningful for benchmark_band
    metric: Optional[dict]   # the CORE_METRICS entry, when benchmark-backed
    description: str         # one line, also injected into extractor prompts


# ── Hand-set weights for the CORE_METRICS-backed benchmark_band factors ──────
# (scored ones only -- "neutral" direction entries are context, not weighted)
_CORE_METRIC_WEIGHTS = {
    "scope_1_emissions":        8,
    "scope_2_emissions":        8,
    "scope_3_emissions":        8,
    "renewable_energy_pct":     6,
    "total_energy_consumption": 5,
    "water_withdrawal":         5,
    "total_waste_generated":    5,
    "female_employees_pct":     5,
    "female_board_pct":         6,
    "employee_turnover_rate":   4,
    "lost_time_injury_rate":    6,
    "board_independence_pct":   8,
    "anti_corruption_policy":   5,
    "whistleblower_mechanism":  4,
    "esg_report_published":     4,
    "third_party_esg_audit":    5,
}

# ── New qualitative/event factors (no CORE_METRICS equivalent) ───────────────
# (key, pillar, weight, direction, description)
_EVENT_FACTORS = [
    ("net_zero_pledge",             "E", 5,  "higher", "Public net-zero / carbon-neutral pledge with a stated target year"),
    ("sbti_commitment",              "E", 6,  "higher", "Science Based Targets initiative commitment or validated target"),
    ("cdp_disclosure",                "E", 4,  "higher", "CDP climate disclosure submitted"),
    ("environmental_controversy",     "E", 10, "lower",  "Reported environmental violation, spill, or pollution controversy"),
    ("sector_emissions_intensity",    "E", 4,  "lower",  "Company's sector emissions intensity vs. country peers (Climate TRACE anchor)"),

    ("labor_controversy",             "S", 10, "lower",  "Reported labor dispute, unsafe conditions, or wage violation"),
    ("human_rights_incident",         "S", 12, "lower",  "Reported human rights incident (BHRRC-tracked)"),
    ("workplace_safety",              "S", 6,  "higher", "Positive workplace safety record or certification"),

    ("regulatory_fines",              "G", 9,  "lower",  "Regulatory fine or sanction against the company"),
    ("litigation",                    "G", 7,  "lower",  "Material litigation (from 10-K Item 3 or news)"),
    ("compliance_certification",      "G", 5,  "higher", "Anti-corruption / compliance certification or program"),
    ("governance_controversy",        "G", 8,  "lower",  "Reported governance/ethics controversy not covered by "
                                                          "litigation, fines, or a specific policy (e.g. human "
                                                          "rights sourcing allegations, executive misconduct)"),
]


def _build_registry() -> dict[str, Factor]:
    registry: dict[str, Factor] = {}

    for m in CORE_METRICS:
        if m["direction"] == "neutral":
            continue  # employee_count, annual_revenue -- context, not scored
        weight = _CORE_METRIC_WEIGHTS.get(m["key"])
        if weight is None:
            raise ValueError(f"CORE_METRICS key '{m['key']}' has no weight in _CORE_METRIC_WEIGHTS")
        registry[m["key"]] = Factor(
            key=m["key"],
            pillar=m["category"],
            weight=weight,
            delta_shape="benchmark_band" if m["benchmark"] else "event",
            direction=m["direction"],
            metric=m,
            description=m["name"],
        )

    for key, pillar, weight, direction, description in _EVENT_FACTORS:
        if key in registry:
            raise ValueError(f"Duplicate factor key '{key}' between CORE_METRICS and _EVENT_FACTORS")
        registry[key] = Factor(
            key=key,
            pillar=pillar,
            weight=weight,
            delta_shape="event",
            direction=direction,
            metric=None,
            description=description,
        )

    return registry


FACTORS: dict[str, Factor] = _build_registry()
FACTOR_KEYS: list[str] = list(FACTORS.keys())


def factors_for_pillar(pillar: str) -> list[Factor]:
    """All registered factors for one pillar ('E'/'S'/'G'), in a stable order."""
    return [f for f in FACTORS.values() if f.pillar == pillar]


def get_factor(key: str) -> Optional[Factor]:
    return FACTORS.get(key)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Factor registry — list all scorable ESG factors")
    sub = ap.add_subparsers(dest="cmd", required=True)
    list_p = sub.add_parser("list")
    list_p.add_argument("--pillar", choices=["E", "S", "G"], default=None)
    args = ap.parse_args()

    if args.cmd == "list":
        pillars = [args.pillar] if args.pillar else ["E", "S", "G"]
        for pillar in pillars:
            fs = factors_for_pillar(pillar)
            print(f"\n=== {pillar} ({len(fs)} factors) ===")
            for f in sorted(fs, key=lambda x: -x.weight):
                print(f"  {f.key:30s} w={f.weight:>4.1f}  {f.delta_shape:15s} {f.direction:8s}  {f.description}")


if __name__ == "__main__":
    _cli()
