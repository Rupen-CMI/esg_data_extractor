"""
scoring.py -- standalone ESG calculator scoring logic.

DELIBERATELY SEPARATE FROM agentic_estimation/. This calculator does not
call compute_formula_scores(), does not touch the peer_anchor/saturation
machinery, and does not import ExtractedClaim. It is a simpler, self-
contained tool: given whatever a user chooses to fill in -- EVERY field is
optional, there is no company identity field at all (see schema.py) --
produce a fast, explainable, honestly-ranged E/S/G estimate.

FIELD LIST IS FINAL -- see plans/ESG_CALCULATOR_PLAN.md section 2 and
schema.py's CalculatorInput. 10 scored fields total.

THREE SCORING MECHANISMS, MATCHED TO WHAT EACH FIELD ACTUALLY IS
(not everything in ESG reduces to "quantity x factor")

  1. MULTIPLICATION -- for genuine physical quantities with a trusted
     per-unit factor (fuel litres, electricity kWh). See data/emission_factors.py
     for sources -- both factors are DIRECTLY VERIFIED against their primary
     source (opened the real DEFRA spreadsheet / read the real IEA page),
     not taken from a search summary. Produces a real kg CO2e figure, then
     converted to a 0-100 "better than baseline" score via a simple revenue-
     normalised threshold. This threshold (_CRUDE_EMISSIONS_BENCHMARK_KG_PER_1000USD)
     is DELIBERATELY CRUDE -- flagged in `basis` -- it is NOT sector-adjusted
     even though industry is now known (see mechanism 4) because the crude
     benchmark and the EXIOBASE baseline vote are two different things: one
     turns a raw kg-CO2e figure into a delta, the other sets the pillar's
     starting point. Combining them into one sector-adjusted emissions
     threshold is a real future improvement, not done here yet.

  2. RATIO -- for fields that are already countable and don't need an
     external benchmark to be meaningful (female_employee_count /
     total_employee_count). Trust comes from showing the user's OWN
     arithmetic back to them, not from comparing to an invented threshold.

  3. TIER / BOOLEAN -- checklist facts. renewable_energy_tier is a coarse
     3-option tier rather than a precise percentage, because we have no
     honest benchmark to compare a precise % against -- a fake-precise
     ratio undermines trust more than an honest coarse tier. The 5 boolean
     fields (2 negative, 3 governance) each carry a fixed point weight WE
     CHOOSE OURSELVES -- there is no external authority publishing "a
     whistleblower policy is worth N points". This must stay visible, not
     be dressed up as sourced.

  4. BASELINE VOTES (country + industry) -- real, sourced starting points
     for all three pillars, set BEFORE any company-specific field is
     considered:
       - country: a frozen World Bank ESG baseline snapshot
         (data/country_baselines.py -- same real source and methodology
         the live pipeline's own country_baseline_agent.py uses, just
         precomputed offline instead of imported live).
       - industry: a frozen EXIOBASE sector-structural-intensity baseline
         (data/industry_baselines.py -- median of the EXIOBASE sub-sectors
         hand-classified into each of the 11 closed industry options, same
         percentile/inversion logic as the live pipeline's
         agentic_estimation/layer_3/exio_lookup.py, precomputed offline).
     When both are supplied they are averaged into one starting point
     (see _add_baseline_votes) -- neither is more authoritative a priori.
     This REPLACES the flat, generic 50 a truly-blank pillar would
     otherwise start from -- real sourced context is genuine information,
     so "nothing entered yet" should center on what we actually know, not
     a made-up neutral midpoint. Every OTHER field a user fills in still
     swings the score the normal way on top of this starting point.

COVERAGE: each pillar reports how many of its possible fields were actually
supplied, 0-1. Surfaced directly in the API response (the Ecomate-style
"coverage" field) rather than only being used internally.

RANGES, NOT POINTS: every pillar returns (low, high) around its point score.
Range width shrinks as more fields are supplied for that pillar and widens
when coverage is thin -- same spirit as the main pipeline's confidence gate,
reimplemented simply here rather than imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from api.v1.esg_calculator.data.emission_factors import (
    FUEL_KG_CO2E_PER_LITRE, grid_factor_for_country,
    DEFRA_SOURCE_URL, IEA_SOURCE_URL,
)
from api.v1.esg_calculator.data.country_baselines import baseline_for
from api.v1.esg_calculator.data.industry_baselines import industry_baseline_for
from api.v1.esg_calculator.schema import (
    CalculatorInput, PillarResult, CalculatorResult, RenewableTier,
    INDUSTRY_OPTIONS,  # re-exported here for routes.py -- canonical
    # definition lives in schema.py so its own field_validator can use it
    # without a circular import (schema.py cannot import scoring.py).
)

_WORLD_BANK_SOURCE_NOTE = (
    "World Bank Sovereign ESG Data, frozen snapshot computed 2026-08-27 -- "
    "see data/country_baselines.py"
)
_EXIOBASE_SOURCE_NOTE = (
    "EXIOBASE sector structural intensity, frozen snapshot computed 2026-08-27 -- "
    "median of matched EXIOBASE sub-sectors, see data/industry_baselines.py"
)

# Country and industry baselines BOTH become pillar starting points (see
# _PillarAccumulator.baseline_override) instead of a flat, generic 50 --
# real, sourced figures are genuine information, so "nothing entered yet"
# should center on what we actually know, not a made-up neutral midpoint.
# When both are present they are averaged into one starting point rather
# than one silently overriding the other -- neither is more authoritative
# than the other a priori. Every OTHER field a user fills in still swings
# the score the normal way, same weights as before -- this only changes
# what the score defaults to before any company-specific field is
# supplied, and narrows the default range accordingly (see to_result()).
def _add_baseline_votes(acc: "_PillarAccumulator", inp: CalculatorInput, pillar_key: str, pillar_label: str):
    parts: list[float] = []

    bl = baseline_for(inp.country)
    if bl is not None:
        country_score = bl[pillar_key]  # 0-100, higher = better, same scale as everything else
        parts.append(country_score)
        acc.basis.append(
            f"Country context ({bl['country']}, {pillar_label}): {country_score:.1f}/100 "
            f"({_WORLD_BANK_SOURCE_NOTE}) -- reflects national development context, "
            f"not this specific company"
        )

    ib = industry_baseline_for(inp.industry)
    if ib is not None:
        industry_score = ib[pillar_key]
        if industry_score is not None:
            parts.append(industry_score)
            acc.basis.append(
                f"Industry context ({inp.industry}, {pillar_label}): {industry_score:.1f}/100 "
                f"({_EXIOBASE_SOURCE_NOTE}, n={ib['n_sectors_matched']} matched EXIOBASE "
                f"sub-sectors) -- reflects this industry's typical structural profile, "
                f"not this specific company"
            )

    if not parts:
        return

    acc.baseline_override = sum(parts) / len(parts)
    acc.has_real_baseline = True
    if len(parts) > 1:
        acc.basis.append(
            f"Combined starting point (country + industry average): "
            f"{acc.baseline_override:.1f}/100 -- before company-specific data"
        )

# Very rough, DELIBERATELY CONSERVATIVE revenue-normalised emissions
# benchmark: kg CO2e per $1000 of revenue, used only to turn a raw
# multiplication result into a 0-100 "better/worse than a rough starting
# point" reading. Not sector-specific (this calculator has no EXIOBASE
# dependency) -- flagged in every result's `basis` list as a crude anchor,
# never presented as a peer comparison. See plan doc open decision #3.
_CRUDE_EMISSIONS_BENCHMARK_KG_PER_1000USD = 50.0

# Renewable tier point weights, as a fraction of the field's max weight.
# NOT sourced -- there is no external authority for "mostly renewable is
# worth X points". See plan doc open decision #1.
_RENEWABLE_TIER_FRACTION: dict[RenewableTier, float] = {
    RenewableTier.NONE: 0.0,
    RenewableTier.SOME: 0.5,
    RenewableTier.MOSTLY: 1.0,
}


@dataclass
class _PillarAccumulator:
    points: float = 0.0
    max_points: float = 0.0
    fields_supplied: int = 0
    fields_possible: int = 0
    basis: list[str] = field(default_factory=list)
    # Set by _add_baseline_votes when a real country and/or industry
    # baseline exists -- the empty/default range should be centered on
    # THIS, not a flat 50, since a real sourced figure is genuine
    # information, not nothing.
    baseline_override: float | None = None
    # True once a real, sourced baseline (country and/or industry) backs
    # this pillar -- narrows the default range even before any company-
    # specific field is filled in, because "we know nothing at all" and
    # "we know the real country/industry context" are not the same amount
    # of uncertainty.
    has_real_baseline: bool = False

    def add(self, supplied: bool, weight: float, contribution: float = 0.0, note: str | None = None):
        self.fields_possible += 1
        self.max_points += weight
        if supplied:
            self.fields_supplied += 1
            self.points += contribution
            if note:
                self.basis.append(note)

    def coverage(self) -> float:
        return round(self.fields_supplied / self.fields_possible, 3) if self.fields_possible else 0.0

    def to_result(self) -> PillarResult:
        # Normalise accumulated points onto a 0-100 scale around the
        # pillar's baseline (a real country/industry figure when we have
        # one, otherwise a flat neutral 50), scaled by how much of the
        # pillar's max possible swing was reached. A pillar with zero
        # fields BEYOND the baseline vote returns exactly that baseline
        # with maximum uncertainty (widest range) rather than a
        # fabricated point estimate.
        baseline = self.baseline_override if self.baseline_override is not None else 50.0
        span = self.max_points if self.max_points else 1.0
        score = baseline + 50.0 * (self.points / span)
        score = max(0.0, min(100.0, score))
        cov = self.coverage()
        # Range width shrinks from a WIDE default down to +-5 (fully
        # supplied) -- simple linear interpolation, not the pipeline's
        # confidence gate. With TRULY zero information (no country, no
        # company fields) the default is the full 0-100 scale (+-50) --
        # anything narrower would imply we know something we don't. A
        # real, sourced country baseline IS genuine information, so it
        # narrows the default to +-20 instead of claiming the same total
        # ignorance as not even knowing the country.
        max_half_width = 20.0 if self.has_real_baseline else 50.0
        half_width = max_half_width - (max_half_width - 5.0) * cov
        # Whole numbers only -- a decimal place would imply a precision
        # this self-disclosed, partially-covered estimate doesn't have.
        return PillarResult(
            score=round(score),
            low=round(max(0.0, score - half_width)),
            high=round(min(100.0, score + half_width)),
            coverage=cov,
            basis=self.basis or ["No fields supplied for this pillar -- baseline only."],
        )


def _score_e(inp: CalculatorInput) -> PillarResult:
    """3 fields: diesel_litres, petrol_litres (jointly one 'fuel' slot),
    electricity_kwh, renewable_energy_tier. Plus country/industry baseline
    votes if supplied -- see _add_baseline_votes."""
    acc = _PillarAccumulator()
    _add_baseline_votes(acc, inp, "e_score", "Environmental")

    # -- Mechanism 1: multiplication (fuel + electricity) --
    total_kg_co2e = 0.0
    any_fuel = False
    if inp.diesel_litres is not None:
        diesel_kg = inp.diesel_litres * FUEL_KG_CO2E_PER_LITRE["diesel"]
        total_kg_co2e += diesel_kg
        any_fuel = True
        acc.basis.append(
            f"Diesel: {inp.diesel_litres:.0f} L x {FUEL_KG_CO2E_PER_LITRE['diesel']} kg CO2e/L "
            f"(DEFRA 2024, average biofuel blend) = {diesel_kg:.0f} kg CO2e -- source: {DEFRA_SOURCE_URL}"
        )
    if inp.petrol_litres is not None:
        petrol_kg = inp.petrol_litres * FUEL_KG_CO2E_PER_LITRE["petrol"]
        total_kg_co2e += petrol_kg
        any_fuel = True
        acc.basis.append(
            f"Petrol: {inp.petrol_litres:.0f} L x {FUEL_KG_CO2E_PER_LITRE['petrol']} kg CO2e/L "
            f"(DEFRA 2024, average biofuel blend) = {petrol_kg:.0f} kg CO2e -- source: {DEFRA_SOURCE_URL}"
        )
    acc.fields_possible += 1
    acc.max_points += 20.0
    if any_fuel:
        acc.fields_supplied += 1

    if inp.electricity_kwh is not None:
        grid_factor, basis_label = grid_factor_for_country(inp.country)
        elec_kg_co2e = inp.electricity_kwh * grid_factor
        total_kg_co2e += elec_kg_co2e
        acc.fields_supplied += 1
        acc.basis.append(
            f"Electricity: {inp.electricity_kwh:.0f} kWh x {grid_factor} kg CO2/kWh "
            f"({basis_label} grid factor, IEA) = {elec_kg_co2e:.0f} kg CO2 -- source: {IEA_SOURCE_URL}"
        )
    acc.fields_possible += 1
    acc.max_points += 20.0

    if (any_fuel or inp.electricity_kwh is not None) and inp.annual_revenue_usd:
        kg_per_1000usd = total_kg_co2e / (inp.annual_revenue_usd / 1000.0)
        # lower emissions per $1000 revenue = better; crude, deliberately
        # not sector-adjusted -- see _CRUDE_EMISSIONS_BENCHMARK_KG_PER_1000USD.
        delta = (_CRUDE_EMISSIONS_BENCHMARK_KG_PER_1000USD - kg_per_1000usd) / \
            _CRUDE_EMISSIONS_BENCHMARK_KG_PER_1000USD
        delta = max(-1.0, min(1.0, delta))
        acc.points += delta * 40.0
        acc.basis.append(
            f"Total {total_kg_co2e:.0f} kg CO2e = {kg_per_1000usd:.1f} kg/$1000 revenue "
            f"(crude flat benchmark: {_CRUDE_EMISSIONS_BENCHMARK_KG_PER_1000USD} kg/$1000, not sector-adjusted)"
        )
    elif any_fuel or inp.electricity_kwh is not None:
        acc.basis.append("Revenue not supplied -- raw emissions computed but not benchmarked.")

    # -- Mechanism 3: renewable tier (coarse, not a precise %) --
    if inp.renewable_energy_tier is not None:
        frac = _RENEWABLE_TIER_FRACTION[inp.renewable_energy_tier]
        acc.add(True, 15.0, frac * 15.0,
                f"Renewable energy: '{inp.renewable_energy_tier.value}' "
                f"(self-reported tier, not independently verified)")
    else:
        acc.add(False, 15.0)

    return acc.to_result()


def _score_s(inp: CalculatorInput) -> PillarResult:
    """4 fields: total_employee_count + female_employee_count (jointly one
    ratio slot), had_fines_or_litigation_3y, missed_sustainability_target.
    Plus country/industry baseline votes if supplied."""
    acc = _PillarAccumulator()
    _add_baseline_votes(acc, inp, "s_score", "Social")

    # -- Mechanism 2: ratio, computed directly, no external benchmark --
    acc.fields_possible += 1
    acc.max_points += 20.0
    if inp.total_employee_count and inp.female_employee_count is not None:
        pct = 100.0 * inp.female_employee_count / inp.total_employee_count
        # Linear against a 0-50% range -- 50% representation scores full
        # marks, not an external norm, just "parity is the natural ceiling
        # for a ratio of two halves of a workforce".
        acc.fields_supplied += 1
        contribution = min(1.0, pct / 50.0) * 20.0
        acc.points += contribution
        acc.basis.append(
            f"Female employees: {inp.female_employee_count} of {inp.total_employee_count} "
            f"= {pct:.1f}% (your own figures, no external benchmark applied)"
        )
    elif inp.total_employee_count == 0:
        acc.basis.append("total_employee_count is 0 -- cannot compute a ratio.")
    elif inp.total_employee_count is not None or inp.female_employee_count is not None:
        acc.basis.append(
            "Both total_employee_count and female_employee_count are needed "
            "to compute a ratio -- only one was supplied."
        )

    # -- Mechanism 3: negative-disclosure booleans --
    acc.fields_possible += 1
    acc.max_points += 25.0
    if inp.had_fines_or_litigation_3y is not None:
        acc.fields_supplied += 1
        if inp.had_fines_or_litigation_3y:
            acc.points -= 25.0
            acc.basis.append("Fines/litigation disclosed in last 3 years -- scored down")
        else:
            acc.points += 10.0
            acc.basis.append("No fines/litigation disclosed in last 3 years")

    acc.fields_possible += 1
    acc.max_points += 15.0
    if inp.missed_sustainability_target is not None:
        acc.fields_supplied += 1
        if inp.missed_sustainability_target:
            acc.points -= 15.0
            acc.basis.append("Missed a stated sustainability/diversity target -- scored down")
        else:
            acc.points += 5.0
            acc.basis.append("No missed sustainability/diversity target disclosed")

    return acc.to_result()


def _score_g(inp: CalculatorInput) -> PillarResult:
    """3 fields, all boolean: anti_corruption_policy,
    esg_responsibility_assigned, whistleblower_mechanism. Plus country/
    industry baseline votes if supplied."""
    acc = _PillarAccumulator()
    _add_baseline_votes(acc, inp, "g_score", "Governance")

    acc.add(inp.anti_corruption_policy is True, 20.0, 20.0, "Anti-corruption policy in place")
    acc.add(inp.esg_responsibility_assigned is True, 20.0, 20.0,
            "A named person has explicit ESG/compliance responsibility")
    acc.add(inp.whistleblower_mechanism is True, 20.0, 20.0,
            "Whistleblower/issue-reporting channel exists")

    return acc.to_result()


def score(inp: CalculatorInput) -> CalculatorResult:
    e, s, g = _score_e(inp), _score_s(inp), _score_g(inp)
    overall_coverage = round((e.coverage + s.coverage + g.coverage) / 3, 3)
    return CalculatorResult(E=e, S=s, G=g, overall_coverage=overall_coverage)
