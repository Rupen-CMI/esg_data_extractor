"""
emission_factors.py -- real, sourced multiplier constants for ESG Calculator
v2. Copied verbatim from api/v1/esg_calculator/data/emission_factors.py (v1) --
these factors are used identically here, only to DERIVE a scope-1/scope-2
claim from a bill (kWh/litres) when the user doesn't have a direct tCO2e
figure -- see schema.py's electricity_kwh/diesel_litres/petrol_litres fields
and scoring.py's claim-construction layer.

WHY THESE NUMBERS AND NOT OTHERS
Every factor below is a real published figure from a named, checkable
source. Nothing here is invented or approximated by guesswork -- if a real
number wasn't confirmed, the field is left as a documented TODO rather than
filled with a plausible-looking placeholder.

SOURCES
    Fuel combustion (Scope 1) -- UK DEFRA/DESNZ Government GHG Conversion
    Factors for Company Reporting, 2024 edition (v1.1, year 2024, next
    publication 2025-06-10 per the workbook's own Introduction sheet).
    Landing page: https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024
    Direct download (the exact file these numbers were read from, "Fuels"
    sheet, row 22 header "Activity | Fuel | Unit | kg CO2e | ..."):
    https://assets.publishing.service.gov.uk/media/6722567487df31a87d8c497e/ghg-conversion-factors-2024-full_set__for_advanced_users__v1_1.xlsx
    VERIFIED 2026-08-26 by opening this exact file and reading the Fuels
    sheet directly (not taken from a search-result summary -- an earlier
    version of this module had wrong values, 2.58354/2.075, that were never
    actually checked against the real workbook; this replaces them).
    The workbook lists TWO variants for each fuel -- "average biofuel
    blend" and "100% mineral" -- the sheet's own guidance text (row 6-7
    area) states forecourt/pump fuel purchases should use the average
    biofuel blend variant, which is what's used below. A newer 2026 edition
    exists on gov.uk (https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2026)
    but has not yet been opened/verified the same way -- TODO: re-verify
    against the 2026 file when it replaces this one.
    Chosen over EPA's Hub because DEFRA publishes the broadest single-file
    coverage (fuels + electricity + transport + waste + materials + water)
    and is the de-facto standard even outside the UK for exactly that reason.

    Electricity grid intensity (Scope 2) -- IEA, "Emissions - Electricity
    2025" analysis (global average), https://www.iea.org/reports/electricity-2025/emissions
    VERIFIED 2026-08-26 via a direct screenshot of the live IEA page,
    quoting the page verbatim: "Over the forecast period of 2025-2027,
    global CO2 intensity is expected to fall by an average of 3.6%
    annually, declining from 445 g CO2/kWh in 2024 to 400 g CO2/kWh in
    2027." -- this is a primary-source confirmation, not a secondary quote.

    DECISION 2026-08-26: use ONLY the global average for every country,
    full stop -- no per-country table. An earlier version of this module
    had a 2-country table (Germany, US) as a "start", but 2 out of ~190
    countries is not real coverage; it just makes the calculator look
    more precise than it is for two arbitrary countries while everyone
    else silently got the same fallback anyway. One honestly-labelled
    global number for everyone is more truthful than a fake sense of
    precision for a random 1% of users. Revisit only if/when the full IEA
    ~150-country table is actually pulled and verified (see TODO below) --
    not with more one-off hand-picked countries.
    TODO: download the actual IEA Emissions Factors 2025 xlsx (requires an
    IEA data-portal login) and replace this single global figure with the
    full ~150-country table, all at once, all verified the same way.

WHAT THIS DOES NOT COVER
    Water, waste, and materials factors (DEFRA also publishes these) are
    not yet added -- the calculator's water/waste fields are compared via
    percentile/benchmark instead of multiplication for now (see
    scoring.py). Add here only once a real DEFRA figure is pulled and
    confirmed, same discipline as above.
"""

from __future__ import annotations

# ── Scope 1: fuel combustion, kg CO2e per litre ─────────────────────────────
# Source: DEFRA/DESNZ 2024 GHG Conversion Factors for Company Reporting
# (v1.1), "Fuels" sheet, "average biofuel blend" rows (the variant DEFRA's
# own guidance says to use for standard forecourt/pump fuel purchases).
# Direct download of the exact file these were read from -- open it and
# check the Fuels sheet yourself:
# https://assets.publishing.service.gov.uk/media/6722567487df31a87d8c497e/ghg-conversion-factors-2024-full_set__for_advanced_users__v1_1.xlsx
FUEL_KG_CO2E_PER_LITRE: dict[str, float] = {
    "diesel": 2.51279,   # "Diesel (average biofuel blend)", Fuels sheet row 72
    "petrol": 2.0844,    # "Petrol (average biofuel blend)", Fuels sheet row 96, aka gasoline
}

# Download link surfaced directly in every API response's `basis` field so
# a user can independently verify these numbers without trusting our word
# for it -- see routes.py / scoring.py.
DEFRA_SOURCE_URL = (
    "https://assets.publishing.service.gov.uk/media/6722567487df31a87d8c497e/"
    "ghg-conversion-factors-2024-full_set__for_advanced_users__v1_1.xlsx"
)

# Natural gas: "Natural gas" variant (not "100% mineral blend" -- DEFRA's
# own guidance, Fuels sheet row 12, says standard mains-grid gas use should
# report under plain "Natural gas"), kWh (Gross CV) basis -- matches how
# most energy bills report consumption (row 16 guidance). Fuels sheet row
# 42, VERIFIED 2026-08-26 the same way as the fuel litres figures above.
# Kept in its own unit rather than force-converted to litres/m3, since that
# conversion depends on gas composition/calorific value and DEFRA's own
# workbook handles it with its own lookup, not reproduced here.
NATURAL_GAS_KG_CO2E_PER_KWH: float = 0.1829

# ── Scope 2: purchased electricity, kg CO2 per kWh ──────────────────────────
# Global average CO2 intensity of electricity generation, 2024 -- IEA
# "Emissions - Electricity 2025" analysis, directly verified (see module
# docstring). Used for EVERY country -- see the DECISION note above for why
# there's no per-country table.
GLOBAL_AVERAGE_KG_PER_KWH: float = 0.445

# Source page for the electricity figures above -- surfaced in API responses
# alongside DEFRA_SOURCE_URL so a user can check both without trusting our
# word for it. Not a direct file download (IEA's page blocks automated
# fetches; this global-average figure was verified via a manual screenshot
# of this exact page, see module docstring) but it is the real, live,
# citable page a person can open themselves.
IEA_SOURCE_URL = "https://www.iea.org/reports/electricity-2025/emissions"


def grid_factor_for_country(iso3: str | None) -> tuple[float, str]:
    """Returns (kg CO2 per kWh, basis). Always the global average -- see
    the DECISION note in the module docstring for why there's no
    per-country table. `iso3` is accepted (not used) so scoring.py's call
    site doesn't need to change if/when a real full country table lands."""
    return GLOBAL_AVERAGE_KG_PER_KWH, "global_average"
