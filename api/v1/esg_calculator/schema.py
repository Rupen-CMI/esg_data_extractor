"""
schema.py -- request/response models for the standalone ESG calculator.
No dependency on api/v1/models.py or agentic_estimation's ExtractedClaim --
this calculator is intentionally a separate, self-contained tool (see
api/v1/esg_calculator/__init__.py and the module docstrings in scoring.py).

FIELD LIST IS FINAL -- see plans/ESG_CALCULATOR_PLAN.md section 2. Every
field here earned its place after a deliberate cut from a ~24-field draft;
do not add fields back without updating the plan doc first.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from api.v1.esg_calculator.data.country_baselines import baseline_for


class RenewableTier(str, Enum):
    NONE = "none"
    SOME = "some"
    MOSTLY = "mostly"


# Canonical, closed industry list -- avoids the free-text sector-matching
# failure documented at length in the live pipeline (peer_anchor abstains
# 33-45% of the time on free-text industry strings). Kept small and
# self-contained; not the same list as bcorp_lookup.industry_category,
# deliberately, since this calculator has no DB dependency at all.
# Lives here (not scoring.py) so schema.py's own validator can enforce it
# without a circular import; re-exported from scoring.py for routes.py.
INDUSTRY_OPTIONS: tuple[str, ...] = (
    "Manufacturing", "Technology", "Financial Services", "Retail",
    "Healthcare", "Energy & Utilities", "Construction & Real Estate",
    "Transportation & Logistics", "Agriculture & Food", "Professional Services",
    "Other",
)


class CalculatorInput(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "country": "DEU",
                    "industry": "Manufacturing",
                    "annual_revenue_usd": 5_000_000,
                    "diesel_litres": 1000,
                    "petrol_litres": 500,
                    "electricity_kwh": 200000,
                    "renewable_energy_tier": "some",
                    "total_employee_count": 100,
                    "female_employee_count": 45,
                    "had_fines_or_litigation_3y": False,
                    "missed_sustainability_target": False,
                    "anti_corruption_policy": True,
                    "esg_responsibility_assigned": True,
                    "whistleblower_mechanism": True,
                }
            ]
        }
    }

    # ── Identity -- context only, not scored directly ───────────────────────
    # NO company_name / NO company identity field of any kind, deliberately.
    # It was never read by any scoring logic (see scoring.py -- only
    # echoed back in the response), so keeping it created a false
    # expectation: a client typing "Google" could reasonably assume we'd
    # look up something real, when in fact nothing company-specific would
    # happen at all. Removing it is the safe/honest choice, not a loss of
    # function -- see plans/ESG_CALCULATOR_PLAN.md for the field list
    # history. This also means EVERY field on this model is optional --
    # there is no longer any required field at all.
    country: Optional[str] = Field(
        None, description="ISO3 code, e.g. 'USA', 'DEU', 'IND'. Must be a "
        "country with a real World Bank ESG baseline on file -- see "
        "GET /calculator/countries for the exact allowed list. Sets each "
        "pillar's starting point (averaged with the industry baseline if "
        "both are supplied) -- see scoring.py's _add_baseline_votes; "
        "electricity emissions still use one global grid factor for every "
        "country regardless (see data/emission_factors.py)."
    )
    industry: Optional[str] = Field(
        None, description="Must be one of INDUSTRY_OPTIONS (this module). "
        "Closed list, not free text. Contributes a real, sourced baseline "
        "vote to each pillar's starting point -- see data/industry_baselines.py "
        "(EXIOBASE sector structural intensity, median per matched sub-sectors)."
    )
    annual_revenue_usd: Optional[float] = Field(
        None, gt=0, description="Used to revenue-normalise raw emissions into "
        "a kg-CO2e-per-$1000-revenue reading. Without it, emissions are "
        "computed but not benchmarked."
    )

    # ── E: Environmental (3 fields) ─────────────────────────────────────────
    diesel_litres: Optional[float] = Field(
        None, ge=0, description="Diesel consumed, litres. Multiplied by a "
        "DEFRA 2024 factor -- see data/emission_factors.py."
    )
    petrol_litres: Optional[float] = Field(
        None, ge=0, description="Petrol/gasoline consumed, litres. Multiplied "
        "by a DEFRA 2024 factor."
    )
    electricity_kwh: Optional[float] = Field(
        None, ge=0, description="Purchased electricity, kWh. Multiplied by "
        "the IEA 2024 global average grid factor."
    )
    renewable_energy_tier: Optional[RenewableTier] = Field(
        None, description="Coarse self-assessed tier, not a precise percentage "
        "-- there is no honest benchmark to compare a precise % against yet."
    )

    # ── S: Social (4 fields) ────────────────────────────────────────────────
    total_employee_count: Optional[int] = Field(
        None, ge=0, description="Denominator for the female-employee ratio."
    )
    female_employee_count: Optional[int] = Field(
        None, ge=0, description="Used with total_employee_count to compute a "
        "female-employee percentage directly -- no external benchmark needed."
    )
    had_fines_or_litigation_3y: Optional[bool] = Field(
        None, description="Negative-disclosure signal (scored down if true)."
    )
    missed_sustainability_target: Optional[bool] = Field(
        None, description="Deliberately a 'safe to admit' negative signal -- "
        "missing a stated target is a smaller admission than misconduct, so "
        "it is more likely to be answered honestly."
    )

    # ── G: Governance (3 fields, all boolean) ───────────────────────────────
    anti_corruption_policy: Optional[bool] = Field(
        None, description="Does a written policy against bribery/corruption exist?"
    )
    esg_responsibility_assigned: Optional[bool] = Field(
        None, description="Does a named person (owner, manager, anyone) have "
        "explicit responsibility for ESG/compliance? The SME-appropriate "
        "substitute for 'board oversight' -- works for a 5-person company."
    )
    whistleblower_mechanism: Optional[bool] = Field(
        None, description="Any way to report misconduct that isn't just "
        "'tell your manager' -- e.g. an anonymous inbox."
    )

    @field_validator("country")
    @classmethod
    def _validate_country(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().upper()
        if not v:
            return None
        if baseline_for(v) is None:
            raise ValueError(
                f"'{v}' is not a country we have a World Bank ESG baseline for -- "
                "see GET /calculator/countries for the allowed list"
            )
        return v

    @field_validator("industry")
    @classmethod
    def _validate_industry(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if v not in INDUSTRY_OPTIONS:
            raise ValueError(
                f"'{v}' is not one of the allowed industries: {', '.join(INDUSTRY_OPTIONS)}"
            )
        return v

    @model_validator(mode="after")
    def _female_not_more_than_total(self):
        if (
            self.total_employee_count is not None
            and self.female_employee_count is not None
            and self.female_employee_count > self.total_employee_count
        ):
            raise ValueError(
                "female_employee_count cannot exceed total_employee_count"
            )
        return self


class PillarResult(BaseModel):
    score: float = Field(..., description="0-100, higher is better")
    low: float
    high: float
    coverage: float = Field(..., description="0-1, how much of this pillar's "
                             "possible signal was actually provided/derivable")
    basis: list[str] = Field(default_factory=list, description="Human-readable "
                              "notes on what drove this pillar's score, with a "
                              "clickable source URL wherever a preset value "
                              "was used")


class CalculatorResult(BaseModel):
    E: PillarResult
    S: PillarResult
    G: PillarResult
    overall_coverage: float
    disclaimer: str = (
        "Self-disclosed, unverified estimate for exploratory use only -- "
        "not an audited or certified ESG rating."
    )
