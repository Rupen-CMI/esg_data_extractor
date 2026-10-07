"""
schema.py -- request/response models for ESG Calculator v2.

See plans/ESG_CALCULATOR_V2_PLAN.md for the full design rationale. This is a
REPLACEMENT product for api/v1/esg_calculator/ (v1), built alongside it (not
in place of it) so the two can be compared directly.

FACTOR SOURCE: every scored field here is a real key from
agentic_estimation/layer_2/factor_registry.py -- not a parallel invented list.
Of the registry's 29 factors: 22 are askable (fields on CalculatorInput
below, after scope_3_emissions was deliberately dropped -- see below), 6 are
LOCKED (never accepted as input -- see LOCKED_FACTORS and plan doc section
2.4/2.1). The locked 6 carry ~40 points of almost entirely negative weight
(human rights, labor controversy, consumer harm, governance controversy,
workplace safety) plus sector_emissions_intensity, that no company
self-reports -- shown to the user as greyed-out, not silently omitted,
because that gap IS the honest argument for the real pipeline.

scope_3_emissions is a 23rd registry factor that IS askable in principle
(benchmark_band, not locked) but was deliberately removed from this
calculator's form entirely (2026-09) -- product decision, not a data-quality
one. It never appears as a field, is never built into a claim, and is not
counted in the 22 above.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from api.v1.esg_calculator_v2.data.country_baselines import baseline_for


# Canonical, closed industry list -- SAME dropdown values the pipeline itself
# uses (factor_registry / sector matching), so the sector passed to
# compute_formula_scores never silently abstains on a free-text mismatch.
# Re-exported from scoring.py's import for routes.py.
INDUSTRY_OPTIONS: tuple[str, ...] = (
    "Manufacturing", "Technology", "Financial Services", "Retail",
    "Healthcare", "Energy & Utilities", "Construction & Real Estate",
    "Transportation & Logistics", "Agriculture & Food", "Professional Services",
    "Other",
)


# The 7 factors that exist in the registry but are NEVER accepted as input --
# allegation-shaped, structurally undisclosable, or (sector_emissions_intensity)
# not user data at all. Exposed via GET /calculator/v2/factors so the frontend
# can render them greyed-out inside their own pillar's section, each with a
# one-line reason. See plan doc section 2.4.
LOCKED_FACTORS: tuple[dict, ...] = (
    {"key": "labor_controversy", "pillar": "S", "weight": 10,
     "reason": "Allegation-shaped; no company self-reports a labor dispute. "
               "Found by the pipeline from news/BHRRC signals."},
    {"key": "human_rights_incident", "pillar": "S", "weight": 12,
     "reason": "Highest single weight in the registry. Found by the pipeline "
               "from BHRRC-tracked incidents, never self-reported."},
    {"key": "consumer_harm_incident", "pillar": "S", "weight": 10,
     "reason": "Allegation-shaped (child safety, privacy, product safety). "
               "Found by the pipeline from news/regulator signals."},
    {"key": "workplace_safety", "pillar": "S", "weight": 6,
     "reason": "Too vague to self-report meaningfully -- lost_time_injury_rate "
               "already captures this with a real number."},
    {"key": "governance_controversy", "pillar": "G", "weight": 8,
     "reason": "Reputational catch-all (executive misconduct, sourcing "
               "allegations). Found by the pipeline, not self-reportable."},
    {"key": "sector_emissions_intensity", "pillar": "E", "weight": 4,
     "reason": "Derived from the sector anchor (Climate TRACE), not user "
               "input -- filled automatically from your industry selection."},
)


class CalculatorInput(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "country": "DEU",
                    "industry": "Manufacturing",
                    "annual_revenue_usd": 5_000_000,
                    "electricity_kwh": 200000,
                    "diesel_litres": 1000,
                    "renewable_energy_pct": 15.0,
                    "female_employees_pct": 45.0,
                    "female_board_pct": 33.0,
                    "regulatory_fines": False,
                    "anti_corruption_policy": True,
                    "whistleblower_mechanism": True,
                }
            ]
        }
    }

    # ── Identity / context -- not scored directly ───────────────────────────
    country: Optional[str] = Field(
        None, description="ISO3 code, e.g. 'USA', 'DEU', 'IND'. Must have a "
        "real World Bank ESG baseline on file -- see GET /calculator/v2/countries. "
        "Sets each pillar's starting point."
    )
    industry: Optional[str] = Field(
        None, description="Must be one of INDUSTRY_OPTIONS. Closed list, not "
        "free text -- the exact sector names compute_formula_scores' peer "
        "matching uses. If blank AND website_url is given, one LLM call infers "
        "it from the site (mapped onto this same closed list, never free text)."
    )
    annual_revenue_usd: Optional[float] = Field(
        None, gt=0, description="Denominator for every revenue-intensity "
        "benchmark band (scope 1/2/3, energy, water, waste)."
    )
    website_url: Optional[str] = Field(
        None, description="Optional. Used ONLY for industry inference when "
        "`industry` is blank, and as context for the closing narrative -- "
        "NOT scraped for ESG evidence. See plan doc section 1.3."
    )

    # ── E: Environmental -- 9 askable + 2 bill-derived helper inputs ────────
    # scope_3_emissions removed from this calculator entirely (2026-09) --
    # not a data-quality change, a deliberate product decision to drop the
    # field, form, and any claim-building for it.
    # Registry factors (direct, when the user has the real figure):
    scope_1_emissions: Optional[float] = Field(
        None, ge=0, description="tCO2e. Benchmark band (15, 250) per $1000 "
        "revenue-normalised. If you don't have this figure directly, supply "
        "electricity_kwh/diesel_litres/petrol_litres below instead -- we "
        "derive an equivalent claim from a verified DEFRA/IEA factor."
    )
    scope_2_emissions: Optional[float] = Field(
        None, ge=0, description="tCO2e. Benchmark band (15, 250)."
    )
    renewable_energy_pct: Optional[float] = Field(
        None, ge=0, le=100, description="% of energy from renewable sources. "
        "Benchmark band (75, 0)."
    )
    total_energy_consumption: Optional[float] = Field(
        None, ge=0, description="GJ. Benchmark band (300, 3500)."
    )
    water_withdrawal: Optional[float] = Field(
        None, ge=0, description="m3. Benchmark band (100, 3000)."
    )
    total_waste_generated: Optional[float] = Field(
        None, ge=0, description="tonnes. Benchmark band (5, 120)."
    )
    net_zero_pledge: Optional[bool] = Field(
        None, description="Public net-zero / carbon-neutral pledge with a "
        "stated target year. Zero-weight in the registry -- context only."
    )
    sbti_commitment: Optional[bool] = Field(
        None, description="Science Based Targets initiative commitment. "
        "Zero-weight -- context only."
    )
    cdp_disclosure: Optional[bool] = Field(
        None, description="CDP climate disclosure submitted. Zero-weight."
    )
    environmental_controversy: Optional[bool] = Field(
        None, description="Any reported environmental violation, spill, or "
        "pollution controversy in the last 3 years? Public-record signal -- "
        "denying costs little, admitting is weighed heavily. See detail field."
    )
    environmental_controversy_detail: Optional[str] = Field(
        None, max_length=500, description="Optional context if you answered "
        "yes above -- feeds the closing narrative, does not change the score."
    )
    # Bill-derived helper inputs -- NOT registry factors. Multiplied by the
    # verified DEFRA/IEA factors (data/emission_factors.py, carried over from
    # v1) to DERIVE a scope-1/scope-2 claim when the user has the underlying
    # bill but not a computed tCO2e figure. Kept from v1 -- its best mechanic.
    electricity_kwh: Optional[float] = Field(
        None, ge=0, description="Purchased electricity, kWh, off a utility "
        "bill. Multiplied by the IEA 2024 global grid factor to derive a "
        "scope_2_emissions-equivalent claim, IF scope_2_emissions itself "
        "wasn't supplied directly."
    )
    diesel_litres: Optional[float] = Field(
        None, ge=0, description="Diesel consumed, litres. Multiplied by a "
        "DEFRA 2024 factor to derive part of a scope_1_emissions-equivalent "
        "claim, IF scope_1_emissions itself wasn't supplied directly."
    )
    petrol_litres: Optional[float] = Field(
        None, ge=0, description="Petrol/gasoline consumed, litres. Same "
        "derivation as diesel_litres."
    )

    # ── S: Social -- 4 askable ───────────────────────────────────────────────
    female_employees_pct: Optional[float] = Field(
        None, ge=0, le=100, description="% of total workforce. Benchmark "
        "band (50, 10)."
    )
    female_board_pct: Optional[float] = Field(
        None, ge=0, le=100, description="% of board/leadership. Benchmark "
        "band (40, 5). Coarse for a very small board -- acknowledged, not "
        "hidden."
    )
    employee_turnover_rate: Optional[float] = Field(
        None, ge=0, le=100, description="Annual %, voluntary + involuntary. "
        "Benchmark band (5, 30)."
    )
    lost_time_injury_rate: Optional[float] = Field(
        None, ge=0, description="Per 200,000 hours worked. Benchmark band "
        "(0.2, 5). Legally tracked in most jurisdictions -- most companies "
        "have this even if nothing else on this form."
    )

    # ── G: Governance -- 8 askable ───────────────────────────────────────────
    board_independence_pct: Optional[float] = Field(
        None, ge=0, le=100, description="% of independent (non-executive, "
        "non-family) board members. Benchmark band (75, 20). Often not a "
        "meaningful figure for a very small/founder-run company -- "
        "acknowledged."
    )
    anti_corruption_policy: Optional[bool] = Field(
        None, description="Written policy against bribery/corruption. "
        "Zero-weight in the real registry -- nearly everyone answers yes, "
        "so this carries almost no scoring signal by design -- but MANDATORY "
        "here with a small +/-3pt calculator-only nudge (2026-09-21 product "
        "decision, see scoring.py's badge-factor nudge comment)."
    )
    whistleblower_mechanism: Optional[bool] = Field(
        None, description="Any reporting channel beyond 'tell your manager'. "
        "Zero-weight in the real registry; MANDATORY here with the same "
        "+/-3pt calculator-only nudge as anti_corruption_policy."
    )
    esg_report_published: Optional[bool] = Field(
        None, description="Has published a standalone ESG/sustainability "
        "report. Zero-weight -- stays optional, no calculator-only nudge."
    )
    third_party_esg_audit: Optional[bool] = Field(
        None, description="Independently assured/audited ESG disclosure. "
        "Zero-weight -- stays optional, no calculator-only nudge."
    )
    compliance_certification: Optional[bool] = Field(
        None, description="Anti-corruption/compliance certification or "
        "formal program. Zero-weight in the real registry; MANDATORY here "
        "with the same +/-3pt calculator-only nudge as anti_corruption_policy."
    )
    regulatory_fines: Optional[bool] = Field(
        None, description="Any regulatory fine or sanction in the last 3 "
        "years? Public-record signal (SEC/regulator filings) -- denying "
        "costs little, admitting is weighed heavily."
    )
    regulatory_fines_detail: Optional[str] = Field(
        None, max_length=500, description="Optional context -- feeds the "
        "narrative, does not change the score."
    )
    litigation: Optional[bool] = Field(
        None, description="Any material litigation in the last 3 years? "
        "Public-record signal (10-K Item 3, court dockets)."
    )
    litigation_detail: Optional[str] = Field(
        None, max_length=500, description="Optional context -- feeds the "
        "narrative, does not change the score."
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
                f"'{v}' is not a country we have a World Bank ESG baseline for "
                "-- see GET /calculator/v2/countries for the allowed list"
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
                f"'{v}' is not one of the allowed industries: "
                f"{', '.join(INDUSTRY_OPTIONS)}"
            )
        return v


class ClaimSummary(BaseModel):
    """One line per claim actually built from the user's input -- the
    calculator's version of the pipeline's claim list. Shown in the UI so a
    user can see their own inputs become the exact objects the real
    formula estimator consumes."""
    factor: str
    pillar: str
    polarity: int
    confidence: float = Field(..., description="0-1, set by HOW the value was "
                               "obtained (bill-derived > direct metric > "
                               "self-reported boolean) -- see scoring.py")
    source_tag: str
    reasoning: str


class PillarResult(BaseModel):
    score: float = Field(..., description="0-100, higher is better")
    low: float
    high: float
    confidence: str = Field(..., description="'high' | 'medium' | 'low'")
    formula_score: Optional[float] = Field(
        None, description="The formula estimator's own pillar score, before "
        "the LLM review and reconciliation -- shown alongside the LLM's read "
        "so the two can be compared, not hidden behind one blended number."
    )
    llm_score: Optional[float] = Field(
        None, description="The LLM review's own pillar score, or null if the "
        "LLM step failed/was unavailable -- see basis for what happened."
    )
    basis: list[str] = Field(default_factory=list, description="Human-readable "
                              "notes on what drove this pillar's score.")


class OverallResult(BaseModel):
    score: float = Field(..., description="0-100, higher is better -- a "
        "fixed-weight E/S/G blend (see scoring.py's _OVERALL_PILLAR_WEIGHTS, "
        "35/35/30 -- a calculator-only choice, NOT the same split "
        "build_esg_json.py uses for the live pipeline's report), NOT a "
        "separate estimate of its own.")
    low: float
    high: float


class CalculatorResult(BaseModel):
    E: PillarResult
    S: PillarResult
    G: PillarResult
    overall: OverallResult
    claims: list[ClaimSummary] = Field(default_factory=list)
    narrative: Optional[str] = Field(
        None, description="LLM-generated explanation of the ALREADY-COMPUTED "
        "scores above -- never used to derive them. Null if the narrative "
        "call failed or wasn't run."
    )
    critics_note: str = (
        "The full pipeline runs an adversarial three-critic review on "
        "uncertain pillars here. Not run in this playground."
    )
    disclaimer: str = (
        "Self-disclosed, unverified estimate for exploratory use only -- "
        "not an audited or certified ESG rating. Blank fields are not scored "
        "as zero: they widen your range and lower confidence. The full "
        "pipeline also checks factors no company self-reports (regulatory "
        "enforcement, labor/human-rights incidents) -- see GET "
        "/calculator/v2/factors for what this playground cannot see."
    )
