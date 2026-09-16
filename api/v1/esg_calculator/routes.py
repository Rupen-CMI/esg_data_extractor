"""
routes.py -- standalone ESG calculator API. No DB session, no dependency
on the live pipeline (agentic_estimation/) -- see package docstring.

Playground tool: EVERY field is optional -- there is no company identity
field at all (removed deliberately, see schema.py's CalculatorInput
docstring: it was never read by any scoring logic, only echoed back, and
kept it created a false expectation that typing a real company's name
would look something up). Missing fields widen the returned score range
rather than blocking a result -- see schema.py's CalculatorInput and
scoring.py's module docstring for the three scoring mechanisms
(multiply / ratio / boolean-tier) and plans/ESG_CALCULATOR_PLAN.md for the
full design rationale.
"""

from fastapi import APIRouter

from api.v1.esg_calculator.data.country_baselines import country_options
from api.v1.esg_calculator.schema import CalculatorInput, CalculatorResult
from api.v1.esg_calculator.scoring import INDUSTRY_OPTIONS, score

router = APIRouter(prefix="/calculator", tags=["ESG Calculator"])


@router.get("/health", summary="Health check")
def health():
    return {"status": "running"}


@router.get(
    "/industries", summary="List allowed industries",
    description="Canonical, closed industry list for the `industry` field on "
    "/calculator/score -- deliberately not free text, to avoid the sector-"
    "matching ambiguity that plagues free-text industry input. Each one has "
    "a real EXIOBASE-derived structural baseline on file (see "
    "data/industry_baselines.py) that contributes to each pillar's "
    "starting point.",
)
def list_industries():
    return {"industries": list(INDUSTRY_OPTIONS)}


@router.get(
    "/countries", summary="List countries with a real ESG baseline",
    description="Every country here has a real World Bank ESG baseline on "
    "file (data/country_baselines.py, frozen 2026-08-27) -- the `country` "
    "field on /calculator/score only accepts values from this list. "
    "Supplying a country sets each pillar's starting point (see "
    "scoring.py's _add_baseline_votes).",
)
def list_countries():
    return {"countries": country_options()}


@router.post(
    "/score", response_model=CalculatorResult, summary="Score inputs",
    description=(
        "Fast, deterministic, no-LLM scoring. EVERY field is optional -- "
        "there is no company identity field of any kind -- and an omitted "
        "field simply widens the returned score range rather than blocking "
        "a result.\n\n"
        "Three scoring mechanisms are used depending on the field:\n"
        "- **Multiply**: fuel litres and electricity kWh are multiplied by "
        "a real, sourced emission factor (DEFRA 2024 for fuel, IEA 2024 "
        "global average for electricity). Every such line in a pillar's "
        "`basis` carries a clickable source URL.\n"
        "- **Ratio**: female_employee_count / total_employee_count is "
        "computed directly -- no external benchmark needed.\n"
        "- **Boolean / tier**: the remaining fields carry point weights "
        "chosen by this tool, not sourced from an external authority -- "
        "this is stated explicitly, not implied to be as rigorous as the "
        "sourced multiply-mechanism fields.\n"
        "- **Baseline vote**: `country` (World Bank ESG data) and `industry` "
        "(EXIOBASE sector structural intensity) each set a real, sourced "
        "starting point for every pillar -- averaged together when both are "
        "supplied -- before any company-specific field adjusts it.\n\n"
        "Every response is labelled self-disclosed and unverified. Expected "
        "latency: well under a second -- pure in-process arithmetic, no "
        "network/DB/LLM call in the request path."
    ),
)
def score_company(payload: CalculatorInput):
    return score(payload)
