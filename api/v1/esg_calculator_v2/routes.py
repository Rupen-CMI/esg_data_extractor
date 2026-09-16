"""
routes.py -- ESG Calculator v2 API. See plans/ESG_CALCULATOR_V2_PLAN.md.

Mounted at /calculator/v2 -- deliberately separate from v1's /calculator,
so the two are simultaneously live and directly comparable (see plan doc
section 8, Migration).
"""

from fastapi import APIRouter

from api.v1.esg_calculator_v2.data.country_baselines import country_options
from api.v1.esg_calculator_v2.schema import (
    CalculatorInput, CalculatorResult, INDUSTRY_OPTIONS, LOCKED_FACTORS,
)
from api.v1.esg_calculator_v2.scoring import score

router = APIRouter(prefix="/calculator/v2", tags=["ESG Calculator v2"])


@router.get("/health", summary="Health check")
def health():
    return {"status": "running"}


@router.get(
    "/industries", summary="List allowed industries",
    description="Same closed list v1 uses -- the exact sector names "
    "compute_formula_scores' peer matching expects.",
)
def list_industries():
    return {"industries": list(INDUSTRY_OPTIONS)}


@router.get(
    "/countries", summary="List countries with a real ESG baseline",
    description="Every country here has a real World Bank ESG baseline on "
    "file. Supplying one sets each pillar's starting point.",
)
def list_countries():
    return {"countries": country_options()}


@router.get(
    "/factors", summary="List all 29 factors, askable and locked",
    description="The full registry this calculator is built from -- 23 "
    "askable (fields on POST /score) and 6 LOCKED (never accepted as "
    "input; found only by the real pipeline). Meant to be rendered "
    "directly: askable factors show as form fields, locked factors show "
    "greyed inside their own pillar's section with the given reason.",
)
def list_factors():
    return {"locked": list(LOCKED_FACTORS)}


@router.post(
    "/score", response_model=CalculatorResult, summary="Score inputs",
    description=(
        "Runs the REAL agentic_estimation pipeline code on claims built from "
        "your input -- compute_formula_scores(), the real confidence gate -- "
        "not a reimplementation. Every field is optional; a blank field "
        "widens the range and lowers confidence, it is never scored as a "
        "negative.\n\n"
        "use_llm=true additionally sends the formula's computed scores + "
        "your data to an LLM for an independent review, then reconciles the "
        "two into a single blended score with a real confidence label. "
        "use_llm=false (default) is fully deterministic -- same real "
        "pipeline code, just one estimator voting instead of two.\n\n"
        "The response's `claims` list shows exactly what was built from your "
        "input and consumed by the formula estimator -- the calculator's "
        "version of the pipeline's own claim objects. `critics_note` states "
        "plainly that the adversarial critic panel is not run here."
    ),
)
def score_company(payload: CalculatorInput, use_llm: bool = False):
    return score(payload, use_llm=use_llm)
