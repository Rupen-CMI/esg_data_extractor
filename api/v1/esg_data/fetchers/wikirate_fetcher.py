"""
WikiRate fetcher — looks up a company by name in WikiRate, pulls all
available metric answers, and maps them to our esg_metric_definitions catalog.
"""

import os
from difflib import SequenceMatcher
from dotenv import load_dotenv
import wikirate4py

load_dotenv()

_api: wikirate4py.API | None = None

def _get_api() -> wikirate4py.API:
    global _api
    if _api is None:
        key = os.getenv("WIKIRATE_API_KEY", "")
        _api = wikirate4py.API(key)
    return _api


# ---------------------------------------------------------------------------
# WikiRate metric name → our catalog key
# Matching is done on the metric card name string (lowercased).
# First keyword match wins — order matters (more specific first).
# ---------------------------------------------------------------------------

WIKIRATE_KEYWORD_MAP: list[tuple[str, str]] = [
    # -----------------------------------------------------------------------
    # Environmental — GHG / Emissions
    # Order matters: more specific patterns first
    # -----------------------------------------------------------------------
    # Most specific scope patterns first — order is critical
    ("combined scope 1, 2 and 3",                  "scope_3_emissions"),
    ("scope 3",                                    "scope_3_emissions"),
    ("scope_3",                                    "scope_3_emissions"),
    ("indirect greenhouse gas (scope 3)",          "scope_3_emissions"),
    ("indirect ghg (scope 3)",                     "scope_3_emissions"),
    ("gri 305-3",                                  "scope_3_emissions"),
    ("greenhouse gas emissions scope 3",           "scope_3_emissions"),
    ("ghg scope 3",                                "scope_3_emissions"),
    ("combined scope 1 and 2",                     "scope_2_emissions"),
    ("scope 2",                                    "scope_2_emissions"),
    ("scope_2",                                    "scope_2_emissions"),
    ("indirect greenhouse gas (scope 2)",          "scope_2_emissions"),
    ("indirect ghg (scope 2)",                     "scope_2_emissions"),
    ("gri 305-2",                                  "scope_2_emissions"),
    ("greenhouse gas emissions scope 2",           "scope_2_emissions"),
    ("ghg scope 2",                                "scope_2_emissions"),
    ("scope 1",                                    "scope_1_emissions"),
    ("scope_1",                                    "scope_1_emissions"),
    ("direct greenhouse gas",                      "scope_1_emissions"),
    ("direct ghg",                                 "scope_1_emissions"),
    ("gri 305-1",                                  "scope_1_emissions"),
    ("greenhouse gas emissions scope 1",           "scope_1_emissions"),
    ("ghg scope 1",                                "scope_1_emissions"),
    # Generic GHG — map to scope 1 as best proxy (most specific patterns above must come first)
    ("greenhouse gas emission",                    "scope_1_emissions"),
    ("ghg emission",                               "scope_1_emissions"),
    ("carbon emission",                            "scope_1_emissions"),
    ("co2 emission",                               "scope_1_emissions"),
    ("ghg per",                                    "scope_1_emissions"),
    ("emissions per dollar",                       "scope_1_emissions"),
    ("emissions per revenue",                      "scope_1_emissions"),

    # -----------------------------------------------------------------------
    # Environmental — Energy
    # -----------------------------------------------------------------------
    ("renewable energy",                           "renewable_energy_pct"),
    ("re100",                                      "renewable_energy_pct"),
    ("decarbonisation",                            "renewable_energy_pct"),
    ("clean energy",                               "renewable_energy_pct"),
    ("energy procurement",                         "renewable_energy_pct"),
    ("financing decarbonisation",                  "renewable_energy_pct"),
    ("energy consumption",                         "total_energy_consumption"),
    ("energy use",                                 "total_energy_consumption"),
    ("energy intensity",                           "total_energy_consumption"),
    ("electricity consumption",                    "total_energy_consumption"),
    ("energy sourced",                             "total_energy_consumption"),

    # -----------------------------------------------------------------------
    # Environmental — Water
    # -----------------------------------------------------------------------
    ("water withdrawal",                           "water_withdrawal"),
    ("water consumption",                          "water_withdrawal"),
    ("water use",                                  "water_withdrawal"),
    ("freshwater",                                 "water_withdrawal"),
    ("water pollution",                            "water_withdrawal"),
    ("water stress",                               "water_stress_exposure_pct"),

    # -----------------------------------------------------------------------
    # Environmental — Waste
    # -----------------------------------------------------------------------
    ("hazardous waste",                            "hazardous_waste_generated"),
    ("waste generated",                            "total_waste_generated"),
    ("total waste",                                "total_waste_generated"),
    ("waste management",                           "total_waste_generated"),
    ("circularity",                                "total_waste_generated"),
    ("plastic use",                                "packaging_recyclable_pct"),
    ("packaging",                                  "packaging_recyclable_pct"),
    ("plastic",                                    "packaging_recyclable_pct"),

    # -----------------------------------------------------------------------
    # Environmental — Nature / Land / Biodiversity
    # -----------------------------------------------------------------------
    ("biodiversity",                               "land_use"),
    ("land use",                                   "land_use"),
    ("land area",                                  "land_use"),
    ("ecosystem",                                  "land_use"),
    ("nature-positive",                            "land_use"),
    ("halting biodiversity",                       "land_use"),
    ("pesticide",                                  "pesticide_use"),

    # -----------------------------------------------------------------------
    # Environmental — Agriculture / Food
    # -----------------------------------------------------------------------
    ("food waste",                                 "food_waste_pct"),
    ("food loss",                                  "food_waste_pct"),
    ("sustainable sourc",                          "sustainable_sourcing_pct"),
    ("responsible sourc",                          "sustainable_sourcing_pct"),
    ("traceability",                               "sustainable_sourcing_pct"),
    ("food safety",                                "food_safety_incidents"),
    ("animal welfare",                             "animal_welfare_policy"),

    # -----------------------------------------------------------------------
    # Social — Workforce / Employees
    # -----------------------------------------------------------------------
    ("number of employees",                        "employee_count"),
    ("total employees",                            "employee_count"),
    ("headcount",                                  "employee_count"),
    ("number of workers",                          "employee_count"),
    ("workforce size",                             "employee_count"),
    ("worker count",                               "employee_count"),
    # WBA / GreenDex employee fields
    ("social benchmark",                           "employee_count"),
    ("worker voice",                               "employee_count"),
    ("employee turnover",                          "employee_turnover_rate"),
    ("staff turnover",                             "employee_turnover_rate"),
    ("turnover rate",                              "employee_turnover_rate"),
    ("workforce redundancy",                       "employee_turnover_rate"),
    ("lost time injury",                           "lost_time_injury_rate"),
    ("ltir",                                       "lost_time_injury_rate"),
    ("trir",                                       "lost_time_injury_rate"),
    ("recordable incident",                        "lost_time_injury_rate"),
    ("training hours",                             "training_hours_per_employee"),
    ("hours of training",                          "training_hours_per_employee"),
    ("worker reskilling",                          "training_hours_per_employee"),
    ("upskilling",                                 "training_hours_per_employee"),

    # -----------------------------------------------------------------------
    # Social — Wages / Labour Rights
    # -----------------------------------------------------------------------
    ("living wage",                                "living_wage_commitment"),
    ("living income",                              "living_wage_commitment"),
    ("minimum wage",                               "living_wage_commitment"),
    ("wage data",                                  "living_wage_commitment"),
    ("workers paid",                               "living_wage_commitment"),
    ("piece rate",                                 "living_wage_commitment"),
    ("freedom of association",                     "whistleblower_mechanism"),
    ("collective bargaining",                      "whistleblower_mechanism"),
    ("forced labour",                              "supplier_audit_coverage_pct"),
    ("forced labor",                               "supplier_audit_coverage_pct"),
    ("child labour",                               "supplier_audit_coverage_pct"),
    ("child labor",                                "supplier_audit_coverage_pct"),
    ("modern slavery",                             "supplier_audit_coverage_pct"),
    ("human rights",                               "supplier_audit_coverage_pct"),
    ("iplc",                                       "supplier_audit_coverage_pct"),
    ("land rights",                                "supplier_audit_coverage_pct"),

    # -----------------------------------------------------------------------
    # Social — Gender / Diversity
    # -----------------------------------------------------------------------
    ("female employees",                           "female_employees_pct"),
    ("women employees",                            "female_employees_pct"),
    ("gender pay gap",                             "female_employees_pct"),
    ("gender benchmark",                           "female_employees_pct"),
    ("gender diversity",                           "female_employees_pct"),
    ("proportion of female",                       "female_employees_pct"),
    ("proportion of male",                         "female_employees_pct"),
    ("geb-",                                       "female_employees_pct"),
    ("violence and harassment",                    "female_employees_pct"),
    ("board gender",                               "female_board_pct"),
    ("women on board",                             "female_board_pct"),
    ("female board",                               "female_board_pct"),
    ("gender parity",                              "female_board_pct"),

    # -----------------------------------------------------------------------
    # Social — Supply Chain
    # -----------------------------------------------------------------------
    ("supplier audit",                             "supplier_audit_coverage_pct"),
    ("supplier assessment",                        "supplier_audit_coverage_pct"),
    ("supply chain transparency",                  "supplier_audit_coverage_pct"),
    ("supply chain wage",                          "supplier_audit_coverage_pct"),
    ("supplier list",                              "supplier_audit_coverage_pct"),
    ("supplied by",                                "supplier_audit_coverage_pct"),
    ("supply chain management",                    "supplier_audit_coverage_pct"),
    ("fair labor",                                 "supplier_audit_coverage_pct"),
    ("fair labour",                                "supplier_audit_coverage_pct"),

    # -----------------------------------------------------------------------
    # Governance — Board
    # -----------------------------------------------------------------------
    ("board independence",                         "board_independence_pct"),
    ("independent director",                       "board_independence_pct"),
    ("board composition",                          "board_independence_pct"),
    ("governance score",                           "board_independence_pct"),
    ("wba+governance",                             "board_independence_pct"),

    # -----------------------------------------------------------------------
    # Governance — Pay
    # -----------------------------------------------------------------------
    ("ceo pay ratio",                              "ceo_pay_ratio"),
    ("executive pay",                              "ceo_pay_ratio"),
    ("executive compensation",                     "ceo_pay_ratio"),

    # -----------------------------------------------------------------------
    # Governance — Ethics / Compliance
    # -----------------------------------------------------------------------
    ("anti-corruption",                            "anti_corruption_policy"),
    ("anti corruption",                            "anti_corruption_policy"),
    ("bribery",                                    "anti_corruption_policy"),
    ("whistleblower",                              "whistleblower_mechanism"),
    ("speak up",                                   "whistleblower_mechanism"),
    ("social dialogue",                            "whistleblower_mechanism"),
    ("stakeholder engagement",                     "whistleblower_mechanism"),

    # -----------------------------------------------------------------------
    # Governance — Disclosure / Reporting
    # -----------------------------------------------------------------------
    ("sustainability report",                      "esg_report_published"),
    ("esg report",                                 "esg_report_published"),
    ("csr report",                                 "esg_report_published"),
    ("fashion transparency index",                 "esg_report_published"),
    ("transparency index",                         "esg_report_published"),
    ("accountability for sustainability",          "esg_report_published"),
    ("sustainability target",                      "esg_report_published"),
    ("sustainability strategy",                    "esg_report_published"),
    ("impact materiality",                         "esg_report_published"),
    ("climate-related risks",                      "esg_report_published"),
    ("climate risk disclosure",                    "esg_report_published"),
    ("third-party assur",                          "third_party_esg_audit"),
    ("third party assur",                          "third_party_esg_audit"),
    ("external audit",                             "third_party_esg_audit"),
    ("external assurance",                         "third_party_esg_audit"),
    ("verified by",                                "third_party_esg_audit"),

    # -----------------------------------------------------------------------
    # Sector — Extractives
    # -----------------------------------------------------------------------
    ("methane",                                    "methane_emissions_intensity"),
    ("spill",                                      "hydrocarbon_spill_volume"),
    ("flare",                                      "flared_gas_volume"),

    # -----------------------------------------------------------------------
    # Sector — Technology
    # -----------------------------------------------------------------------
    ("data breach",                                "customer_data_breaches"),
    ("cybersecurity incident",                     "customer_data_breaches"),
    ("data privacy complaint",                     "data_privacy_complaints"),
    ("privacy complaint",                          "data_privacy_complaints"),
    ("e-waste",                                    "e_waste_recycled"),
    ("electronic waste",                           "e_waste_recycled"),
    ("power usage effectiveness",                  "data_center_pue"),
    ("pue",                                        "data_center_pue"),

    # -----------------------------------------------------------------------
    # Sector — Financials
    # -----------------------------------------------------------------------
    ("financed emission",                          "financed_emissions"),
    ("portfolio emission",                         "financed_emissions"),

    # -----------------------------------------------------------------------
    # Sector — Healthcare / Pharma
    # -----------------------------------------------------------------------
    ("patient safety",                             "patient_safety_incidents"),
    ("product recall",                             "medical_device_recalls"),
    ("drug recall",                                "medical_device_recalls"),
    ("r&d spend",                                  "r_and_d_spend_pct"),
    ("research and development spend",             "r_and_d_spend_pct"),
    ("research & development",                     "r_and_d_spend_pct"),

    # -----------------------------------------------------------------------
    # Sector — Transportation
    # -----------------------------------------------------------------------
    ("fleet fuel",                                 "fleet_fuel_efficiency"),
    ("fuel efficiency",                            "fleet_fuel_efficiency"),
    ("nox",                                        "nox_sox_emissions"),
    ("sox",                                        "nox_sox_emissions"),
    ("electric vehicle",                           "fleet_ev_pct"),
    ("ev fleet",                                   "fleet_ev_pct"),
    ("low-emission vehicle",                       "fleet_ev_pct"),
]


def _map_metric_name(raw_metric: str) -> str | None:
    """Return the catalog key for a WikiRate metric card name, or None."""
    lower = raw_metric.lower().replace("+", " ").replace("_", " ")
    for keyword, catalog_key in WIKIRATE_KEYWORD_MAP:
        if keyword.lower() in lower:
            return catalog_key
    return None


def _best_company_match(candidates, target_name: str):
    """Pick the candidate whose name is closest to target_name."""
    if not candidates:
        return None
    target = target_name.lower().strip()

    def _score(c) -> float:
        name = c.name.lower().strip()
        base = SequenceMatcher(None, name, target).ratio()
        # Boost if target words are a subset of candidate words (e.g. "Patagonia" in "Patagonia Inc.")
        target_words = set(target.split())
        name_words = set(name.split())
        if target_words and target_words.issubset(name_words):
            base += 0.15
        # Penalise if candidate has extra prefix/suffix tokens not in target
        extra_tokens = name_words - target_words
        if extra_tokens and not target_words.issubset(name_words):
            base -= 0.05 * len(extra_tokens)
        return base

    scored = sorted(candidates, key=_score, reverse=True)
    best = scored[0]
    return best if _score(best) > 0.5 else None


def _parse_numeric(value: str) -> float | None:
    if value is None:
        return None
    try:
        cleaned = str(value).replace(",", "").replace("%", "").strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def fetch_wikirate(company_name: str, timeout: int = 20) -> dict:
    """
    Main entry point.

    Returns:
        {
            source: "wikirate",
            found: bool,
            wikirate_id: int | None,
            wikirate_name: str | None,
            values: list[dict],
            error: str | None,
        }
    """
    api = _get_api()
    base = {
        "source": "wikirate",
        "found": False,
        "wikirate_id": None,
        "wikirate_name": None,
        "values": [],
        "error": None,
    }

    # --- Step 1: find company in WikiRate ---
    try:
        candidates = api.get_companies(name=company_name, limit=5)
        company = _best_company_match(candidates, company_name)
    except Exception as e:
        base["error"] = f"company lookup failed: {e}"
        return base

    if not company:
        return base

    base["found"] = True
    base["wikirate_id"] = company.id
    base["wikirate_name"] = company.name

    # --- Step 2: fetch all answers for this company ---
    try:
        answers = api.get_answers(company=company.id, limit=100)
    except Exception as e:
        base["error"] = f"answers fetch failed: {e}"
        return base

    # --- Step 3: map metric names to catalog keys ---
    seen_keys: set[str] = set()
    values = []

    for answer in answers:
        raw_metric = str(answer.metric) if answer.metric else ""
        catalog_key = _map_metric_name(raw_metric)
        if not catalog_key:
            continue
        # Deduplicate — keep the most recent year if same key appears twice
        dedup_key = (catalog_key, str(answer.year))
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        raw_value = str(answer.value) if answer.value is not None else None
        values.append({
            "metric_key": catalog_key,
            "value": raw_value,
            "numeric_value": _parse_numeric(raw_value),
            "reporting_year": int(answer.year) if answer.year else None,
        })

    base["values"] = values
    return base
