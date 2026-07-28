# ESG Data Pipeline — Progress Report

## 1. Task Overview

**Task Assigned to: Rupen Kumar**

The objective is to **fetch and provide ESG (Environmental, Social, Governance) data for the key players of a given market** — pillar scores (E/S/G), risk ratings, and the underlying metrics for each company.

---

## 2. Procured Data (public ESG sources)

Where companies already have disclosed or assessed ESG data, we use it directly instead of estimating. Sources ingested:

- **B Corp (B Lab)** — impact assessment scores
- **Upright Project** — net-impact metrics
- **World Bank** — country-level ESG baselines
- **Corporate disclosures / GRI / CDP** — directly reported metrics

**Currently on hand:**
- **20,000+ company ESG records** (B Corp ~10.3k + Upright ~10.1k)
- **210** country ESG baselines (World Bank)
- **~100** ESG metric definitions (GRI / SASB aligned)

When a market's key players match a company in this pool, their **real data is used directly**; only undisclosed companies go through estimation.

---

## 3. The Estimation Agent

Most niche-market companies (small/private brands, subsidiaries) don't disclose ESG data. For these, an **agentic estimation pipeline** produces size-aware estimates.

**When it runs:** automatically in the background when a market's ESG data is requested, for any unprocessed company. The API responds immediately; the frontend polls until results are ready.

**How it works (stages):**
1. **Signal gathering** — real-world evidence from news, RSS feeds, and ESG sources (CDP, SBTi, Wikipedia, sustainability reports). Cached for reuse.
2. **Company metadata** — size, employees, revenue, sector, country (anchors estimates to actual scale).
3. **Country ESG baseline** — World Bank context for the company's country.
4. **LLM estimation** — estimates core ESG metrics in native units, each with a confidence score, strictly anchored to company size and sector.
5. **Evaluator agent** — validates values and corrects any that are clearly implausible.
6. **Explainability agent** — plain-English summary of how the scores were derived.

**Scoring — fixed benchmark bands:** Metric values are scored 0–100 against **fixed real-world benchmarks** (not by comparing companies to each other), so each company is judged on its own merit. Emissions/energy/water are **normalised by revenue** first, so large firms aren't penalised for scale. Scores roll up into E/S/G pillars (weighted 40/35/25) → total score and a Leader / Follower / Laggard rating.

---

## 4. Final JSON Response (example)

Each company returns pillar scores, risk ratings, data source, and ~18 core metrics — each tagged with `estimated`, `confidence`, and a value.

```json
{
    "market": "Plant-Based Cheese Market",
    "industry_avg_esg_score": 65.6,
    "total_companies": 10,
    "scoring_method": "fixed_benchmark_revenue_intensity",
    "companies": [
        {
            "name": "Bel Group",
            "country": "France",
            "data_source": "agentic_estimated",
            "rating": "Leader",
            "esg_scores": {
                "environment": { "score": 85.4, "risk": "Low" },
                "social":      { "score": 74.9, "risk": "Medium" },
                "governance":  { "score": 94.5, "risk": "Low" },
                "total":       { "score": 84.0, "risk": "Low" }
            },
            "environmental_metrics": {
                "scope_1_emissions": { "value": "15,000 tCO2e", "score": 100.0, "estimated": true, "confidence": 0.6 },
                "renewable_energy_pct": { "value": "25%", "score": 33.3, "estimated": true, "confidence": 0.5 },
                "water_withdrawal": { "value": "3,000,000 m3", "score": 73.9, "estimated": true, "confidence": 0.5 }
                // ... + scope 2/3, energy, waste
            },
            "social_metrics": {
                "female_employees_pct": { "value": "45%", "score": 87.5, "estimated": true, "confidence": 0.5 }
                // ... + board %, turnover, injury rate, employee count
            },
            "governance_metrics": {
                "anti_corruption_policy": { "value": "Yes", "score": 100.0, "estimated": true, "confidence": 0.8 }
                // ... + board independence, whistleblower, ESG report, audit, revenue
            },
            "metrics_estimated": 18,
            "reporting_year": 2026
        }
        // ... one object per company
    ]
}
```

Each metric carries `estimated`, `confidence`, a unit-formatted `value`, and its `score` (0–100). `data_source` is `reported`, `agentic_estimated`, or `mixed`.

---

## 5. Status & Issues Addressed

The model is **currently under active testing**. Issues surfaced and resolved:

- **Wrong / implausible estimates** → added the **Evaluator agent** to validate and correct values.
- **Large companies unfairly compared to small ones** → added **revenue-based normalisation** and **fixed scoring benchmark bands**, so companies are judged on absolute, size-independent merit.
- **Excessive 404s** during signal gathering → expected "page not found" noise, trimmed and quieted.
- **429 rate-limit errors** → traced to **GDELT** (and a defunct Reuters feed); both **removed** from the sources.
- Smaller fixes: signal caching (fewer external calls), keeping the API responsive during estimation, and DB connection stability.

---

## Next Steps
Improve estimate accuracy (values currently skew optimistic), and optionally add sector-specific metrics/bands for market-specific tables.

***All data (company dataset, tested companies, metric definitions and values etc.) are present in the NeonDB database.***