"""
yfinance fetcher — ticker discovery + sustainability scores.

Yahoo Finance deprecated the sustainability endpoint (~2023), so
ticker.sustainability is usually empty. We still try it and store
whatever comes back. The main value here is confirming whether a
company is publicly traded (has_public_esg flag) and finding its ticker.
"""

import yfinance as yf
from difflib import SequenceMatcher

YFINANCE_TO_CATALOG: dict[str, str] = {
    "environmentScore":    "sustainalytics_environment_score",
    "socialScore":         "sustainalytics_social_score",
    "governanceScore":     "sustainalytics_governance_score",
    "totalEsg":            "sustainalytics_total_esg_score",
    "highestControversy":  "sustainalytics_controversy_level",
}


def _name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_ticker(company_name: str) -> tuple[str | None, str | None]:
    """
    Return (ticker, matched_longname) or (None, None).
    Only accepts a match if name similarity > 0.5 and it is an Equity.
    """
    try:
        quotes = yf.Search(company_name, max_results=5).quotes
        for q in quotes:
            if q.get("quoteType") != "EQUITY":
                continue
            longname = q.get("longname") or q.get("shortname") or ""
            if _name_similarity(company_name, longname) > 0.5:
                return q["symbol"], longname
    except Exception:
        pass
    return None, None


def _parse_sustainability(sus_df) -> list[dict]:
    """Convert sustainability DataFrame rows we care about into value dicts."""
    values = []
    if sus_df is None or sus_df.empty:
        return values
    data = sus_df.to_dict().get("Value", {})
    for yf_key, catalog_key in YFINANCE_TO_CATALOG.items():
        raw = data.get(yf_key)
        if raw is None:
            continue
        try:
            num = float(raw)
        except (TypeError, ValueError):
            num = None
        values.append({
            "metric_key": catalog_key,
            "value": str(raw),
            "numeric_value": num,
            "reporting_year": None,
        })
    return values


def fetch_yfinance(company_name: str, existing_ticker: str | None = None) -> dict:
    """
    Main entry point.

    Returns:
        {
            source: "yfinance",
            found: bool,
            ticker: str | None,
            matched_name: str | None,
            values: list[dict],
            error: str | None,
        }
    """
    ticker_sym = existing_ticker
    matched_name = None

    if not ticker_sym:
        ticker_sym, matched_name = _find_ticker(company_name)

    if not ticker_sym:
        return {
            "source": "yfinance",
            "found": False,
            "ticker": None,
            "matched_name": None,
            "values": [],
            "error": None,
        }

    try:
        t = yf.Ticker(ticker_sym)
        values = _parse_sustainability(t.sustainability)
        return {
            "source": "yfinance",
            "found": True,
            "ticker": ticker_sym,
            "matched_name": matched_name,
            "values": values,
            "error": None,
        }
    except Exception as e:
        return {
            "source": "yfinance",
            "found": True,
            "ticker": ticker_sym,
            "matched_name": matched_name,
            "values": [],
            "error": str(e),
        }
