"""
run_signals.py — Fetch and display ESG signals for key players in a market.

Usage:
    python agentic_estimation/run_signals.py "Automotive Infotainment"
    python agentic_estimation/run_signals.py "Pharmaceutical Packaging" --companies "Amcor,Berry Global,Sealed Air"
    python agentic_estimation/run_signals.py "Solar Energy" --limit 3
"""

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from agentic_estimation.layer_1.signal_agent import fetch_signals_for_companies

# ── Default key players per well-known markets (used when --companies not given) ─

_DEFAULT_PLAYERS: dict[str, list[str]] = {
    "automotive infotainment": [
        "Harman International", "Panasonic Automotive", "Continental AG",
        "Bosch", "Visteon Corporation",
    ],
    "solar energy": [
        "First Solar", "SunPower", "Canadian Solar", "JinkoSolar", "Enphase Energy",
    ],
    "pharmaceutical packaging": [
        "Amcor", "Berry Global", "Sealed Air", "Gerresheimer", "AptarGroup",
    ],
    "electric vehicles": [
        "Tesla", "BYD", "Rivian", "NIO", "Lucid Motors",
    ],
}


def _resolve_companies(market: str, companies_arg: str | None, limit: int) -> list[str]:
    """Return company list from --companies arg, default map, or prompt the user."""
    if companies_arg:
        return [c.strip() for c in companies_arg.split(",") if c.strip()][:limit]

    key = market.lower().strip()
    for known_key, players in _DEFAULT_PLAYERS.items():
        if known_key in key or key in known_key:
            return players[:limit]

    print(f"[run_signals] No default companies for '{market}'.")
    print("  Pass them with: --companies \"Company A,Company B,Company C\"")
    sys.exit(1)


def _divider(char: str = "=", width: int = 70) -> str:
    return char * width


def _print_source(name: str, text: str, verbose: bool) -> None:
    max_len = 600 if verbose else 250
    preview = text[:max_len].replace("\n", " | ")
    truncated = "..." if len(text) > max_len else ""
    print(f"  [{name}]")
    print(f"    {preview}{truncated}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ESG signals for key players in a market."
    )
    parser.add_argument("market", help="Market name, e.g. 'Automotive Infotainment'")
    parser.add_argument(
        "--companies", "-c",
        default=None,
        help="Comma-separated company names (overrides defaults)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=5,
        help="Max companies to process (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show more text per source (600 chars instead of 250)",
    )
    parser.add_argument(
        "--sources", "-s",
        default=None,
        help="Comma-separated source names to show (e.g. 'gdelt,wikipedia,sbti'). Shows all if omitted.",
    )
    args = parser.parse_args()

    companies   = _resolve_companies(args.market, args.companies, args.limit)
    filter_srcs = {s.strip().lower() for s in args.sources.split(",")} if args.sources else None

    print()
    print(_divider())
    print(f"  ESG Signal Gathering")
    print(f"  Market   : {args.market}")
    print(f"  Companies: {', '.join(companies)}")
    if filter_srcs:
        print(f"  Sources  : {', '.join(sorted(filter_srcs))}")
    print(_divider())
    print()

    t0 = time.perf_counter()

    def on_progress(msg: str) -> None:
        print(f"  {msg}")

    results = fetch_signals_for_companies(companies, args.market, on_progress=on_progress)

    elapsed = round(time.perf_counter() - t0, 1)
    print()
    print(_divider())
    print(f"  Done in {elapsed}s")
    print(_divider())

    # ── Per-company output ────────────────────────────────────────────────────
    for company in companies:
        signals = results.get(company, {})

        if filter_srcs:
            signals = {k: v for k, v in signals.items() if k in filter_srcs}

        print()
        print(_divider("-"))
        status = f"{len(signals)} sources" if signals else "NO SIGNALS FOUND"
        print(f"  {company}  ({status})")
        print(_divider("-"))

        if not signals:
            print("  [no signals returned for this company]")
            continue

        for source_name, text in sorted(signals.items()):
            _print_source(source_name, text, args.verbose)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(_divider())
    print("  SUMMARY")
    print(_divider())

    all_sources: set[str] = set()
    for s in results.values():
        all_sources.update(s.keys())

    col = max((len(c) for c in companies), default=10) + 2
    header = f"  {'Company':<{col}}" + "".join(f"{s[:10]:<12}" for s in sorted(all_sources))
    print(header)
    print("  " + "-" * (col + 12 * len(all_sources)))

    for company in companies:
        signals = results.get(company, {})
        row = f"  {company:<{col}}"
        for src in sorted(all_sources):
            mark = "YES" if src in signals else "-"
            row += f"{mark:<12}"
        print(row)

    print()


if __name__ == "__main__":
    main()
