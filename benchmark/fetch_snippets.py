"""
Fetch DuckDuckGo snippets ONCE for all markets and cache them to snippets.json.
Every model in the benchmark is then scored on identical input.
"""

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddgs import DDGS
from utils import read_market_from_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "snippets.json")


def fetch(market: str, country: str = "USA", max_results: int = 3) -> str:
    query = f"{market} top 5 key players in {country}"
    snippets = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            snippets.append(r.get("body") or "")
    return "\n".join(snippets)


def main():
    markets = read_market_from_file(os.path.join(ROOT, "input.txt"))
    data = {}
    for m in markets:
        print(f"Fetching: {m}")
        try:
            data[m] = fetch(m)
        except Exception as e:
            print(f"  failed: {e}")
            data[m] = ""
        time.sleep(random.uniform(3, 6))  # polite delay, avoid rate limits
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(data)} markets -> {OUT}")


if __name__ == "__main__":
    main()
