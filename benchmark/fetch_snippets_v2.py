"""
Fetch DuckDuckGo snippets for a country x industry grid (markets_v2.json),
caching to snippets_v2.json. Each entry keyed "Industry @ Country".
"""

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ddgs import DDGS

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "markets_v2.json")
OUT = os.path.join(HERE, "snippets_v2.json")


def fetch(industry: str, country: str, max_results: int = 3) -> str:
    query = f"{industry} top 5 key players in {country}"
    snippets = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            snippets.append(r.get("body") or "")
    return "\n".join(snippets)


def main():
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    data = {}
    for country in cfg["countries"]:
        for industry in cfg["industries"]:
            key = f"{industry} @ {country}"
            print(f"Fetching: {key}")
            try:
                data[key] = fetch(industry, country)
            except Exception as e:
                print(f"  failed: {e}")
                data[key] = ""
            time.sleep(random.uniform(3, 6))
    json.dump(data, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved {len(data)} markets -> {OUT}")


if __name__ == "__main__":
    main()
