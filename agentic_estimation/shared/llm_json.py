"""
llm_json.py — shared JSON-extraction helper for parsing LLM responses.

Reasoning models emit chain-of-thought prose before their final JSON answer.
This scans for the LAST balanced top-level {...} object in the response text,
which is reliably the final answer rather than a draft mentioned mid-reasoning.

Was previously duplicated near-verbatim across pillar_extractors.py,
market_climate_trace_mapper.py, and metric_estimation_agent.py; consolidated
here so a parsing fix only needs to happen once.
"""

import json
import re
from typing import Optional


def extract_json_object(raw: str) -> Optional[dict]:
    if not raw:
        return None
    text = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Fast path: whole thing is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Scan for balanced { ... } blocks, keep the last one that parses to a dict
    best = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = text[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        if isinstance(obj, dict):
                            best = obj
                    except Exception:
                        pass
    return best
