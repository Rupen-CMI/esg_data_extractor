"""
OpenCode Zen client (OpenAI-compatible) for extracting company names from
search snippets.

Base URL:  https://opencode.ai/zen/v1
Auth:      Authorization: Bearer <ZEN_API_KEY>

The free models served by Zen are reasoning models, so we give a generous
max_tokens budget and read the answer from `choices[0].message.content`.
"""

import os
import random
import re
import json
import threading
import time
import requests

ZEN_BASE_URL = "https://opencode.ai/zen/v1"

# Free model IDs available on Zen (from GET /v1/models).
FREE_MODELS = [
    "deepseek-v4-flash-free",
    "qwen3.6-plus-free",
    "minimax-m3-free",
    "mimo-v2.5-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
]

# Default model used by the pipeline (chosen by user after benchmarking).
DEFAULT_MODEL = "deepseek-v4-flash-free"

# Models that are actually usable on the free tier (qwen/minimax promos ended;
# nemotron is too slow/unstable). Ordered as a sensible fallback chain.
USABLE_FREE_MODELS = [
    "deepseek-v4-flash-free",
    "north-mini-code-free",
    "mimo-v2.5-free",
]


def _load_api_key() -> str:
    key = os.environ.get("ZEN_API_KEY")
    if key:
        return key.strip()
    # Fallback: read the bare key from .env next to this file.
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Support either "ZEN_API_KEY=sk-..." or a bare "sk-..." line.
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" in line:
                _, _, val = line.partition("=")
                return val.strip()
            return line
    raise RuntimeError("No Zen API key found. Set ZEN_API_KEY or put it in .env")


_EXTRACTION_PROMPT = (
    "You are a precise information extraction engine.\n"
    "From the search-result text below, extract ONLY the names of companies / "
    "organizations that are explicitly named as key players, manufacturers, or "
    "competitors in the market.\n"
    "Rules:\n"
    "- Use the full, canonical company name (e.g. 'Tesla' -> 'Tesla, Inc.' is "
    "NOT required; keep the name as written but drop trailing punctuation).\n"
    "- Do NOT invent companies that are not present in the text.\n"
    "- Do NOT include market-research firms, report publishers, generic phrases, "
    "or ticker symbols.\n"
    "- Return STRICT JSON only, no prose, in this exact shape:\n"
    '  {"companies": ["Name A", "Name B"]}\n\n'
    "Market: __MARKET__\n"
    "Search text:\n"
    "\"\"\"\n__CONTEXT__\n\"\"\"\n"
)


def _extract_json_array(text: str):
    """Best-effort parse of a {"companies":[...]} object from model output."""
    if not text:
        return []
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    # Try direct object parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "companies" in obj:
            return [str(c).strip() for c in obj["companies"] if str(c).strip()]
        if isinstance(obj, list):
            return [str(c).strip() for c in obj if str(c).strip()]
    except Exception:
        pass
    # Find the first JSON object substring containing "companies".
    m = re.search(r"\{.*?\"companies\".*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return [str(c).strip() for c in obj.get("companies", []) if str(c).strip()]
        except Exception:
            pass
    return []


def extract_companies(market: str, context: str, model: str = DEFAULT_MODEL,
                      api_key: str | None = None, max_tokens: int = 2000,
                      timeout: int = 120) -> dict:
    """
    Call a Zen model to extract company names from snippet text.

    Returns a dict with: companies, raw, latency_s, usage, model, ok, error.
    """
    api_key = api_key or _load_api_key()
    prompt = _EXTRACTION_PROMPT.replace("__MARKET__", market).replace("__CONTEXT__", context)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{ZEN_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        companies = _extract_json_array(content)
        return {
            "model": model,
            "ok": True,
            "error": None,
            "companies": companies,
            "raw": content,
            "latency_s": round(latency, 2),
            "usage": data.get("usage", {}),
        }
    except Exception as e:
        return {
            "model": model,
            "ok": False,
            "error": str(e),
            "companies": [],
            "raw": "",
            "latency_s": round(time.perf_counter() - started, 2),
            "usage": {},
        }


# Jitter added before every call (including the first) to spread out request
# bursts from concurrent workers -- found live that a tight cluster of
# same-instant calls (3-7 workers firing together) was a real contributor to
# 429s. Small enough to not meaningfully slow a single call, large enough to
# de-synchronise a thread pool.
_JITTER_RANGE_S = (0.3, 1.5)

# Process-wide minimum gap between the START of any two requests, across ALL
# threads -- jitter alone staggers when calls start but doesn't cap overall
# throughput; multiple worker threads (pillar_extractors' 3-way pool,
# calibration_harness' N workers, market_climate_trace_mapper's pool) all call
# this function concurrently, so the rate limit has to be enforced with a
# shared lock, not a per-call sleep. Same pacing discipline as
# climate_trace_harvester.py's _MIN_GAP, applied here across threads.
_MIN_GAP_S = 2.0
_rate_lock = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    global _last_call_at
    with _rate_lock:
        wait = _last_call_at + _MIN_GAP_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def call_with_prompt(prompt: str, model: str = DEFAULT_MODEL,
                     max_tokens: int = 2000, timeout: int = 120,
                     system: str | None = None,
                     retries: int = 3,
                     disable_thinking: bool = True) -> dict:
    """
    Call a Zen model with a prompt. No cross-model fallback -- pinned to
    `model` only, retried in place on failure. Found live that falling back
    to a different model (e.g. north-mini-code-free) could return HTTP 200
    with a syntactically valid envelope but completely empty content AND
    reasoning_content -- a silent non-answer that a fallback chain would mask.
    Real calibration/comparison work needs every call attributable to one
    known model, not a chain that can silently hand off to a weaker one.
    """
    api_key = _load_api_key()
    sys_msg = system or ("You are a JSON-only extraction engine. You must respond with ONLY "
                         "valid JSON. No thinking, no explanation, no markdown, no prose.")

    started = time.perf_counter()
    last_error = ""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        **({"thinking": {"type": "disabled"}} if disable_thinking else {}),
    }
    for attempt in range(retries + 1):
        _throttle()  # process-wide minimum gap across all threads
        time.sleep(random.uniform(*_JITTER_RANGE_S))  # + jitter, de-synchronise bursts
        try:
            resp = requests.post(
                f"{ZEN_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            if resp.status_code in (503, 429) and attempt < retries:
                wait = 8 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            # Reasoning models split into reasoning_content (think) + content (answer).
            # Prefer content (the final answer); fall back to reasoning_content if content empty.
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            if not content and not reasoning:
                # Found live under real concurrent load: the model can return
                # HTTP 200 with a syntactically valid response envelope but
                # BOTH content and reasoning_content empty -- a silent
                # non-answer that previously returned ok=True, causing every
                # downstream caller to see a "successful" call with nothing
                # to parse. Treat exactly like a retryable failure.
                last_error = f"empty completion from {model} (status {resp.status_code})"
                if attempt < retries:
                    time.sleep(8 * (attempt + 1))
                    continue
                break  # retries exhausted
            return {
                "ok": True,
                "raw": content,
                "reasoning": reasoning,
                "error": None,
                "model_used": model,
                "latency_s": round(time.perf_counter() - started, 2),
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                time.sleep(8 * (attempt + 1))

    return {"ok": False, "raw": "", "error": last_error,
            "model_used": None, "latency_s": round(time.perf_counter() - started, 2)}


if __name__ == "__main__":
    sample = (
        "Some of the major players in the US cloud computing market include "
        "Google, Salesforce, Microsoft, AWS, and Oracle."
    )
    print(json.dumps(extract_companies("Cloud Computing Market", sample), indent=2))
