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
#
# deepseek-v4-flash-free DELIBERATELY REMOVED from this list 2026-10-06 (user
# instruction) -- opencode.ai's free tier was intermittently 429-throttling
# and the model was implicated in at least one unexplained 400 Bad Request
# on a real production call (pillar_extractors.py, Toyota E/S/G extraction,
# all 3 pillars failed simultaneously). Do not re-add until explicitly told
# to.
FREE_MODELS = [
    "qwen3.6-plus-free",
    "minimax-m3-free",
    "mimo-v2.5-free",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
]

# Default model used by the pipeline. Ollama's gpt-oss:120b-cloud, not an
# opencode.ai free model -- see "Ollama routing" below for why (separate
# backend/quota from opencode.ai's throttled free tier).
DEFAULT_MODEL = "gpt-oss:120b-cloud"

# Models that are actually usable on the free tier (qwen/minimax promos ended;
# nemotron is too slow/unstable). Ordered as a sensible fallback chain.
#
# NOTE 2026-08-18: "north-mini-code-free" is CONFIRMED DEAD -- a live GET
# /v1/models call no longer lists it; calling it returns 401, not a real auth
# failure (the same key works fine for every other model). Live model list as
# of today: deepseek-v4-flash-free, mimo-v2.5-free, hy3-free,
# nemotron-3-ultra-free, nemotron-3.5-lightning-free, laguna-s-2.1-free. This
# constant is left stale (not corrected) because nothing in the live pipeline
# reads it for fallback selection today -- pillar_extractors.py is pinned to
# DEFAULT_MODEL only, per call_with_prompt's own no-cross-model-fallback
# design. Fix if/when a real fallback-chain caller is built.
#
# deepseek-v4-flash-free REMOVED 2026-10-06, same reason as FREE_MODELS above.
USABLE_FREE_MODELS = [
    "north-mini-code-free",
    "mimo-v2.5-free",
]

# ── Ollama routing (2026-08-18) ──────────────────────────────────────────────
# opencode.ai's free tier has been intermittently 429-throttling this session
# (confirmed: fresh 429s on an otherwise-idle process, not something our own
# concurrency caused -- looks like shared free-tier load, out of our control).
# gpt-oss:120b-cloud, pulled via local Ollama, runs on Ollama's OWN cloud
# infrastructure -- a genuinely separate backend/quota from opencode.ai, not
# just a different model name on the same throttled host. Verified live:
# responds correctly, produces valid closed-factor-list JSON, no subscription
# wall (unlike deepseek-v4-flash:cloud, which 402s without one).
#
# Routing is by SUFFIX, not an explicit allowlist -- any "*-cloud" or "*:cloud"
# style name reaching call_with_prompt is treated as an Ollama model, so this
# doesn't need updating every time a new Ollama cloud model is pulled.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CLOUD_MODEL = "gpt-oss:120b-cloud"


def _is_ollama_model(model: str) -> bool:
    return model.endswith("-cloud") or ":cloud" in model or model.endswith(":cloud")


# Separate rate limiter + separate 429 counter from opencode.ai's -- these are
# different hosts with different quotas; conflating them would either starve
# Ollama calls with opencode.ai's pacing for no reason, or (worse) let an
# opencode.ai throttle silently absorb budget that should have been Ollama's
# own tripwire, masking a real Ollama-side ban risk. "We don't want to get
# banned here too" (user, 2026-08-18) -- this exists so Ollama gets its own
# accounting, not opencode.ai's leftovers.
_OLLAMA_MIN_GAP_S = float(os.getenv("ESG_OLLAMA_MIN_GAP", "2.0"))
_ollama_rate_lock = threading.Lock()
_ollama_last_call_at = 0.0
_ollama_error_lock = threading.Lock()
_ollama_error_count = 0
_OLLAMA_ERROR_THRESHOLD = 3   # same threshold discipline as signal_agent's RateLimitTripped


class OllamaRateLimitTripped(RuntimeError):
    """Same spirit as signal_agent.RateLimitTripped, kept separate: a cascade
    of Ollama-side errors (which could mean a cloud-quota exhaustion, not
    just a transient network blip) should stop the run, not retry into it."""


def _ollama_throttle() -> None:
    """Same reserve-then-sleep + randomised-gap discipline as opencode.ai's
    _throttle() (zen_client.py above) -- a fixed gap is a detectable, uniform
    request-spacing signature, and this run uses 3 concurrent workers
    (pillar_extractors' per-pillar pool), so bursts are a real risk here too,
    not just on the host that already got hammered tonight."""
    global _ollama_last_call_at
    with _ollama_rate_lock:
        now = time.monotonic()
        gap = _OLLAMA_MIN_GAP_S + random.uniform(0, 1.0)
        wait = _ollama_last_call_at + gap - now
        _ollama_last_call_at = max(now, _ollama_last_call_at + gap)
    if wait > 0:
        time.sleep(wait)
    time.sleep(random.uniform(0.3, 1.5))   # + jitter before the call itself


def _call_ollama_cloud(prompt: str, model: str, system: str, max_tokens: int, timeout: int) -> dict:
    """Ollama /api/generate, non-streaming. CLEAN ABORT ON ANY ERROR, no
    retry -- same discipline as opencode.ai's path (user decision,
    2026-08-18): do not hammer a host that just failed us, cloud quota
    exhaustion looks identical to a transient blip from here and both
    deserve the same caution."""
    global _ollama_error_count
    _ollama_throttle()
    started = time.perf_counter()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code != 200:
            with _ollama_error_lock:
                _ollama_error_count += 1
                count = _ollama_error_count
            print(f"[zen_client] Ollama HTTP {resp.status_code} from {model} "
                  f"({count}/{_OLLAMA_ERROR_THRESHOLD}) -- aborting immediately, no retry",
                  flush=True)
            if count >= _OLLAMA_ERROR_THRESHOLD:
                raise OllamaRateLimitTripped(
                    f"aborting: {count} Ollama errors (threshold {_OLLAMA_ERROR_THRESHOLD}) "
                    f"on {model} -- possible quota exhaustion, stopping rather than risk a ban."
                )
            return {"ok": False, "raw": "", "error": f"Ollama HTTP {resp.status_code}",
                    "model_used": None, "latency_s": round(time.perf_counter() - started, 2)}
        data = resp.json()
        # Ollama reasoning models (gpt-oss, qwen3) split "thinking" (chain of
        # thought) from "response" (the final answer) -- same shape issue
        # zen_client already handles for opencode.ai's reasoning_content.
        # "response" is what to parse; "thinking" is context only.
        content = data.get("response") or ""
        thinking = data.get("thinking") or ""
        if not content:
            with _ollama_error_lock:
                _ollama_error_count += 1
            return {"ok": False, "raw": "", "error": "empty Ollama response",
                    "model_used": None, "latency_s": round(time.perf_counter() - started, 2)}
        return {"ok": True, "raw": content, "reasoning": thinking, "error": None,
                "model_used": model, "latency_s": round(time.perf_counter() - started, 2)}
    except OllamaRateLimitTripped:
        raise
    except Exception as exc:
        with _ollama_error_lock:
            _ollama_error_count += 1
            count = _ollama_error_count
        print(f"[zen_client] Ollama call failed: {type(exc).__name__}: {exc} "
              f"({count}/{_OLLAMA_ERROR_THRESHOLD})", flush=True)
        if count >= _OLLAMA_ERROR_THRESHOLD:
            raise OllamaRateLimitTripped(
                f"aborting: {count} Ollama errors (threshold {_OLLAMA_ERROR_THRESHOLD})"
            )
        return {"ok": False, "raw": "", "error": str(exc),
                "model_used": None, "latency_s": round(time.perf_counter() - started, 2)}


def _load_api_key() -> str:
    key = os.environ.get("ZEN_API_KEY")
    if key:
        return key.strip()
    # Fallback: read from .env next to this file. .env is a shared,
    # multi-variable file (DB_URL, ASYNC_DB_URL, ZEN_API_KEY, ...) -- must
    # find the ZEN_API_KEY= line specifically, NOT just return the first
    # non-empty/non-comment line. A prior version did exactly that and
    # silently returned "neondb" (from DB_NAME=neondb, .env's first line)
    # as the "API key" whenever ZEN_API_KEY wasn't set as a real env var,
    # producing a 401 that looked like an account/quota problem but was
    # actually this parser reading the wrong line entirely.
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                name, _, val = line.partition("=")
                if name.strip() == "ZEN_API_KEY":
                    return val.strip()
            elif line.startswith("sk-"):
                # Bare "sk-..." line with no "KEY=" prefix at all.
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
#
# Overridable via ESG_LLM_MIN_GAP for runs where wall-clock matters more than
# headroom: this gap is the single largest cost in a scoring run (150 companies
# x ~8 calls is ~50 min at 2.0s, ~35 min at 1.5s). Lower it only when watching
# the run -- 429s here feed _note_rate_limit and can trip the abort guard, and a
# tripped run costs far more than the minutes saved.
_MIN_GAP_S = float(os.getenv("ESG_LLM_MIN_GAP", "2.0"))
_rate_lock = threading.Lock()
_last_call_at = 0.0

# ADAPTIVE BACKOFF -- DEAD CODE as of 2026-08-18, kept for now, no callers.
# call_with_prompt() used to call penalise() on every 429 so a throttle would
# slow the whole run rather than just the one call that hit it. Removed: user
# decision was clean-abort-no-retry instead (retrying, even slower, is still
# load on a host that just told us to stop). If a future caller wants a
# soft-degrade path again, this is still here and still correct; nothing
# currently invokes it.
_PENALTY_MAX = float(os.getenv("ESG_LLM_PENALTY_MAX", "8.0"))
_PENALTY_DECAY_S = float(os.getenv("ESG_LLM_PENALTY_DECAY", "600"))
_penalty = 1.0
_penalty_set_at = 0.0


def _current_penalty() -> float:
    """Penalty multiplier, decaying linearly to 1.0 over _PENALTY_DECAY_S."""
    global _penalty
    if _penalty <= 1.0:
        return 1.0
    age = time.monotonic() - _penalty_set_at
    if age >= _PENALTY_DECAY_S:
        _penalty = 1.0
        return 1.0
    return 1.0 + (_penalty - 1.0) * (1.0 - age / _PENALTY_DECAY_S)


def penalise(factor: float = 2.0) -> None:
    """Slow every subsequent LLM call after a throttle. Compounds, capped."""
    global _penalty, _penalty_set_at
    with _rate_lock:
        _penalty = min(_current_penalty() * factor, _PENALTY_MAX)
        _penalty_set_at = time.monotonic()
    print(f"[zen_client] backing off x{_penalty:.1f} after throttle", flush=True)


def _throttle() -> None:
    global _last_call_at
    with _rate_lock:
        # Jitter the GAP itself, not just the post-gap sleep below. A fixed
        # 2.0s gap means the request stream has a perfectly regular period
        # regardless of the per-call jitter that follows it -- the same bot
        # signature every other limiter in this pipeline was audited to remove.
        gap = (_MIN_GAP_S + random.uniform(0, 1.0)) * _current_penalty()
        wait = _last_call_at + gap - time.monotonic()
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

    Ollama routing: if `model` matches _is_ollama_model() (e.g.
    "gpt-oss:120b-cloud"), this transparently routes to local Ollama's
    /api/generate instead of opencode.ai -- separate rate limiter, separate
    error counter, same clean-abort-no-retry discipline. Every existing
    caller (pillar_extractors.py etc.) is unaffected unless it's explicitly
    passed an Ollama-style model name.
    """
    if _is_ollama_model(model):
        return _call_ollama_cloud(prompt, model, system or "", max_tokens, timeout)

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
            if resp.status_code in (503, 429):
                # CLEAN ABORT, NO RETRY (user decision, 2026-08-18): retrying
                # after a 429 -- even with backoff -- is still additional load
                # on a host that just told us to stop, and risks worsening
                # whatever cooldown/ban window we're already in. Previously
                # this retried up to `retries` times with an 8/16/24s wait and
                # a decaying penalise() multiplier; that whole soft-recovery
                # path is removed. One 429 now ends the call immediately and
                # feeds the SHARED tripwire signal_agent uses for evidence
                # hosts, so a throttled LLM run is visible the same way a
                # throttled web-fetch run already is -- not a quiet gap.
                try:
                    from agentic_estimation.layer_1.signal_agent import _note_rate_limit
                    _note_rate_limit(ZEN_BASE_URL)   # may raise RateLimitTripped
                except ImportError:
                    pass                            # zen_client is usable standalone
                print(f"[zen_client] HTTP {resp.status_code} from {model} -- "
                      f"aborting immediately, no retry", flush=True)
                return {"ok": False, "raw": "", "error": f"HTTP {resp.status_code} (no retry)",
                        "model_used": None, "latency_s": round(time.perf_counter() - started, 2)}
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
            # The abort signal must never be downgraded to "last_error" and
            # retried -- that is precisely how the tripwire was defeated on the
            # evidence path, and it reappeared here because this file lives
            # outside layer_1/ and the earlier audit never scanned it.
            if type(e).__name__ == "RateLimitTripped":
                raise
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
