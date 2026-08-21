"""Disk cache for outbound evidence fetches.

THE PROBLEM IT SOLVES: nothing cached HTTP responses, so every run re-fetched
everything. A 150-company gather spends ~25s per company in DDG alone (9
DDG-backed sources through one 2.75s-average limiter), and a crash at company
140 re-paid for the first 139 on restart. Re-running a corpus to compare two
scoring configurations paid the entire request bill a second time, against
hosts that had already throttled us once.

The cheapest request is the one never sent. This is the single biggest
reduction in rate-limit exposure available, because it attacks request VOLUME
rather than pacing.

WHAT IS CACHED: successful bodies only.

Errors are deliberately NOT cached. A 429/503 is a statement about our request
RATE, not about the resource -- caching it would turn a transient throttle into
a permanent hole in the corpus, and the tripwire would stop seeing the throttles
it exists to count. A 404 is likewise skipped: Wikipedia slug probing generates
them by design, and a company's page may appear later.

TTL, not permanence: evidence goes stale, and a cache with no expiry silently
freezes a corpus at whatever the web looked like the first time it ran.

NOT SHARED WITH THE PREDICTION CACHE in calibration_harness. That one stores
scores; this one stores bytes. Keeping them separate means clearing one to
force a re-score does not also discard hours of fetching.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("http_cache")

_CACHE_DIR = Path(os.getenv("ESG_HTTP_CACHE_DIR", "raw_esg_data/http_cache"))
_TTL_S = int(os.getenv("ESG_HTTP_CACHE_TTL", str(14 * 24 * 3600)))
_ENABLED = os.getenv("ESG_HTTP_CACHE", "1") != "0"

# Bodies above this are not worth the disk round-trip for the sources here
# (search snippets, RSS, small JSON). PDFs go through report_coverage, which
# has its own on-disk store.
_MAX_BODY = 4 * 1024 * 1024

_stats = {"hit": 0, "miss": 0, "store": 0, "skip": 0}
_lock = threading.Lock()


def _key(namespace: str, url: str, params: Optional[dict] = None) -> str:
    """Stable key. Params are sorted so dict ordering cannot split the cache."""
    blob = f"{namespace}|{url}|{json.dumps(params or {}, sort_keys=True, default=str)}"
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _path(key: str) -> Path:
    # Two-level fan-out: a flat directory with tens of thousands of entries is
    # slow to stat on Windows.
    return _CACHE_DIR / key[:2] / f"{key}.json"


def get(namespace: str, url: str, params: Optional[dict] = None) -> Optional[str]:
    """Cached body, or None on miss/expiry."""
    if not _ENABLED:
        return None
    p = _path(_key(namespace, url, params))
    if not p.exists():
        with _lock:
            _stats["miss"] += 1
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None                     # corrupt entry -> treat as a miss
    if time.time() - rec.get("t", 0) > _TTL_S:
        with _lock:
            _stats["miss"] += 1
        return None
    with _lock:
        _stats["hit"] += 1
    return rec.get("body")


def put(namespace: str, url: str, body: str, params: Optional[dict] = None) -> None:
    """Store a SUCCESSFUL body. Callers must not pass error responses."""
    if not _ENABLED or body is None:
        return
    if len(body) > _MAX_BODY:
        with _lock:
            _stats["skip"] += 1
        return
    p = _path(_key(namespace, url, params))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"t": time.time(), "url": url, "body": body}),
                       encoding="utf-8")
        tmp.replace(p)                  # atomic: a crash mid-write cannot
                                        # leave a truncated entry that later
                                        # reads as valid JSON
        with _lock:
            _stats["store"] += 1
    except Exception as exc:
        log.debug("cache write failed for %s: %s", url[:80], exc)


def stats() -> dict:
    with _lock:
        s = dict(_stats)
    total = s["hit"] + s["miss"]
    s["hit_rate"] = round(100.0 * s["hit"] / total, 1) if total else 0.0
    return s


def report() -> str:
    s = stats()
    return (f"http cache: {s['hit']} hits / {s['miss']} misses "
            f"({s['hit_rate']}%), {s['store']} stored")
