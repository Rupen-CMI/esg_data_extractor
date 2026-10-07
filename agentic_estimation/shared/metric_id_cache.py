"""
metric_id_cache.py — process-wide cache for esg_metric_definitions.key -> id.

esg_metric_definitions is seeded once by migration (001/005) and nothing in
the live pipeline ever inserts a new row into it at runtime -- same kind of
small, static reference data peer_anchor.py and climate_trace_anchor.py
already cache at module scope (_distribution_cache, _owners_cache,
_upright_labels_cache). This table just never got the same treatment:
scoring_agent.py, evaluator_agent.py, and metric_estimation_agent.py each
independently re-ran `SELECT id, key FROM esg_metric_definitions WHERE key =
ANY(...)` on every call.

One process-wide dict, populated from a single query on first use (keyed by
whatever superset of keys the first caller asks for, extended on demand if a
later caller asks for a key not yet cached) -- every subsequent call for an
already-cached key is free.
"""

from __future__ import annotations

from typing import Iterable
from uuid import UUID

import asyncpg

_cache: dict[str, UUID] = {}


async def get_metric_ids(conn: asyncpg.Connection, keys: Iterable[str]) -> dict[str, UUID]:
    """{key: id} for every key in `keys`. Queries the DB only for keys not
    already cached; raises RuntimeError naming any key missing from
    esg_metric_definitions entirely (same contract the 3 call sites this
    replaces already had)."""
    keys = list(keys)
    missing_from_cache = [k for k in keys if k not in _cache]
    if missing_from_cache:
        rows = await conn.fetch(
            "SELECT id, key FROM esg_metric_definitions WHERE key = ANY($1::text[])",
            missing_from_cache,
        )
        for row in rows:
            _cache[row["key"]] = UUID(str(row["id"]))

    result = {k: _cache[k] for k in keys if k in _cache}
    missing = set(keys) - set(result.keys())
    if missing:
        raise RuntimeError(
            f"Missing metric definitions in DB: {missing}. "
            "Run the SQL seed to insert them into esg_metric_definitions."
        )
    return result
