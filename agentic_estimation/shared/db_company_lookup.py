"""
db_company_lookup.py — single shared helper for resolving a company name to
its `companies.id` via asyncpg, used by every CLI entry point that accepts a
bare company name instead of a UUID (orchestrator.py, scoring_agent.py,
evaluator_agent.py, explainability_agent.py -- all `_cli()`/`run_for_company_name`
paths only; the live graph.py/API path always carries company_id already and
never calls this).

Replaces 4 independently-duplicated copies of the same lookup, each of which
used `name ILIKE '%{company}%'` -- a leading wildcard that cannot use
companies.name's existing btree index (full sequential scan on every call),
with no ORDER BY, so an ambiguous match (e.g. "Toyota" vs "Toyota Boshoku
Corporation") picked whichever row Postgres's scan happened to return first,
not a deterministic one.
"""

from __future__ import annotations

import os
from typing import Optional
from uuid import UUID

import asyncpg


def _async_db_url() -> str:
    db_url = os.environ.get("ASYNC_DB_URL", "")
    if not db_url:
        raise RuntimeError("ASYNC_DB_URL not set")
    return db_url.replace("postgresql+asyncpg://", "postgresql://")


async def resolve_company_id(conn: asyncpg.Connection, company_name: str
                              ) -> Optional[tuple[UUID, str]]:
    """(id, canonical_name) for the best match on `company_name`, or None.

    Exact case-insensitive match first (uses companies.name's index, and is
    unambiguous by construction -- at most one row since the column is
    UNIQUE). Falls back to a prefix match (`'name%'`, still index-friendly,
    unlike a leading-wildcard `'%name%'`) ordered by name length then
    alphabetically, so the shortest/most-likely-canonical match wins
    deterministically instead of depending on scan order."""
    row = await conn.fetchrow(
        "SELECT id, name FROM companies WHERE lower(name) = lower($1) LIMIT 1",
        company_name,
    )
    if row is None:
        row = await conn.fetchrow(
            "SELECT id, name FROM companies WHERE name ILIKE $1 "
            "ORDER BY length(name), name LIMIT 1",
            f"{company_name}%",
        )
    if row is None:
        return None
    return UUID(str(row["id"])), row["name"]


async def resolve_company_id_standalone(company_name: str
                                         ) -> Optional[tuple[UUID, str]]:
    """Same as resolve_company_id, but opens and closes its own connection --
    for the CLI entry points, which each run standalone (no pipeline-wide
    connection to share)."""
    conn = await asyncpg.connect(_async_db_url())
    try:
        return await resolve_company_id(conn, company_name)
    finally:
        await conn.close()
