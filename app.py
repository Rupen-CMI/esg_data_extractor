import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.v1.key_players.routes import router as key_players_router
from api.v1.esg_data.routes import router as esg_data_router
from api.v1.esg_calculator.routes import router as esg_calculator_router
from api.v1.esg_calculator_v2.routes import router as esg_calculator_v2_router

log = logging.getLogger("app")


# ESG Calculator v2's first /score call was paying ~2.6-2.8s TWICE
# (country_baseline_agent's DB load + exio_lookup's DB load), each a
# one-time psycopg2.connect() + SELECT that then caches in-process for
# every later call. Traced live (2026-09-15): normally invisible after the
# first request, but felt as "every edit is slow" under `uvicorn --reload`,
# which respawns the worker (and every in-memory cache with it) on file
# changes. Running both loads at startup instead moves that unavoidable
# cost off the user's first real request onto server boot -- same total
# work, better-timed. Each is best-effort: a prewarm failure must never
# block the app from starting (the caches still lazy-load correctly on
# first real use either way).
def _prewarm_country_baseline_cache() -> None:
    try:
        from agentic_estimation.layer_1.country_baseline_agent import get_country_baseline_with_fallback
        get_country_baseline_with_fallback("USA")  # any real country forces the one-time cache load
    except Exception:
        log.warning("country baseline pre-warm failed (non-fatal, lazy-loads on first use)", exc_info=True)


def _prewarm_exio_cache() -> None:
    try:
        from agentic_estimation.layer_3.exio_lookup import exio_e_vote
        exio_e_vote("Manufacturing")  # any real sector string forces the one-time cache load
    except Exception:
        log.warning("EXIOBASE pre-warm failed (non-fatal, lazy-loads on first use)", exc_info=True)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import anyio
    # Both loads are blocking DB calls (psycopg2, not asyncpg) -- run off
    # the event loop thread so they don't block it, and in parallel with
    # each other rather than serially (each is ~2.6-2.8s; run together
    # startup only pays the slower of the two, not the sum).
    async with anyio.create_task_group() as tg:
        tg.start_soon(anyio.to_thread.run_sync, _prewarm_country_baseline_cache)
        tg.start_soon(anyio.to_thread.run_sync, _prewarm_exio_cache)
    yield


app = FastAPI(title="ESG Data Extractor", lifespan=_lifespan)

# CORS -- demo_ui/frontend2.0.html's ESG Calculator tab calls /calculator/*
# with a live fetch() from wherever that HTML file happens to be opened
# from (file://, a separate static server, etc.), not necessarily this same
# origin. Wide open ("*") because this is a demo/playground backend with no
# auth and no state-changing side effects, not a production API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(key_players_router)
app.include_router(esg_data_router)
app.include_router(esg_calculator_router)
app.include_router(esg_calculator_v2_router)

# demo_ui/ -- the tabbed pipeline-walkthrough demo (frontend2.0.html, the
# CURRENT design -- demo_ui/index.html is an older page, kept for now but
# not the default). Its ESG Calculator tab is a genuinely live tab calling
# this app's /calculator/* endpoints.
#
# Mounted at /demo (not just relying on "/") so static assets resolve at
# predictable paths (/demo/frontend2.0.js etc.) regardless of what's
# registered at root. /demo bare (no filename) still falls through to
# StaticFiles' own html=True default, which is demo_ui/index.html (the OLD
# page) -- that default can't be pointed at frontend2.0.html without
# renaming files, so instead "/" itself redirects straight to
# frontend2.0.html, sidestepping the ambiguity entirely: '/' is now the one
# blessed "open this and you get the current design" URL.
#
# Routers are registered above this mount, so they still win on exact-path
# matches (e.g. /calculator/score) -- a Mount only catches what nothing
# else claimed first. Safe to mount at "/" for that reason, but /demo is
# kept too so direct links to it keep working.
_demo_ui_dir = Path(__file__).resolve().parent / "demo_ui"
if _demo_ui_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_ui_dir), html=True), name="demo_ui")

    @app.get("/", include_in_schema=False)
    def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/demo/frontend2.0.html")

# app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
