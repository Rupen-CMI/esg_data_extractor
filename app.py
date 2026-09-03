from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.v1.key_players.routes import router as key_players_router
from api.v1.esg_data.routes import router as esg_data_router
from api.v1.esg_calculator.routes import router as esg_calculator_router

app = FastAPI(title="ESG Data Extractor")

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
