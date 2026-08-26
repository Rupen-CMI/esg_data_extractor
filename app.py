from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.v1.key_players.routes import router as key_players_router
from api.v1.esg_data.routes import router as esg_data_router
from api.v1.esg_calculator.routes import router as esg_calculator_router

app = FastAPI(title="ESG Data Extractor")

app.include_router(key_players_router)
app.include_router(esg_data_router)
app.include_router(esg_calculator_router)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
