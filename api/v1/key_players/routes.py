from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from database import get_db

from .services import fetch_by_market, process_batch, run_pipeline, seed_markets

router = APIRouter(prefix="/key_players", tags=["KEY PLAYERS"])


@router.get("/health")
def get_health():
    return {"status": "running!"}


@router.post("/")
async def get_key_players():
    print("Processing single batch \n\n")
    response = await process_batch()
    return {"has_more": response}

@router.post("/get_players_by_market")
async def get_players_by_market(request: Request):
    data = await request.json()

    market_name = data.get("market_name")
    response = await fetch_by_market(market_name)

    return response


@router.post("/start-pipeline", status_code=202)
def start_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline)
    return {"status": "pipeline started"}


@router.post("/seed-markets", status_code=201)
def seed_markets_endpoint(db: Session = Depends(get_db)):
    try:
        result = seed_markets(db)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))