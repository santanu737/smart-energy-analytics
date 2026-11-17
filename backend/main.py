"""FastAPI app: JSON API under /api plus the static dashboard at /.

Start:  python -m uvicorn backend.main:app --reload --port 8000
Docs :  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.auth import (authenticate_user, create_access_token, get_current_user,
                          security_warnings)
from backend.schemas import LoginRequest, TokenResponse
from backend.services import HORIZONS, DataService, NotAvailable
from src import config

service: DataService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    for w in security_warnings():
        print(f"WARNING: {w}")
    service = DataService()
    yield


app = FastAPI(title="Smart Campus Energy Analytics API", version="1.0.0", lifespan=lifespan)

origins = os.getenv("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins],
                   allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])


@app.exception_handler(NotAvailable)
async def not_available_handler(request: Request, exc: NotAvailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def svc() -> DataService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service is still loading")
    return service


def ts(value: str | None, name: str) -> pd.Timestamp | None:
    """Parse a date or datetime query parameter, or answer 422."""
    if value in (None, ""):
        return None
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail=f"'{name}' is not a valid date: {value!r}")


def resolution_check(resolution: str) -> str:
    if resolution not in ("hourly", "daily"):
        raise HTTPException(status_code=422, detail="resolution must be 'hourly' or 'daily'")
    return resolution


auth = [Depends(get_current_user)]


# ------------------------------------------------------------------ open
@app.get("/api/health")
def health():
    loaded = service.loaded if service else {}
    return {"status": "ok" if service else "loading", "campus": config.CAMPUS_SITE_ID, "loaded": loaded}


@app.post("/api/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"access_token": create_access_token(user["username"]), "token_type": "bearer"}


# ------------------------------------------------------------- protected
@app.get("/api/campus/summary", dependencies=auth)
def campus_summary(start: str | None = None, end: str | None = None):
    r = svc().get_summary(ts(start, "start"), ts(end, "end"))
    if not r:
        raise HTTPException(status_code=404, detail="No data in that date range")
    return r


@app.get("/api/campus/history", dependencies=auth)
def campus_history(start: str | None = None, end: str | None = None, resolution: str = "hourly"):
    return svc().get_campus_history(ts(start, "start"), ts(end, "end"), resolution_check(resolution))


@app.get("/api/buildings", dependencies=auth)
def buildings():
    return svc().get_buildings()


@app.get("/api/buildings/{building_id}/history", dependencies=auth)
def building_history(building_id: str, start: str | None = None, end: str | None = None,
                     resolution: str = "hourly"):
    s = svc()
    if not s.has_building(building_id):
        raise HTTPException(status_code=404, detail=f"Unknown building: {building_id}")
    return s.get_building_history(building_id, ts(start, "start"), ts(end, "end"), resolution_check(resolution))


@app.get("/api/buildings/{building_id}/profile", dependencies=auth)
def building_profile(building_id: str):
    s = svc()
    if not s.has_building(building_id):
        raise HTTPException(status_code=404, detail=f"Unknown building: {building_id}")
    return s.get_building_profile(building_id)


@app.get("/api/forecast", dependencies=auth)
def forecast(horizon: int = 1, as_of: str | None = None, points: int = Query(48, ge=1, le=336)):
    if horizon not in HORIZONS:
        raise HTTPException(status_code=422, detail=f"horizon must be one of {list(HORIZONS)}")
    r = svc().get_forecast(horizon, ts(as_of, "as_of"), points)
    if not r:
        raise HTTPException(status_code=404, detail="as_of is after the last available hour")
    return r


@app.get("/api/anomalies", dependencies=auth)
def anomalies(building_id: str | None = None, start: str | None = None, end: str | None = None,
              source: str | None = None, limit: int = Query(50, ge=1, le=1000)):
    if source not in (None, "isolation_forest", "zero", "flat_run"):
        raise HTTPException(status_code=422, detail="source must be isolation_forest, zero or flat_run")
    return svc().get_anomalies(building_id, ts(start, "start"), ts(end, "end"), source, limit)


@app.get("/api/ranking", dependencies=auth)
def ranking(metric: str = "total", limit: int = Query(10, ge=1, le=200)):
    if metric not in ("total", "eui", "baseload"):
        raise HTTPException(status_code=422, detail="metric must be total, eui or baseload")
    return svc().get_ranking(metric, limit)


@app.get("/api/usage-types", dependencies=auth)
def usage_types():
    return svc().get_usage_types()


@app.get("/api/weather", dependencies=auth)
def weather(start: str | None = None, end: str | None = None, resolution: str = "hourly"):
    return svc().get_weather(ts(start, "start"), ts(end, "end"), resolution_check(resolution))


@app.get("/api/cost-carbon", dependencies=auth)
def cost_carbon(start: str | None = None, end: str | None = None,
                tariff: float = Query(config.TARIFF_PER_KWH, ge=0),
                co2_factor: float = Query(config.CO2_KG_PER_KWH, ge=0)):
    r = svc().get_summary(ts(start, "start"), ts(end, "end"))
    if not r:
        raise HTTPException(status_code=404, detail="No data in that date range")
    kwh = r["total_kwh"]
    return {"total_kwh": kwh, "tariff": tariff, "co2_factor": co2_factor,
            "estimated_cost": kwh * tariff, "estimated_co2_kg": kwh * co2_factor,
            "note": "Tariff and CO2 factor are illustrative defaults."}


@app.get("/api/model-info", dependencies=auth)
def model_info():
    return svc().get_model_info()


@app.get("/api/replay/next", dependencies=auth)
def replay_next():
    return svc().get_replay_next()


@app.post("/api/replay/reset", dependencies=auth)
def replay_reset():
    return svc().reset_replay()


# The dashboard. Mounted last so it never hides an /api route.
if (config.PROJECT_ROOT / "frontend").exists():
    app.mount("/", StaticFiles(directory=str(config.PROJECT_ROOT / "frontend"), html=True), name="frontend")
