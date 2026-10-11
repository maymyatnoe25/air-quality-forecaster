from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import get_forecast

app = FastAPI(title="Malaysia Air Quality Forecaster", version="1.0.0")


class Point(BaseModel):
    horizon_h: int
    target_time: str
    pm25: float
    low: float
    high: float
    category: str
    alert: str


class ForecastResponse(BaseModel):
    city: str
    issued_at: str
    current_pm25: float
    current_category: str
    forecasts: list[Point]


@lru_cache(maxsize=1)
def _cached(hour_key: str):
    return get_forecast()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast", response_model=ForecastResponse)
def forecast():
    try:
        return _cached(str(pd.Timestamp.now(tz="UTC").floor("h")))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Forecast unavailable: {e}")