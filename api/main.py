"""
FastAPI service for the NEO Hazard Predictor.

Endpoints:
- GET  /                -> health check
- POST /predict         -> predict hazard for manually provided features
- GET  /predict/live     -> fetch live asteroids from NASA and predict on them
"""

from datetime import date, timedelta
import traceback

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nasa_client import get_live_asteroids

app = FastAPI(title="NEO Hazard Predictor API")

# Allow the dashboard (hosted on Vercel) to call this API from the browser.
# Wide open for demo purposes — tighten to your exact Vercel domain later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts once at startup
model = joblib.load("model/neo_model.pkl")
scaler = joblib.load("model/neo_scaler.pkl")
feature_order = joblib.load("model/neo_feature_order.pkl")


class NeoFeatures(BaseModel):
    est_diameter_min: float
    est_diameter_max: float
    relative_velocity: float
    miss_distance: float
    absolute_magnitude: float


def predict_one(features: dict) -> dict:
    """Runs the model on a single dict of the 5 features."""
    row = [[features[col] for col in feature_order]]
    scaled = scaler.transform(row)

    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    return {
        "hazardous": bool(prediction),
        "hazard_probability": round(probability, 4),
    }


@app.get("/")
def health_check():
    return {"status": "ok", "model": "Random Forest (tuned)", "features": feature_order}


@app.post("/predict")
def predict(features: NeoFeatures):
    """Predict hazard status from manually provided asteroid features."""
    result = predict_one(features.dict())
    return {**features.dict(), **result}


@app.get("/predict/live")
def predict_live(start_date: str = None, end_date: str = None):
    """
    Fetch real asteroids from NASA's API for a date range and predict
    hazard status for each. Defaults to today through +6 days if no
    dates are given (NASA's feed endpoint allows a max 7-day window).
    """
    if start_date is None:
        start_date = date.today().isoformat()
    if end_date is None:
        end_date = (date.today() + timedelta(days=6)).isoformat()

    try:
        asteroids = get_live_asteroids(start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NASA API error: {e}")

    results = []
    for a in asteroids:
        try:
            pred = predict_one(a)
            results.append({**a, **pred})
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error on asteroid {a.get('id')}: {e} | features={a} | traceback={traceback.format_exc()}",
            )

    # Most likely hazardous first
    results.sort(key=lambda r: r["hazard_probability"], reverse=True)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "count": len(results),
        "asteroids": results,
    }
