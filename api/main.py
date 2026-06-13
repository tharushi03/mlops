from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_single

# ── App Setup ───────────────────────────────────────────────────
app = FastAPI(
    title="Flood Risk Prediction API",
    description="ML Opsidian Genesis — Flood Risk Score Prediction for Sri Lanka",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema ───────────────────────────────────────────────
class FloodRiskInput(BaseModel):
    district                      : str
    latitude                      : float
    longitude                     : float
    elevation_m                   : float
    distance_to_river_m           : float
    landcover                     : str
    soil_type                     : str
    water_supply                  : str
    electricity                   : str
    road_quality                  : str
    population_density_per_km2    : float
    built_up_percent              : float
    urban_rural                   : str
    rainfall_7d_mm                : float
    monthly_rainfall_mm           : float
    drainage_index                : float
    ndvi                          : float
    ndwi                          : float
    water_presence_flag           : str
    historical_flood_count        : Optional[float] = 0.0
    infrastructure_score          : Optional[float] = 0.5
    nearest_hospital_km           : Optional[float] = 5.0
    nearest_evac_km               : Optional[float] = 3.0
    inundation_area_sqm           : Optional[float] = 5000.0
    seasonal_index                : Optional[float] = 0.5
    terrain_roughness_index       : Optional[float] = 0.5
    socioeconomic_status_index    : Optional[float] = 0.5
    extreme_weather_index         : Optional[float] = 0.5

# ── Response Schema ──────────────────────────────────────────────
class FloodRiskOutput(BaseModel):
    flood_risk_score  : float
    risk_level        : str
    risk_color        : str
    timestamp         : str
    message           : str

# ── Helper ──────────────────────────────────────────────────────
def get_risk_level(score: float):
    if score < 0.3:
        return "Low Risk", "green"
    elif score < 0.5:
        return "Moderate Risk", "yellow"
    elif score < 0.7:
        return "High Risk", "orange"
    else:
        return "Very High Risk", "red"

def log_prediction(input_data: dict, score: float):
    """Log predictions for monitoring."""
    os.makedirs('monitoring/logs', exist_ok=True)
    log_entry = {
        "timestamp" : datetime.now().isoformat(),
        "input"     : input_data,
        "prediction": score
    }
    log_file = f"monitoring/logs/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# ── Routes ───────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message"    : "Flood Risk Prediction API",
        "version"    : "1.0.0",
        "status"     : "running",
        "docs"       : "/docs",
        "health"     : "/health"
    }

@app.get("/health")
def health():
    return {
        "status"    : "healthy",
        "timestamp" : datetime.now().isoformat()
    }

@app.post("/predict", response_model=FloodRiskOutput)
def predict(input_data: FloodRiskInput):
    try:
        # Convert to dict and add derived features
        data = input_data.model_dump()
        data['distance_to_river_m_log1p']        = np.log1p(data['distance_to_river_m'])
        data['population_density_per_km2_log1p'] = np.log1p(data['population_density_per_km2'])
        data['rainfall_7d_mm_log1p']             = np.log1p(data['rainfall_7d_mm'])
        data['monthly_rainfall_mm_log1p']        = np.log1p(data['monthly_rainfall_mm'])
        data['nearest_hospital_km_log1p']        = np.log1p(data['nearest_hospital_km'])
        data['nearest_evac_km_log1p']            = np.log1p(data['nearest_evac_km'])
        data['elevation_m_yeojohnson']           = data['elevation_m'] ** 0.5
        data['drainage_index_yeojohnson']        = data['drainage_index']
        data['ndvi_qmap']                        = data['ndvi']
        data['ndwi_qmap']                        = data['ndwi']
        data['built_up_percent_qmap']            = data['built_up_percent']

        # Get prediction
        score = predict_single(data)
        risk_level, risk_color = get_risk_level(score)

        # Log prediction
        log_prediction(data, score)

        return FloodRiskOutput(
            flood_risk_score = round(score, 4),
            risk_level       = risk_level,
            risk_color       = risk_color,
            timestamp        = datetime.now().isoformat(),
            message          = f"Flood risk assessment complete for {input_data.district} district"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/history")
def prediction_history():
    """Return last 50 predictions for monitoring."""
    log_file = f"monitoring/logs/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    if not os.path.exists(log_file):
        return {"predictions": [], "count": 0}

    predictions = []
    with open(log_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))

    return {
        "predictions" : predictions[-50:],
        "count"       : len(predictions)
    }

@app.get("/model/info")
def model_info():
    return {
        "model_type"     : "CatBoostRegressor",
        "loss_function"  : "MAE",
        "oof_mae"        : 0.17882,
        "public_score"   : 0.38241,
        "features_used"  : 45,
        "training_rows"  : 20886,
        "cv_folds"       : 5,
        "target"         : "flood_risk_score (0-1)"
    }