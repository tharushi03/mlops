import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

DROP_COLS = [
    'record_id', 'place_name', 'generation_date',
    'reason_not_good_to_live',
    'flood_occurrence_current_event',
    'is_good_to_live',
    'is_synthetic',
]

CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply',
    'electricity', 'road_quality', 'urban_rural', 'water_presence_flag'
]

def load_model():
    model = CatBoostRegressor()
    model.load_model('models/catboost_model.cbm')
    encoders = joblib.load('models/encoders.pkl')
    medians  = joblib.load('models/medians.pkl')
    return model, encoders, medians

def predict_single(input_dict: dict):
    """Predict flood risk for a single location input dict."""
    model, encoders, medians = load_model()

    df = pd.DataFrame([input_dict])

    # Drop unused cols if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

    # Feature engineering
    from src.preprocess import (add_geography_features,
                                 add_rainfall_features,
                                 add_v3_features)
    df = add_geography_features(df)
    df = add_rainfall_features(df)
    df = add_v3_features(df)

    # District target encoding — use global mean as fallback
    df['district_target_enc'] = 0.5

    # Label encode
    for col in CAT_COLS:
        if col in df.columns:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )

    # Fill nulls with saved medians
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)

    prediction = model.predict(df)
    return float(np.clip(prediction[0], 0.001, 0.999))

if __name__ == '__main__':
    # Test with sample input
    sample = {
        'district'                    : 'Colombo',
        'latitude'                    : 6.9271,
        'longitude'                   : 79.8612,
        'elevation_m'                 : 10.0,
        'distance_to_river_m'         : 300.0,
        'landcover'                   : 'Urban',
        'soil_type'                   : 'Clay',
        'water_supply'                : 'Municipal',
        'electricity'                 : 'Grid',
        'road_quality'                : 'Good (paved)',
        'population_density_per_km2'  : 3500.0,
        'built_up_percent'            : 80.0,
        'urban_rural'                 : 'Urban',
        'rainfall_7d_mm'              : 120.0,
        'monthly_rainfall_mm'         : 300.0,
        'drainage_index'              : 0.4,
        'ndvi'                        : 0.2,
        'ndwi'                        : 0.5,
        'water_presence_flag'         : 'Likely',
        'historical_flood_count'      : 3.0,
        'infrastructure_score'        : 0.6,
        'nearest_hospital_km'         : 2.0,
        'nearest_evac_km'             : 1.5,
        'inundation_area_sqm'         : 5000,
        'seasonal_index'              : 0.7,
        'terrain_roughness_index'     : 0.3,
        'socioeconomic_status_index'  : 0.5,
        'extreme_weather_index'       : 0.8,
        'distance_to_river_m_log1p'   : np.log1p(300.0),
        'population_density_per_km2_log1p': np.log1p(3500.0),
        'rainfall_7d_mm_log1p'        : np.log1p(120.0),
        'monthly_rainfall_mm_log1p'   : np.log1p(300.0),
        'nearest_hospital_km_log1p'   : np.log1p(2.0),
        'nearest_evac_km_log1p'       : np.log1p(1.5),
        'elevation_m_yeojohnson'      : 2.5,
        'drainage_index_yeojohnson'   : 0.4,
        'ndvi_qmap'                   : 0.2,
        'ndwi_qmap'                   : 0.5,
        'built_up_percent_qmap'       : 80.0,
    }
    result = predict_single(sample)
    print(f"Predicted flood risk score: {result:.4f}")