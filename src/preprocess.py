import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
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

def add_geography_features(df):
    df = df.copy()
    df['low_elevation_flag']   = (df['elevation_m'] < 50).astype(float)
    df['elevation_risk']       = 1 / (df['elevation_m'].clip(lower=1))
    df['near_river_flag']      = (df['distance_to_river_m'] < 500).astype(float)
    df['river_proximity_risk'] = 1 / (df['distance_to_river_m'].clip(lower=1))
    df['flat_terrain_flag']    = (df['terrain_roughness_index'] < df['terrain_roughness_index'].median()).astype(float)
    df['low_elev_near_river']  = df['low_elevation_flag'] * df['river_proximity_risk']
    return df

def add_rainfall_features(df):
    df = df.copy()
    df['rain7d_x_low_elev']       = df['rainfall_7d_mm'] * df['low_elevation_flag']
    df['monthly_rain_x_low_elev'] = df['monthly_rainfall_mm'] * df['low_elevation_flag']
    df['rain7d_x_near_river']     = df['rainfall_7d_mm'] * df['near_river_flag']
    df['rain7d_x_drainage']       = df['rainfall_7d_mm'] / (df['drainage_index'].clip(lower=0.01))
    df['rainfall_spike_ratio']    = df['rainfall_7d_mm'] / (df['monthly_rainfall_mm'].clip(lower=0.1))
    df['rain_x_extreme_weather']  = df['rainfall_7d_mm'] * df['extreme_weather_index']
    df['rainfall_spike_log']      = np.log1p(df['rainfall_spike_ratio'])
    return df

def add_v3_features(df):
    df = df.copy()
    df['ndwi_x_inundation']  = df['ndwi'] * df['inundation_area_sqm']
    df['rain_x_inundation']  = df['monthly_rainfall_mm'] * df['inundation_area_sqm']
    df['pop_x_inundation']   = df['population_density_per_km2'] * df['inundation_area_sqm']
    df['river_x_ndwi']       = df['distance_to_river_m'] * df['ndwi']
    df['terrain_x_extreme']  = df['terrain_roughness_index'] * df['extreme_weather_index']
    return df

def add_target_encoding(X_train, y_train, X_test):
    global_mean = y_train.mean()
    kf_te = KFold(n_splits=5, shuffle=True, random_state=0)
    X_train['district_target_enc'] = 0.0
    X_test['district_target_enc']  = 0.0

    for tr_idx, val_idx in kf_te.split(X_train):
        X_tr_fold = X_train.iloc[tr_idx].copy()
        y_tr_fold = y_train.iloc[tr_idx]
        dist_mean = y_tr_fold.groupby(X_tr_fold['district']).mean()
        X_train.iloc[val_idx, X_train.columns.get_loc('district_target_enc')] = \
            X_train.iloc[val_idx]['district'].map(dist_mean).fillna(global_mean)

    dist_mean_full = y_train.groupby(X_train['district']).mean()
    X_test['district_target_enc'] = X_test['district'].map(dist_mean_full).fillna(global_mean)

    return X_train, X_test

def preprocess(train_path, test_path, save_encoders=True):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    y        = train['flood_risk_score']
    test_ids = test['record_id']

    X      = train.drop(columns=DROP_COLS + ['flood_risk_score'])
    X_test = test.drop(columns=DROP_COLS)

    # Feature engineering
    X      = add_geography_features(X)
    X_test = add_geography_features(X_test)
    X      = add_rainfall_features(X)
    X_test = add_rainfall_features(X_test)
    X      = add_v3_features(X)
    X_test = add_v3_features(X_test)

    # Target encoding
    X, X_test = add_target_encoding(X, y, X_test)

    # Label encode categoricals
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X[col]      = le.transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    # Fill nulls
    medians = {}
    for col in X.columns:
        if X[col].isnull().any():
            median_val  = X[col].median()
            X[col]      = X[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
            medians[col] = median_val

    if save_encoders:
        os.makedirs('models', exist_ok=True)
        joblib.dump(encoders, 'models/encoders.pkl')
        joblib.dump(medians,  'models/medians.pkl')

    print(f"✅ Preprocessing done! X shape: {X.shape}")
    return X, y, X_test, test_ids

if __name__ == '__main__':
    preprocess('data/train.csv', 'data/test.csv')