import json
import os
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

LOG_DIR = 'monitoring/logs'

def get_log_file(date=None):
    if date is None:
        date = datetime.now().strftime('%Y%m%d')
    return f"{LOG_DIR}/predictions_{date}.jsonl"

def load_predictions(days=7):
    """Load predictions from the last N days."""
    all_preds = []
    for i in range(days):
        from datetime import timedelta
        date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        log_file = get_log_file(date)
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        all_preds.append(json.loads(line))
                    except:
                        continue
    return all_preds

def get_summary_stats():
    """Get summary statistics of predictions."""
    preds = load_predictions()
    if not preds:
        return {
            "total_predictions" : 0,
            "avg_risk_score"    : None,
            "high_risk_count"   : 0,
            "low_risk_count"    : 0,
            "message"           : "No predictions yet"
        }

    scores = [p['prediction'] for p in preds]
    districts = [p['input'].get('district', 'Unknown') for p in preds]

    high_risk  = sum(1 for s in scores if s >= 0.7)
    low_risk   = sum(1 for s in scores if s < 0.3)
    mod_risk   = sum(1 for s in scores if 0.3 <= s < 0.7)

    # District breakdown
    district_scores = defaultdict(list)
    for p in preds:
        d = p['input'].get('district', 'Unknown')
        district_scores[d].append(p['prediction'])

    district_avg = {
        d: round(float(np.mean(scores)), 4)
        for d, scores in district_scores.items()
    }

    return {
        "total_predictions"  : len(scores),
        "avg_risk_score"     : round(float(np.mean(scores)), 4),
        "max_risk_score"     : round(float(np.max(scores)), 4),
        "min_risk_score"     : round(float(np.min(scores)), 4),
        "std_risk_score"     : round(float(np.std(scores)), 4),
        "high_risk_count"    : high_risk,
        "moderate_risk_count": mod_risk,
        "low_risk_count"     : low_risk,
        "district_breakdown" : district_avg,
        "last_prediction_at" : preds[-1]['timestamp'] if preds else None
    }

def check_data_drift(current_input: dict):
    """Simple drift detection — flag if input is outside training ranges."""
    # Training data approximate ranges
    EXPECTED_RANGES = {
        'elevation_m'              : (0, 2500),
        'distance_to_river_m'      : (0, 50000),
        'rainfall_7d_mm'           : (0, 500),
        'monthly_rainfall_mm'      : (0, 1000),
        'ndwi'                     : (-1, 1),
        'ndvi'                     : (-1, 1),
        'drainage_index'           : (0, 1),
        'population_density_per_km2': (0, 15000),
    }

    drift_warnings = []
    for feature, (min_val, max_val) in EXPECTED_RANGES.items():
        if feature in current_input:
            val = current_input[feature]
            if val < min_val or val > max_val:
                drift_warnings.append({
                    "feature"  : feature,
                    "value"    : val,
                    "expected" : f"{min_val} to {max_val}"
                })

    return {
        "drift_detected" : len(drift_warnings) > 0,
        "warnings"       : drift_warnings
    }

if __name__ == '__main__':
    stats = get_summary_stats()
    print(json.dumps(stats, indent=2))