# 🌊 Flood Risk Prediction System — ML Opsidian Genesis

ML-powered flood risk assessment for Sri Lanka using production-grade MLOps practices.

## 🏆 Competition
ML Opsidian: Genesis — Final Round  
University of Colombo School of Computing | IEEE Student Branch

## 🔗 Live Demo
- **API:** https://flood-risk-api-y2pz.onrender.com
- **API Docs:** https://flood-risk-api-y2pz.onrender.com/docs
- **Health Check:** https://flood-risk-api-y2pz.onrender.com/health

## 🏗️ System Architecture

Raw Data → Preprocessing Pipeline → CatBoost Model → FastAPI → Frontend UI

↓

MLflow Tracking

↓

Prediction Logging

## 🤖 ML Model
- **Algorithm:** CatBoostRegressor with MAE loss function
- **OOF MAE:** 0.17882
- **Public Leaderboard Score:** 0.38241
- **Cross Validation:** 5-Fold KFold
- **Key Feature:** District-level out-of-fold target encoding

## ⚙️ MLOps Components

| Component | Technology |
|---|---|
| Model Training | CatBoost + 5-Fold CV |
| Experiment Tracking | MLflow |
| API Serving | FastAPI + Uvicorn |
| Containerization | Docker |
| Cloud Deployment | Render |
| Prediction Logging | Custom JSONL logger |
| Frontend | HTML/CSS/JavaScript |

## 📁 Project Structure

ml-opsidian-genesis/
├── src/
│   ├── preprocess.py      # Feature engineering pipeline
│   ├── train.py           # Model training + MLflow tracking
│   ├── predict.py         # Inference logic
│   └── monitor.py         # Monitoring utilities
├── api/
│   └── main.py            # FastAPI endpoints
├── frontend/
│   └── index.html         # Web UI
├── models/                # Trained model artifacts
├── monitoring/            # Prediction logs
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

## 🚀 Setup Instructions

### Local Development
```bash
# Clone the repo
git clone https://github.com/tharushi03/mlops.git
cd mlops

# Install dependencies
pip install -r requirements.txt

# Add data files to data/ folder
# (train.csv, test.csv, sample_submission.csv)

# Train the model
python -m src.train

# Start the API
python -m uvicorn api.main:app --reload

# Open frontend
open frontend/index.html
```

### Docker
```bash
docker-compose up --build
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Get flood risk prediction |
| `/predictions/history` | GET | Recent predictions log |
| `/model/info` | GET | Model metadata |

## 📊 Key Features
- Real-time flood risk scoring (0–1 scale)
- Risk level classification (Low/Moderate/High/Very High)
- Prediction logging for monitoring
- Interactive web UI with visual risk meter
- Production-ready REST API with auto-generated docs
