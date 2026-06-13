import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import mlflow
import mlflow.catboost
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess

def train(train_path='data/train.csv', test_path='data/test.csv'):

    # ── Preprocessing ──────────────────────────────────────────
    X, y, X_test, test_ids = preprocess(train_path, test_path)

    # ── MLflow Experiment ───────────────────────────────────────
    mlflow.set_experiment("flood-risk-prediction")

    with mlflow.start_run(run_name="catboost_mae_slow"):

        params = {
            'iterations'           : 10000,
            'learning_rate'        : 0.005,
            'depth'                : 6,
            'l2_leaf_reg'          : 3,
            'loss_function'        : 'MAE',
            'random_seed'          : 42,
            'verbose'              : 1000,
            'early_stopping_rounds': 300,
        }

        # Log params to MLflow
        mlflow.log_params(params)

        # ── 5-Fold Cross Validation ─────────────────────────────
        kf         = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds  = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        fold_maes  = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

            oof_preds[val_idx] = model.predict(X_val)
            test_preds        += model.predict(X_test) / 5

            fold_mae = mean_absolute_error(y_val, np.clip(oof_preds[val_idx], 0, 1))
            fold_maes.append(fold_mae)
            print(f"Fold {fold+1} MAE: {fold_mae:.5f}")

            # Log each fold score
            mlflow.log_metric(f"fold_{fold+1}_mae", fold_mae)

        # ── Overall Metrics ─────────────────────────────────────
        oof_mae = mean_absolute_error(y, np.clip(oof_preds, 0, 1))
        print(f"\n✅ Overall OOF MAE: {oof_mae:.5f}")

        # Log final metrics
        mlflow.log_metric("oof_mae", oof_mae)
        mlflow.log_metric("oof_mae_std", np.std(fold_maes))

        # ── Save Model ──────────────────────────────────────────
        os.makedirs('models', exist_ok=True)
        model.save_model('models/catboost_model.cbm')
        joblib.dump(test_preds, 'models/test_preds.pkl')
        joblib.dump(oof_preds,  'models/oof_preds.pkl')

        # Log model to MLflow
        mlflow.catboost.log_model(model, "catboost_model")
        mlflow.log_artifact('models/encoders.pkl')
        mlflow.log_artifact('models/medians.pkl')

        print("✅ Model saved to models/catboost_model.cbm")

        # ── Save Submission ─────────────────────────────────────
        sub = pd.read_csv('data/sample_submission.csv')
        sub['flood_risk_score'] = np.clip(test_preds, 0.001, 0.999)
        sub.to_csv('models/submission.csv', index=False)
        mlflow.log_artifact('models/submission.csv')

        print("✅ Submission saved to models/submission.csv")
        print(f"\n📊 MLflow Run ID: {mlflow.active_run().info.run_id}")

    return model, oof_mae, test_preds

if __name__ == '__main__':
    train()