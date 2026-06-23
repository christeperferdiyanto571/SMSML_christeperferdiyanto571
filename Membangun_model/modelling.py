"""
modelling.py
Melatih model Random Forest Regressor menggunakan MLflow autolog
untuk dataset California Housing yang sudah dipreprocessing.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Konfigurasi MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Housing_Price_Prediction"

def load_data(data_dir: str = "housing_preprocessing"):
    """Load preprocessed train dan test data."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    X_test = test_df.drop('MedHouseVal', axis=1)
    y_test = test_df['MedHouseVal']

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    """Melatih model menggunakan MLflow autolog."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        logger.info("Training RandomForestRegressor...")

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"MSE:  {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE:  {mae:.4f}")
        logger.info(f"R2:   {r2:.4f}")

        print(f"\n=== Model Evaluation ===")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, X_test, y_train, y_test)
    print("\nTraining completed! Check MLflow UI at http://127.0.0.1:5000")
