"""
modelling_tuning.py
Melatih model dengan hyperparameter tuning menggunakan Optuna,
manual logging MLflow, dan integrasi DagsHub.
Kriteria Advanced: manual logging + minimal 2 artefak tambahan.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI - Sesuaikan dengan akun DagsHub Anda
# ============================================================
DAGSHUB_OWNER = "christeperferdiyanto571"
DAGSHUB_REPO  = "SMSML_Chris-Teper-Ferdiyanto" # Ganti dengan nama repo DagsHub
EXPERIMENT_NAME = "Housing_Price_Tuning"
# ============================================================


def load_data(data_dir: str = "housing_preprocessing"):
    """Load preprocessed train dan test data."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))

    X_train = train_df.drop('MedHouseVal', axis=1)
    y_train = train_df['MedHouseVal']
    X_test  = test_df.drop('MedHouseVal', axis=1)
    y_test  = test_df['MedHouseVal']

    logger.info(f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ---- Artefak 1: Feature Importance Plot ----
def save_feature_importance_plot(model, feature_names: list, path: str = "feature_importance.png"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved: {path}")
    return path


# ---- Artefak 2: Prediction vs Actual Plot ----
def save_prediction_plot(y_test, y_pred, path: str = "prediction_vs_actual.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, s=10, color='steelblue')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Prediction vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Prediction plot saved: {path}")
    return path


# ---- Artefak 3: Residual Distribution Plot ----
def save_residual_plot(y_test, y_pred, path: str = "residual_distribution.png"):
    residuals = y_test.values - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=50, kde=True, color='coral')
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel("Residuals")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Residual plot saved: {path}")
    return path


def objective(trial, X_train, y_train):
    """Optuna objective function untuk hyperparameter tuning."""
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
        'max_depth':         trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': -1,
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
    return scores.mean()


def train_with_tuning(X_train, X_test, y_train, y_test, n_trials: int = 30):
    """Hyperparameter tuning + manual MLflow logging ke DagsHub."""

    # Init DagsHub + MLflow
    dagshub.init(repo_owner=DAGSHUB_OWNER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Hyperparameter Tuning dengan Optuna ---
    logger.info("Starting hyperparameter tuning with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info(f"Best params: {best_params}")

    # --- Training dengan parameter terbaik + MLflow manual logging ---
    with mlflow.start_run(run_name="RandomForest_Tuned_Advanced"):

        best_params['random_state'] = 42
        best_params['n_jobs'] = -1

        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Metrik ---
        mse    = mean_squared_error(y_test, y_pred)
        rmse   = np.sqrt(mse)
        mae    = mean_absolute_error(y_test, y_pred)
        r2     = r2_score(y_test, y_pred)
        mape   = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-10))) * 100
        cv_r2  = cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()

        # Manual logging — params
        mlflow.log_params(best_params)
        mlflow.log_param("n_trials_optuna", n_trials)

        # Manual logging — metrics (autolog + tambahan)
        mlflow.log_metric("mse",    mse)
        mlflow.log_metric("rmse",   rmse)
        mlflow.log_metric("mae",    mae)
        mlflow.log_metric("r2",     r2)
        mlflow.log_metric("mape",   mape)
        mlflow.log_metric("cv_r2",  cv_r2)

        # Manual logging — model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ---- Artefak Tambahan 1: Feature Importance Plot ----
        fi_path = save_feature_importance_plot(model, list(X_train.columns))
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # ---- Artefak Tambahan 2: Prediction vs Actual Plot ----
        pred_path = save_prediction_plot(y_test, y_pred)
        mlflow.log_artifact(pred_path, artifact_path="plots")

        # ---- Artefak Tambahan 3: Residual Distribution ----
        resid_path = save_residual_plot(y_test, y_pred)
        mlflow.log_artifact(resid_path, artifact_path="plots")

        # ---- Artefak Tambahan 4: Best Params JSON ----
        params_path = "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact(params_path)

        # ---- Artefak Tambahan 5: Optuna Trials CSV ----
        trials_df = study.trials_dataframe()
        trials_path = "optuna_trials.csv"
        trials_df.to_csv(trials_path, index=False)
        mlflow.log_artifact(trials_path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        print(f"\n=== Best Model Evaluation ===")
        print(f"MSE:    {mse:.4f}")
        print(f"RMSE:   {rmse:.4f}")
        print(f"MAE:    {mae:.4f}")
        print(f"R2:     {r2:.4f}")
        print(f"MAPE:   {mape:.4f}%")
        print(f"CV R2:  {cv_r2:.4f}")

    return model, best_params


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model, best_params = train_with_tuning(X_train, X_test, y_train, y_test, n_trials=30)
    print("\nTraining with tuning completed!")
    print("Check DagsHub MLflow UI for experiment tracking.")
