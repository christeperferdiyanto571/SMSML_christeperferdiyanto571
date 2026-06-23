"""
inference.py
Script untuk melakukan inferensi menggunakan model yang sudah dilatih.
Mendukung single prediction dan batch prediction.

Cara menjalankan:
    # Single prediction
    python inference.py --mode single

    # Batch prediction dari CSV
    python inference.py --mode batch --input data.csv --output predictions.csv

    # Jalankan sebagai API server (FastAPI)
    python inference.py --mode serve
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================================================
# Pydantic schema untuk API input
# ====================================================
class HousingFeatures(BaseModel):
    MedInc:     float = 3.0
    HouseAge:   float = 20.0
    AveRooms:   float = 5.0
    AveBedrms:  float = 1.0
    Population: float = 1000.0
    AveOccup:   float = 3.0
    Latitude:   float = 37.0
    Longitude:  float = -120.0

class PredictionResponse(BaseModel):
    prediction:      float
    unit:            str
    latency_ms:      float
    model_version:   str


# ====================================================
# Load model
# ====================================================
def load_model(model_path: str = "MLProject/artifacts/model"):
    """Load model dari path MLflow artifacts."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Pastikan Anda sudah menjalankan training terlebih dahulu."
        )
    model = mlflow.sklearn.load_model(model_path)
    logger.info(f"Model loaded from: {model_path}")
    return model


# ====================================================
# Prediksi fungsi
# ====================================================
def predict_single(model, features: dict) -> dict:
    """Prediksi untuk 1 sample."""
    df = pd.DataFrame([features])
    start = time.time()
    pred = model.predict(df)[0]
    latency = (time.time() - start) * 1000  # ms

    return {
        "prediction": round(float(pred), 4),
        "unit": "100,000 USD",
        "latency_ms": round(latency, 3),
        "model_version": "1.0.0"
    }


def predict_batch(model, input_csv: str, output_csv: str) -> None:
    """Prediksi batch dari file CSV."""
    df = pd.read_csv(input_csv)

    # Hapus kolom target jika ada
    if 'MedHouseVal' in df.columns:
        df = df.drop('MedHouseVal', axis=1)

    start = time.time()
    predictions = model.predict(df)
    latency = time.time() - start

    df['MedHouseVal_predicted'] = predictions
    df.to_csv(output_csv, index=False)

    logger.info(f"Batch prediction completed: {len(df)} samples in {latency:.3f}s")
    logger.info(f"Results saved to: {output_csv}")
    print(f"Predictions saved to {output_csv}")
    print(f"Sample predictions:\n{df['MedHouseVal_predicted'].head()}")


# ====================================================
# FastAPI server
# ====================================================
def create_app(model_path: str = "MLProject/artifacts/model") -> FastAPI:
    app = FastAPI(
        title="Housing Price Prediction API",
        description="API untuk prediksi harga rumah menggunakan Random Forest model",
        version="1.0.0"
    )

    model = load_model(model_path)

    @app.get("/")
    def root():
        return {"message": "Housing Price Prediction API is running", "status": "healthy"}

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": "RandomForestRegressor", "version": "1.0.0"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: HousingFeatures):
        try:
            result = predict_single(model, features.dict())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch")
    def predict_batch_endpoint(data: list):
        try:
            df = pd.DataFrame(data)
            predictions = model.predict(df).tolist()
            return {"predictions": predictions, "count": len(predictions)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ====================================================
# CLI
# ====================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Housing Price Prediction Inference")
    parser.add_argument('--mode',       choices=['single', 'batch', 'serve'], default='single')
    parser.add_argument('--model_path', default='MLProject/artifacts/model')
    parser.add_argument('--input',      default='housing_preprocessing/test.csv')
    parser.add_argument('--output',     default='predictions_output.csv')
    parser.add_argument('--host',       default='0.0.0.0')
    parser.add_argument('--port',       type=int, default=8080)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'single':
        model = load_model(args.model_path)
        sample = {
            "MedInc": 5.0, "HouseAge": 25.0, "AveRooms": 6.0,
            "AveBedrms": 1.2, "Population": 1500.0, "AveOccup": 3.0,
            "Latitude": 37.5, "Longitude": -122.0
        }
        result = predict_single(model, sample)
        print("\n=== Single Prediction ===")
        print(json.dumps(result, indent=2))

    elif args.mode == 'batch':
        model = load_model(args.model_path)
        predict_batch(model, args.input, args.output)

    elif args.mode == 'serve':
        app = create_app(args.model_path)
        logger.info(f"Starting API server at http://{args.host}:{args.port}")
        logger.info("API docs available at http://localhost:8080/docs")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
