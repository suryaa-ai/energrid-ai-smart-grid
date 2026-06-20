from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
import datetime
import os

# Initialize FastAPI
app = FastAPI(title="EnerGrid AI API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key (read from environment or use default)
API_KEY = os.getenv("ENERGRID_API_KEY", "test-api-key-123")

# Verify API Key from Header (X-API-Key)
def verify_api_key(api_key: str = Header(..., alias="X-API-Key")):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Load models once at startup
forecasting_model = tf.keras.models.load_model('models/cnn_lstm_real.h5', compile=False)
autoencoder = tf.keras.models.load_model('models/autoencoder_real.h5', compile=False)
scaler = joblib.load('data/processed/scaler_real.pkl')

# Helper functions
def get_recent_predictions(window=24):
    X_test = np.load('data/processed/X_test_real.npy')
    y_test = np.load('data/processed/y_test_real.npy')
    X_window = X_test[-window:]
    pred_norm = forecasting_model.predict(X_window, verbose=0).flatten()
    load_min = scaler.min_[0]
    load_max = scaler.scale_[0] + load_min
    pred_denorm = pred_norm * (load_max - load_min) + load_min
    actual_denorm = y_test[-window:] * (load_max - load_min) + load_min
    timestamps = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(window-1, -1, -1)]
    return {
        "timestamps": [ts.isoformat() for ts in timestamps],
        "actual": actual_denorm.tolist(),
        "predicted": pred_denorm.tolist()
    }

def get_future_forecast(horizon=24):
    X_test = np.load('data/processed/X_test_real.npy')
    last_24_seq = X_test[-24:]
    if horizon > 24:
        horizon = 24
    pred_norm = forecasting_model.predict(last_24_seq[:horizon], verbose=0).flatten()
    load_min = scaler.min_[0]
    load_max = scaler.scale_[0] + load_min
    pred_denorm = pred_norm * (load_max - load_min) + load_min
    timestamps = [datetime.datetime.now() + datetime.timedelta(hours=i+1) for i in range(horizon)]
    return {
        "timestamps": [ts.isoformat() for ts in timestamps],
        "forecast": pred_denorm.tolist()
    }

def get_anomalies():
    X_normal_test = np.load('data/processed/X_normal_test_real.npy')
    X_anomaly_test = np.load('data/processed/X_anomaly_test_real.npy')
    X_all = np.vstack([X_normal_test, X_anomaly_test])
    reconstructions = autoencoder.predict(X_all, verbose=0)
    errors = np.mean((X_all - reconstructions) ** 2, axis=1)
    normal_errors = errors[:len(X_normal_test)]
    threshold = normal_errors.mean() + 2 * normal_errors.std()
    anomaly_indices = np.where(errors > threshold)[0]
    return {
        "anomaly_count": int(len(anomaly_indices)),
        "threshold": float(threshold),
        "anomaly_indices": anomaly_indices.tolist()
    }

def get_metrics():
    df = pd.read_csv('data/raw/PJM_Load_hourly.csv', parse_dates=['Datetime'])
    df.rename(columns={'Datetime': 'timestamp', 'PJM_Load_MW': 'load_MW'}, inplace=True)
    return {
        "total_energy_gwh": float(df['load_MW'].sum() / 1000),
        "avg_load_mw": float(df['load_MW'].mean()),
        "peak_load_mw": float(df['load_MW'].max()),
        "anomaly_rate_percent": 4.5
    }

# API Endpoints
@app.get("/api/forecast", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def forecast(request: Request, horizon: int = 24):
    recent = get_recent_predictions()
    future = get_future_forecast(horizon)
    return {"recent": recent, "future": future}

@app.get("/api/anomalies", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def anomalies(request: Request):
    return get_anomalies()

@app.get("/api/metrics", dependencies=[Depends(verify_api_key)])
@limiter.limit("100/minute")
async def metrics(request: Request):
    return get_metrics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)