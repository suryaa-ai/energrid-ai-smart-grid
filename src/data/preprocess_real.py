import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_real():
    print("=" * 60)
    print("🚀 PREPROCESSING REAL PJM DATA")
    print("=" * 60)

    # --- 1. Load real data ---
    # Kaggle file: 'PJM_Load_hourly.csv'
    df = pd.read_csv('data/raw/PJM_Load_hourly.csv')
    
    # Rename columns to match our pipeline
    df = df.rename(columns={
        'Datetime': 'timestamp',
        'PJM_Load_MW': 'load_MW'
    })
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"✅ Loaded: {df.shape} records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # --- 2. Add synthetic temperature/humidity (optional) ---
    np.random.seed(42)
    df['temperature_C'] = 25 + 10 * np.sin(2 * np.pi * df['timestamp'].dt.month / 12) + np.random.normal(0, 3, len(df))
    df['humidity_percent'] = np.random.uniform(40, 85, len(df))

    # --- 3. Inject synthetic anomalies (for evaluation) ---
    n_anomalies = int(0.05 * len(df))
    anomaly_indices = np.random.choice(len(df), size=n_anomalies, replace=False)
    df['is_anomaly'] = 0
    for idx in anomaly_indices:
        if np.random.rand() > 0.5:
            df.loc[idx, 'load_MW'] *= np.random.uniform(1.5, 3.0)
        else:
            df.loc[idx, 'load_MW'] *= np.random.uniform(0.2, 0.5)
        df.loc[idx, 'is_anomaly'] = 1
    print(f"   Injected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)")

    # --- 4. Feature engineering ---
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Lag features (24h, 168h)
    df['load_lag_1'] = df['load_MW'].shift(1)
    df['load_lag_24'] = df['load_MW'].shift(24)
    df['load_lag_168'] = df['load_MW'].shift(168)

    # Rolling stats
    df['load_rolling_mean_24'] = df['load_MW'].rolling(window=24, min_periods=1).mean()
    df['load_rolling_std_24'] = df['load_MW'].rolling(window=24, min_periods=1).std()

    # Drop rows with NaN (from lags)
    df = df.dropna()
    print(f"   After feature engineering: {df.shape}")

    # --- 5. Normalize ---
    scaler = MinMaxScaler()
    features_to_scale = ['load_MW', 'temperature_C', 'humidity_percent',
                         'load_lag_1', 'load_lag_24', 'load_lag_168',
                         'load_rolling_mean_24', 'load_rolling_std_24']
    df_scaled = df.copy()
    df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    joblib.dump(scaler, 'data/processed/scaler_real.pkl')

    # --- 6. Split data (80/20 time series) ---
    split_idx = int(len(df_scaled) * 0.8)
    train = df_scaled.iloc[:split_idx]
    test = df_scaled.iloc[split_idx:]

    # --- 7. Prepare sequences for forecasting ---
    sequence_length = 24
    forecast_features = [col for col in train.columns if col not in ['timestamp', 'is_anomaly', 'load_MW']]
    X_train, y_train = [], []
    for i in range(len(train) - sequence_length):
        X_train.append(train[forecast_features].iloc[i:i+sequence_length].values)
        y_train.append(train['load_MW'].iloc[i+sequence_length])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test, y_test = [], []
    for i in range(len(test) - sequence_length):
        X_test.append(test[forecast_features].iloc[i:i+sequence_length].values)
        y_test.append(test['load_MW'].iloc[i+sequence_length])
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # --- 8. Prepare anomaly detection data ---
    # Use only normal samples for training autoencoder
    X_normal_train = train[train['is_anomaly'] == 0][['load_MW', 'temperature_C', 'humidity_percent']].values
    X_anomaly_train = train[train['is_anomaly'] == 1][['load_MW', 'temperature_C', 'humidity_percent']].values

    X_normal_test = test[test['is_anomaly'] == 0][['load_MW', 'temperature_C', 'humidity_percent']].values
    X_anomaly_test = test[test['is_anomaly'] == 1][['load_MW', 'temperature_C', 'humidity_percent']].values

    # --- 9. Save everything ---
    processed_path = Path('data/processed')
    processed_path.mkdir(exist_ok=True)

    train.to_csv(processed_path / 'train_processed_real.csv', index=False)
    test.to_csv(processed_path / 'test_processed_real.csv', index=False)

    np.save(processed_path / 'X_train_real.npy', X_train)
    np.save(processed_path / 'y_train_real.npy', y_train)
    np.save(processed_path / 'X_test_real.npy', X_test)
    np.save(processed_path / 'y_test_real.npy', y_test)

    np.save(processed_path / 'X_normal_train_real.npy', X_normal_train)
    np.save(processed_path / 'X_anomaly_train_real.npy', X_anomaly_train)
    np.save(processed_path / 'X_normal_test_real.npy', X_normal_test)
    np.save(processed_path / 'X_anomaly_test_real.npy', X_anomaly_test)

    with open(processed_path / 'forecast_features_real.txt', 'w') as f:
        for feat in forecast_features:
            f.write(f"{feat}\n")
    with open(processed_path / 'anomaly_features_real.txt', 'w') as f:
        f.write("load_MW\ntemperature_C\nhumidity_percent\n")

    print("\n" + "=" * 60)
    print("✅ REAL DATA PREPROCESSING COMPLETE!")
    print(f"   Forecasting: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"   Anomaly: normal train {X_normal_train.shape}, anomaly train {X_anomaly_train.shape}")
    print("=" * 60)

if __name__ == "__main__":
    preprocess_real()