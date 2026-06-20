# src/data/simple_preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path

print("🚀 Simple Preprocessing Starting...")

# Load data
df = pd.read_csv("data/raw/energy_consumption_2023.csv", parse_dates=['timestamp'])
print(f"Loaded data: {df.shape}")

# Create simple features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Create sequences for LSTM (24-hour sequences)
sequence_length = 24
X, y = [], []

for i in range(len(df) - sequence_length):
    X.append(df[['load_MW', 'temperature_C', 'hour', 'day_of_week']].iloc[i:i+sequence_length].values)
    y.append(df['load_MW'].iloc[i+sequence_length])

X = np.array(X)
y = np.array(y)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Save
Path("data/processed").mkdir(exist_ok=True)
np.save("data/processed/X_simple.npy", X)
np.save("data/processed/y_simple.npy", y)

print("✅ Simple preprocessing complete!")