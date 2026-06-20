import numpy as np 
import tensorflow as tf 
import pandas as pd 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout 
import os 
 
print("=" * 60) 
print("?? CNN-LSTM with REAL DATA") 
print("=" * 60) 
 
# Load your actual data 
df = pd.read_csv("data/raw/energy_consumption_2023.csv", parse_dates=['timestamp']) 
print(f"? Data loaded: {df.shape}") 
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}") 
 
# Simple preprocessing 
df['hour'] = df['timestamp'].dt.hour 
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24) 
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24) 
 
# Normalize load 
load_min, load_max = df['load_MW'].min(), df['load_MW'].max() 
df['load_normalized'] = (df['load_MW'] - load_min) / (load_max - load_min) 
 
# Create sequences (24-hour windows) 
sequence_length = 24 
X, y = [], [] 
for i in range(len(df) - sequence_length): 
    X.append(df[['load_normalized', 'temperature_C', 'hour_sin', 'hour_cos']].iloc[i:i+sequence_length].values) 
    y.append(df['load_normalized'].iloc[i+sequence_length]) 
 
X = np.array(X) 
y = np.array(y) 
print(f"? Sequences created: X={X.shape}, y={y.shape}") 
 
# Split data (80-10-10) 
train_size = int(len(X) * 0.8) 
val_size = int(len(X) * 0.1) 
 
X_train = X[:train_size] 
X_val = X[train_size:train_size + val_size] 
X_test = X[train_size + val_size:] 
 
y_train = y[:train_size] 
y_val = y[train_size:train_size + val_size] 
y_test = y[train_size + val_size:] 
 
print(f"?? Data splits:") 
print(f"   Training: {len(X_train)} samples") 
print(f"   Validation: {len(X_val)} samples") 
print(f"   Testing: {len(X_test)} samples") 
 
# Build CNN-LSTM model 
model = Sequential([ 
    Conv1D(64, 3, activation='relu', input_shape=(sequence_length, 4), padding='same'), 
    Dropout(0.2), 
    LSTM(100, return_sequences=True), 
    Dropout(0.2), 
    LSTM(50), 
    Dropout(0.2), 
    Dense(25, activation='relu'), 
    Dense(1) 
]) 
 
model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
print("? Model built!") 
 
# Train 
print("?? Training (10 epochs)...") 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1) 
 
# Evaluate 
loss, mae = model.evaluate(X_test, y_test, verbose=0) 
print(f"?? Test Results:") 
print(f"   MSE: {loss:.6f}") 
print(f"   MAE: {mae:.6f}") 
 
# Convert predictions back to original scale 
y_pred = model.predict(X_test, verbose=0).flatten() 
y_test_original = y_test * (load_max - load_min) + load_min 
y_pred_original = y_pred * (load_max - load_min) + load_min 
 
print(f"?? Sample predictions (MW):") 
for i in range(5): 
    print(f"   Actual: {y_test_original[i]:.1f} MW, Predicted: {y_pred_original[i]:.1f} MW, Error: {abs(y_test_original[i]-y_pred_original[i]):.1f} MW") 
 
# Save model 
os.makedirs('models', exist_ok=True) 
model.save('models/cnn_lstm_real.h5') 
print("?? Model saved: models/cnn_lstm_real.h5") 
 
print("\\n" + "=" * 60) 
print("? CNN-LSTM WITH REAL DATA - COMPLETE!") 
print("=" * 60) 
