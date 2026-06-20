import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from pathlib import Path

print("=" * 60)
print("🔋 Training CNN-LSTM on REAL PJM Data")
print("=" * 60)

# Load preprocessed real data
X_train = np.load('data/processed/X_train_real.npy')
y_train = np.load('data/processed/y_train_real.npy')
X_test = np.load('data/processed/X_test_real.npy')
y_test = np.load('data/processed/y_test_real.npy')

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# Build model (using the same architecture as your paper)
input_shape = (X_train.shape[1], X_train.shape[2])
model = Sequential([
    Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(32, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(100, return_sequences=True, dropout=0.2),
    BatchNormalization(),
    LSTM(50, dropout=0.2),
    BatchNormalization(),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mape'])
model.summary()

# Train
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=1)

# Evaluate
loss, mae, mape = model.evaluate(X_test, y_test, verbose=0)
print(f"\n📊 Test Results (Real Data):")
print(f"   MSE  = {loss:.6f}")
print(f"   MAE  = {mae:.6f}")
print(f"   MAPE = {mape:.2f}%")

# Save model
Path('models').mkdir(exist_ok=True)
model.save('models/cnn_lstm_real.h5')
print("✅ Model saved: models/cnn_lstm_real.h5")