# train_simple_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pathlib import Path

X_train = np.load('data/processed/X_train_real.npy')
y_train = np.load('data/processed/y_train_real.npy')
X_test = np.load('data/processed/X_test_real.npy')
y_test = np.load('data/processed/y_test_real.npy')

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
Path('models').mkdir(exist_ok=True)
model.save('models/simple_lstm.h5')
print("✅ Simple LSTM saved")