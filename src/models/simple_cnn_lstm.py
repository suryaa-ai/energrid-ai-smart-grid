import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout 
import matplotlib.pyplot as plt 
 
print("=" * 60) 
print("?? SIMPLE CNN-LSTM Model Test") 
print("=" * 60) 
 
# Create synthetic data 
n_samples = 1000 
sequence_length = 24 
n_features = 4 
 
X_train = np.random.randn(int(n_samples * 0.7), sequence_length, n_features) 
X_val = np.random.randn(int(n_samples * 0.15), sequence_length, n_features) 
X_test = np.random.randn(int(n_samples * 0.15), sequence_length, n_features) 
 
y_train = np.random.randn(int(n_samples * 0.7)) 
y_val = np.random.randn(int(n_samples * 0.15)) 
y_test = np.random.randn(int(n_samples * 0.15)) 
 
print("Data created!") 
 
# Build model 
model = Sequential([ 
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(sequence_length, n_features)), 
    Dropout(0.2), 
    LSTM(50, dropout=0.2), 
    Dense(25, activation='relu'), 
    Dense(1) 
]) 
 
model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
model.summary() 
 
# Train 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=1) 
 
# Evaluate 
loss, mae = model.evaluate(X_test, y_test, verbose=0) 
print(f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}") 
 
model.save('models/simple_cnn_lstm.h5') 
print("Model saved!") 
