import numpy as np
import tensorflow as tf
import joblib

print("Loading model...")
model = tf.keras.models.load_model('models/cnn_lstm_real.h5', compile=False)
print("✅ Model loaded")

print("Loading test data...")
X_test = np.load('data/processed/X_test_real.npy')
y_test = np.load('data/processed/y_test_real.npy')
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print("Loading scaler...")
scaler = joblib.load('data/processed/scaler_real.pkl')
print("✅ Scaler loaded")

print("Predicting last 24 sequences...")
pred_norm = model.predict(X_test[-24:], verbose=0).flatten()
load_min = scaler.min_[0]
load_max = scaler.scale_[0] + load_min
pred_denorm = pred_norm * (load_max - load_min) + load_min
actual_denorm = y_test[-24:] * (load_max - load_min) + load_min

print("Predictions (first 5):", pred_denorm[:5])
print("Actuals (first 5):", actual_denorm[:5])
print("✅ Test passed")