# src/models/simple_cnn_lstm.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

print("=" * 60)
print("🔋 SIMPLE CNN-LSTM Model Test")
print("=" * 60)

# Create synthetic data for testing
print("📊 Creating synthetic data...")
n_samples = 1000
sequence_length = 24
n_features = 4

# Create dummy data
X_train = np.random.randn(int(n_samples * 0.7), sequence_length, n_features)
X_val = np.random.randn(int(n_samples * 0.15), sequence_length, n_features)
X_test = np.random.randn(int(n_samples * 0.15), sequence_length, n_features)

y_train = np.random.randn(int(n_samples * 0.7))
y_val = np.random.randn(int(n_samples * 0.15))
y_test = np.random.randn(int(n_samples * 0.15))

print(f"Data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")

# Build simple model
print("\n🔨 Building CNN-LSTM model...")
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', 
           input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(50, dropout=0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Train
print("\n🚀 Training model (5 epochs for testing)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32,
    verbose=1
)

# Evaluate
print("\n📊 Evaluating model...")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# Make predictions
y_pred = model.predict(X_test, verbose=0).flatten()

# Plot sample predictions
plt.figure(figsize=(10, 5))
sample_size = min(50, len(y_test))
indices = np.random.choice(len(y_test), sample_size, replace=False)

plt.plot(y_test[indices], 'b-', label='Actual', linewidth=2, alpha=0.7)
plt.plot(y_pred[indices], 'r--', label='Predicted', linewidth=2, alpha=0.7)
plt.title(f'Actual vs Predicted (Sample of {sample_size})')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/simple_model_predictions.png')
plt.show()

print("\n" + "=" * 60)
print("✅ Simple CNN-LSTM Test Complete!")
print("=" * 60)

# Save model
model.save('models/simple_cnn_lstm.h5')
print("💾 Model saved to: models/simple_cnn_lstm.h5")