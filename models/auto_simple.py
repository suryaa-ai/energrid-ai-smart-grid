import numpy as np
import tensorflow as tf
import pandas as pd

print("AUTOENCODER TRAINING")
print("=" * 40)

# Load data
df = pd.read_csv("data/raw/energy_consumption_2023.csv")
print(f"Data: {len(df)} records")

# Get normal data
normal = df[df['is_anomaly'] == 0]
X = normal[['load_MW']].values

# Normalize
X_norm = (X - X.min()) / (X.max() - X.min())

# Build autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input_layer = Input(shape=(1,))
encoded = Dense(2, activation='relu')(input_layer)
decoded = Dense(1, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
print("Training autoencoder...")
autoencoder.fit(X_norm, X_norm, epochs=3, batch_size=32, verbose=1)

# Test
pred = autoencoder.predict(X_norm, verbose=0)
error = np.mean((X_norm - pred) ** 2)
print(f"Reconstruction error: {error:.6f}")

# Save
import os
os.makedirs('models', exist_ok=True)
autoencoder.save('models/auto_simple.h5')
print("Model saved!")
print("✅ AUTOENCODER COMPLETE!")