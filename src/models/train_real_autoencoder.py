import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from pathlib import Path

print("=" * 60)
print("🔍 Training Autoencoder on REAL PJM Data")
print("=" * 60)

# Load normal data (no anomalies)
X_normal_train = np.load('data/processed/X_normal_train_real.npy')
print(f"Normal training samples: {X_normal_train.shape}")

# Build autoencoder
input_dim = X_normal_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(8, activation='relu')(input_layer)
encoded = Dense(4, activation='relu')(encoded)
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train
history = autoencoder.fit(X_normal_train, X_normal_train,
                          epochs=50,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=1)

# Save model
Path('models').mkdir(exist_ok=True)
autoencoder.save('models/autoencoder_real.h5')
print("✅ Autoencoder saved: models/autoencoder_real.h5")