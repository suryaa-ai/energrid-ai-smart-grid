import numpy as np 
import tensorflow as tf 
import pandas as pd 
 
print("AUTOENCODER - FINAL WORKING VERSION") 
print("=" * 50) 
 
# Load data 
df = pd.read_csv("data/raw/energy_consumption_2023.csv") 
print(f"Data loaded: {len(df)} records") 
 
# Separate normal and anomaly 
normal_data = df[df['is_anomaly'] == 0] 
anomaly_data = df[df['is_anomaly'] == 1] 
print(f"Normal samples: {len(normal_data)}") 
print(f"Anomaly samples: {len(anomaly_data)}") 
 
# Use only load data for simplicity 
X_normal = normal_data[['load_MW']].values 
X_anomaly = anomaly_data[['load_MW']].values 
 
# Normalize between 0 and 1 
min_val = X_normal.min() 
max_val = X_normal.max() 
X_normal_norm = (X_normal - min_val) / (max_val - min_val) 
X_anomaly_norm = (X_anomaly - min_val) / (max_val - min_val) 
 
# Build simple autoencoder 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
 
input_layer = Input(shape=(1,)) 
encoded = Dense(2, activation='relu')(input_layer) 
decoded = Dense(1, activation='sigmoid')(encoded) 
 
autoencoder = Model(input_layer, decoded) 
autoencoder.compile(optimizer='adam', loss='mse') 
 
# Train on normal data only 
print("Training autoencoder on normal data...") 
history = autoencoder.fit(X_normal_norm, X_normal_norm, epochs=5, batch_size=32, verbose=1) 
 
# Calculate reconstruction errors 
normal_pred = autoencoder.predict(X_normal_norm, verbose=0) 
anomaly_pred = autoencoder.predict(X_anomaly_norm, verbose=0) 
 
normal_error = np.mean((X_normal_norm - normal_pred) ** 2, axis=1) 
anomaly_error = np.mean((X_anomaly_norm - anomaly_pred) ** 2, axis=1) 
 
# Set threshold (mean + 2 standard deviations) 
threshold = normal_error.mean() + 2 * normal_error.std() 
 
# Count how many anomalies are detected 
detected = 0 
for err in anomaly_error: 
    if err 
        detected = detected + 1 
 
print("\\n=== RESULTS ===") 
print(f"Normal data reconstruction error: {normal_error.mean():.6f}") 
print(f"Anomaly data reconstruction error: {anomaly_error.mean():.6f}") 
print(f"Detection threshold: {threshold:.6f}") 
print(f"Anomalies detected: {detected} out of {len(anomaly_data)}") 
print(f"Detection rate: {detected/len(anomaly_data)*100:.1f}%") 
 
# Save the model 
import os 
os.makedirs('models', exist_ok=True) 
autoencoder.save('models/autoencoder_final.h5') 
print("\\n? Autoencoder saved to: models/autoencoder_final.h5") 
print("? AUTOENCODER TRAINING COMPLETE!") 
