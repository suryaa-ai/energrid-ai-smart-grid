import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, LSTM, Dense 
 
print("? TensorFlow version:", tf.__version__) 
 
# Create simple data 
X = np.random.randn(100, 24, 4) 
y = np.random.randn(100) 
 
# Build simple model 
model = Sequential([ 
    Conv1D(32, 3, activation='relu', input_shape=(24, 4)), 
    LSTM(50), 
    Dense(1) 
]) 
 
model.compile(optimizer='adam', loss='mse') 
print("? Model built successfully!") 
 
# Quick training 
print("Training for 2 epochs...") 
model.fit(X, y, epochs=2, batch_size=16, verbose=1) 
 
print("? Test complete! CNN-LSTM works!") 
