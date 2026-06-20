import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Flatten, BatchNormalization 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 
import numpy as np 
from pathlib import Path 
import matplotlib.pyplot as plt 
 
class CNNLSTMModel: 
    def __init__(self, input_shape, model_name="cnn_lstm_forecaster"): 
        self.input_shape = input_shape 
        self.model_name = model_name 
        self.model = None 
        self.history = None 
 
