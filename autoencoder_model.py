import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.callbacks import EarlyStopping 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
import os 
 
print("=" * 60) 
print("?? AUTOENCODER FOR ANOMALY DETECTION") 
print("=" * 60) 
 
# Load data 
df = pd.read_csv("data/raw/energy_consumption_2023.csv") 
print(f"? Data loaded: {df.shape}") 
