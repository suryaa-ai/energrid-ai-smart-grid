# shap_explainer.py
import shap
import numpy as np
import joblib
import tensorflow as tf

_explainer = None
_background = None
_feature_names = None

def init_explainer():
    global _explainer, _background, _feature_names
    if _explainer is not None:
        return
    model = tf.keras.models.load_model('models/cnn_lstm_real.h5', compile=False)
    _scaler = joblib.load('data/processed/scaler_real.pkl')
    with open('data/processed/forecast_features_real.txt', 'r') as f:
        _feature_names = [line.strip() for line in f.readlines()]
    X_train = np.load('data/processed/X_train_real.npy')
    idx = np.random.choice(len(X_train), 200, replace=False)
    _background = X_train[idx]
    _explainer = shap.GradientExplainer(model, _background)
    print("✅ SHAP explainer initialized")

def explain_prediction(x_input):
    if _explainer is None:
        init_explainer()
    shap_values = _explainer.shap_values(x_input, nsamples=100)
    shap_vals = shap_values[0][0]
    return shap_vals, _feature_names