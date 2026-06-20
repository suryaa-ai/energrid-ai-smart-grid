# ⚡ EnerGrid AI: Unified Smart Grid Forecasting & Anomaly Platform

An end-to-end deep learning framework designed to solve critical power grid operational challenges: predicting electricity load demand 24 hours in advance and identifying structural equipment/transmission anomalies in real time. 

Research paper covering this framework has been submitted to the **IEEE SEGE 2026** conference.

---

## 📊 Performance Metrics & Benchmarks

* **Load Forecasting:** Achieved a **4.2% Mean Absolute Percentage Error (MAPE)** evaluated on 32,896 real-world PJM hourly load records, outperforming baseline LSTM models (5.8%) and traditional ARIMA frameworks (8.1%).
* **Anomaly Detection:** Unsupervised Autoencoder architecture achieved **91% Precision**, **85% Recall**, and a **0.88 F1-Score**.
* **Operational Efficiency:** Ultra-low inference latency running at **<30ms per record**, making the system fully capable of processing live stream pipelines.
* **Ensemble Optimization:** Integrating a secondary validation layer (CNN-LSTM + XGBoost + Prophet) reduced edge forecasting error to **3.9%** during high-volatility demand spikes.

---

## 🛠️ Technical Architecture & Stack

* **Core Programming:** `Python 3.x`
* **Deep Learning Frameworks:** `TensorFlow` | `Keras` | `Scikit-learn`
* **Network Topologies:** Hybrid `CNN-LSTM` (Spatial-Temporal feature extraction) | Unsupervised `Autoencoders`
* **Data Processing Engines:** `Pandas` | `NumPy` | `Min-Max Scaling` | `Joblib`
* **Real-time Pipeline Integrations:** `OpenWeatherMap API` (Weather-load correlation analytics) | `Telegram Bot API` (Instant operator alert dispatch within 2-4 seconds)
* **Visualization & Frontend:** `Streamlit Dashboard UI` | `Plotly Interactive Graphs` | `Matplotlib`
* **Data Persistence & DevOps:** `MongoDB` | `Git` | `Streamlit Cloud Deployment`

---

## 🖥️ Modular System Structure

The interactive Streamlit operator interface is partitioned into 5 independent production modules:
1. **Operator Dashboard:** Real-time visual tracking of grid loads, system health statuses, and dynamic parameter dials.
2. **Model Analytics:** Deep-dive validation metrics showing training loss, precision-recall graphs, and baseline comparative error curves.
3. **Data Explorer:** Automated preprocessing pipelines allowing operators to inspect historical data streams, engineer time-series lag features, and export cleaned datasets.
4. **Explainable AI (XAI):** Powered by SHAP to correlate grid anomalies with extreme weather fluctuations, proving a 68% environmental correlation accuracy.
5. **System Settings:** Configurable tariff thresholds (₹) and carbon footprint emission variables paired with Telegram bot alert handlers containing automated retry logic.

---

## 🚀 How To Run & Deploy Locally

### 1. Clone the repository
```bash
git clone https://github.com
cd EnerGrid-AI
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Operator Interface
```bash
streamlit run app.py
```

---

## 🤝 Contact & Profile Links
* **Developer:** Suryaa S
* **LinkedIn:** [://linkedin.com](https://://linkedin.com)
* **GitHub Profile:** [://github.com](https://://github.com)
