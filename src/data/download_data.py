import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os

class DataDownloader:
    def __init__(self):
        self.data_path = Path("data/raw")
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def create_sample_energy_data(self, start_date='2023-01-01', end_date='2023-12-31'):
        """Create synthetic energy consumption data with patterns and anomalies"""
        print("🔧 Creating synthetic energy dataset...")
        
        # Generate hourly timestamps (use 'h' instead of 'H')
        dates = pd.date_range(start=start_date, end=end_date, freq='h')
        n_samples = len(dates)
        
        np.random.seed(42)  # For reproducibility
        
        # Extract time features
        hour = dates.hour
        day_of_week = dates.dayofweek
        month = dates.month
        
        # 1. Base load pattern (MW)
        base_load = 500
        
        # 2. Daily pattern (sinusoidal - peaks at 6 PM)
        daily_pattern = 150 * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # 3. Weekly pattern (lower on weekends)
        weekly_factor = np.where(day_of_week < 5, 1.0, 0.7)
        
        # 4. Seasonal pattern
        seasonal_pattern = 100 * np.sin(2 * np.pi * (month - 1) / 12)
        
        # 5. Random noise
        noise = np.random.normal(0, 25, n_samples)
        
        # 6. Trend
        trend = np.linspace(0, 50, n_samples)
        
        # Combine all components - CONVERT TO NUMPY ARRAY
        load = np.array((base_load + daily_pattern + seasonal_pattern) * weekly_factor + trend + noise)
        
        # Add temperature
        temperature = 25 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3, n_samples)
        
        # Add humidity
        humidity = np.random.uniform(40, 85, n_samples)
        
        # Create anomalies (5% of data)
        n_anomalies = int(0.05 * n_samples)
        anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
        
        is_anomaly = np.zeros(n_samples, dtype=int)
        
        # Apply anomalies - NOW load IS A NUMPY ARRAY
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'drop', 'noise'])
            if anomaly_type == 'spike':
                load[idx] *= np.random.uniform(1.5, 3.0)
            elif anomaly_type == 'drop':
                load[idx] *= np.random.uniform(0.2, 0.5)
            else:
                load[idx] += np.random.uniform(-200, 200)
            is_anomaly[idx] = 1
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'load_MW': np.round(load, 2),
            'temperature_C': np.round(temperature, 1),
            'humidity_percent': np.round(humidity, 1),
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': (day_of_week >= 5).astype(int),
            'is_holiday': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
            'is_anomaly': is_anomaly
        })
        
        # Save to CSV
        output_file = self.data_path / "energy_consumption_2023.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Dataset created!")
        print(f"   - Records: {len(df):,}")
        print(f"   - Anomalies: {df['is_anomaly'].sum()} points ({df['is_anomaly'].mean()*100:.1f}%)")
        print(f"   - Saved to: {output_file}")
        
        return df

if __name__ == "__main__":
    downloader = DataDownloader()
    df = downloader.create_sample_energy_data()
    print("\n📊 First 5 rows:")
    print(df.head())