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
        
    def build_model(self):
        """Build hybrid CNN-LSTM model"""
        print(f"🔨 Building CNN-LSTM model with input shape: {self.input_shape}")
        
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   input_shape=self.input_shape, padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            # LSTM layers for temporal dependencies
            LSTM(100, return_sequences=True, dropout=0.2),
            BatchNormalization(),
            
            LSTM(50, dropout=0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(100, activation='relu'),
            Dropout(0.2),
            
            Dense(50, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1)
        ])
        
        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        self.model = model
        model.summary()
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with callbacks"""
        print(f"🚀 Training model for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(f'models/{self.model_name}_best.h5', 
                          monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("📊 Evaluating model...")
        
        loss, mae, mape = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Loss (MSE): {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test MAPE: {mape:.2f}%")
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        return {
            'loss': loss,
            'mae': mae,
            'mape': mape,
            'y_pred': y_pred.flatten(),
            'y_true': y_test
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # MAPE plot
        axes[2].plot(self.history.history['mape'], label='Train MAPE')
        axes[2].plot(self.history.history['val_mape'], label='Val MAPE')
        axes[2].set_title('Mean Absolute Percentage Error')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('MAPE (%)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, sample_size=100):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(14, 6))
        
        # Plot a sample of predictions
        indices = np.random.choice(len(y_true), min(sample_size, len(y_true)), replace=False)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        x_range = np.arange(len(sample_true))
        
        plt.plot(x_range, sample_true, 'b-', label='Actual', linewidth=2, alpha=0.7)
        plt.plot(x_range, sample_pred, 'r--', label='Predicted', linewidth=2, alpha=0.7)
        plt.fill_between(x_range, sample_true, sample_pred, color='gray', alpha=0.2)
        
        plt.title(f'Actual vs Predicted Load Values (Sample of {sample_size})', fontsize=14, fontweight='bold')
        plt.xlabel('Sample Index')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{self.model_name}_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path='models/cnn_lstm_model.h5'):
        """Save the trained model"""
        self.model.save(path)
        print(f"✅ Model saved to: {path}")
    
    def load_model(self, path='models/cnn_lstm_model.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from: {path}")

def main():
    """Main function to run the model pipeline"""
    print("=" * 60)
    print("🔋 CNN-LSTM Load Forecasting Model")
    print("=" * 60)
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    
    # First, check if preprocessed data exists
    print("📥 Checking for preprocessed data...")
    
    # Try simple data creation if no preprocessed data
    try:
        import pandas as pd
        
        # Load raw data
        df = pd.read_csv("data/raw/energy_consumption_2023.csv", parse_dates=['timestamp'])
        print(f"✅ Loaded raw data: {df.shape}")
        
        # Simple preprocessing
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Normalize load data (simple min-max scaling)
        load_min = df['load_MW'].min()
        load_max = df['load_MW'].max()
        df['load_normalized'] = (df['load_MW'] - load_min) / (load_max - load_min)
        
        # Create sequences (24-hour windows)
        sequence_length = 24
        X, y = [], []
        
        print(f"🔧 Creating sequences with length {sequence_length}...")
        
        for i in range(len(df) - sequence_length):
            # Use 4 features for simplicity
            X.append(df[['load_normalized', 'temperature_C', 'hour', 'day_of_week']].iloc[i:i+sequence_length].values)
            y.append(df['load_normalized'].iloc[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Sequences created: X={X.shape}, y={y.shape}")
        
        # Split data (80-10-10)
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.1)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        print(f"\n📊 Data splits:")
        print(f"   - Training: {len(X_train)} samples")
        print(f"   - Validation: {len(X_val)} samples")
        print(f"   - Testing: {len(X_test)} samples")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Creating dummy data for testing...")
        
        # Create dummy data for testing
        n_samples = 1000
        sequence_length = 24
        n_features = 4
        
        X_train = np.random.randn(int(n_samples * 0.7), sequence_length, n_features)
        X_val = np.random.randn(int(n_samples * 0.15), sequence_length, n_features)
        X_test = np.random.randn(int(n_samples * 0.15), sequence_length, n_features)
        
        y_train = np.random.randn(int(n_samples * 0.7))
        y_val = np.random.randn(int(n_samples * 0.15))
        y_test = np.random.randn(int(n_samples * 0.15))
        
        print(f"Using dummy data: X_train={X_train.shape}")
    
    # Build and train model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = CNNLSTMModel(input_shape)
    model.build_model()
    
    # Train with fewer epochs for testing
    print("\n🚀 Starting training (10 epochs for testing)...")
    history = model.train(X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    
    # Plot results
    model.plot_training_history()
    model.plot_predictions(results['y_true'], results['y_pred'], sample_size=50)
    
    # Save model
    model.save_model()
    
    print("\n" + "=" * 60)
    print("✅ CNN-LSTM Model Training Complete!")
    print("=" * 60)
    
    # Show final metrics
    print(f"\n🎯 Final Performance Metrics:")
    print(f"   - Mean Squared Error (MSE): {results['loss']:.4f}")
    print(f"   - Mean Absolute Error (MAE): {results['mae']:.4f}")
    print(f"   - Mean Absolute Percentage Error: {results['mape']:.2f}%")
    
    return model, results

if __name__ == "__main__":
    model, results = main()