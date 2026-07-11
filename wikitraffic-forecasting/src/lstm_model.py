"""
LSTM model for time series forecasting.
Implements deep learning approach with Long Short-Term Memory networks.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, but handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM functionality will be limited.")


class LSTMTimeSeriesModel:
    """LSTM-based time series forecasting model."""
    
    def __init__(self, lookback=30, lstm_units=[50, 50], dropout_rate=0.4, 
                 learning_rate=0.001, batch_size=32, epochs=100):
        """
        Initialize LSTM model.
        
        Args:
            lookback (int): Number of past days to use for prediction
            lstm_units (list): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Please install it with: pip install tensorflow")
            
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.is_fitted = False
        self.training_history = None
        
    def _create_model(self, input_shape):
        """
        Create LSTM model architecture.
        
        Args:
            input_shape (tuple): Input shape for the model
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate
        ))
        model.add(BatchNormalization())
        
        # Second LSTM layer (if specified)
        if len(self.lstm_units) > 1:
            model.add(LSTM(
                units=self.lstm_units[1],
                return_sequences=False,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate
            ))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(units=25, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_sequences(self, data, target_col='log_traffic'):
        """
        Prepare sequences for LSTM training.
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            
        Returns:
            tuple: (X_seq, y_seq) - sequences and targets
        """
        sequences = []
        targets = []
        
        # Group by page and access type
        for (page, access), group in data.groupby(['Page', 'Access']):
            group = group.sort_values('Date')
            
            # Create sequences
            for i in range(self.lookback, len(group)):
                seq = group.iloc[i-self.lookback:i][target_col].values
                target = group.iloc[i][target_col]
                
                sequences.append(seq)
                targets.append(target)
        
        X_seq = np.array(sequences)
        y_seq = np.array(targets)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        return X_seq, y_seq
    
    def fit(self, X_seq, y_seq, validation_split=0.2, verbose=1):
        """
        Train the LSTM model.
        
        Args:
            X_seq (array-like): Input sequences
            y_seq (array-like): Target values
            validation_split (float): Fraction of data for validation
            verbose (int): Verbosity level
        """
        print("Training LSTM model...")
        print(f"Input shape: {X_seq.shape}")
        print(f"Target shape: {y_seq.shape}")
        
        # Create model
        self.model = self._create_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Important for time series
        )
        
        self.is_fitted = True
        print("LSTM model training completed!")
    
    def predict(self, X_seq):
        """
        Make predictions using the trained model.
        
        Args:
            X_seq (array-like): Input sequences
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()


def train_lstm_model(data, lookback=30, test_size=0.2, target_col='log_traffic'):
    """
    Complete training and evaluation pipeline for LSTM model.
    
    Args:
        data (pd.DataFrame): Input data
        lookback (int): Number of past days to use for prediction
        test_size (float): Fraction of data for testing
        target_col (str): Target column name
        
    Returns:
        tuple: (trained_model, predictions, metrics)
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping LSTM model.")
        return None, None, None
    
    # Prepare sequences
    lstm_model = LSTMTimeSeriesModel(lookback=lookback)
    X_seq, y_seq = lstm_model.prepare_sequences(data, target_col)
    
    print(f"Prepared {len(X_seq)} sequences with lookback={lookback}")
    
    # Split data
    split_idx = int(len(X_seq) * (1 - test_size))
    X_train_seq = X_seq[:split_idx]
    X_test_seq = X_seq[split_idx:]
    y_train_seq = y_seq[:split_idx]
    y_test_seq = y_seq[split_idx:]
    
    print(f"Training sequences: {len(X_train_seq)}")
    print(f"Test sequences: {len(X_test_seq)}")
    
    # Train model
    lstm_model.fit(X_train_seq, y_train_seq, validation_split=0.2)
    
    # Make predictions
    y_pred = lstm_model.predict(X_test_seq)
    
    # Calculate metrics
    from evaluation import TimeSeriesEvaluator
    evaluator = TimeSeriesEvaluator()
    metrics = evaluator.calculate_metrics(y_test_seq, y_pred, "LSTM")
    
    print(f"\nLSTM Model Results:")
    print(f"SMAPE: {metrics['smape']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    return lstm_model, y_pred, metrics


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import WikiTrafficPreprocessor
    
    # Create sample data
    preprocessor = WikiTrafficPreprocessor()
    result = preprocessor.preprocess_pipeline(sample_size=1000)
    
    # Train LSTM model
    lstm_model, predictions, metrics = train_lstm_model(
        result['data'], lookback=30
    )
    
    if lstm_model is not None:
        print("LSTM model training and evaluation completed!")
    else:
        print("LSTM model skipped due to missing dependencies.")
