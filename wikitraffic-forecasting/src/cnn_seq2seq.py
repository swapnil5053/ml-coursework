"""
Sequence-to-Sequence CNN with causal convolutions for time series forecasting.
Implements advanced CNN architecture with dilated convolutions and residual connections.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, but handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, Dense, Dropout, Add, Activation, 
        BatchNormalization, GlobalAveragePooling1D, Reshape
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. CNN functionality will be limited.")


class CausalConv1D(tf.keras.layers.Layer):
    """Causal 1D Convolution layer for time series."""
    
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super(CausalConv1D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        
    def build(self, input_shape):
        # Calculate padding to ensure causal convolution
        self.padding = (self.kernel_size - 1) * self.dilation_rate
        
        self.conv = Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding='valid'
        )
        
    def call(self, inputs):
        # Apply padding to the left
        padded_inputs = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])
        return self.conv(padded_inputs)


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with dilated causal convolution."""
    
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Causal convolution
        self.causal_conv = CausalConv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            dilation_rate=self.dilation_rate
        )
        
        # Batch normalization
        self.batch_norm1 = BatchNormalization()
        
        # Activation
        self.activation = Activation('relu')
        
        # Dropout
        self.dropout = Dropout(self.dropout_rate)
        
        # 1x1 convolution for residual connection
        self.conv_1x1 = Conv1D(filters=self.filters, kernel_size=1)
        
        # Second batch normalization
        self.batch_norm2 = BatchNormalization()
        
    def call(self, inputs, training=None):
        # First causal convolution
        x = self.causal_conv(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        # Second causal convolution (1x1)
        x = self.conv_1x1(x)
        x = self.batch_norm2(x, training=training)
        
        # Residual connection
        residual = self.conv_1x1(inputs) if inputs.shape[-1] != self.filters else inputs
        x = Add()([x, residual])
        
        return self.activation(x)


class Seq2SeqCNNModel:
    """Sequence-to-Sequence CNN model with causal convolutions."""
    
    def __init__(self, input_length=128, output_length=64, filters=64, 
                 kernel_size=3, dilation_rates=[1, 2, 4, 8, 16, 32, 64, 128],
                 dropout_rate=0.1, learning_rate=0.001, batch_size=32, epochs=100):
        """
        Initialize Seq2Seq CNN model.
        
        Args:
            input_length (int): Length of input sequences
            output_length (int): Length of output sequences
            filters (int): Number of filters in convolutional layers
            kernel_size (int): Kernel size for convolutions
            dilation_rates (list): Dilation rates for residual blocks
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN model. Please install it with: pip install tensorflow")
            
        self.input_length = input_length
        self.output_length = output_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.is_fitted = False
        self.training_history = None
    
    def _create_model(self):
        """Create Seq2Seq CNN model architecture."""
        
        # Encoder
        encoder_inputs = Input(shape=(self.input_length, 1), name='encoder_inputs')
        
        # Initial convolution
        x = Conv1D(filters=self.filters, kernel_size=1, padding='same')(encoder_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Residual blocks with increasing dilation rates
        skip_connections = []
        for i, dilation_rate in enumerate(self.dilation_rates):
            residual_block = ResidualBlock(
                filters=self.filters,
                kernel_size=self.kernel_size,
                dilation_rate=dilation_rate,
                dropout_rate=self.dropout_rate,
                name=f'residual_block_{i}'
            )
            x = residual_block(x)
            
            # Collect skip connections
            skip_connections.append(x)
        
        # Global average pooling for encoder output
        encoder_output = GlobalAveragePooling1D()(x)
        
        # Decoder
        # Repeat the encoder output for each time step
        decoder_inputs = Input(shape=(self.output_length,), name='decoder_inputs')
        
        # Expand decoder inputs
        decoder_expanded = Reshape((self.output_length, 1))(decoder_inputs)
        
        # Decoder layers
        decoder_hidden = Dense(self.filters, activation='relu')(decoder_expanded)
        decoder_hidden = BatchNormalization()(decoder_hidden)
        decoder_hidden = Dropout(self.dropout_rate)(decoder_hidden)
        
        # Add encoder context
        encoder_context = Dense(self.output_length)(encoder_output)
        encoder_context = Reshape((self.output_length, 1))(encoder_context)
        
        # Combine decoder and encoder information
        combined = Add()([decoder_hidden, encoder_context])
        
        # Final decoder layers
        decoder_hidden = Dense(self.filters // 2, activation='relu')(combined)
        decoder_hidden = BatchNormalization()(decoder_hidden)
        decoder_hidden = Dropout(self.dropout_rate)(decoder_hidden)
        
        # Output layer
        decoder_outputs = Dense(1, activation='linear')(decoder_hidden)
        
        # Create model
        model = Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=decoder_outputs,
            name='seq2seq_cnn'
        )
        
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
        Prepare sequences for Seq2Seq training.
        
        Args:
            data (pd.DataFrame): Input data
            target_col (str): Target column name
            
        Returns:
            tuple: (encoder_inputs, decoder_inputs, targets)
        """
        encoder_sequences = []
        decoder_sequences = []
        targets = []
        
        # Group by page and access type
        for (page, access), group in data.groupby(['Page', 'Access']):
            group = group.sort_values('Date')
            
            # Create sequences
            for i in range(self.input_length, len(group) - self.output_length + 1):
                # Encoder input: past input_length days
                encoder_seq = group.iloc[i-self.input_length:i][target_col].values
                
                # Decoder input: zeros (teacher forcing will be handled in training)
                decoder_seq = np.zeros(self.output_length)
                
                # Target: next output_length days
                target_seq = group.iloc[i:i+self.output_length][target_col].values
                
                encoder_sequences.append(encoder_seq)
                decoder_sequences.append(decoder_seq)
                targets.append(target_seq)
        
        encoder_inputs = np.array(encoder_sequences)
        decoder_inputs = np.array(decoder_sequences)
        targets = np.array(targets)
        
        # Reshape for CNN
        encoder_inputs = encoder_inputs.reshape((encoder_inputs.shape[0], encoder_inputs.shape[1], 1))
        targets = targets.reshape((targets.shape[0], targets.shape[1], 1))
        
        return encoder_inputs, decoder_inputs, targets
    
    def fit(self, encoder_inputs, decoder_inputs, targets, validation_split=0.2, verbose=1):
        """
        Train the Seq2Seq CNN model.
        
        Args:
            encoder_inputs (array-like): Encoder input sequences
            decoder_inputs (array-like): Decoder input sequences
            targets (array-like): Target sequences
            validation_split (float): Fraction of data for validation
            verbose (int): Verbosity level
        """
        print("Training Seq2Seq CNN model...")
        print(f"Encoder input shape: {encoder_inputs.shape}")
        print(f"Decoder input shape: {decoder_inputs.shape}")
        print(f"Target shape: {targets.shape}")
        
        # Create model
        self.model = self._create_model()
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
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
                'best_seq2seq_cnn_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            [encoder_inputs, decoder_inputs], targets,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Important for time series
        )
        
        self.is_fitted = True
        print("Seq2Seq CNN model training completed!")
    
    def predict(self, encoder_inputs, decoder_inputs=None):
        """
        Make predictions using the trained model.
        
        Args:
            encoder_inputs (array-like): Encoder input sequences
            decoder_inputs (array-like): Decoder input sequences (optional)
            
        Returns:
            array: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if decoder_inputs is None:
            # Create zero decoder inputs for inference
            decoder_inputs = np.zeros((encoder_inputs.shape[0], self.output_length))
        
        predictions = self.model.predict([encoder_inputs, decoder_inputs], verbose=0)
        return predictions


def train_seq2seq_cnn_model(data, input_length=128, output_length=64, test_size=0.2, target_col='log_traffic'):
    """
    Complete training and evaluation pipeline for Seq2Seq CNN model.
    
    Args:
        data (pd.DataFrame): Input data
        input_length (int): Length of input sequences
        output_length (int): Length of output sequences
        test_size (float): Fraction of data for testing
        target_col (str): Target column name
        
    Returns:
        tuple: (trained_model, predictions, metrics)
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping Seq2Seq CNN model.")
        return None, None, None
    
    # Prepare sequences
    cnn_model = Seq2SeqCNNModel(input_length=input_length, output_length=output_length)
    encoder_inputs, decoder_inputs, targets = cnn_model.prepare_sequences(data, target_col)
    
    print(f"Prepared {len(encoder_inputs)} sequences")
    print(f"Input length: {input_length}, Output length: {output_length}")
    
    # Split data
    split_idx = int(len(encoder_inputs) * (1 - test_size))
    
    X_train_enc = encoder_inputs[:split_idx]
    X_test_enc = encoder_inputs[split_idx:]
    X_train_dec = decoder_inputs[:split_idx]
    X_test_dec = decoder_inputs[split_idx:]
    y_train = targets[:split_idx]
    y_test = targets[split_idx:]
    
    print(f"Training sequences: {len(X_train_enc)}")
    print(f"Test sequences: {len(X_test_enc)}")
    
    # Train model
    cnn_model.fit(X_train_enc, X_train_dec, y_train, validation_split=0.2)
    
    # Make predictions
    y_pred = cnn_model.predict(X_test_enc, X_test_dec)
    
    # Flatten for evaluation
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate metrics
    from evaluation import TimeSeriesEvaluator
    evaluator = TimeSeriesEvaluator()
    metrics = evaluator.calculate_metrics(y_test_flat, y_pred_flat, "Seq2Seq CNN")
    
    print(f"\nSeq2Seq CNN Model Results:")
    print(f"SMAPE: {metrics['smape']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    
    return cnn_model, y_pred_flat, metrics


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import WikiTrafficPreprocessor
    
    # Create sample data
    preprocessor = WikiTrafficPreprocessor()
    result = preprocessor.preprocess_pipeline(sample_size=1000)
    
    # Train Seq2Seq CNN model
    cnn_model, predictions, metrics = train_seq2seq_cnn_model(
        result['data'], input_length=64, output_length=32
    )
    
    if cnn_model is not None:
        print("Seq2Seq CNN model training and evaluation completed!")
    else:
        print("Seq2Seq CNN model skipped due to missing dependencies.")
