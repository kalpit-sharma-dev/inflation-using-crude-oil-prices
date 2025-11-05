"""
LSTM Model Implementation
Long Short-Term Memory neural network for time series forecasting
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import config
import warnings

warnings.filterwarnings('ignore')


def create_sequences(data, sequence_length):
    """
    Create sequences for LSTM training
    
    Parameters:
    -----------
    data : np.array
        Time series data
    sequence_length : int
        Number of time steps to look back
    
    Returns:
    --------
    tuple
        (X, y) where X is input sequences and y is target values
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)


def build_lstm_model(sequence_length, units=None, dropout=0.2):
    """
    Build LSTM model architecture
    
    Parameters:
    -----------
    sequence_length : int
        Length of input sequences
    units : list, optional
        Number of units in each LSTM layer
    dropout : float
        Dropout rate
    
    Returns:
    --------
    tf.keras.Model
        Compiled LSTM model
    """
    if units is None:
        units = config.LSTM_CONFIG['units']
    
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        units=units[0],
        return_sequences=True if len(units) > 1 else False,
        input_shape=(sequence_length, 1)
    ))
    model.add(Dropout(dropout))
    
    # Additional LSTM layers
    for i in range(1, len(units)):
        return_sequences = i < len(units) - 1
        model.add(LSTM(units=units[i], return_sequences=return_sequences))
        model.add(Dropout(dropout))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    optimizer = Adam(learning_rate=config.LSTM_CONFIG['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_lstm_model(X_train, y_train, X_val=None, y_val=None, 
                     sequence_length=None, epochs=None, batch_size=None,
                     validation_split=None, verbose=1):
    """
    Train LSTM model
    
    Parameters:
    -----------
    X_train : np.array
        Training input sequences
    y_train : np.array
        Training target values
    X_val : np.array, optional
        Validation input sequences
    y_val : np.array, optional
        Validation target values
    sequence_length : int, optional
        Sequence length (if None, inferred from X_train)
    epochs : int, optional
        Number of training epochs
    batch_size : int, optional
        Batch size
    validation_split : float, optional
        Validation split ratio
    verbose : int
        Verbosity level
    
    Returns:
    --------
    tuple
        (trained_model, training_history)
    """
    print("="*60)
    print("BUILDING AND TRAINING LSTM MODEL")
    print("="*60)
    
    if sequence_length is None:
        sequence_length = config.LSTM_CONFIG['sequence_length']
    if epochs is None:
        epochs = config.LSTM_CONFIG['epochs']
    if batch_size is None:
        batch_size = config.LSTM_CONFIG['batch_size']
    if validation_split is None:
        validation_split = config.LSTM_CONFIG['validation_split']
    
    # Build model
    model = build_lstm_model(
        sequence_length=sequence_length,
        units=config.LSTM_CONFIG['units'],
        dropout=config.LSTM_CONFIG['dropout']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if (X_val is not None or validation_split > 0) else 'loss',
            patience=config.LSTM_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss' if (X_val is not None or validation_split > 0) else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        validation_split = 0
    
    # Train model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose
    )
    
    print("\nTraining completed!")
    
    return model, history


def prepare_lstm_data(data, sequence_length=None, scaler=None, fit_scaler=True):
    """
    Prepare data for LSTM training
    
    Parameters:
    -----------
    data : pd.Series or np.array
        Time series data
    sequence_length : int, optional
        Sequence length
    scaler : MinMaxScaler, optional
        Pre-fitted scaler
    fit_scaler : bool
        Whether to fit scaler on data
    
    Returns:
    --------
    tuple
        (scaled_data, scaler, sequences_X, sequences_y)
    """
    if sequence_length is None:
        sequence_length = config.LSTM_CONFIG['sequence_length']
    
    # Convert to numpy array
    if isinstance(data, pd.Series):
        data = data.values
    
    # Reshape for scaling
    data = data.reshape(-1, 1)
    
    # Scale data
    if scaler is None:
        scaler = MinMaxScaler()
    
    if fit_scaler:
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    
    # Create sequences
    X, y = create_sequences(scaled_data.flatten(), sequence_length)
    
    # Reshape X for LSTM (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return scaled_data, scaler, X, y


def forecast_lstm(model, last_sequence, steps, scaler):
    """
    Generate multi-step forecasts using LSTM model
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained LSTM model
    last_sequence : np.array
        Last sequence from training data
    steps : int
        Number of steps ahead to forecast
    scaler : MinMaxScaler
        Scaler used for normalization
    
    Returns:
    --------
    np.array
        Forecasted values (denormalized)
    """
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Predict next step
        next_pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        forecasts.append(next_pred[0, 0])
        
        # Update sequence: remove first element, append prediction
        current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
    
    # Denormalize forecasts
    forecasts = np.array(forecasts).reshape(-1, 1)
    forecasts = scaler.inverse_transform(forecasts).flatten()
    
    return forecasts


def evaluate_lstm_model(model, X_test, y_test, scaler, test_data_index=None):
    """
    Evaluate LSTM model on test data
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained LSTM model
    X_test : np.array
        Test input sequences
    y_test : np.array
        Test target values
    scaler : MinMaxScaler
        Scaler used for normalization
    test_data_index : pd.Index, optional
        Index for test data (for creating DataFrame)
    
    Returns:
    --------
    dict
        Dictionary containing predictions and evaluation metrics
    """
    print("\nEvaluating LSTM model...")
    
    # Generate predictions
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Denormalize predictions and actual values
    y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    y_pred = scaler.inverse_transform(y_pred_reshaped).flatten()
    y_test_denorm = scaler.inverse_transform(y_test_reshaped).flatten()
    
    # Create Series if index provided
    if test_data_index is not None:
        y_pred = pd.Series(y_pred, index=test_data_index[-len(y_pred):])
        y_test_denorm = pd.Series(y_test_denorm, index=test_data_index[-len(y_test_denorm):])
    
    # Calculate metrics
    from src.model_evaluation import calculate_metrics
    metrics = calculate_metrics(y_test_denorm, y_pred)
    
    results = {
        'forecasts': y_pred,
        'actual': y_test_denorm,
        'metrics': metrics,
        'model': model
    }
    
    return results


def save_lstm_model(model, scaler, filepath=None):
    """
    Save LSTM model and scaler to disk
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained LSTM model
    scaler : MinMaxScaler
        Fitted scaler
    filepath : str, optional
        Base path to save model (without extension)
    """
    if filepath is None:
        os.makedirs(config.PATHS['models'], exist_ok=True)
        filepath = os.path.join(config.PATHS['models'], 'lstm_model')
    
    # Save model
    model.save(f"{filepath}.h5")
    
    # Save scaler
    with open(f"{filepath}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"LSTM model saved to: {filepath}.h5")
    print(f"Scaler saved to: {filepath}_scaler.pkl")


def load_lstm_model(filepath=None):
    """
    Load LSTM model and scaler from disk
    
    Parameters:
    -----------
    filepath : str, optional
        Base path to model (without extension)
    
    Returns:
    --------
    tuple
        (model, scaler)
    """
    if filepath is None:
        filepath = os.path.join(config.PATHS['models'], 'lstm_model')
    
    # Load model
    model = tf.keras.models.load_model(f"{filepath}.h5")
    
    # Load scaler
    with open(f"{filepath}_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


if __name__ == "__main__":
    # Example usage
    from src.data_preprocessing import preprocess_data
    
    # Preprocess data
    data, _ = preprocess_data()
    
    # Prepare data for LSTM (use inflation rate)
    if 'Inflation_Rate' in data.columns:
        target = data['Inflation_Rate']
        
        # Split data
        train_size = int(len(target) * config.DATA_CONFIG['train_split'])
        train_data = target[:train_size]
        test_data = target[train_size:]
        
        # Prepare LSTM data
        _, scaler, X_train, y_train = prepare_lstm_data(
            train_data,
            sequence_length=config.LSTM_CONFIG['sequence_length'],
            fit_scaler=True
        )
        
        # Prepare test data
        _, _, X_test, y_test = prepare_lstm_data(
            test_data,
            sequence_length=config.LSTM_CONFIG['sequence_length'],
            scaler=scaler,
            fit_scaler=False
        )
        
        # Train LSTM model
        model, history = train_lstm_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        results = evaluate_lstm_model(
            model, X_test, y_test, scaler,
            test_data_index=test_data.index
        )
        
        print("\nLSTM Model Metrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")

