"""
Configuration file for Time Series Analysis Project
Forecasting India's Inflation Using Crude Oil Prices
"""

# Data Configuration
DATA_CONFIG = {
    'start_date': '2010-01-01',
    'end_date': '2025-09-30',
    'frequency': 'monthly',
    'train_split': 0.7,  # 70% for training, 30% for testing
}

# ARIMA Model Configuration
ARIMA_CONFIG = {
    'max_p': 5,
    'max_d': 2,
    'max_q': 5,
    'seasonal': False,
    'trend': 'c',  # constant term
}

# LSTM Model Configuration
LSTM_CONFIG = {
    'sequence_length': 12,  # Use 12 months (1 year) of historical data
    'units': [64, 32],  # Number of units in each LSTM layer
    'dropout': 0.2,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping_patience': 15,
    'learning_rate': 0.001,
}

# Evaluation Metrics
METRICS = ['RMSE', 'MAE', 'MAPE']

# File Paths
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'figures': 'results/figures',
    'models': 'results/models',
    'metrics': 'results/metrics',
}

# Plot Configuration
PLOT_CONFIG = {
    'figsize': (12, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8',
}

