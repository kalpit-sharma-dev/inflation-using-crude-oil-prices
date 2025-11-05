"""
Data Preprocessing Module
Handles missing values, normalization, stationarity tests, and feature engineering
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config


def load_data(file_path=None):
    """
    Load merged data from CSV file
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the data file. If None, uses default path
    
    Returns:
    --------
    pd.DataFrame
        Loaded data with Date index
    """
    if file_path is None:
        file_path = os.path.join(config.PATHS['raw_data'], 'merged_data.csv')
    
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return data


def handle_missing_values(data):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    
    Returns:
    --------
    pd.DataFrame
        Data with missing values handled
    """
    print("Handling missing values...")
    
    # Check for missing values
    missing_count = data.isnull().sum()
    if missing_count.sum() > 0:
        print(f"Missing values found:\n{missing_count}")
        
        # Forward fill, then backward fill
        data = data.ffill().bfill()
        
        # If still missing, interpolate
        data = data.interpolate(method='linear')
        
        print("Missing values handled using forward fill, backward fill, and interpolation")
    else:
        print("No missing values found")
    
    return data


def calculate_inflation_rate(cpi_data):
    """
    Calculate inflation rate (YoY) from CPI data
    
    Parameters:
    -----------
    cpi_data : pd.Series
        CPI index values
    
    Returns:
    --------
    pd.Series
        Year-over-year inflation rate
    """
    # Calculate YoY inflation rate
    inflation_rate = ((cpi_data - cpi_data.shift(12)) / cpi_data.shift(12)) * 100
    
    return inflation_rate


def test_stationarity(timeseries, title="Time Series"):
    """
    Perform Augmented Dickey-Fuller test for stationarity
    
    Parameters:
    -----------
    timeseries : pd.Series
        Time series data to test
    title : str
        Title for the test output
    
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    print(f"\n{'='*60}")
    print(f"ADF Test for {title}")
    print(f"{'='*60}")
    
    # Perform ADF test
    adf_result = adfuller(timeseries.dropna(), autolag='AIC')
    
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    print(f"ADF Statistic: {adf_statistic:.6f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.6f}")
    
    is_stationary = p_value < 0.05
    print(f"\nResult: {'Series is STATIONARY' if is_stationary else 'Series is NON-STATIONARY'}")
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }


def make_stationary(data, column, diff_order=1):
    """
    Make a time series stationary by differencing
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    column : str
        Column name to make stationary
    diff_order : int
        Order of differencing
    
    Returns:
    --------
    pd.Series
        Differenced (stationary) series
    """
    diff_data = data[column].diff(diff_order)
    return diff_data


def normalize_data(data, method='minmax'):
    """
    Normalize data using specified method
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data to normalize
    method : str
        Normalization method: 'minmax', 'zscore', or 'robust'
    
    Returns:
    --------
    tuple
        (normalized_data, scaler_info) for inverse transformation
    """
    if method == 'minmax':
        # Min-Max scaling to [0, 1]
        data_min = data.min()
        data_max = data.max()
        normalized = (data - data_min) / (data_max - data_min)
        scaler_info = {'min': data_min, 'max': data_max, 'method': 'minmax'}
        
    elif method == 'zscore':
        # Z-score normalization
        data_mean = data.mean()
        data_std = data.std()
        normalized = (data - data_mean) / data_std
        scaler_info = {'mean': data_mean, 'std': data_std, 'method': 'zscore'}
        
    elif method == 'robust':
        # Robust scaling (using median and IQR)
        data_median = data.median()
        q75 = data.quantile(0.75)
        q25 = data.quantile(0.25)
        iqr = q75 - q25
        normalized = (data - data_median) / iqr
        scaler_info = {'median': data_median, 'iqr': iqr, 'method': 'robust'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, scaler_info


def denormalize_data(normalized_data, scaler_info):
    """
    Denormalize data using scaler information
    
    Parameters:
    -----------
    normalized_data : pd.Series or np.array
        Normalized data
    scaler_info : dict
        Scaler information from normalize_data
    
    Returns:
    --------
    pd.Series or np.array
        Denormalized data
    """
    method = scaler_info['method']
    
    if method == 'minmax':
        denormalized = normalized_data * (scaler_info['max'] - scaler_info['min']) + scaler_info['min']
    elif method == 'zscore':
        denormalized = normalized_data * scaler_info['std'] + scaler_info['mean']
    elif method == 'robust':
        denormalized = normalized_data * scaler_info['iqr'] + scaler_info['median']
    
    return denormalized


def create_lag_features(data, target_col, max_lags=12):
    """
    Create lag features for time series
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    target_col : str
        Target column name
    max_lags : int
        Maximum number of lags to create
    
    Returns:
    --------
    pd.DataFrame
        Data with lag features
    """
    data_with_lags = data.copy()
    
    for lag in range(1, max_lags + 1):
        data_with_lags[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
    
    return data_with_lags


def preprocess_data(data_file=None, save_processed=True):
    """
    Main preprocessing function
    
    Parameters:
    -----------
    data_file : str, optional
        Path to raw data file
    save_processed : bool
        Whether to save processed data
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for modeling
    """
    print("="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    # Load data
    data = load_data(data_file)
    print(f"\nLoaded data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Handle missing values
    data = handle_missing_values(data)
    
    # Calculate inflation rate from CPI
    if 'CPI' in data.columns:
        data['Inflation_Rate'] = calculate_inflation_rate(data['CPI'])
        print(f"\nCalculated inflation rate (YoY %)")
        print(f"Inflation rate statistics:")
        print(data['Inflation_Rate'].describe())
    
    # Test stationarity
    print("\n" + "="*60)
    print("STATIONARITY TESTS")
    print("="*60)
    
    stationarity_results = {}
    
    if 'Brent_Price' in data.columns:
        stationarity_results['Brent_Price'] = test_stationarity(
            data['Brent_Price'], 
            title="Brent Crude Oil Price"
        )
    
    if 'Inflation_Rate' in data.columns:
        stationarity_results['Inflation_Rate'] = test_stationarity(
            data['Inflation_Rate'],
            title="Inflation Rate"
        )
    
    # Drop rows with NaN after differencing/inflation calculation
    data = data.dropna()
    
    # Create processed data directory
    os.makedirs(config.PATHS['processed_data'], exist_ok=True)
    
    # Save processed data
    if save_processed:
        processed_file = os.path.join(config.PATHS['processed_data'], 'processed_data.csv')
        data.to_csv(processed_file)
        print(f"\nProcessed data saved to: {processed_file}")
    
    # Save stationarity test results
    stationarity_file = os.path.join(config.PATHS['processed_data'], 'stationarity_results.txt')
    with open(stationarity_file, 'w') as f:
        f.write("STATIONARITY TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        for col, result in stationarity_results.items():
            f.write(f"{col}:\n")
            f.write(f"  ADF Statistic: {result['adf_statistic']:.6f}\n")
            f.write(f"  p-value: {result['p_value']:.6f}\n")
            f.write(f"  Is Stationary: {result['is_stationary']}\n\n")
    
    print(f"\nStationarity results saved to: {stationarity_file}")
    print("\nPreprocessing completed successfully!")
    
    return data, stationarity_results


if __name__ == "__main__":
    processed_data, stationarity = preprocess_data()
    print(f"\nFinal data shape: {processed_data.shape}")
    print(f"\nFirst few rows:")
    print(processed_data.head())
    print(f"\nData columns: {processed_data.columns.tolist()}")

