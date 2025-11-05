"""
ARIMA Model Implementation
Autoregressive Integrated Moving Average model for time series forecasting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import itertools
import warnings
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import config

warnings.filterwarnings('ignore')


def find_optimal_arima_params(data, max_p=5, max_d=2, max_q=5, seasonal=False):
    """
    Find optimal ARIMA parameters using grid search with AIC
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    max_p : int
        Maximum AR order
    max_d : int
        Maximum differencing order
    max_q : int
        Maximum MA order
    seasonal : bool
        Whether to use seasonal ARIMA
    
    Returns:
    --------
    tuple
        Best (p, d, q) parameters and AIC score
    """
    print("Finding optimal ARIMA parameters...")
    
    best_aic = np.inf
    best_params = None
    best_model = None
    
    # Remove NaN values
    data_clean = data.dropna()
    
    # Grid search
    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)
    
    total_combinations = len(p_range) * len(d_range) * len(q_range)
    print(f"Testing {total_combinations} parameter combinations...")
    
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            model = ARIMA(data_clean, order=(p, d, q))
            fitted_model = model.fit()
            aic = fitted_model.aic
            
            if aic < best_aic:
                best_aic = aic
                best_params = (p, d, q)
                best_model = fitted_model
                
        except Exception as e:
            continue
    
    print(f"Best ARIMA parameters: {best_params} with AIC: {best_aic:.4f}")
    return best_params, best_aic, best_model


def plot_acf_pacf(data, lags=40, figsize=(12, 6)):
    """
    Plot ACF and PACF for ARIMA parameter selection
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    lags : int
        Number of lags to plot
    figsize : tuple
        Figure size
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF plot
    acf_values = acf(data.dropna(), nlags=lags)
    axes[0].bar(range(len(acf_values)), acf_values)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
    axes[0].axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    
    # PACF plot
    pacf_values = pacf(data.dropna(), nlags=lags)
    axes[1].bar(range(len(pacf_values)), pacf_values)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].axhline(y=1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
    axes[1].axhline(y=-1.96/np.sqrt(len(data)), color='red', linestyle='--', linewidth=0.5)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    
    plt.tight_layout()
    return fig


def build_arima_model(data, order=None, auto_select=True):
    """
    Build and train ARIMA model
    
    Parameters:
    -----------
    data : pd.Series
        Training time series data
    order : tuple, optional
        (p, d, q) order. If None, will auto-select
    auto_select : bool
        Whether to automatically select optimal parameters
    
    Returns:
    --------
    fitted ARIMA model
    """
    print("="*60)
    print("BUILDING ARIMA MODEL")
    print("="*60)
    
    data_clean = data.dropna()
    
    if auto_select and order is None:
        # Auto-select parameters
        order, aic, model = find_optimal_arima_params(
            data_clean,
            max_p=config.ARIMA_CONFIG['max_p'],
            max_d=config.ARIMA_CONFIG['max_d'],
            max_q=config.ARIMA_CONFIG['max_q']
        )
        print(f"\nSelected ARIMA order: {order}")
    else:
        if order is None:
            order = (1, 1, 1)  # Default
        print(f"Using ARIMA order: {order}")
        model = ARIMA(data_clean, order=order)
        model = model.fit()
    
    print("\nModel Summary:")
    print(model.summary())
    
    return model, order


def forecast_arima(model, steps, return_conf_int=True, alpha=0.05):
    """
    Generate forecasts using ARIMA model
    
    Parameters:
    -----------
    model : fitted ARIMA model
        Trained ARIMA model
    steps : int
        Number of steps ahead to forecast
    return_conf_int : bool
        Whether to return confidence intervals
    alpha : float
        Significance level for confidence intervals
    
    Returns:
    --------
    pd.Series or tuple
        Forecasts (and confidence intervals if requested)
    """
    forecast_result = model.forecast(steps=steps, alpha=alpha)
    
    if return_conf_int:
        conf_int = model.get_forecast(steps=steps).conf_int(alpha=alpha)
        return forecast_result, conf_int
    else:
        return forecast_result


def evaluate_arima_model(model, test_data, train_data=None):
    """
    Evaluate ARIMA model on test data
    
    Parameters:
    -----------
    model : fitted ARIMA model
        Trained ARIMA model
    test_data : pd.Series
        Test data for evaluation
    train_data : pd.Series, optional
        Training data (for fitting continuation)
    
    Returns:
    --------
    dict
        Dictionary containing predictions and evaluation metrics
    """
    print("\nEvaluating ARIMA model...")
    
    # Generate predictions for test period
    n_test = len(test_data)
    forecasts, conf_int = forecast_arima(model, steps=n_test, return_conf_int=True)
    
    # Align forecasts with test data index
    forecasts = pd.Series(forecasts, index=test_data.index)
    
    # Calculate residuals
    residuals = test_data - forecasts
    
    # Calculate metrics
    from src.model_evaluation import calculate_metrics
    metrics = calculate_metrics(test_data, forecasts)
    
    results = {
        'forecasts': forecasts,
        'conf_int': conf_int,
        'residuals': residuals,
        'metrics': metrics,
        'model': model
    }
    
    return results


def diagnose_arima_model(model, residuals):
    """
    Perform diagnostic tests on ARIMA model residuals
    
    Parameters:
    -----------
    model : fitted ARIMA model
        Trained ARIMA model
    residuals : pd.Series
        Model residuals
    
    Returns:
    --------
    dict
        Diagnostic test results
    """
    print("\n" + "="*60)
    print("ARIMA MODEL DIAGNOSTICS")
    print("="*60)
    
    # Ljung-Box test for residual autocorrelation
    lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
    
    print("\nLjung-Box Test for Residual Autocorrelation:")
    print(lb_test)
    
    # Check if residuals are white noise (p-value > 0.05)
    p_value = lb_test['lb_pvalue'].iloc[-1]
    is_white_noise = p_value > 0.05
    
    print(f"\nResiduals are {'white noise' if is_white_noise else 'NOT white noise'} (p-value: {p_value:.4f})")
    
    diagnostics = {
        'ljung_box_test': lb_test,
        'is_white_noise': is_white_noise,
        'residual_mean': residuals.mean(),
        'residual_std': residuals.std()
    }
    
    return diagnostics


def save_arima_model(model, order, filepath=None):
    """
    Save ARIMA model to disk
    
    Parameters:
    -----------
    model : fitted ARIMA model
        Trained ARIMA model
    order : tuple
        (p, d, q) order
    filepath : str, optional
        Path to save model
    """
    if filepath is None:
        os.makedirs(config.PATHS['models'], exist_ok=True)
        filepath = os.path.join(config.PATHS['models'], 'arima_model.pkl')
    
    model_data = {
        'model': model,
        'order': order
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"ARIMA model saved to: {filepath}")


def load_arima_model(filepath=None):
    """
    Load ARIMA model from disk
    
    Parameters:
    -----------
    filepath : str, optional
        Path to model file
    
    Returns:
    --------
    tuple
        (model, order)
    """
    if filepath is None:
        filepath = os.path.join(config.PATHS['models'], 'arima_model.pkl')
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model'], model_data['order']


if __name__ == "__main__":
    # Example usage
    from src.data_preprocessing import preprocess_data
    
    # Preprocess data
    data, _ = preprocess_data()
    
    # Prepare data for ARIMA (use inflation rate)
    if 'Inflation_Rate' in data.columns:
        target = data['Inflation_Rate']
        
        # Split data
        train_size = int(len(target) * config.DATA_CONFIG['train_split'])
        train_data = target[:train_size]
        test_data = target[train_size:]
        
        # Build and train ARIMA model
        model, order = build_arima_model(train_data, auto_select=True)
        
        # Evaluate model
        results = evaluate_arima_model(model, test_data, train_data)
        
        print("\nARIMA Model Metrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric}: {value:.4f}")

