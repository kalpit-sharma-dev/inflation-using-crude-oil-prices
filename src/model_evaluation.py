"""
Model Evaluation Module
Calculate evaluation metrics and compare models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import config


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAE value
    """
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAPE value (in percentage)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Avoid division by zero
    mask = actual != 0
    if mask.sum() == 0:
        return np.inf
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_metrics(actual, predicted):
    """
    Calculate all evaluation metrics
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    metrics = {
        'RMSE': calculate_rmse(actual, predicted),
        'MAE': calculate_mae(actual, predicted),
        'MAPE': calculate_mape(actual, predicted)
    }
    
    return metrics


def compare_models(arima_results, lstm_results, save_results=True):
    """
    Compare ARIMA and LSTM model performance
    
    Parameters:
    -----------
    arima_results : dict
        ARIMA model evaluation results
    lstm_results : dict
        LSTM model evaluation results
    save_results : bool
        Whether to save comparison results
    
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Extract metrics
    arima_metrics = arima_results['metrics']
    lstm_metrics = lstm_results['metrics']
    
    # Create comparison table
    comparison = pd.DataFrame({
        'ARIMA': [
            arima_metrics['RMSE'],
            arima_metrics['MAE'],
            arima_metrics['MAPE']
        ],
        'LSTM': [
            lstm_metrics['RMSE'],
            lstm_metrics['MAE'],
            lstm_metrics['MAPE']
        ]
    }, index=['RMSE', 'MAE', 'MAPE (%)'])
    
    # Calculate improvement
    comparison['Improvement (%)'] = ((comparison['ARIMA'] - comparison['LSTM']) / 
                                     comparison['ARIMA'] * 100)
    
    print("\nComparison Table:")
    print(comparison.round(4))
    
    # Determine best model for each metric
    print("\nBest Model by Metric:")
    for metric in ['RMSE', 'MAE', 'MAPE (%)']:
        if comparison.loc[metric, 'ARIMA'] < comparison.loc[metric, 'LSTM']:
            best = 'ARIMA'
            improvement = -comparison.loc[metric, 'Improvement (%)']
        else:
            best = 'LSTM'
            improvement = comparison.loc[metric, 'Improvement (%)']
        print(f"  {metric}: {best} (improvement: {improvement:.2f}%)")
    
    # Save results
    if save_results:
        os.makedirs(config.PATHS['metrics'], exist_ok=True)
        
        # Save as CSV
        csv_file = os.path.join(config.PATHS['metrics'], 'model_comparison.csv')
        comparison.to_csv(csv_file)
        print(f"\nComparison saved to: {csv_file}")
        
        # Save as JSON
        json_file = os.path.join(config.PATHS['metrics'], 'model_comparison.json')
        comparison_dict = {
            'comparison_table': comparison.to_dict(),
            'arima_metrics': arima_metrics,
            'lstm_metrics': lstm_metrics
        }
        with open(json_file, 'w') as f:
            json.dump(comparison_dict, f, indent=4, default=str)
        print(f"Detailed results saved to: {json_file}")
    
    return comparison


def plot_predictions(actual, arima_pred, lstm_pred, title="Model Predictions Comparison"):
    """
    Plot actual vs predicted values for both models
    
    Parameters:
    -----------
    actual : pd.Series or np.array
        Actual values
    arima_pred : pd.Series or np.array
        ARIMA predictions
    lstm_pred : pd.Series or np.array
        LSTM predictions
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Convert to Series if needed
    if not isinstance(actual, pd.Series):
        actual = pd.Series(actual)
    if not isinstance(arima_pred, pd.Series):
        arima_pred = pd.Series(arima_pred)
    if not isinstance(lstm_pred, pd.Series):
        lstm_pred = pd.Series(lstm_pred)
    
    # Plot 1: Time series comparison
    axes[0].plot(actual.index, actual.values, label='Actual', linewidth=2, color='black')
    axes[0].plot(arima_pred.index, arima_pred.values, label='ARIMA', 
                 linewidth=1.5, linestyle='--', alpha=0.8)
    axes[0].plot(lstm_pred.index, lstm_pred.values, label='LSTM', 
                 linewidth=1.5, linestyle='--', alpha=0.8)
    axes[0].set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Inflation Rate (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plots
    axes[1].scatter(actual.values, arima_pred.values, alpha=0.6, 
                    label='ARIMA', s=50)
    axes[1].scatter(actual.values, lstm_pred.values, alpha=0.6, 
                    label='LSTM', s=50, marker='^')
    
    # Perfect prediction line
    min_val = min(actual.min(), arima_pred.min(), lstm_pred.min())
    max_val = max(actual.max(), arima_pred.max(), lstm_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_title(f'{title} - Scatter Plot', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.PATHS['figures'], exist_ok=True)
    plot_file = os.path.join(config.PATHS['figures'], 'model_predictions_comparison.png')
    plt.savefig(plot_file, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    return fig


def plot_residuals(arima_residuals, lstm_residuals, titles=None):
    """
    Plot residuals for both models
    
    Parameters:
    -----------
    arima_residuals : pd.Series or np.array
        ARIMA model residuals
    lstm_residuals : pd.Series or np.array
        LSTM model residuals
    titles : dict, optional
        Dictionary with 'arima' and 'lstm' keys for plot titles
    """
    if titles is None:
        titles = {'arima': 'ARIMA Residuals', 'lstm': 'LSTM Residuals'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convert to Series if needed
    if not isinstance(arima_residuals, pd.Series):
        arima_residuals = pd.Series(arima_residuals)
    if not isinstance(lstm_residuals, pd.Series):
        lstm_residuals = pd.Series(lstm_residuals)
    
    # ARIMA residuals - time series
    axes[0, 0].plot(arima_residuals.index, arima_residuals.values, 
                    color='blue', linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].set_title(titles['arima'] + ' - Time Series', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ARIMA residuals - histogram
    axes[0, 1].hist(arima_residuals.values, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(titles['arima'] + ' - Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # LSTM residuals - time series
    axes[1, 0].plot(lstm_residuals.index, lstm_residuals.values, 
                    color='green', linewidth=1)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_title(titles['lstm'] + ' - Time Series', fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # LSTM residuals - histogram
    axes[1, 1].hist(lstm_residuals.values, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title(titles['lstm'] + ' - Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.PATHS['figures'], exist_ok=True)
    plot_file = os.path.join(config.PATHS['figures'], 'model_residuals.png')
    plt.savefig(plot_file, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
    print(f"Residuals plot saved to: {plot_file}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    actual = pd.Series(np.random.randn(100) + 5, 
                       index=pd.date_range('2020-01-01', periods=100, freq='M'))
    arima_pred = actual + np.random.randn(100) * 0.5
    lstm_pred = actual + np.random.randn(100) * 0.3
    
    # Calculate metrics
    arima_metrics = calculate_metrics(actual, arima_pred)
    lstm_metrics = calculate_metrics(actual, lstm_pred)
    
    print("ARIMA Metrics:", arima_metrics)
    print("LSTM Metrics:", lstm_metrics)
    
    # Compare models
    comparison = compare_models(
        {'metrics': arima_metrics},
        {'metrics': lstm_metrics}
    )

