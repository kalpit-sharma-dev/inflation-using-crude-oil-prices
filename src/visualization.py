"""
Visualization Module
Create various plots and visualizations for the analysis
"""

import pandas as pd
import numpy as np
import matplotlib
# Set non-interactive backend to avoid GUI issues
matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

# Set style
try:
    plt.style.use(config.PLOT_CONFIG['style'])
except:
    # Fallback if style doesn't exist
    plt.style.use('default')
sns.set_palette("husl")


def plot_time_series(data, columns=None, title="Time Series Plot", save_path=None):
    """
    Plot time series data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    columns : list, optional
        Columns to plot (if None, plots all numeric columns)
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=config.PLOT_CONFIG['figsize'])
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        ax.plot(data.index, data[col], label=col, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_correlation(data, columns=None, title="Correlation Analysis", save_path=None):
    """
    Plot correlation heatmap
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data for correlation analysis
    columns : list, optional
        Columns to include in correlation
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_data = data[columns]
    correlation_matrix = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_data_distribution(data, columns=None, title="Data Distribution", save_path=None):
    """
    Plot distribution of data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to plot
    columns : list, optional
        Columns to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]
    
    for i, col in enumerate(columns):
        ax = axes[i]
        data[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title(f'{col} Distribution', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_oil_inflation_relationship(oil_data, inflation_data, save_path=None):
    """
    Plot relationship between crude oil prices and inflation
    
    Parameters:
    -----------
    oil_data : pd.Series
        Crude oil price data
    inflation_data : pd.Series
        Inflation rate data
    save_path : str, optional
        Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(oil_data.index, oil_data.values, 'b-', label='Brent Crude Oil Price', linewidth=2)
    line2 = ax1_twin.plot(inflation_data.index, inflation_data.values, 'r-', 
                          label='Inflation Rate', linewidth=2)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Oil Price (USD)', color='b')
    ax1_twin.set_ylabel('Inflation Rate (%)', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title('Brent Crude Oil Prices vs Inflation Rate Over Time', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot with lag
    ax2 = axes[1]
    # Align data
    common_index = oil_data.index.intersection(inflation_data.index)
    oil_aligned = oil_data.loc[common_index]
    inflation_aligned = inflation_data.loc[common_index]
    
    # Try different lags
    max_corr = 0
    best_lag = 0
    for lag in range(0, 13):
        if lag == 0:
            corr = oil_aligned.corr(inflation_aligned)
        else:
            oil_lagged = oil_aligned.shift(lag)
            corr = oil_lagged.corr(inflation_aligned)
        
        if abs(corr) > abs(max_corr):
            max_corr = corr
            best_lag = lag
    
    if best_lag > 0:
        oil_lagged = oil_aligned.shift(best_lag)
        ax2.scatter(oil_lagged, inflation_aligned, alpha=0.6, s=50)
        ax2.set_xlabel(f'Brent Crude Oil Price (lagged by {best_lag} months)')
        title_suffix = f' (Best correlation at lag {best_lag}: {max_corr:.3f})'
    else:
        ax2.scatter(oil_aligned, inflation_aligned, alpha=0.6, s=50)
        ax2.set_xlabel('Brent Crude Oil Price')
        title_suffix = f' (Correlation: {max_corr:.3f})'
    
    ax2.set_ylabel('Inflation Rate (%)')
    ax2.set_title('Oil Price vs Inflation Rate' + title_suffix, 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line
    if best_lag > 0:
        z = np.polyfit(oil_lagged.dropna(), inflation_aligned[oil_lagged.dropna().index], 1)
    else:
        z = np.polyfit(oil_aligned.dropna(), inflation_aligned[oil_aligned.dropna().index], 1)
    p = np.poly1d(z)
    
    x_line = np.linspace(ax2.get_xlim()[0], ax2.get_xlim()[1], 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_forecast_comparison(actual, arima_forecast, lstm_forecast, 
                            arima_conf_int=None, title="Forecast Comparison", save_path=None):
    """
    Plot forecast comparison between ARIMA and LSTM models
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    arima_forecast : pd.Series
        ARIMA forecasts
    lstm_forecast : pd.Series
        LSTM forecasts
    arima_conf_int : pd.DataFrame, optional
        ARIMA confidence intervals
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot actual
    ax.plot(actual.index, actual.values, 'k-', label='Actual', linewidth=2.5)
    
    # Plot ARIMA forecast
    ax.plot(arima_forecast.index, arima_forecast.values, 'b--', 
            label='ARIMA Forecast', linewidth=2, alpha=0.8)
    
    # Plot ARIMA confidence intervals if provided
    if arima_conf_int is not None:
        ax.fill_between(arima_forecast.index, 
                       arima_conf_int.iloc[:, 0].values,
                       arima_conf_int.iloc[:, 1].values,
                       alpha=0.2, color='blue', label='ARIMA 95% CI')
    
    # Plot LSTM forecast
    ax.plot(lstm_forecast.index, lstm_forecast.values, 'g--', 
            label='LSTM Forecast', linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Inflation Rate (%)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range('2010-01-01', periods=100, freq='M')
    np.random.seed(42)
    
    oil_data = pd.Series(np.random.randn(100).cumsum() + 60, index=dates)
    inflation_data = pd.Series(np.random.randn(100).cumsum() + 5, index=dates)
    
    plot_oil_inflation_relationship(oil_data, inflation_data)

