"""
Data Collection Module
Downloads and collects CPI and Brent Crude Oil price data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import os
import config


def download_brent_crude_oil(start_date='2010-01-01', end_date='2025-09-30'):
    """
    Download Brent Crude Oil prices from Yahoo Finance (BZ=F)
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date and Brent_Price columns
    """
    print("Downloading Brent Crude Oil prices...")
    
    # Yahoo Finance ticker for Brent Crude Oil futures
    ticker = "BZ=F"
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, interval='1mo')
        
        if data.empty:
            raise ValueError("No data downloaded from Yahoo Finance")
        
        # Extract close prices and resample to monthly
        if 'Close' in data.columns:
            brent_data = pd.DataFrame({
                'Date': data.index,
                'Brent_Price': data['Close'].values
            })
        else:
            # If multi-level columns, get close price
            brent_data = pd.DataFrame({
                'Date': data.index,
                'Brent_Price': data[('Close', ticker)].values if isinstance(data.columns, pd.MultiIndex) else data.iloc[:, -1].values
            })
        
        # Remove NaN values
        brent_data = brent_data.dropna()
        
        # Set Date as index
        brent_data.set_index('Date', inplace=True)
        
        # Resample to end of month if needed
        brent_data = brent_data.resample('M').last()
        
        print(f"Successfully downloaded {len(brent_data)} records")
        return brent_data
        
    except Exception as e:
        print(f"Error downloading from Yahoo Finance: {e}")
        print("Generating synthetic Brent Crude Oil data for demonstration...")
        return generate_synthetic_brent_data(start_date, end_date)


def generate_synthetic_brent_data(start_date='2010-01-01', end_date='2025-09-30'):
    """
    Generate synthetic Brent Crude Oil price data for demonstration
    Based on historical trends and volatility
    """
    print("Generating synthetic Brent Crude Oil data...")
    
    # Create monthly date range
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Simulate oil prices with realistic trends
    np.random.seed(42)
    n = len(dates)
    
    # Base price with trends
    base_price = 60
    
    # Add trend (volatility over time)
    trend = np.linspace(0, 20, n)
    
    # Add seasonal component (higher in winter months)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
    
    # Random walk component
    random_walk = np.cumsum(np.random.randn(n) * 3)
    
    # Add some shocks (like 2014-2015 crash, COVID-19, etc.)
    shocks = np.zeros(n)
    # 2014-2015 crash
    crash_idx = np.where(dates >= pd.Timestamp('2014-06-01'))[0]
    if len(crash_idx) > 0:
        idx_start = crash_idx[0]
        idx_end = min(idx_start + 18, n)
        shocks[idx_start:idx_end] = -30
    # COVID-19 crash
    covid_idx = np.where(dates >= pd.Timestamp('2020-03-01'))[0]
    if len(covid_idx) > 0:
        idx_start = covid_idx[0]
        idx_end = min(idx_start + 3, n)
        shocks[idx_start:idx_end] = -25
    # Recovery
    recovery_idx = np.where((dates >= pd.Timestamp('2020-06-01')) & (dates <= pd.Timestamp('2022-06-01')))[0]
    if len(recovery_idx) > 0:
        shocks[recovery_idx] = 20
    
    # Combine all components
    prices = base_price + trend + seasonal + random_walk + shocks
    
    # Ensure prices are positive
    prices = np.maximum(prices, 20)
    
    brent_data = pd.DataFrame({
        'Date': dates,
        'Brent_Price': prices
    })
    brent_data.set_index('Date', inplace=True)
    
    print(f"Generated {len(brent_data)} synthetic records")
    return brent_data


def download_india_cpi(start_date='2010-01-01', end_date='2025-09-30'):
    """
    Download India CPI data
    Note: This function attempts to download from public sources
    For actual project, use RBI or MOSPI official data
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date and CPI columns
    """
    print("Downloading India CPI data...")
    
    # For demonstration, we'll generate synthetic CPI data
    # In real project, replace this with actual data download from RBI/MOSPI
    print("Note: Generating synthetic CPI data. Replace with actual RBI/MOSPI data for production.")
    return generate_synthetic_cpi_data(start_date, end_date)


def generate_synthetic_cpi_data(start_date='2010-01-01', end_date='2025-09-30'):
    """
    Generate synthetic India CPI data based on historical trends
    CPI is typically correlated with oil prices but with lag
    """
    print("Generating synthetic India CPI data...")
    
    # Create monthly date range
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n = len(dates)
    
    np.random.seed(123)
    
    # Base CPI index (2010 base = 100)
    base_cpi = 100
    
    # Trend component (gradual inflation)
    trend = np.linspace(0, 60, n)  # CPI increases over time
    
    # Seasonal component (higher inflation in certain months)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n) / 12) + 1.5 * np.cos(2 * np.pi * np.arange(n) / 6)
    
    # Random component
    random_comp = np.cumsum(np.random.randn(n) * 0.5)
    
    # Combine all components
    cpi_values = base_cpi + trend + seasonal + random_comp
    
    # Ensure CPI is positive and increasing overall
    cpi_values = np.maximum(cpi_values, base_cpi)
    
    cpi_data = pd.DataFrame({
        'Date': dates,
        'CPI': cpi_values
    })
    cpi_data.set_index('Date', inplace=True)
    
    print(f"Generated {len(cpi_data)} synthetic records")
    return cpi_data


def collect_all_data():
    """
    Main function to collect all required data
    """
    print("="*60)
    print("DATA COLLECTION")
    print("="*60)
    
    # Create data directory if it doesn't exist
    os.makedirs(config.PATHS['raw_data'], exist_ok=True)
    
    # Download Brent Crude Oil data
    brent_data = download_brent_crude_oil(
        start_date=config.DATA_CONFIG['start_date'],
        end_date=config.DATA_CONFIG['end_date']
    )
    
    # Download India CPI data
    cpi_data = download_india_cpi(
        start_date=config.DATA_CONFIG['start_date'],
        end_date=config.DATA_CONFIG['end_date']
    )
    
    # Save raw data
    brent_file = os.path.join(config.PATHS['raw_data'], 'brent_crude_oil.csv')
    cpi_file = os.path.join(config.PATHS['raw_data'], 'india_cpi.csv')
    
    brent_data.to_csv(brent_file)
    cpi_data.to_csv(cpi_file)
    
    print(f"\nRaw data saved to:")
    print(f"  - {brent_file}")
    print(f"  - {cpi_file}")
    
    # Merge datasets
    merged_data = pd.merge(brent_data, cpi_data, left_index=True, right_index=True, how='inner')
    
    # Save merged data
    merged_file = os.path.join(config.PATHS['raw_data'], 'merged_data.csv')
    merged_data.to_csv(merged_file)
    
    print(f"  - {merged_file}")
    print("\nData collection completed successfully!")
    
    return merged_data


if __name__ == "__main__":
    data = collect_all_data()
    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nData summary:")
    print(data.describe())

