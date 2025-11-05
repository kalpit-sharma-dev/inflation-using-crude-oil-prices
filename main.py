"""
Main script to run the complete analysis pipeline
Forecasting India's Inflation Using Crude Oil Prices
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use non-interactive backend (fixes Tkinter issues on Windows)
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require GUI

# Import project modules
from src.data_collection import collect_all_data
from src.data_preprocessing import preprocess_data
from src.arima_model import build_arima_model, evaluate_arima_model, save_arima_model
from src.lstm_model import prepare_lstm_data, train_lstm_model, evaluate_lstm_model, save_lstm_model
from src.model_evaluation import compare_models, plot_predictions, plot_residuals
from src.visualization import plot_oil_inflation_relationship, plot_forecast_comparison
import config

def main():
    """
    Main function to run the complete analysis pipeline
    """
    print("="*80)
    print("FORECASTING INDIA'S INFLATION USING CRUDE OIL PRICES")
    print("A Comparative Study of ARIMA and LSTM Models")
    print("="*80)
    
    # Step 1: Data Collection
    print("\n" + "="*80)
    print("STEP 1: DATA COLLECTION")
    print("="*80)
    raw_data = collect_all_data()
    
    # Step 2: Data Preprocessing
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)
    processed_data, stationarity_results = preprocess_data()
    
    # Step 3: Prepare Target Variable
    print("\n" + "="*80)
    print("STEP 3: PREPARING TARGET VARIABLE")
    print("="*80)
    target_variable = 'Inflation_Rate'
    target_data = processed_data[target_variable].dropna()
    
    # Split data
    train_size = int(len(target_data) * config.DATA_CONFIG['train_split'])
    train_data = target_data[:train_size]
    test_data = target_data[train_size:]
    
    print(f"Total data points: {len(target_data)}")
    print(f"Training data: {len(train_data)} ({len(train_data)/len(target_data)*100:.1f}%)")
    print(f"Test data: {len(test_data)} ({len(test_data)/len(target_data)*100:.1f}%)")
    
    # Step 4: ARIMA Model
    print("\n" + "="*80)
    print("STEP 4: BUILDING ARIMA MODEL")
    print("="*80)
    arima_model, arima_order = build_arima_model(train_data, auto_select=True)
    arima_results = evaluate_arima_model(arima_model, test_data, train_data)
    
    print("\nARIMA Model Performance:")
    for metric, value in arima_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save ARIMA model
    save_arima_model(arima_model, arima_order)
    
    # Step 5: LSTM Model
    print("\n" + "="*80)
    print("STEP 5: BUILDING LSTM MODEL")
    print("="*80)
    
    # Prepare LSTM data
    _, scaler, X_train, y_train = prepare_lstm_data(
        train_data,
        sequence_length=config.LSTM_CONFIG['sequence_length'],
        fit_scaler=True
    )
    
    _, _, X_test, y_test = prepare_lstm_data(
        test_data,
        sequence_length=config.LSTM_CONFIG['sequence_length'],
        scaler=scaler,
        fit_scaler=False
    )
    
    # Train LSTM
    lstm_model, lstm_history = train_lstm_model(
        X_train, y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=config.LSTM_CONFIG['epochs'],
        batch_size=config.LSTM_CONFIG['batch_size']
    )
    
    # Evaluate LSTM
    lstm_results = evaluate_lstm_model(
        lstm_model, X_test, y_test, scaler,
        test_data_index=test_data.index
    )
    
    print("\nLSTM Model Performance:")
    for metric, value in lstm_results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save LSTM model
    save_lstm_model(lstm_model, scaler)
    
    # Step 6: Model Comparison
    print("\n" + "="*80)
    print("STEP 6: MODEL COMPARISON")
    print("="*80)
    comparison_table = compare_models(arima_results, lstm_results, save_results=True)
    
    # Step 7: Generate Visualizations
    print("\n" + "="*80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("="*80)
    print("Note: Using non-interactive backend - plots will be saved to files only")
    
    # Oil-Inflation relationship
    plot_oil_inflation_relationship(
        processed_data['Brent_Price'],
        processed_data['Inflation_Rate'],
        save_path=os.path.join(config.PATHS['figures'], 'oil_inflation_relationship.png')
    )
    
    # Forecast comparison
    plot_forecast_comparison(
        test_data,
        arima_results['forecasts'],
        lstm_results['forecasts'],
        arima_conf_int=arima_results.get('conf_int', None),
        title='Forecast Comparison: ARIMA vs LSTM',
        save_path=os.path.join(config.PATHS['figures'], 'forecast_comparison.png')
    )
    
    # Predictions comparison
    plot_predictions(
        test_data,
        arima_results['forecasts'],
        lstm_results['forecasts'],
        title='Model Predictions Comparison'
    )
    
    # Residuals comparison
    arima_residuals = arima_results['residuals']
    lstm_residuals = lstm_results['actual'] - lstm_results['forecasts']
    plot_residuals(arima_residuals, lstm_residuals)
    
    print("\nAll visualizations saved to results/figures/ directory")
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*80)
    print(f"\nData Overview:")
    print(f"  - Total data points: {len(processed_data)}")
    print(f"  - Date range: {processed_data.index.min()} to {processed_data.index.max()}")
    print(f"  - Training size: {len(train_data)} ({len(train_data)/len(target_data)*100:.1f}%)")
    print(f"  - Test size: {len(test_data)} ({len(test_data)/len(target_data)*100:.1f}%)")
    
    print(f"\nARIMA Model:")
    print(f"  - Order: {arima_order}")
    for metric, value in arima_results['metrics'].items():
        print(f"  - {metric}: {value:.4f}")
    
    print(f"\nLSTM Model:")
    print(f"  - Sequence length: {config.LSTM_CONFIG['sequence_length']}")
    print(f"  - Architecture: {config.LSTM_CONFIG['units']}")
    for metric, value in lstm_results['metrics'].items():
        print(f"  - {metric}: {value:.4f}")
    
    print("\n" + "="*80)
    print("All results saved in the 'results' directory")
    print("="*80)

if __name__ == "__main__":
    main()

