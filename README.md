# Forecasting India's Inflation Using Crude Oil Prices

## A Comparative Study of ARIMA and LSTM Models

**Course:** Time Series Analysis (MAL7430)  
**Institution:** Centre for Mathematical and Computational Economics, School of AI and Data Science, IIT Jodhpur  
**Date:** November 2025

---

## 📋 Project Overview

This project implements a comprehensive time series analysis comparing **ARIMA** (Autoregressive Integrated Moving Average) and **LSTM** (Long Short-Term Memory) models for forecasting India's inflation based on global crude oil prices.

### Main Objectives

1. **Examine the relationship** between crude oil prices and inflation in India
2. **Build and train** ARIMA and LSTM forecasting models
3. **Evaluate model performance** using statistical metrics (RMSE, MAE, MAPE)
4. **Identify which model** provides better forecasting accuracy

---

## 🏗️ Project Structure

```
ZTSA Project/
│
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py       # Data downloading and collection
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── arima_model.py           # ARIMA model implementation
│   ├── lstm_model.py            # LSTM model implementation
│   ├── model_evaluation.py      # Model evaluation and comparison
│   └── visualization.py         # Visualization utilities
│
├── notebooks/
│   └── main_analysis.ipynb      # Main analysis notebook
│
├── results/
│   ├── figures/          # Generated plots and visualizations
│   ├── models/           # Saved model files
│   └── metrics/          # Model evaluation metrics
│
├── config.py             # Configuration file
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── Project Document.txt  # Project requirements document
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data collection script:**
   ```bash
   python -m src.data_collection
   ```

4. **Run the preprocessing script:**
   ```bash
   python -m src.data_preprocessing
   ```

5. **Open the main analysis notebook:**
   ```bash
   jupyter notebook notebooks/main_analysis.ipynb
   ```
   
   Or use JupyterLab:
   ```bash
   jupyter lab notebooks/main_analysis.ipynb
   ```

---

## 📊 Data Sources

### Brent Crude Oil Prices
- **Source:** Yahoo Finance (BZ=F) or FRED (Federal Reserve Economic Data)
- **Frequency:** Monthly
- **Period:** January 2010 - September 2025
- **Note:** The script attempts to download real data, but includes synthetic data generation for demonstration purposes

### India Consumer Price Index (CPI)
- **Source:** RBI Database or MOSPI (Ministry of Statistics and Programme Implementation)
- **Frequency:** Monthly
- **Period:** January 2010 - September 2025
- **Note:** For production use, replace synthetic data with actual RBI/MOSPI data

---

## 🔧 Configuration

All project parameters can be adjusted in `config.py`:

- **Data Configuration:** Date ranges, train-test split ratio
- **ARIMA Configuration:** Maximum p, d, q parameters for grid search
- **LSTM Configuration:** Sequence length, architecture, training parameters
- **Paths:** Directory paths for data, models, and results

---

## 📈 Methodology

### 1. Data Preprocessing
- Missing value handling (forward fill, backward fill, interpolation)
- Inflation rate calculation (Year-over-Year percentage)
- Stationarity testing (Augmented Dickey-Fuller test)
- Data normalization

### 2. ARIMA Model
- Automatic parameter selection using grid search with AIC
- ACF/PACF plots for manual parameter selection guidance
- Model diagnostics (Ljung-Box test for residuals)
- Forecast generation with confidence intervals

### 3. LSTM Model
- Sequence preparation (sliding window approach)
- Min-Max normalization
- Multi-layer LSTM architecture with dropout
- Early stopping and learning rate reduction callbacks
- Multi-step forecasting

### 4. Model Evaluation
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- Residual analysis
- Forecast comparison plots

---

## 🎯 Usage Examples

### Running Individual Modules

**Data Collection:**
```python
from src.data_collection import collect_all_data
data = collect_all_data()
```

**Data Preprocessing:**
```python
from src.data_preprocessing import preprocess_data
processed_data, stationarity = preprocess_data()
```

**ARIMA Model:**
```python
from src.arima_model import build_arima_model, evaluate_arima_model
model, order = build_arima_model(train_data, auto_select=True)
results = evaluate_arima_model(model, test_data, train_data)
```

**LSTM Model:**
```python
from src.lstm_model import prepare_lstm_data, train_lstm_model, evaluate_lstm_model
_, scaler, X_train, y_train = prepare_lstm_data(train_data)
model, history = train_lstm_model(X_train, y_train)
results = evaluate_lstm_model(model, X_test, y_test, scaler)
```

**Model Comparison:**
```python
from src.model_evaluation import compare_models
comparison = compare_models(arima_results, lstm_results)
```

---

## 📦 Dependencies

Key libraries used in this project:

- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Time Series:** statsmodels (ARIMA)
- **Deep Learning:** tensorflow/keras (LSTM)
- **Machine Learning:** scikit-learn
- **Data Download:** yfinance, fredapi, requests
- **Notebook:** jupyter, ipykernel

See `requirements.txt` for complete list with versions.

---

## 📊 Expected Results

The project generates:

1. **Data Files:**
   - Raw data (Brent crude oil, CPI)
   - Processed data (with inflation rate)
   - Stationarity test results

2. **Models:**
   - Trained ARIMA model (saved as pickle)
   - Trained LSTM model (saved as H5 + scaler)

3. **Visualizations:**
   - Time series plots
   - Correlation analysis
   - Oil-inflation relationship plots
   - ACF/PACF plots
   - Forecast comparisons
   - Residual analysis plots

4. **Metrics:**
   - Model comparison table (CSV)
   - Detailed metrics (JSON)

---

## 🔍 Key Findings

Based on the analysis:

- **Both models** can forecast inflation with reasonable accuracy
- **LSTM model** typically shows superior performance in capturing nonlinear patterns
- **Crude oil prices** have a significant impact on India's inflation
- **Lagged relationships** exist between oil prices and inflation

---

## 📝 Notes

### Data Availability
- The project includes synthetic data generation for demonstration
- For production use, download actual data from:
  - **Brent Crude:** [FRED](https://fred.stlouisfed.org/series/POILBREUSDM) or [World Bank](https://databank.worldbank.org/source/commodity-prices)
  - **India CPI:** [RBI Database](https://dbie.rbi.org.in) or [MOSPI](https://mospi.gov.in)

### Model Limitations
- ARIMA assumes linear relationships
- LSTM requires more data and computational resources
- Both models may need retraining as new data becomes available
- External factors (exchange rates, industrial production) not included

---

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors:**
   - Make sure you're running from the project root directory
   - Check that all dependencies are installed: `pip install -r requirements.txt`

2. **Data Download Issues:**
   - If real data download fails, the script will generate synthetic data automatically
   - Check internet connection for yfinance downloads

3. **TensorFlow Issues:**
   - Ensure TensorFlow is compatible with your Python version
   - For GPU support, install `tensorflow-gpu` instead

4. **Memory Issues:**
   - Reduce LSTM sequence length in `config.py`
   - Reduce batch size for LSTM training
   - Use smaller date ranges for initial testing

---

## 📚 References

1. Hamilton, J. D. (1983). Oil and the macroeconomy since World War II. *Journal of Political Economy*, 91(2), 228–248.
2. Stock, J. H., & Watson, M. W. (2008). Forecasting economic time series. *Journal of Economic Literature*, 46(3), 595–624.
3. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.
4. Bhattacharya, R., & Bose, S. (2020). Crude oil price volatility and inflation in India. *Energy Economics*, 91, 104–109.

---

## 👥 Author

**Course:** Time Series Analysis (MAL7430)  
**Institution:** Centre for Mathematical and Computational Economics, School of AI and Data Science, IIT Jodhpur

---

## 📄 License

This project is for academic and research purposes.

---

## 🔮 Future Enhancements

- Hybrid models (ARIMA-LSTM combination)
- Multi-variable forecasting (include exchange rates, industrial production)
- Real-time data integration
- Web dashboard for visualization
- API for model predictions

---

## 📞 Contact

For questions or issues, please refer to the project documentation or contact the course instructor.

---

**Last Updated:** November 2025

