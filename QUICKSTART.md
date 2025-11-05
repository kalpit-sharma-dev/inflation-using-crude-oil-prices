# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline

**Option A: Using the main script (Recommended)**
```bash
python main.py
```

This will:
- ✅ Collect/download data
- ✅ Preprocess data
- ✅ Train ARIMA model
- ✅ Train LSTM model
- ✅ Compare models
- ✅ Generate all visualizations
- ✅ Save results

**Option B: Using Jupyter Notebook (For Interactive Analysis)**
```bash
jupyter notebook notebooks/main_analysis.ipynb
```

Then run all cells in the notebook.

### Step 3: View Results

After running, check the following directories:

- **`results/figures/`** - All generated plots and visualizations
- **`results/metrics/`** - Model comparison tables and metrics
- **`results/models/`** - Saved trained models

### Step 4: Customize (Optional)

Edit `config.py` to adjust:
- Date ranges
- Train-test split ratio
- ARIMA parameters (max p, d, q)
- LSTM architecture and training parameters

---

## 📊 What You'll Get

### Data Files
- `data/raw/brent_crude_oil.csv` - Brent crude oil prices
- `data/raw/india_cpi.csv` - India CPI data
- `data/raw/merged_data.csv` - Combined dataset
- `data/processed/processed_data.csv` - Preprocessed data with inflation rate

### Models
- `results/models/arima_model.pkl` - Trained ARIMA model
- `results/models/lstm_model.h5` - Trained LSTM model
- `results/models/lstm_model_scaler.pkl` - LSTM scaler

### Visualizations
- `results/figures/oil_inflation_relationship.png` - Oil-inflation correlation
- `results/figures/correlation_matrix.png` - Correlation heatmap
- `results/figures/forecast_comparison.png` - Model forecast comparison
- `results/figures/model_predictions_comparison.png` - Prediction comparison
- `results/figures/model_residuals.png` - Residual analysis
- And more...

### Metrics
- `results/metrics/model_comparison.csv` - Performance comparison table
- `results/metrics/model_comparison.json` - Detailed metrics in JSON

---

## ⚠️ Important Notes

### Data
- The script will attempt to download real data from Yahoo Finance
- If download fails, synthetic data will be generated automatically
- For production use, replace with actual RBI/MOSPI data

### System Requirements
- **RAM:** At least 4GB recommended (8GB for smoother LSTM training)
- **Disk Space:** ~500MB for data and models
- **Time:** Full pipeline takes ~5-10 minutes (depending on system)

### First Run
- The first run may take longer due to data download and model training
- Subsequent runs will be faster if data is already downloaded

---

## 🔧 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root directory
cd "C:\ZTSA Project"

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Data Download Issues
- The script will automatically generate synthetic data if download fails
- Check internet connection
- Firewall may block yfinance downloads

### TensorFlow Issues
```bash
# For CPU-only systems
pip install tensorflow

# For GPU support
pip install tensorflow-gpu
```

### Memory Issues
- Reduce LSTM batch size in `config.py`
- Reduce sequence length in `config.py`
- Use smaller date range for testing

---

## 📚 Next Steps

1. **Explore the Notebook:** Open `notebooks/main_analysis.ipynb` for detailed analysis
2. **Modify Parameters:** Edit `config.py` to experiment with different settings
3. **Add Real Data:** Replace synthetic data with actual RBI/MOSPI data
4. **Extend Models:** Add more features or try hybrid models

---

## 💡 Tips

- **For Quick Testing:** Reduce date range in `config.py` (e.g., last 5 years)
- **For Better Accuracy:** Use actual RBI/MOSPI data instead of synthetic
- **For Faster Training:** Reduce LSTM epochs in `config.py`
- **For Production:** Implement proper error handling and logging

---

## 📞 Need Help?

1. Check the main `README.md` for detailed documentation
2. Review the `Project Document.txt` for project requirements
3. Examine the code comments in source files
4. Check the notebook for usage examples

---

**Happy Forecasting! 📈**

