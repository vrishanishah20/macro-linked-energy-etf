# Implementation Guide: Macro-Linked Energy ETF

**Author:** Vrishani Shah
**Course:** MSDS 451 - Financial Engineering
**Date:** November 2025

---

## Overview

This guide provides step-by-step instructions for running the complete analysis pipeline for the Macro-Linked Energy ETF project.

## Prerequisites

### 1. Python Environment

Ensure you have Python 3.8+ installed. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. FRED API Key

You need a FRED API key to download macroeconomic data:

1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Request an API key (free)
3. Set as environment variable:

```bash
export FRED_API_KEY='your_key_here'  # On Mac/Linux
set FRED_API_KEY=your_key_here      # On Windows
```

Alternatively, edit `configs/config.py` and set `FRED_API_KEY` directly.

---

## Project Structure

```
macro-linked-energy-etf/
├── src/
│   ├── data_pipeline.py         # Data acquisition & cleaning
│   ├── macro_features.py        # Feature engineering
│   └── portfolio_optimizer.py   # Portfolio optimization & backtesting
├── notebooks/
│   ├── backtest_results.ipynb   # Main backtesting notebook
│   └── results_visualization.ipynb  # Visualizations & analysis
├── data/                        # Output data & results
├── configs/
│   └── config.py                # Configuration parameters
└── docs/
    ├── checkpoint1.md
    └── IMPLEMENTATION_GUIDE.md (this file)
```

---

## Running the Analysis

### Option 1: Run Python Modules Directly

#### Step 1: Data Pipeline

```bash
cd src
python data_pipeline.py
```

This will:
- Fetch price data from Yahoo Finance (WTI, XLE, XOM, CVX, IEF, SHY, SPY)
- Fetch macro data from FRED (Fed Funds, CPI, VIX, etc.)
- Merge and clean the data
- Save to `../data/prices.csv` and `../data/merged_data.csv`

#### Step 2: Feature Engineering

```bash
python macro_features.py
```

This will:
- Calculate rolling momentum, volatility, drawdown
- Compute macro sensitivities and correlations
- Create composite macro score
- Save to `../data/features_data.csv`

#### Step 3: Portfolio Optimization

```bash
python portfolio_optimizer.py
```

This will:
- Run backtests for all strategies
- Calculate performance metrics
- Save results to `../data/backtest_*.csv` and `../data/performance_metrics.csv`

### Option 2: Run Jupyter Notebooks (Recommended)

#### Step 1: Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

#### Step 2: Run Backtesting Notebook

Open `backtest_results.ipynb` and run all cells. This notebook will:
1. Execute the complete data pipeline
2. Build all features
3. Run backtests for 6 strategies:
   - SPY Buy-and-Hold (benchmark)
   - Equal Weight
   - Volatility Parity
   - Risk Parity
   - Mean-Variance (Markowitz)
   - Macro-Adaptive
4. Calculate performance metrics
5. Generate cumulative performance, rolling metrics, and drawdown charts

#### Step 3: Run Visualization Notebook

Open `results_visualization.ipynb` and run all cells. This notebook will:
1. Create correlation heatmaps
2. Plot risk-return scatter
3. Analyze performance by macro regime
4. Compare weight allocations
5. Perform statistical tests
6. Generate distribution plots

---

## Expected Outputs

After running the complete pipeline, you should have:

### Data Files (in `data/` directory)

- `prices.csv` - Raw price data
- `merged_data.csv` - Merged prices + macro indicators
- `features_data.csv` - Complete feature set
- `backtest_spy_benchmark.csv` - SPY benchmark results
- `backtest_equal_weight.csv` - Equal-weight strategy results
- `backtest_volatility_parity.csv` - Vol-parity results
- `backtest_risk_parity.csv` - Risk-parity results
- `backtest_mean_variance.csv` - Mean-variance optimization results
- `backtest_macro_adaptive.csv` - Macro-adaptive results
- `performance_metrics.csv` - Summary performance metrics

### Figures (in `data/` directory)

- `cumulative_performance.png` - Equity curves
- `rolling_metrics.png` - Rolling Sharpe and volatility
- `drawdown_analysis.png` - Drawdown plots
- `asset_correlation_heatmap.png` - Asset return correlations
- `macro_correlation_heatmap.png` - Asset vs. macro factor correlations
- `risk_return_scatter.png` - Risk-return scatter plot
- `performance_comparison_bars.png` - Bar chart comparisons
- `macro_adaptive_weights.png` - Weight evolution over time
- `regime_performance.png` - Performance by macro regime
- `return_distributions.png` - Return distribution histograms

---

## Customization

### Modify Parameters

Edit `configs/config.py` to change:
- Date range for backtesting
- Rebalancing frequency
- Transaction costs
- Macro score weights
- Risk parameters

### Add New Assets

In `src/data_pipeline.py`, modify the `tickers` dictionary:

```python
self.tickers = {
    'WTI': 'CL=F',
    'XLE': 'XLE',
    # Add your ticker here:
    'NEW_ASSET': 'TICKER'
}
```

### Add New Strategies

In `src/portfolio_optimizer.py`, create a new method in the `PortfolioOptimizer` class:

```python
def your_strategy(self, **kwargs):
    # Your allocation logic here
    weights = np.array([...])
    return weights
```

Then backtest it in the notebook.

---

## Troubleshooting

### Common Issues

1. **FRED API errors**
   - Check that your API key is set correctly
   - Verify you have an active internet connection
   - Some FRED series may have limited historical data

2. **Missing data warnings**
   - Some assets may have shorter price histories
   - The pipeline will forward-fill and drop rows with excessive missing values
   - Check the console output for specific warnings

3. **Optimization convergence issues**
   - Risk parity and mean-variance optimization may not always converge
   - If this happens, the code falls back to equal-weight
   - Consider adjusting the lookback window or initial weights

4. **Memory issues with large datasets**
   - If running into memory issues, reduce the date range
   - Use shorter lookback windows for rolling calculations

---

## Performance Expectations

### Runtime

- Data Pipeline: ~2-5 minutes
- Feature Engineering: ~1-2 minutes
- Backtesting: ~3-5 minutes
- Total: ~10 minutes for complete analysis

### Hardware

- Minimum: 4GB RAM, dual-core CPU
- Recommended: 8GB+ RAM, quad-core CPU

---

## Next Steps

After running the analysis:

1. Review performance metrics in `performance_metrics.csv`
2. Examine charts to understand strategy behavior
3. Analyze regime-specific performance
4. Interpret results and write up discussion section
5. Prepare final term paper with findings

---

## References

- **Data Sources:**
  - Yahoo Finance: https://finance.yahoo.com
  - FRED: https://fred.stlouisfed.org

- **Python Libraries:**
  - yfinance: https://github.com/ranaroussi/yfinance
  - fredapi: https://github.com/mortada/fredapi
  - PyPortfolioOpt: https://pyportfolioopt.readthedocs.io

---

## Support

For questions or issues:
- Review the inline code documentation
- Check console output for error messages
- Consult the checkpoint documents in `docs/`

**Good luck with your analysis!**
