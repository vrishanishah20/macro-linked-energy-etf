# Macro-Linked Energy ETF: Comprehensive Analysis and Business Viability Assessment

**A rigorous empirical study evaluating whether macro-informed allocation strategies can generate superior risk-adjusted returns in the energy sector**

**Author:** Vrishani Shah
**Course:** MSDS 451 – Financial Engineering
**Institution:** Northwestern University
**Date:** December 2024
**Professor** Dr.Tom Miller

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Executive Summary

This project implements and evaluates a macro-linked energy sector ETF using **22.4 years of historical data** (2002-2024), encompassing multiple allocation strategies, Monte Carlo simulation, walk-forward validation, and comprehensive fee analysis.

### Key Findings

**Business Conclusion: NOT VIABLE as standalone investment fund**

- **SPY outperformed** all energy strategies by 4.4-6.4% annually
- **Best energy strategy** (Volatility Parity): 4.44% vs SPY's 8.81%
- **After typical fees** (2% + 20%): Net returns drop to 1.92% annually
- **Underperformance:** -6.87% vs SPY after fees
- **High volatility:** 20%+ volatility without compensating returns
- **Severe drawdowns:** -63% maximum drawdown vs -60% for SPY

### Value of Research

While the fund failed commercial viability tests, the project successfully demonstrates:
- Complete quantitative investment pipeline implementation
- Rigorous backtesting methodology (22+ years)
- Monte Carlo robustness testing (2,000 scenarios)
- Walk-forward validation (17 out-of-sample windows)
- Comprehensive fee structure analysis
- Importance of empirical testing before launching investment products

---

## 🎯 Research Objective

**Can macroeconomic signals (oil prices, interest rates, volatility) improve risk-adjusted returns in energy sector portfolios compared to passive market benchmarks?**

This study tests multiple allocation strategies:
1. Equal-weight
2. Volatility parity
3. Risk parity
4. Mean-variance optimization (Markowitz)
5. Macro-adaptive weighting

Against the **S&P 500 (SPY)** benchmark.

---

## 📊 Project Structure

```
macro-linked-energy-etf/
├── src/
│   ├── data_pipeline.py           # Data acquisition & cleaning
│   ├── macro_features.py          # Feature engineering (149 features)
│   ├── portfolio_optimizer.py     # Portfolio allocation strategies
│   ├── monte_carlo.py             # Monte Carlo simulation
│   ├── walk_forward.py            # Walk-forward backtesting
│   ├── fee_analysis.py            # Fee impact analysis
│   ├── comprehensive_analysis.py  # Complete analysis pipeline
│   └── generate_results.py        # Visualization generation
├── notebooks/
│   └── backtest_results.ipynb     # Interactive analysis notebook
├── data/                          # All results (CSVs & charts)
├── configs/
│   └── config.py                  # Configuration parameters
├── docs/
│   ├── checkpoint1.md             # Project checkpoints
│   └── IMPLEMENTATION_GUIDE.md    # How to run the analysis
└── requirements.txt               # Python dependencies
```

---
## Contents of the Final Report

The final report contains:
- Introduction
- Literature Review
- Methods
- Data sources
- Feature engineering (drawdowns, beta, Sharpe, macro sensitivities, etc.)
- Portfolio strategies
- Backtesting design
- Monte Carlo simulation
- Walk-forward validation
- Fee structure modeling
- Results
- Full-period performance
- CAPM Alpha/Beta
- Monte Carlo outcomes
- Walk-forward out-of-sample performance
- Fee impact
- Management Recommendation
- Conclusion
Appendices
References

---

## 📈 Data Sources

### Time Period
- **Time Period:** July 2002 - December 2024 (22.4 years, 5,634 trading days)
- **Reason for difference:** Data availability for some assets

### Assets
**Yahoo Finance (daily adjusted prices):**
- **WTI Crude Oil** (`CL=F`) - Commodity exposure
- **XLE** - Energy Select Sector SPDR Fund
- **XOM** - ExxonMobil Corporation
- **CVX** - Chevron Corporation
- **IEF** - iShares 7-10 Year Treasury Bond ETF (diversifier)
- **SPY** - S&P 500 ETF (benchmark)

**FRED API (macroeconomic indicators):**
- Federal Funds Rate (`FEDFUNDS`)
- Consumer Price Index (`CPIAUCSL`)
- Industrial Production (`INDPRO`)
- Unemployment Rate (`UNRATE`)
- GDP (`GDP`)
- 10Y-2Y Treasury Spread (`T10Y2Y`)
- VIX Volatility Index (`VIXCLS`)
- 10-Year Treasury Rate (`DGS10`)

---

## 🔬 Methodology

### 1. Feature Engineering (149 Features Created)

**Rolling Indicators:**
- Momentum (21, 63, 126-day windows)
- Volatility (annualized, multiple windows)
- Drawdowns (current and max over various periods)
- Beta to SPY (market sensitivity)
- Sharpe ratios (risk-adjusted performance)

**Macro Sensitivity Metrics:**
- Response to Federal Funds Rate changes
- Correlation with inflation (CPI)
- VIX sensitivity
- Yield curve exposure

**Composite Macro Score:**
```
C_t = 0.4 × z(WTI_return) - 0.3 × z(ΔFedFunds) - 0.3 × z(ΔVIX)
```

**Regime Classification:**
- **Bullish:** C_t > 0.75 (214 days, 3.8%)
- **Neutral:** -0.75 ≤ C_t ≤ 0.75 (4,870 days, 86.4%)
- **Bearish:** C_t < -0.75 (277 days, 4.9%)

### 2. Portfolio Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **SPY Benchmark** | Buy-and-hold S&P 500 | No rebalancing |
| **Equal Weight** | 1/N allocation | 20% each asset |
| **Volatility Parity** | Inverse-volatility weights | 126-day lookback |
| **Risk Parity** | Equal risk contribution | Optimized numerically |
| **Mean-Variance** | Markowitz optimization | Max Sharpe ratio |
| **Macro-Adaptive** | Dynamic tilts based on C_t | ±20% max tilt |

**Rebalancing:** Monthly
**Transaction costs:** 10 basis points per rebalance

### 3. Robustness Testing

**Monte Carlo Simulation (2,000 scenarios):**
- 1,000 multivariate normal scenarios
- 1,000 bootstrap resampling scenarios
- Tests strategy across synthetic data preserving historical correlation structure

**Walk-Forward Backtesting (17 windows):**
- Train period: 5 years
- Test period: 1 year
- Step size: 1 year
- Avoids overfitting by using true out-of-sample data

### 4. Fee Structure Analysis

**Management Fees:** 0%, 1%, 2%, 3%, 4% annually on AUM
**Performance Fees:** 0%, 5%, 10%, 15%, 20%, 25% on returns above benchmark
**Methodology:** High-water mark with hurdle rate

### 5. Performance Metrics

- Total and annualized returns
- Sharpe ratio (risk-adjusted returns)
- Sortino ratio (downside risk-adjusted)
- Maximum drawdown and Calmar ratio
- **Alpha:** Rp - [Rf + β(Rm - Rf)]
- **Beta:** Cov(Rp, Rm) / Var(Rm)
- Tracking error
- Win rate and consistency

---

## 🎯 Results

### Full-Period Performance (2002-2024)

| Strategy | Total Return | Annual Return | Volatility | Sharpe | Alpha | Beta | Max DD |
|----------|--------------|---------------|------------|--------|-------|------|--------|
| **SPY Benchmark** 🏆 | **559.91%** | **8.81%** | 18.92% | **0.44** | — | 1.00 | -59.58% |
| **Volatility Parity** | 164.39% | 4.44% | 13.52% | 0.31 | +0.49% | 0.45 | **-42.82%** |
| **Macro-Adaptive** | 155.50% | 4.28% | 21.38% | 0.25 | -0.42% | 0.71 | -62.93% |
| **Equal Weight** | 156.50% | 4.30% | 20.99% | 0.25 | -0.38% | 0.69 | -63.98% |
| **Mean-Variance** | 69.02% | 2.38% | **6.89%** | 0.21 | **+2.40%** | -0.11 | -26.72% |

### Monte Carlo Simulation Results

**Equal-Weight Strategy (1,000 scenarios):**

| Metric | Mean | Median | 5th Percentile | 95th Percentile |
|--------|------|--------|----------------|-----------------|
| Annual Return | 7.46% | 4.40% | -24.97% | 50.78% |
| Sharpe Ratio | 0.34 | 0.31 | -1.27 | 2.01 |
| Max Drawdown | -19.86% | -18.97% | -33.34% | -10.11% |

**Interpretation:** Wide outcome dispersion indicates high uncertainty and regime-dependence.

### Walk-Forward Validation

| Metric | Value |
|--------|-------|
| Number of Windows | 17 |
| Average Annual Return | 0.09% |
| Worst Window | -29.36% (2014-2015 oil crash) |
| Best Window | +16.92% (2011-2012) |
| Positive Windows | 58.8% |

**Conclusion:** Minimal out-of-sample profitability; strategy fails robustness test.

### Fee Impact Analysis

**Macro-Adaptive Strategy with Different Fee Structures:**

| Fee Structure | Gross Return | Net Return | Fee Drag |
|---------------|--------------|------------|----------|
| No Fees | 5.25% | 5.25% | 0% |
| 1% Management | 5.25% | 4.21% | 1.04% |
| 2% Management | 5.25% | 3.17% | 2.08% |
| **2% + 20% Performance** | 5.25% | **1.92%** | **3.33%** |
| 2% + 25% Performance | 5.25% | 1.61% | 3.64% |

**Critical Finding:** Typical hedge fund fees (2% + 20%) reduce investor returns by **63%**.

### Business Viability Assessment

```
═══════════════════════════════════════════════════════════
                  INVESTMENT COMPARISON
═══════════════════════════════════════════════════════════

SPY ETF (passive):
  Annual Return:   8.81%
  After Fees:      8.78% (0.03% expense ratio)

Best Energy Strategy (Macro-Adaptive):
  Gross Return:    4.28%
  Net Return:      1.92% (after 2% + 20% fees)

UNDERPERFORMANCE:  -6.87% annually

═══════════════════════════════════════════════════════════
                    BUSINESS VERDICT
═══════════════════════════════════════════════════════════

STATUS: NOT VIABLE AS STANDALONE FUND

RECOMMENDATION: DO NOT LAUNCH

Rationale:
  • Massive underperformance vs low-cost index funds
  • Fees destroy already poor returns
  • High volatility without compensating gains
  • Severe sector concentration risk
  • No competitive advantage or unique value proposition
  • Legal/reputational risk of sustained underperformance

═══════════════════════════════════════════════════════════
```

---

## 💡 Key Insights

### Why Energy Strategies Failed

1. **Sector Concentration Risk**
   - All assets correlated during crashes
   - No diversification benefit within energy sector

2. **Multiple Severe Crashes**
   - 2008 Financial Crisis: -59% drawdown
   - 2014-2016 Oil Crash: WTI from $100 → $26
   - 2020 COVID Crash: WTI briefly negative

3. **Structural Headwinds**
   - Tech sector dominance drove SPY
   - Energy sector transition (renewables)
   - Demand shocks and oversupply

4. **Fee Impact**
   - 2% + 20% structure cuts returns by 2/3
   - Compounds dramatically over 22 years

5. **Macro Signals Insufficient**
   - Tactical tilts couldn't overcome sector headwinds
   - Crash magnitudes overwhelmed allocation benefits

### What Worked (Relatively)

1. **Volatility Parity** - Best risk-adjusted returns among energy strategies
2. **Treasury Diversification** - IEF reduced drawdowns during crashes
3. **Mean-Variance** - Lowest volatility (6.89%) and positive alpha (+2.40%)

### Lessons Learned

1. ✅ **Diversification is crucial** - sector concentration is dangerous
2. ✅ **Fees matter enormously** - even small fees compound over time
3. ✅ **Passive indexing is hard to beat** - especially after fees
4. ✅ **Rigorous testing essential** - prevented launching a failed product
5. ✅ **Alpha is rare** - most active strategies underperform

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

### FRED API Key

Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html):

```bash
export FRED_API_KEY='your_key_here'
```

### Quick Start

**Option 1: Run Complete Analysis**

```bash
cd src
FRED_API_KEY='your_key' python comprehensive_analysis.py
```

This generates:
- Full-period backtesting (2002-2024)
- Monte Carlo simulation (2,000 scenarios)
- Walk-forward validation (17 windows)
- Fee analysis (30 scenarios)
- Alpha & Beta calculations
- Business viability assessment
- 11 publication-ready charts

**Option 2: Step-by-Step**

```bash
# 1. Fetch data
python data_pipeline.py

# 2. Engineer features
python macro_features.py

# 3. Run backtests
python generate_results.py

# 4. Monte Carlo & walk-forward
python monte_carlo.py
python walk_forward.py

# 5. Fee analysis
python fee_analysis.py
```

**Option 3: Interactive Notebook**

```bash
cd notebooks
jupyter notebook backtest_results.ipynb
```

---

## 📁 Output Files

All results saved to `/data/`:

### Performance Metrics
- `full_period_metrics.csv` - Complete performance table
- `alpha_beta_analysis.csv` - Alpha & Beta for each strategy
- `business_viability.csv` - Investment recommendation
- `summary_table.csv` - Rankings and comparisons

### Monte Carlo
- `monte_carlo_normal_results.csv` - 1,000 scenarios (parametric)
- `monte_carlo_bootstrap_results.csv` - 1,000 scenarios (non-parametric)
- `monte_carlo_*_summary.csv` - Statistical summaries

### Walk-Forward
- `walk_forward_results.csv` - 17 out-of-sample test windows
- `walk_forward_summary.csv` - Aggregated metrics

### Fee Analysis
- `fee_impact_analysis.csv` - Net returns under various fee structures
- `fee_sensitivity_grid.csv` - Complete fee sensitivity matrix

### Visualizations (PNG)
- `cumulative_performance.png` - Equity curves (2002-2024)
- `rolling_metrics.png` - Rolling Sharpe & volatility
- `drawdown_analysis.png` - Drawdown from peak
- `risk_return_scatter.png` - Risk-return positioning
- `performance_comparison_bars.png` - Bar chart comparisons
- `asset_correlation_heatmap.png` - Correlation matrix
- `return_distributions.png` - Return histograms
- `fee_impact_chart.png` - Fee structure impact
- `monte_carlo_distribution.png` - MC simulation results
- `walk_forward_performance.png` - Out-of-sample performance
- `alpha_beta_scatter.png` - Alpha vs Beta positioning

---

## 📚 References

### Energy Markets
- Bouchouev, I. (2023). *Virtual Barrels: Quantitative Trading in the Oil Market.* Springer.
- Downey, M. (2009). *Oil 101.* Wooden Table Press.
- Edwards, D. W. (2017). *Energy Trading & Investing (2nd ed.).* McGraw-Hill.
- Hamilton, J. D. (2009). "Causes and Consequences of the Oil Shock of 2007–08." *Brookings Papers on Economic Activity.*

### Portfolio Theory
- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance.*
- Sharpe, W. F. (1963). "Capital Asset Prices." *Journal of Finance.*
- Qian, E. (2005). "Risk Parity Portfolios." *PanAgora Asset Management.*
- Asness, C. et al. (2012). "Leverage Aversion and Risk Parity." *Financial Analysts Journal.*

### Active Management
- Grinold, R., & Kahn, R. (2000). *Active Portfolio Management.* McGraw-Hill.
- Greyserman, A., & Kaminski, K. (2014). *Trend Following with Managed Futures.*
- Antonacci, G. (2014). *Dual Momentum Investing.*

### Backtesting & Validation
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Trivedi, P., & Kyal, H. (2021). *Python for Finance Cookbook.* Packt.
- Chan, E. (2020). *Quantitative Trading (2nd ed.).* Wiley.

---

## 🏆 Academic Context

**Course:** MSDS 451 – Financial Engineering
**Program:** Master of Science in Data Science
**Institution:** Northwestern University
**Instructor:** [Instructor Name]
**Term:** Fall 2024

### Learning Objectives Demonstrated

✅ Implement complete quantitative investment pipeline
✅ Apply portfolio optimization theory (Markowitz, risk parity)
✅ Conduct rigorous empirical backtesting
✅ Perform Monte Carlo simulation for robustness
✅ Execute walk-forward validation to avoid overfitting
✅ Analyze fee impact on investor returns
✅ Calculate Alpha and Beta using CAPM framework
✅ Make data-driven business recommendations
✅ Communicate complex quantitative results effectively

---

## 🤝 Contributing

This is an academic research project. For questions or suggestions:
- Open an issue on GitHub
- Contact: vrishani.shah@northwestern.edu

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.

- **NOT investment advice**
- **NOT a recommendation** to trade or invest
- Past performance does not predict future results
- All investments carry risk of loss
- Consult a qualified financial advisor before making investment decisions

The conclusion that this fund is "not viable" is based on historical backtesting and does not constitute financial, legal, or professional advice.

---
## GenAI Tools Used

This project used generative AI tools (ChatGPT GPT-5.0, OpenAI, 2025) to assist with:
- Literature review summarization
- Report structure & writing refinement
- Debugging Python code
- Organizing repository structure
- Creating final README.md
All quantitative analysis, modeling, programming, results generation, and conclusions were performed by the author.
AI-generated content was verified, corrected, and validated manually.
Conversation logs are preserved in the submitted screenshots and ChatGPT session transcripts.

---

## 🙏 Acknowledgments

- **FRED API** for macroeconomic data
- **Yahoo Finance** for market data
- **Northwestern University MSDS Program** for academic support
- **Open-source Python community** for excellent libraries (pandas, numpy, scipy, matplotlib, seaborn)

---

**Last Updated:** December 2024
**Status:** ✅ Complete - Final Analysis
**Results:** ❌ Fund Not Viable for Launch

---

