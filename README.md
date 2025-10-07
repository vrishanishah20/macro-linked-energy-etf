# macro-linked-energy-etf
A data science project exploring how oil price shocks, macroeconomic indicators, and news sentiment affect energy sector equities (XOM, CVX), forming the foundation of a rules-based, actively managed ETF.


## Introduction

The purpose of this research is to explore how macro-level events—particularly oil price movements, inventory reports, and major economic indicators—affect the performance of leading U.S. energy sector equities such as **ExxonMobil (XOM)** and **Chevron (CVX)**.  
This study aligns with **Grinold and Kahn’s (2000)** insight that “superior forecasts come from superior information.” In this context, superior information means combining well-structured macroeconomic data with event-based signals such as OPEC announcements or crude-oil inventory surprises.

The ultimate goal is to design a **rules-based, actively managed ETF** that dynamically adjusts exposure to the energy sector based on these macro signals. Potential users of this knowledge include:
- Quantitative fund managers and analysts seeking event-driven trading strategies.
- Individual investors interested in systematic sector-rotation ETFs.
- Financial researchers exploring oil-equity linkages and macro-sensitivity.

---

## Literature Review

Previous studies have shown that **oil prices and energy equities** exhibit strong but asymmetric relationships. In particular, oil price increases tend to boost the revenues of integrated energy firms like ExxonMobil and Chevron, while price declines often reduce margins and investor sentiment (Hamilton, 2009).

Key references shaping this study include:
- **Grinold & Kahn (2000)** – *Active Portfolio Management*: Emphasizes forecasting alpha using superior information.
- **Markowitz (1952, 1956)** and **Sharpe (1963)** – Foundational works on portfolio selection and risk-adjusted returns.
- **Greyserman & Kaminski (2014)** – Trend-following and risk management frameworks.
- **Hamilton (2009)** – Empirical analysis of oil shocks and macroeconomic effects.
- **Covel (2017)** and **Clenow (2023)** – Practical systematic trading and risk-control techniques.

Recent ETF and hedge-fund literature (Zuckerman, 2019; Harris, 2023) demonstrates the growing interest in algorithmic management strategies that use structured macro and sentiment data for real-time portfolio adjustment.  
This project builds upon that foundation by integrating **macroeconomic**, **commodity**, and **news sentiment** data into a single predictive framework for energy equities.

---

## Methods

### Data Sources
1. **Yahoo Finance API (`yfinance`)**
   - Tickers: XOM, CVX  
   - Data: Daily adjusted close prices, 2018–2024  
   - Frequency: Daily

2. **Federal Reserve Economic Data (FRED API)**
   - Indicators: Federal Funds Rate, CPI, Industrial Production  
   - Provides macroeconomic context and regime indicators.

3. **U.S. Energy Information Administration (EIA API)**
   - Variables: Daily WTI spot prices, weekly U.S. crude inventories  
   - Captures supply–demand shocks relevant to oil-sensitive stocks.

4. **Reuters / Investing.com RSS Feeds**
   - News related to OPEC decisions, energy policy, and global oil events.  
   - Processed with **VADER** for daily sentiment scoring.

### Analytical Workflow
1. **Data Alignment**
   - All sources merged on a **common business-day index**.  
   - Weekly and monthly macro data forward-filled to match daily trading data.

2. **Signal Construction**
   - **Oil Shock Signal (OSS):** z-score of daily WTI changes.  
   - **Inventory Surprise (IS):** difference between actual inventory change and 52-week seasonal average.  
   - **Sentiment (SNT):** average daily VADER score from energy news.  
   - **Rate Regime (RR):** 3-month change in Federal Funds Rate.  

3. **Composite Score (C<sub>t</sub>):**  
- If C<sub>t</sub> ≥ 0.75 → Overweight energy positions  
- If C<sub>t</sub> ≤ -0.75 → Underweight or hedge  
- Otherwise → Neutral  

4. **ETF Strategy Rules**
- Portfolio tilts applied to a baseline energy exposure (e.g., XLE or equal-weight XOM/CVX).  
- 10-day moving-average filter to smooth signals.  
- Max position change: 15 percentage points per day.  
- Target volatility: 12% annualized.

5. **Evaluation**
- Correlation and lag analysis between macro signals and returns.  
- Event studies around major WTI shocks and OPEC news.  
- Regression of future returns on composite signals.  
- Metrics: Information coefficient (IC), hit rate, and Sharpe ratio.

---

## Results (Preliminary)

Initial exploratory analysis yielded several observations:
- **Strong correlation** between WTI daily returns and energy stock performance (ρ ≈ 0.65).  
- **Asymmetric effects:** Negative oil shocks have a larger downside impact on equities than positive shocks have upside impact.  
- **Inventory surprises** correlate more strongly during high-volatility regimes.  
- **Sentiment signals** add predictive value on ambiguous oil-movement days (neutral or small ΔWTI).  

Figures produced so far:
- `figures/price_series_xom_cvx_wti.png` – Price trajectories 2018–2024  
- `figures/event_study_wti.png` – Average energy stock reaction ±5 days around major oil shocks  
- `figures/corr_heatmap.png` – Correlation among macro, sentiment, and return features  

These results indicate that macro-event signals have potential as **timing and risk-control mechanisms** for an actively managed ETF.

---

## Conclusions

The proposed **Macro-Linked Energy ETF** aims to translate macroeconomic and event-based insights into a transparent, rules-based trading framework.  
Early results support the feasibility of a **composite macro-sentiment score** as a driver for dynamic sector weighting.  

**Concerns and next steps:**
- Data availability and synchronization (especially EIA and RSS timing) remain challenges.  
- The backtesting framework must incorporate **transaction costs and slippage**.  
- Further robustness tests needed to confirm predictive power out-of-sample.  

Future work will include completing the ETF **prospectus**, formalizing the algorithmic trading rules, and validating performance against benchmark indices.

---

## References

- Carver, C. (2015). *Systematic Trading: A Unique New Method for Designing Trading and Investing Systems*.  
- Clenow, A. (2023). *Trading Evolved: Anyone Can Build Killer Trading Strategies in Python*.  
- Covel, M. (2017). *Trend Following*.  
- Grinold, R., & Kahn, R. (2000). *Active Portfolio Management*.  
- Greyserman, A., & Kaminski, K. (2014). *Trend Following with Managed Futures*.  
- Hamilton, J. (2009). *Causes and Consequences of the Oil Shock of 2007–08*, *Brookings Papers on Economic Activity*.  
- Markowitz, H. (1952). *Portfolio Selection*, *Journal of Finance*.  
- Sharpe, W. (1963). *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk*.  
- Zuckerman, G. (2019). *The Man Who Solved the Market: How Jim Simons Launched the Quant Revolution*.  
- Harris, J. (2023). *Commodity ETFs Explained*.  

---

**Author:** Vrishani Shah  
**Course:** MSDS 413 – Time Series and Forecasting  
**Instructor:** Tom Miller  
**Date:** October 2025
