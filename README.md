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
