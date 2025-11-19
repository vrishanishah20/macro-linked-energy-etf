"""
Configuration file for Macro-Linked Energy ETF project.
"""

import os

# =============================================================================
# API KEYS
# =============================================================================

# FRED API Key - Get yours at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Date range for backtesting 
START_DATE = "1999-01-01"
END_DATE = "2024-12-31"

# Asset tickers
TICKERS = {
    'WTI': 'CL=F',      # WTI Crude Oil futures
    'XLE': 'XLE',       # Energy Select Sector SPDR Fund
    'XOM': 'XOM',       # ExxonMobil
    'CVX': 'CVX',       # Chevron
    'IEF': 'IEF',       # iShares 7-10 Year Treasury Bond ETF
    'SHY': 'SHY',       # iShares 1-3 Year Treasury Bond ETF
    'SPY': 'SPY'        # S&P 500 ETF (benchmark)
}

# FRED series IDs
FRED_SERIES = {
    'FEDFUNDS': 'Federal Funds Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'INDPRO': 'Industrial Production Index',
    'UNRATE': 'Unemployment Rate',
    'GDP': 'Gross Domestic Product',
    'T10Y2Y': '10Y-2Y Treasury Spread',
    'DCOILWTICO': 'WTI Spot Price (FRED)',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'VIXCLS': 'VIX Index'
}

# =============================================================================
# PORTFOLIO PARAMETERS
# =============================================================================

# Risk-free rate (annual)
RISK_FREE_RATE = 0.02  # 2%

# Rebalancing frequency
REBALANCE_FREQ = 'M'  # M=Monthly, W=Weekly, Q=Quarterly

# Transaction costs
TRANSACTION_COST = 0.001  # 10 basis points

# Target volatility for vol-targeting strategies
TARGET_VOLATILITY = 0.12  # 12% annual

# Macro-adaptive parameters
MACRO_WEIGHTS = {
    'wti': 0.4,
    'rates': 0.3,
    'vix': 0.3
}

MACRO_TILT_MAGNITUDE = 0.20  # 20% max tilt

MACRO_THRESHOLDS = {
    'bullish': 0.75,
    'bearish': -0.75
}

# =============================================================================
# OPTIMIZATION PARAMETERS
# =============================================================================

# Lookback windows (in trading days)
LOOKBACK_SHORT = 63   # ~3 months
LOOKBACK_MEDIUM = 126 # ~6 months
LOOKBACK_LONG = 252   # ~1 year

# Optimization constraints
MIN_WEIGHT = 0.0  # No short selling
MAX_WEIGHT = 1.0  # Max position size
MAX_LEVERAGE = 2.0  # Maximum leverage

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# Directory paths
DATA_DIR = '../data'
OUTPUT_DIR = '../data'
FIGURE_DIR = '../data'

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
