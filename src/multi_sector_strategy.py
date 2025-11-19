"""
Multi-Sector Macro-Adaptive Strategy
=====================================
Improved version diversified across multiple sectors.

This addresses the key failure of the energy-only fund:
sector concentration risk.

Author: Vrishani Shah
Course: MSDS 451 - Financial Engineering
"""

import pandas as pd
import numpy as np

# PROPOSED PORTFOLIO
MULTI_SECTOR_PORTFOLIO = {
    # Energy (20%)
    'XLE': 0.10,   # Energy sector
    'XOM': 0.05,   # Exxon
    'CVX': 0.05,   # Chevron

    # Technology (20%)
    'QQQ': 0.15,   # Nasdaq 100
    'MSFT': 0.05,  # Microsoft

    # Healthcare (15%)
    'XLV': 0.15,   # Healthcare sector

    # Financials (15%)
    'XLF': 0.15,   # Financial sector

    # Consumer (10%)
    'XLY': 0.10,   # Consumer discretionary

    # Bonds (20%)
    'IEF': 0.10,   # 7-10yr Treasury
    'TLT': 0.10,   # 20+ yr Treasury
}

# MACRO SIGNALS FOR EACH SECTOR
SECTOR_SIGNALS = {
    'Energy': ['WTI', 'Industrial_Production', 'Dollar_Index'],
    'Technology': ['Nasdaq_Momentum', 'Rate_Regime', 'Innovation_Index'],
    'Healthcare': ['Aging_Demographics', 'FDA_Approvals', 'Insurance_Spending'],
    'Financials': ['Yield_Curve', 'Credit_Spreads', 'Fed_Policy'],
    'Consumer': ['Consumer_Confidence', 'Employment', 'Retail_Sales'],
    'Bonds': ['Inflation', 'Fed_Funds', 'Recession_Probability']
}

# EXPECTED IMPROVEMENT
EXPECTED_METRICS = {
    'Energy_Only_Fund': {
        'annual_return': 4.28,
        'volatility': 21.38,
        'sharpe': 0.25,
        'max_drawdown': -62.93,
        'alpha': -0.42
    },
    'Multi_Sector_Fund': {
        'annual_return': 7.5,      # Estimated
        'volatility': 15.0,         # Reduced via diversification
        'sharpe': 0.45,            # Improved
        'max_drawdown': -40.0,     # Better downside protection
        'alpha': 0.5               # Positive alpha from rotation
    },
    'SPY_Benchmark': {
        'annual_return': 8.81,
        'volatility': 18.92,
        'sharpe': 0.44,
        'max_drawdown': -59.58,
        'alpha': 0.0
    }
}

def compare_strategies():
    """
    Compare multi-sector approach to energy-only and SPY.
    """
    df = pd.DataFrame(EXPECTED_METRICS).T

    print("="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(df.round(2))
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print("\nEnergy-Only Fund:")
    print("  ❌ 4.28% return (trails SPY by 4.53%)")
    print("  ❌ 21.38% volatility (higher than SPY)")
    print("  ❌ Negative alpha (-0.42%)")
    print("  ❌ Severe drawdown (-62.93%)")
    print("  VERDICT: NOT VIABLE")

    print("\nMulti-Sector Fund (Proposed):")
    print("  ✅ 7.5% return (only 1.3% behind SPY)")
    print("  ✅ 15.0% volatility (lower than SPY)")
    print("  ✅ Positive alpha (+0.5%)")
    print("  ✅ Better drawdown (-40% vs -59.58%)")
    print("  VERDICT: POTENTIALLY VIABLE")

    print("\nWith 1% + 10% fees:")
    multi_net = 7.5 - 1.0 - (7.5 - 2.0) * 0.10  # Rough estimate
    spy_net = 8.81 - 0.03
    print(f"  Multi-Sector net return: {multi_net:.2f}%")
    print(f"  SPY net return: {spy_net:.2f}%")
    print(f"  Gap: {multi_net - spy_net:.2f}%")

    if multi_net >= spy_net * 0.95:  # Within 5% of SPY
        print("\n  ✅ COMPETITIVE with SPY after fees!")
    else:
        print("\n  🟡 Still trails SPY but much better than energy-only")

if __name__ == "__main__":
    compare_strategies()
