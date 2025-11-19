"""
Fee Structure Analysis
======================
Analyzes impact of management and performance fees on investor returns.

Management Fee: 1-4% annually on AUM
Performance Fee: 5-25% on returns above benchmark (high-water mark)

Author: Vrishani Shah
Course: MSDS 451 - Financial Engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeeAnalyzer:
    """
    Analyzes impact of fees on portfolio returns.
    """

    def __init__(self, portfolio_returns: pd.Series,
                 benchmark_returns: pd.Series = None):
        """
        Initialize fee analyzer.

        Parameters:
        -----------
        portfolio_returns : pd.Series
            Daily portfolio returns
        benchmark_returns : pd.Series, optional
            Benchmark returns for performance fee calculation
        """
        self.returns = portfolio_returns
        self.benchmark = benchmark_returns
        self.dates = portfolio_returns.index

    def apply_management_fee(self, annual_fee_pct: float) -> pd.Series:
        """
        Apply annual management fee to returns.

        Parameters:
        -----------
        annual_fee_pct : float
            Annual management fee as percentage (e.g., 2.0 for 2%)

        Returns:
        --------
        pd.Series
            Returns after management fees
        """
        # Convert annual fee to daily
        daily_fee = annual_fee_pct / 100 / 252

        # Subtract daily fee from returns
        returns_after_fee = self.returns - daily_fee

        return returns_after_fee

    def apply_performance_fee(self, annual_fee_pct: float,
                             hurdle_rate: float = 0.0,
                             use_high_water_mark: bool = True) -> pd.Series:
        """
        Apply performance fee on returns above hurdle (high-water mark).

        Parameters:
        -----------
        annual_fee_pct : float
            Performance fee percentage (e.g., 20 for 20% of profits)
        hurdle_rate : float
            Minimum return before performance fee applies (annual %)
        use_high_water_mark : bool
            If True, fee only charged on returns above previous peak

        Returns:
        --------
        pd.Series
            Returns after performance fees
        """
        cumulative_value = (1 + self.returns).cumprod()
        returns_after_perf_fee = self.returns.copy()

        if use_high_water_mark:
            high_water_mark = 1.0
            daily_hurdle = (1 + hurdle_rate / 100) ** (1 / 252) - 1

            for i, date in enumerate(self.dates):
                current_value = cumulative_value.iloc[i]

                if current_value > high_water_mark:
                    # Calculate profit above high-water mark
                    profit = current_value - high_water_mark

                    # Apply performance fee
                    fee = profit * (annual_fee_pct / 100)

                    # Update high-water mark
                    high_water_mark = current_value

                    # Adjust return for fee
                    # Fee is taken from the gain, so we reduce the return proportionally
                    if i > 0:
                        prev_value = cumulative_value.iloc[i - 1]
                        daily_gain = current_value - prev_value
                        fee_on_gain = daily_gain * (annual_fee_pct / 100)
                        returns_after_perf_fee.iloc[i] -= fee_on_gain / prev_value

        return returns_after_perf_fee

    def apply_combined_fees(self, management_fee_pct: float,
                           performance_fee_pct: float,
                           hurdle_rate: float = 0.0) -> pd.Series:
        """
        Apply both management and performance fees.

        Parameters:
        -----------
        management_fee_pct : float
            Annual management fee (%)
        performance_fee_pct : float
            Performance fee on excess returns (%)
        hurdle_rate : float
            Hurdle rate for performance fee (annual %)

        Returns:
        --------
        pd.Series
            Returns after all fees
        """
        # First apply management fee
        returns_after_mgmt = self.apply_management_fee(management_fee_pct)

        # Then apply performance fee
        temp_analyzer = FeeAnalyzer(returns_after_mgmt, self.benchmark)
        returns_after_all_fees = temp_analyzer.apply_performance_fee(
            performance_fee_pct, hurdle_rate
        )

        return returns_after_all_fees

    def calculate_fee_impact(self, management_fee_pct: float,
                            performance_fee_pct: float) -> Dict:
        """
        Calculate comprehensive fee impact metrics.

        Parameters:
        -----------
        management_fee_pct : float
            Annual management fee (%)
        performance_fee_pct : float
            Performance fee (%)

        Returns:
        --------
        dict
            Fee impact metrics
        """
        # Gross returns (no fees)
        gross_value = (1 + self.returns).cumprod()
        gross_total_return = gross_value.iloc[-1] - 1
        gross_annual_return = (1 + gross_total_return) ** (252 / len(self.returns)) - 1

        # Net returns (after fees)
        net_returns = self.apply_combined_fees(management_fee_pct, performance_fee_pct)
        net_value = (1 + net_returns).cumprod()
        net_total_return = net_value.iloc[-1] - 1
        net_annual_return = (1 + net_total_return) ** (252 / len(net_returns)) - 1

        # Calculate fee drag
        fee_drag_total = gross_total_return - net_total_return
        fee_drag_annual = gross_annual_return - net_annual_return

        return {
            'gross_total_return_pct': gross_total_return * 100,
            'net_total_return_pct': net_total_return * 100,
            'gross_annual_return_pct': gross_annual_return * 100,
            'net_annual_return_pct': net_annual_return * 100,
            'total_fee_drag_pct': fee_drag_total * 100,
            'annual_fee_drag_pct': fee_drag_annual * 100,
            'management_fee_pct': management_fee_pct,
            'performance_fee_pct': performance_fee_pct
        }

    def fee_sensitivity_analysis(self, management_fees: List[float],
                                performance_fees: List[float]) -> pd.DataFrame:
        """
        Analyze sensitivity to different fee structures.

        Parameters:
        -----------
        management_fees : list
            List of management fee percentages to test
        performance_fees : list
            List of performance fee percentages to test

        Returns:
        --------
        pd.DataFrame
            Grid of net returns for different fee combinations
        """
        results = []

        for mgmt_fee in management_fees:
            for perf_fee in performance_fees:
                impact = self.calculate_fee_impact(mgmt_fee, perf_fee)
                results.append({
                    'management_fee': mgmt_fee,
                    'performance_fee': perf_fee,
                    'gross_annual_return': impact['gross_annual_return_pct'],
                    'net_annual_return': impact['net_annual_return_pct'],
                    'fee_drag': impact['annual_fee_drag_pct']
                })

        return pd.DataFrame(results)


def calculate_alpha_beta(portfolio_returns: pd.Series,
                        benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate explicit Alpha and Beta.

    Parameters:
    -----------
    portfolio_returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    risk_free_rate : float
        Annual risk-free rate

    Returns:
    --------
    dict
        Alpha and Beta metrics
    """
    # Align returns
    aligned_returns = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()

    # Calculate Beta (covariance / variance)
    covariance = aligned_returns['portfolio'].cov(aligned_returns['benchmark'])
    benchmark_variance = aligned_returns['benchmark'].var()
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

    # Calculate Alpha (Jensen's alpha)
    # Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    portfolio_return = aligned_returns['portfolio'].mean() * 252  # Annualized
    benchmark_return = aligned_returns['benchmark'].mean() * 252
    alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))

    # Additional metrics
    portfolio_vol = aligned_returns['portfolio'].std() * np.sqrt(252)
    benchmark_vol = aligned_returns['benchmark'].std() * np.sqrt(252)
    correlation = aligned_returns['portfolio'].corr(aligned_returns['benchmark'])

    return {
        'alpha_annual': alpha * 100,  # Annual alpha in %
        'beta': beta,
        'correlation': correlation,
        'portfolio_volatility': portfolio_vol * 100,
        'benchmark_volatility': benchmark_vol * 100,
        'tracking_error': (aligned_returns['portfolio'] - aligned_returns['benchmark']).std() * np.sqrt(252) * 100
    }


def main():
    """
    Example usage of fee analysis.
    """
    print("="*80)
    print("FEE STRUCTURE ANALYSIS")
    print("="*80)

    # Load data
    features_data = pd.read_csv('../data/features_data.csv', index_col=0, parse_dates=True)

    # Example: Load one strategy's results
    strategy_data = pd.read_csv('../data/backtest_equal_weight.csv', index_col=0, parse_dates=True)
    portfolio_returns = strategy_data['portfolio_return']
    spy_returns = features_data['SPY_RETURN'].reindex(portfolio_returns.index).fillna(0)

    # Initialize fee analyzer
    fee_analyzer = FeeAnalyzer(portfolio_returns, spy_returns)

    # Test different fee structures
    print("\n" + "="*80)
    print("FEE IMPACT ANALYSIS")
    print("="*80)

    fee_scenarios = [
        (0, 0, "No Fees"),
        (1, 0, "1% Management Only"),
        (2, 0, "2% Management Only"),
        (1, 20, "1% Management + 20% Performance"),
        (2, 20, "2% Management + 20% Performance"),
        (2, 25, "2% Management + 25% Performance")
    ]

    for mgmt, perf, label in fee_scenarios:
        impact = fee_analyzer.calculate_fee_impact(mgmt, perf)
        print(f"\n{label}:")
        print(f"  Gross Annual Return: {impact['gross_annual_return_pct']:.2f}%")
        print(f"  Net Annual Return: {impact['net_annual_return_pct']:.2f}%")
        print(f"  Fee Drag: {impact['annual_fee_drag_pct']:.2f}%")

    # Calculate Alpha and Beta
    print("\n" + "="*80)
    print("ALPHA & BETA ANALYSIS")
    print("="*80)

    alpha_beta = calculate_alpha_beta(portfolio_returns, spy_returns)
    for key, value in alpha_beta.items():
        print(f"  {key}: {value:.2f}")

    # Fee sensitivity grid
    print("\n" + "="*80)
    print("FEE SENSITIVITY ANALYSIS")
    print("="*80)

    sensitivity = fee_analyzer.fee_sensitivity_analysis(
        management_fees=[0, 1, 2, 3, 4],
        performance_fees=[0, 5, 10, 15, 20, 25]
    )

    sensitivity.to_csv('../data/fee_sensitivity.csv', index=False)
    print("\n✓ Saved: ../data/fee_sensitivity.csv")


if __name__ == "__main__":
    main()
