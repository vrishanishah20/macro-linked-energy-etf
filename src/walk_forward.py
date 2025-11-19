"""
Tests strategies on rolling windows to avoid overfitting.
Train on in-sample period, test on out-of-sample period.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
import warnings

warnings.filterwarnings('ignore')


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.
    """

    def __init__(self, returns_data: pd.DataFrame,
                 train_period_years: int = 5,
                 test_period_years: int = 1,
                 step_years: int = 1):
        """
        Initialize walk-forward backtester.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Historical returns data
        train_period_years : int
            Training window size in years
        test_period_years : int
            Testing window size in years
        step_years : int
            Step size for rolling window in years
        """
        self.returns = returns_data
        self.train_days = train_period_years * 252
        self.test_days = test_period_years * 252
        self.step_days = step_years * 252

        self.train_period_years = train_period_years
        self.test_period_years = test_period_years
        self.step_years = step_years

    def generate_windows(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test window pairs for walk-forward analysis.

        Returns:
        --------
        list
            List of (train_data, test_data) tuples
        """
        windows = []
        start_idx = 0

        while start_idx + self.train_days + self.test_days <= len(self.returns):
            train_end_idx = start_idx + self.train_days
            test_end_idx = train_end_idx + self.test_days

            train_data = self.returns.iloc[start_idx:train_end_idx]
            test_data = self.returns.iloc[train_end_idx:test_end_idx]

            windows.append((train_data, test_data))

            start_idx += self.step_days

        return windows

    def backtest_strategy(self, strategy_func: Callable,
                         optimize_func: Callable = None,
                         **strategy_kwargs) -> pd.DataFrame:
        """
        Run walk-forward backtest for a strategy.

        Parameters:
        -----------
        strategy_func : callable
            Function that returns portfolio weights given returns data
        optimize_func : callable, optional
            Function to optimize strategy parameters on training data
        **strategy_kwargs
            Additional arguments for strategy

        Returns:
        --------
        pd.DataFrame
            Test period results for each window
        """
        windows = self.generate_windows()
        print(f"\nWalk-Forward Backtest: {len(windows)} windows")
        print(f"  Train: {self.train_period_years} years, Test: {self.test_period_years} year(s)")
        print(f"  Step: {self.step_years} year(s)")

        all_results = []

        for i, (train_data, test_data) in enumerate(windows):
            window_num = i + 1
            train_start = train_data.index[0].strftime('%Y-%m')
            train_end = train_data.index[-1].strftime('%Y-%m')
            test_start = test_data.index[0].strftime('%Y-%m')
            test_end = test_data.index[-1].strftime('%Y-%m')

            print(f"\n  Window {window_num}/{len(windows)}:")
            print(f"    Train: {train_start} to {train_end}")
            print(f"    Test:  {test_start} to {test_end}")

            # Optimize parameters on training data if optimization function provided
            if optimize_func:
                optimized_params = optimize_func(train_data)
                strategy_kwargs.update(optimized_params)

            # Get weights from strategy (trained on train_data)
            weights = strategy_func(train_data, **strategy_kwargs)

            # Test on out-of-sample data
            test_portfolio_returns = (test_data * weights).sum(axis=1)

            # Calculate metrics
            total_return = (1 + test_portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(test_portfolio_returns)) - 1
            annual_vol = test_portfolio_returns.std() * np.sqrt(252)
            sharpe = test_portfolio_returns.mean() / test_portfolio_returns.std() * np.sqrt(252) if test_portfolio_returns.std() > 0 else 0

            # Drawdown
            cumulative = (1 + test_portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            all_results.append({
                'window': window_num,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'total_return': total_return * 100,
                'annual_return': annual_return * 100,
                'annual_volatility': annual_vol * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100
            })

            print(f"      Return: {annual_return*100:.2f}%, Sharpe: {sharpe:.2f}, DD: {max_drawdown*100:.2f}%")

        results_df = pd.DataFrame(all_results)
        return results_df

    def aggregate_results(self, results: pd.DataFrame) -> Dict:
        """
        Aggregate walk-forward results across all windows.

        Parameters:
        -----------
        results : pd.DataFrame
            Results from backtest_strategy

        Returns:
        --------
        dict
            Aggregated metrics
        """
        aggregate = {
            'avg_annual_return': results['annual_return'].mean(),
            'std_annual_return': results['annual_return'].std(),
            'avg_sharpe_ratio': results['sharpe_ratio'].mean(),
            'avg_volatility': results['annual_volatility'].mean(),
            'avg_max_drawdown': results['max_drawdown'].mean(),
            'worst_window_return': results['annual_return'].min(),
            'best_window_return': results['annual_return'].max(),
            'percent_positive_windows': (results['annual_return'] > 0).mean() * 100,
            'total_windows': len(results)
        }

        return aggregate


def main():
    """
    Example usage of walk-forward backtesting.
    """
    print("="*80)
    print("WALK-FORWARD BACKTESTING")
    print("="*80)

    # Load data
    features_data = pd.read_csv('../data/features_data.csv', index_col=0, parse_dates=True)

    portfolio_assets = ['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN', 'CVX_RETURN', 'IEF_RETURN']
    portfolio_returns = features_data[portfolio_assets].copy()
    portfolio_returns.columns = ['WTI', 'XLE', 'XOM', 'CVX', 'IEF']
    portfolio_returns = portfolio_returns.dropna()

    print(f"\nData period: {portfolio_returns.index[0]} to {portfolio_returns.index[-1]}")
    print(f"Total days: {len(portfolio_returns)}")

    # Initialize walk-forward backtester
    wf_backtester = WalkForwardBacktester(
        portfolio_returns,
        train_period_years=5,
        test_period_years=1,
        step_years=1
    )

    # Test equal-weight strategy
    def equal_weight_strategy(train_data):
        return np.ones(len(train_data.columns)) / len(train_data.columns)

    results = wf_backtester.backtest_strategy(equal_weight_strategy)

    # Aggregate results
    print("\n" + "="*80)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("="*80)

    aggregate = wf_backtester.aggregate_results(results)
    for key, value in aggregate.items():
        print(f"  {key}: {value:.2f}")

    # Save results
    results.to_csv('../data/walk_forward_results.csv', index=False)
    pd.DataFrame([aggregate]).to_csv('../data/walk_forward_summary.csv', index=False)

    print("\n✓ Results saved to:")
    print("  - ../data/walk_forward_results.csv")
    print("  - ../data/walk_forward_summary.csv")


if __name__ == "__main__":
    main()
