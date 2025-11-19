"""
Monte Carlo Simulation for Portfolio Robustness Testing
Generates synthetic return scenarios based on historical parameters
and tests portfolio strategies for robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """
    Monte Carlo simulation for testing portfolio strategy robustness.
    """

    def __init__(self, historical_returns: pd.DataFrame, n_simulations: int = 1000):
        """
        Initialize Monte Carlo simulator.

        Parameters:
        -----------
        historical_returns : pd.DataFrame
            Historical asset returns used to estimate parameters
        n_simulations : int
            Number of simulation runs
        """
        self.historical_returns = historical_returns
        self.n_simulations = n_simulations
        self.n_assets = len(historical_returns.columns)
        self.asset_names = historical_returns.columns.tolist()

        # Estimate parameters from historical data
        self._estimate_parameters()

    def _estimate_parameters(self):
        """Estimate mean, covariance, and distribution parameters from historical data."""
        self.mean_returns = self.historical_returns.mean()
        self.cov_matrix = self.historical_returns.cov()
        self.std_returns = self.historical_returns.std()

        # Test for normality
        self.is_normal = {}
        for col in self.historical_returns.columns:
            _, p_value = stats.normaltest(self.historical_returns[col].dropna())
            self.is_normal[col] = p_value > 0.05

    def generate_scenarios(self, n_periods: int = 252,
                          method: str = 'multivariate_normal') -> List[pd.DataFrame]:
        """
        Generate Monte Carlo scenarios.

        Parameters:
        -----------
        n_periods : int
            Number of periods to simulate (default: 252 trading days = 1 year)
        method : str
            'multivariate_normal': Assumes normal distribution with correlation
            'bootstrap': Bootstrap resampling from historical data
            'garch': GARCH-based simulation (captures volatility clustering)

        Returns:
        --------
        list
            List of DataFrames, each containing one simulated scenario
        """
        scenarios = []

        print(f"Generating {self.n_simulations} Monte Carlo scenarios ({method})...")

        for i in range(self.n_simulations):
            if method == 'multivariate_normal':
                scenario = self._simulate_multivariate_normal(n_periods)
            elif method == 'bootstrap':
                scenario = self._simulate_bootstrap(n_periods)
            else:
                raise ValueError(f"Unknown method: {method}")

            scenarios.append(scenario)

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{self.n_simulations} scenarios")

        return scenarios

    def _simulate_multivariate_normal(self, n_periods: int) -> pd.DataFrame:
        """
        Simulate returns using multivariate normal distribution.
        Preserves correlation structure.
        """
        # Generate correlated random returns
        simulated = np.random.multivariate_normal(
            mean=self.mean_returns.values,
            cov=self.cov_matrix.values,
            size=n_periods
        )

        return pd.DataFrame(simulated, columns=self.asset_names)

    def _simulate_bootstrap(self, n_periods: int) -> pd.DataFrame:
        """
        Simulate returns using bootstrap resampling.
        Non-parametric approach that preserves actual return distribution.
        """
        # Randomly sample with replacement from historical returns
        sample_indices = np.random.choice(
            len(self.historical_returns),
            size=n_periods,
            replace=True
        )

        return self.historical_returns.iloc[sample_indices].reset_index(drop=True)

    def test_strategy(self, strategy_func, scenarios: List[pd.DataFrame],
                     **strategy_kwargs) -> pd.DataFrame:
        """
        Test a portfolio strategy across all Monte Carlo scenarios.

        Parameters:
        -----------
        strategy_func : callable
            Function that takes returns and returns portfolio weights
        scenarios : list
            List of simulated return DataFrames
        **strategy_kwargs
            Additional arguments for strategy function

        Returns:
        --------
        pd.DataFrame
            Results for each scenario (returns, Sharpe, drawdown, etc.)
        """
        print(f"\nTesting strategy across {len(scenarios)} scenarios...")

        results = []

        for i, scenario in enumerate(scenarios):
            # Get weights from strategy
            weights = strategy_func(scenario, **strategy_kwargs)

            # Calculate portfolio returns
            portfolio_returns = (scenario * weights).sum(axis=1)

            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

            # Max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            results.append({
                'scenario': i + 1,
                'total_return': total_return * 100,
                'annual_return': annual_return * 100,
                'annual_volatility': annual_vol * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100
            })

            if (i + 1) % 100 == 0:
                print(f"  Tested {i + 1}/{len(scenarios)} scenarios")

        results_df = pd.DataFrame(results)
        return results_df

    def summarize_results(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize Monte Carlo results with confidence intervals.

        Parameters:
        -----------
        results : pd.DataFrame
            Results from test_strategy

        Returns:
        --------
        pd.DataFrame
            Summary statistics with percentiles
        """
        summary = pd.DataFrame({
            'Mean': results.mean(),
            'Median': results.median(),
            'Std Dev': results.std(),
            '5th Percentile': results.quantile(0.05),
            '25th Percentile': results.quantile(0.25),
            '75th Percentile': results.quantile(0.75),
            '95th Percentile': results.quantile(0.95),
            'Min': results.min(),
            'Max': results.max()
        })

        return summary.drop('scenario', errors='ignore')

    def probability_of_outperformance(self, strategy_results: pd.DataFrame,
                                     benchmark_results: pd.DataFrame,
                                     metric: str = 'annual_return') -> float:
        """
        Calculate probability that strategy outperforms benchmark.

        Parameters:
        -----------
        strategy_results : pd.DataFrame
            Monte Carlo results for strategy
        benchmark_results : pd.DataFrame
            Monte Carlo results for benchmark
        metric : str
            Metric to compare (e.g., 'annual_return', 'sharpe_ratio')

        Returns:
        --------
        float
            Probability of outperformance (0 to 1)
        """
        outperformance = (strategy_results[metric] > benchmark_results[metric]).mean()
        return outperformance


def main():
    """
    Example usage of Monte Carlo simulation.
    """
    print("="*80)
    print("MONTE CARLO SIMULATION")
    print("="*80)

    # Load historical data
    features_data = pd.read_csv('../data/features_data.csv', index_col=0, parse_dates=True)

    portfolio_assets = ['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN', 'CVX_RETURN', 'IEF_RETURN']
    portfolio_returns = features_data[portfolio_assets].copy()
    portfolio_returns.columns = ['WTI', 'XLE', 'XOM', 'CVX', 'IEF']
    portfolio_returns = portfolio_returns.dropna()

    print(f"\nHistorical data: {len(portfolio_returns)} days")
    print(f"Assets: {list(portfolio_returns.columns)}")

    # Initialize simulator
    simulator = MonteCarloSimulator(portfolio_returns, n_simulations=1000)

    # Generate scenarios
    scenarios = simulator.generate_scenarios(n_periods=252, method='multivariate_normal')

    # Test equal-weight strategy
    def equal_weight_strategy(returns):
        return np.ones(len(returns.columns)) / len(returns.columns)

    results = simulator.test_strategy(equal_weight_strategy, scenarios)

    # Summarize
    print("\n" + "="*80)
    print("MONTE CARLO RESULTS SUMMARY")
    print("="*80)
    summary = simulator.summarize_results(results)
    print(summary.round(2))

    # Save results
    results.to_csv('../data/monte_carlo_results.csv', index=False)
    summary.to_csv('../data/monte_carlo_summary.csv')

    print("\n✓ Results saved to:")
    print("  - ../data/monte_carlo_results.csv")
    print("  - ../data/monte_carlo_summary.csv")


if __name__ == "__main__":
    main()
