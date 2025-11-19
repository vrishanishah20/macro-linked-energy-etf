"""
Implements multiple allocation strategies:
1. Equal-weight baseline
2. Volatility-parity weighting
3. Macro-adaptive weighting
4. Mean-variance optimization (Markowitz)
5. Risk-parity

Author: Vrishani Shah
Course: MSDS 451 - Financial Engineering
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization class implementing multiple allocation strategies.
    """

    def __init__(self, returns_data: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the portfolio optimizer.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            DataFrame of asset returns (columns = assets, rows = dates)
        risk_free_rate : float
            Annual risk-free rate (default: 2%)
        """
        self.returns = returns_data
        self.n_assets = len(returns_data.columns)
        self.asset_names = returns_data.columns.tolist()
        self.risk_free_rate = risk_free_rate

    def equal_weight(self) -> np.ndarray:
        """
        Equal-weight allocation (1/N rule).

        Returns:
        --------
        np.ndarray
            Equal weights for all assets
        """
        return np.ones(self.n_assets) / self.n_assets

    def volatility_parity(self, lookback: int = 126) -> np.ndarray:
        """
        Inverse-volatility weighting (volatility parity).
        Assets with lower volatility get higher weight.

        Parameters:
        -----------
        lookback : int
            Lookback period for volatility estimation

        Returns:
        --------
        np.ndarray
            Volatility-parity weights
        """
        # Calculate rolling volatilities
        recent_returns = self.returns.tail(lookback)
        volatilities = recent_returns.std() * np.sqrt(252)

        # Inverse volatility weights
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

        return weights.values

    def risk_parity(self, lookback: int = 126, max_iter: int = 1000) -> np.ndarray:
        """
        Risk parity allocation - equalizes risk contribution across assets.

        Parameters:
        -----------
        lookback : int
            Lookback period for covariance estimation
        max_iter : int
            Maximum iterations for optimization

        Returns:
        --------
        np.ndarray
            Risk parity weights
        """
        recent_returns = self.returns.tail(lookback)
        cov_matrix = recent_returns.cov() * 252  # Annualized

        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset."""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib

        def objective(weights, cov_matrix):
            """Minimize difference in risk contributions."""
            rc = risk_contribution(weights, cov_matrix)
            target_rc = np.ones(len(weights)) / len(weights)
            return np.sum((rc - target_rc * rc.sum()) ** 2)

        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter}
        )

        return result.x if result.success else initial_weights

    def mean_variance_optimization(self,
                                   lookback: int = 126,
                                   target_return: Optional[float] = None,
                                   risk_aversion: float = 1.0) -> np.ndarray:
        """
        Markowitz mean-variance optimization.

        Parameters:
        -----------
        lookback : int
            Lookback period for parameter estimation
        target_return : float, optional
            Target annual return. If None, maximizes Sharpe ratio
        risk_aversion : float
            Risk aversion parameter (only used if target_return is None)

        Returns:
        --------
        np.ndarray
            Optimal weights
        """
        recent_returns = self.returns.tail(lookback)

        # Estimate parameters
        mean_returns = recent_returns.mean() * 252  # Annualized
        cov_matrix = recent_returns.cov() * 252  # Annualized

        if target_return is None:
            # Maximize Sharpe ratio
            def objective(weights):
                port_return = weights @ mean_returns
                port_vol = np.sqrt(weights @ cov_matrix @ weights)
                sharpe = (port_return - self.risk_free_rate) / port_vol
                return -sharpe  # Minimize negative Sharpe

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        else:
            # Minimize variance for target return
            def objective(weights):
                return weights @ cov_matrix @ weights

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: w @ mean_returns - target_return}
            ]

        # Bounds: no short-selling
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial_weights = np.ones(self.n_assets) / self.n_assets

        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        return result.x if result.success else initial_weights

    def macro_adaptive_weights(self,
                              macro_score: float,
                              base_weights: Optional[np.ndarray] = None,
                              energy_indices: Optional[List[int]] = None,
                              safe_indices: Optional[List[int]] = None,
                              tilt_magnitude: float = 0.20) -> np.ndarray:
        """
        Macro-adaptive allocation based on macro regime score.

        Parameters:
        -----------
        macro_score : float
            Composite macro score (positive = bullish, negative = bearish)
        base_weights : np.ndarray, optional
            Base allocation (default: equal-weight)
        energy_indices : list, optional
            Indices of energy assets in the portfolio
        safe_indices : list, optional
            Indices of safe-haven assets (e.g., treasuries)
        tilt_magnitude : float
            Maximum tilt as proportion of portfolio (default: 20%)

        Returns:
        --------
        np.ndarray
            Macro-adjusted weights
        """
        if base_weights is None:
            base_weights = self.equal_weight()
        else:
            base_weights = np.array(base_weights)

        weights = base_weights.copy()

        # Default: assume first 4 assets are energy, last 2 are safe havens
        if energy_indices is None:
            energy_indices = list(range(min(4, self.n_assets)))
        if safe_indices is None:
            safe_indices = list(range(max(0, self.n_assets - 2), self.n_assets))

        # Calculate tilt based on macro score
        if macro_score > 0.75:
            # Bullish: overweight energy
            tilt = min(macro_score, 1.5) * tilt_magnitude
            for idx in energy_indices:
                weights[idx] += tilt / len(energy_indices)
            for idx in safe_indices:
                weights[idx] -= tilt / len(safe_indices)

        elif macro_score < -0.75:
            # Bearish: underweight energy, overweight safe havens
            tilt = min(abs(macro_score), 1.5) * tilt_magnitude
            for idx in energy_indices:
                weights[idx] -= tilt / len(energy_indices)
            for idx in safe_indices:
                weights[idx] += tilt / len(safe_indices)

        # Ensure no negative weights and rescale
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()

        return weights

    def target_volatility_scaling(self,
                                  weights: np.ndarray,
                                  target_vol: float = 0.12,
                                  lookback: int = 63) -> float:
        """
        Calculate leverage/scaling factor to achieve target portfolio volatility.

        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
        target_vol : float
            Target annual volatility (default: 12%)
        lookback : int
            Lookback period for volatility estimation

        Returns:
        --------
        float
            Scaling factor (leverage multiplier)
        """
        recent_returns = self.returns.tail(lookback)
        cov_matrix = recent_returns.cov() * 252

        current_vol = np.sqrt(weights @ cov_matrix @ weights)

        if current_vol > 0:
            scaling = target_vol / current_vol
        else:
            scaling = 1.0

        # Cap leverage
        return min(max(scaling, 0.5), 2.0)


class PortfolioBacktester:
    """
    Backtesting engine for portfolio strategies.
    """

    def __init__(self,
                 returns_data: pd.DataFrame,
                 rebalance_freq: str = 'M',
                 transaction_cost: float = 0.001):
        """
        Initialize backtester.

        Parameters:
        -----------
        returns_data : pd.DataFrame
            Asset returns data
        rebalance_freq : str
            Rebalancing frequency ('D', 'W', 'M', 'Q')
        transaction_cost : float
            Transaction cost as proportion of trade value
        """
        self.returns = returns_data
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost

    def backtest_strategy(self,
                         strategy_name: str,
                         weight_function,
                         macro_scores: Optional[pd.Series] = None,
                         **kwargs) -> pd.DataFrame:
        """
        Backtest a portfolio strategy.

        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        weight_function : callable
            Function that returns portfolio weights
        macro_scores : pd.Series, optional
            Macro scores for adaptive strategies
        **kwargs
            Additional arguments for weight function

        Returns:
        --------
        pd.DataFrame
            Backtest results with columns: returns, weights, portfolio_value
        """
        print(f"Backtesting {strategy_name}...")

        # Get rebalancing dates
        rebalance_dates = self.returns.resample(self.rebalance_freq).last().index

        # Initialize results
        results = []
        current_weights = None
        portfolio_value = 1.0

        for i, date in enumerate(self.returns.index):
            # Check if rebalancing date
            if date in rebalance_dates or current_weights is None:
                # Calculate new weights
                optimizer = PortfolioOptimizer(
                    self.returns.loc[:date],
                    risk_free_rate=kwargs.get('risk_free_rate', 0.02)
                )

                if 'macro' in strategy_name.lower() and macro_scores is not None:
                    # Macro-adaptive strategy
                    if date in macro_scores.index:
                        current_score = macro_scores.loc[date]
                        current_weights = weight_function(current_score, **kwargs)
                    else:
                        current_weights = optimizer.equal_weight()
                else:
                    # Static strategy
                    current_weights = weight_function(**kwargs)

                # Apply transaction costs if not first period
                if i > 0:
                    portfolio_value *= (1 - self.transaction_cost)

            # Calculate portfolio return
            if date in self.returns.index:
                daily_returns = self.returns.loc[date].values
                portfolio_return = np.sum(current_weights * daily_returns)

                # Update portfolio value
                portfolio_value *= (1 + portfolio_return)

                # Store results
                results.append({
                    'date': date,
                    'portfolio_return': portfolio_return,
                    'portfolio_value': portfolio_value,
                    'weights': current_weights.copy()
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)

        # Add weight columns
        for i, asset in enumerate(self.returns.columns):
            results_df[f'weight_{asset}'] = results_df['weights'].apply(lambda x: x[i])

        results_df.drop('weights', axis=1, inplace=True)

        print(f"  {strategy_name} complete: Final value = ${portfolio_value:.2f}")

        return results_df

    def calculate_performance_metrics(self, results: pd.DataFrame,
                                     risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate performance metrics for backtest results.

        Parameters:
        -----------
        results : pd.DataFrame
            Backtest results
        risk_free_rate : float
            Annual risk-free rate

        Returns:
        --------
        dict
            Performance metrics
        """
        returns = results['portfolio_return']
        values = results['portfolio_value']

        # Annualized return
        total_return = values.iloc[-1] / values.iloc[0] - 1
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Volatility
        annual_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_returns = returns - risk_free_rate / 252
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'Total Return (%)': total_return * 100,
            'Annual Return (%)': annual_return * 100,
            'Annual Volatility (%)': annual_vol * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_drawdown * 100,
            'Calmar Ratio': calmar
        }


def main():
    """
    Example usage of portfolio optimization.
    """
    # Load data
    print("Loading data...")
    features_data = pd.read_csv('../data/features_data.csv', index_col=0, parse_dates=True)

    # Select asset returns
    asset_returns = features_data[['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN',
                                   'CVX_RETURN', 'IEF_RETURN', 'SPY_RETURN']]
    asset_returns = asset_returns.dropna()

    # Get macro scores
    macro_scores = features_data['MACRO_SCORE'] if 'MACRO_SCORE' in features_data.columns else None

    # Initialize backtester
    backtester = PortfolioBacktester(asset_returns, rebalance_freq='M')

    # Define strategies
    strategies = {}

    # 1. Equal-weight
    optimizer = PortfolioOptimizer(asset_returns)
    strategies['Equal Weight'] = backtester.backtest_strategy(
        'Equal Weight',
        lambda: optimizer.equal_weight()
    )

    # 2. Volatility Parity
    strategies['Volatility Parity'] = backtester.backtest_strategy(
        'Volatility Parity',
        lambda: optimizer.volatility_parity()
    )

    # 3. Mean-Variance Optimization
    strategies['Mean-Variance'] = backtester.backtest_strategy(
        'Mean-Variance',
        lambda: optimizer.mean_variance_optimization()
    )

    # 4. Macro-Adaptive (if macro scores available)
    if macro_scores is not None:
        strategies['Macro-Adaptive'] = backtester.backtest_strategy(
            'Macro-Adaptive',
            optimizer.macro_adaptive_weights,
            macro_scores=macro_scores,
            energy_indices=[0, 1, 2, 3],
            safe_indices=[4]
        )

    # Calculate and display performance metrics
    print("\n" + "="*80)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*80)

    metrics_df = pd.DataFrame()
    for name, results in strategies.items():
        metrics = backtester.calculate_performance_metrics(results)
        metrics_df[name] = pd.Series(metrics)

    print(metrics_df.round(2))

    # Save results
    for name, results in strategies.items():
        filename = f'../data/backtest_{name.replace(" ", "_").lower()}.csv'
        results.to_csv(filename)
        print(f"\nSaved {name} results to: {filename}")

    metrics_df.to_csv('../data/performance_metrics.csv')
    print("\nSaved performance metrics to: ../data/performance_metrics.csv")


if __name__ == "__main__":
    main()
