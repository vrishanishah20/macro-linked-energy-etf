"""
Computes rolling indicators, macro sensitivities, and correlation metrics.

Features:
1. Rolling momentum, volatility, and drawdown
2. Macro sensitivity metrics (beta to oil, rates, etc.)
3. Correlation matrices and PCA analysis
4. Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class MacroFeatureEngine:
    """
    Feature engineering class for creating macro-linked indicators.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the feature engine.

        Parameters:
        -----------
        data : pd.DataFrame
            Merged dataset with prices, returns, and macro indicators
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)

    def calculate_momentum(self, returns_col: str, windows: List[int] = [21, 63, 126]) -> pd.DataFrame:
        """
        Calculate momentum indicators over multiple windows.

        Parameters:
        -----------
        returns_col : str
            Column name containing returns
        windows : list
            List of lookback windows in days

        Returns:
        --------
        pd.DataFrame
            Momentum features
        """
        momentum = pd.DataFrame(index=self.data.index)

        for window in windows:
            momentum[f'{returns_col}_MOM_{window}'] = (
                self.data[returns_col].rolling(window).sum()
            )

        return momentum

    def calculate_volatility(self, returns_col: str, windows: List[int] = [21, 63, 126]) -> pd.DataFrame:
        """
        Calculate rolling volatility (annualized).

        Parameters:
        -----------
        returns_col : str
            Column name containing returns
        windows : list
            List of lookback windows in days

        Returns:
        --------
        pd.DataFrame
            Volatility features
        """
        volatility = pd.DataFrame(index=self.data.index)

        for window in windows:
            volatility[f'{returns_col}_VOL_{window}'] = (
                self.data[returns_col].rolling(window).std() * np.sqrt(252)
            )

        return volatility

    def calculate_drawdown(self, price_col: str) -> pd.DataFrame:
        """
        Calculate running drawdown from peak.

        Parameters:
        -----------
        price_col : str
            Column name containing prices

        Returns:
        --------
        pd.DataFrame
            Drawdown metrics
        """
        drawdown = pd.DataFrame(index=self.data.index)

        # Calculate cumulative max (running peak)
        cummax = self.data[price_col].cummax()

        # Calculate drawdown
        drawdown[f'{price_col}_DRAWDOWN'] = (self.data[price_col] / cummax - 1) * 100

        # Calculate max drawdown over rolling windows
        for window in [63, 126, 252]:
            rolling_max = self.data[price_col].rolling(window).max()
            drawdown[f'{price_col}_MAX_DD_{window}'] = (
                (self.data[price_col] / rolling_max - 1) * 100
            ).rolling(window).min()

        return drawdown

    def calculate_rolling_beta(self, asset_return: str, market_return: str,
                              windows: List[int] = [63, 126, 252]) -> pd.DataFrame:
        """
        Calculate rolling beta (sensitivity) to market/factor.

        Parameters:
        -----------
        asset_return : str
            Asset return column name
        market_return : str
            Market/factor return column name
        windows : list
            List of lookback windows

        Returns:
        --------
        pd.DataFrame
            Rolling beta metrics
        """
        betas = pd.DataFrame(index=self.data.index)

        for window in windows:
            # Calculate rolling covariance and variance
            cov = self.data[asset_return].rolling(window).cov(self.data[market_return])
            var = self.data[market_return].rolling(window).var()

            betas[f'BETA_{asset_return}_vs_{market_return}_{window}'] = cov / var

        return betas

    def calculate_rolling_correlation(self, col1: str, col2: str,
                                     windows: List[int] = [63, 126, 252]) -> pd.DataFrame:
        """
        Calculate rolling correlation between two series.

        Parameters:
        -----------
        col1, col2 : str
            Column names to correlate
        windows : list
            List of lookback windows

        Returns:
        --------
        pd.DataFrame
            Rolling correlation metrics
        """
        correlations = pd.DataFrame(index=self.data.index)

        for window in windows:
            correlations[f'CORR_{col1}_vs_{col2}_{window}'] = (
                self.data[col1].rolling(window).corr(self.data[col2])
            )

        return correlations

    def calculate_sharpe_ratio(self, returns_col: str,
                              risk_free_col: Optional[str] = None,
                              windows: List[int] = [63, 126, 252]) -> pd.DataFrame:
        """
        Calculate rolling Sharpe ratio.

        Parameters:
        -----------
        returns_col : str
            Return column name
        risk_free_col : str, optional
            Risk-free rate column (annualized %)
        windows : list
            List of lookback windows

        Returns:
        --------
        pd.DataFrame
            Sharpe ratio metrics
        """
        sharpe = pd.DataFrame(index=self.data.index)

        # Convert risk-free rate to daily if provided
        if risk_free_col and risk_free_col in self.data.columns:
            rf_daily = self.data[risk_free_col] / 252 / 100
        else:
            rf_daily = 0

        excess_returns = self.data[returns_col] - rf_daily

        for window in windows:
            mean_excess = excess_returns.rolling(window).mean()
            std_excess = excess_returns.rolling(window).std()
            sharpe[f'{returns_col}_SHARPE_{window}'] = (
                mean_excess / std_excess * np.sqrt(252)
            )

        return sharpe

    def calculate_macro_sensitivities(self, asset_return: str,
                                     macro_factors: List[str],
                                     window: int = 126) -> pd.DataFrame:
        """
        Calculate rolling sensitivities to macro factors.

        Parameters:
        -----------
        asset_return : str
            Asset return column
        macro_factors : list
            List of macro factor column names
        window : int
            Lookback window

        Returns:
        --------
        pd.DataFrame
            Macro sensitivity metrics
        """
        sensitivities = pd.DataFrame(index=self.data.index)

        for factor in macro_factors:
            if factor not in self.data.columns:
                continue

            # Calculate rolling regression coefficient
            factor_change = self.data[factor].diff()

            # Rolling covariance / variance
            cov = self.data[asset_return].rolling(window).cov(factor_change)
            var = factor_change.rolling(window).var()

            sensitivities[f'SENSITIVITY_{asset_return}_vs_{factor}'] = cov / var

        return sensitivities

    def create_composite_macro_score(self,
                                     wti_return: str = 'WTI_RETURN',
                                     fedfunds_col: str = 'FEDFUNDS_CHANGE',
                                     vix_col: str = 'VIXCLS',
                                     weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Create composite macro score for regime identification.

        C_t = w1 * z(WTI_return) - w2 * z(FFR_change) - w3 * z(VIX_change)

        Parameters:
        -----------
        wti_return : str
            WTI return column
        fedfunds_col : str
            Federal funds rate change column
        vix_col : str
            VIX column
        weights : dict, optional
            Custom weights for each component

        Returns:
        --------
        pd.DataFrame
            Composite score
        """
        if weights is None:
            weights = {'wti': 0.4, 'rates': 0.3, 'vix': 0.3}

        composite = pd.DataFrame(index=self.data.index)

        # Z-score normalization function
        def zscore(series, window=126):
            return (series - series.rolling(window).mean()) / series.rolling(window).std()

        # Component scores
        components = {}

        if wti_return in self.data.columns:
            components['wti_z'] = zscore(self.data[wti_return])

        if fedfunds_col in self.data.columns:
            components['rates_z'] = -zscore(self.data[fedfunds_col])  # Negative: tight policy is bearish

        if vix_col in self.data.columns:
            vix_change = self.data[vix_col].pct_change()
            components['vix_z'] = -zscore(vix_change)  # Negative: rising VIX is bearish

        # Combine into composite score
        composite['MACRO_SCORE'] = 0
        if 'wti_z' in components:
            composite['MACRO_SCORE'] += weights['wti'] * components['wti_z']
        if 'rates_z' in components:
            composite['MACRO_SCORE'] += weights['rates'] * components['rates_z']
        if 'vix_z' in components:
            composite['MACRO_SCORE'] += weights['vix'] * components['vix_z']

        # Create regime indicator
        composite['MACRO_REGIME'] = pd.cut(
            composite['MACRO_SCORE'],
            bins=[-np.inf, -0.75, 0.75, np.inf],
            labels=['BEARISH', 'NEUTRAL', 'BULLISH']
        )

        # Add individual components
        for name, series in components.items():
            composite[f'COMPONENT_{name.upper()}'] = series

        return composite

    def calculate_correlation_matrix(self, columns: List[str],
                                     method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for specified columns.

        Parameters:
        -----------
        columns : list
            List of column names
        method : str
            Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        available_cols = [col for col in columns if col in self.data.columns]
        return self.data[available_cols].corr(method=method)

    def perform_pca(self, columns: List[str],
                   n_components: int = 3) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform Principal Component Analysis.

        Parameters:
        -----------
        columns : list
            Columns to include in PCA
        n_components : int
            Number of principal components

        Returns:
        --------
        tuple
            (principal_components_df, pca_model)
        """
        available_cols = [col for col in columns if col in self.data.columns]
        data_clean = self.data[available_cols].dropna()

        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clean)

        # Fit PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)

        # Create DataFrame
        pc_df = pd.DataFrame(
            components,
            index=data_clean.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )

        return pc_df, pca

    def build_all_features(self, asset_returns: List[str],
                          price_cols: List[str],
                          macro_factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Build comprehensive feature set.

        Parameters:
        -----------
        asset_returns : list
            List of return column names
        price_cols : list
            List of price column names
        macro_factors : list, optional
            List of macro factor column names

        Returns:
        --------
        pd.DataFrame
            Complete feature dataset
        """
        print("Building comprehensive feature set...")

        all_features = pd.DataFrame(index=self.data.index)

        # 1. Momentum features
        print("  Calculating momentum...")
        for ret_col in asset_returns:
            if ret_col in self.data.columns:
                momentum = self.calculate_momentum(ret_col)
                all_features = pd.concat([all_features, momentum], axis=1)

        # 2. Volatility features
        print("  Calculating volatility...")
        for ret_col in asset_returns:
            if ret_col in self.data.columns:
                volatility = self.calculate_volatility(ret_col)
                all_features = pd.concat([all_features, volatility], axis=1)

        # 3. Drawdown features
        print("  Calculating drawdowns...")
        for price_col in price_cols:
            if price_col in self.data.columns:
                drawdown = self.calculate_drawdown(price_col)
                all_features = pd.concat([all_features, drawdown], axis=1)

        # 4. Beta and correlation features
        print("  Calculating betas and correlations...")
        if 'SPY_RETURN' in self.data.columns:
            for ret_col in asset_returns:
                if ret_col != 'SPY_RETURN' and ret_col in self.data.columns:
                    beta = self.calculate_rolling_beta(ret_col, 'SPY_RETURN')
                    all_features = pd.concat([all_features, beta], axis=1)

        # 5. Sharpe ratios
        print("  Calculating Sharpe ratios...")
        rf_col = 'FEDFUNDS' if 'FEDFUNDS' in self.data.columns else None
        for ret_col in asset_returns:
            if ret_col in self.data.columns:
                sharpe = self.calculate_sharpe_ratio(ret_col, rf_col)
                all_features = pd.concat([all_features, sharpe], axis=1)

        # 6. Macro sensitivities
        if macro_factors:
            print("  Calculating macro sensitivities...")
            for ret_col in asset_returns:
                if ret_col in self.data.columns:
                    sens = self.calculate_macro_sensitivities(ret_col, macro_factors)
                    all_features = pd.concat([all_features, sens], axis=1)

        # 7. Composite macro score
        print("  Creating composite macro score...")
        macro_score = self.create_composite_macro_score()
        all_features = pd.concat([all_features, macro_score], axis=1)

        # Merge with original data
        complete_data = pd.concat([self.data, all_features], axis=1)

        print(f"Feature engineering complete: {len(all_features.columns)} new features created")

        return complete_data


def main():
    """
    Example usage of feature engineering.
    """
    # Load data
    print("Loading data...")
    data = pd.read_csv('../data/merged_data.csv', index_col=0, parse_dates=True)

    # Initialize feature engine
    feature_engine = MacroFeatureEngine(data)

    # Define asset returns and price columns
    asset_returns = [col for col in data.columns if '_RETURN' in col]
    price_cols = ['WTI', 'XLE', 'XOM', 'CVX', 'IEF', 'SHY', 'SPY']

    # Define macro factors
    macro_factors = ['FEDFUNDS', 'CPIAUCSL', 'VIXCLS', 'INDPRO', 'T10Y2Y']

    # Build all features
    features_data = feature_engine.build_all_features(
        asset_returns=asset_returns,
        price_cols=price_cols,
        macro_factors=macro_factors
    )

    # Save to CSV
    output_file = '../data/features_data.csv'
    features_data.to_csv(output_file)
    print(f"\nFeatures saved to: {output_file}")

    # Print correlation matrix for key assets
    print("\n" + "="*60)
    print("CORRELATION MATRIX (Returns)")
    print("="*60)
    key_returns = ['WTI_RETURN', 'XLE_RETURN', 'XOM_RETURN', 'CVX_RETURN', 'SPY_RETURN']
    corr_matrix = feature_engine.calculate_correlation_matrix(key_returns)
    print(corr_matrix.round(3))

    # Print macro score summary
    print("\n" + "="*60)
    print("MACRO REGIME DISTRIBUTION")
    print("="*60)
    if 'MACRO_REGIME' in features_data.columns:
        print(features_data['MACRO_REGIME'].value_counts())


if __name__ == "__main__":
    main()
