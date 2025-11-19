"""
Fetches, cleans, and merges asset prices and macroeconomic indicators.

Data Sources:
1. Yahoo Finance: WTI, XLE, XOM, CVX, IEF/SHY, SPY
2. FRED API: Macro indicators (inflation, GDP, interest rates, etc.)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os

warnings.filterwarnings('ignore')


class DataPipeline:
    """
    Main data pipeline class for fetching and processing market and macro data.
    """

    def __init__(self, fred_api_key: Optional[str] = None,
                 start_date: str = "2015-01-01",
                 end_date: Optional[str] = None):
        """
        Initialize the data pipeline.

        Parameters:
        -----------
        fred_api_key : str, optional
            FRED API key. If None, will try to get from environment variable.
        start_date : str
            Start date for data collection (YYYY-MM-DD format)
        end_date : str, optional
            End date for data collection. If None, uses today's date.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

        # Initialize FRED API
        if fred_api_key is None:
            fred_api_key = os.environ.get('FRED_API_KEY', '')
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None

        # Asset tickers
        self.tickers = {
            'WTI': 'CL=F',          # WTI Crude Oil futures
            'XLE': 'XLE',           # Energy Select Sector SPDR Fund
            'XOM': 'XOM',           # ExxonMobil
            'CVX': 'CVX',           # Chevron
            'IEF': 'IEF',           # iShares 7-10 Year Treasury Bond ETF
            'SHY': 'SHY',           # iShares 1-3 Year Treasury Bond ETF
            'SPY': 'SPY'            # S&P 500 ETF (benchmark)
        }

        # FRED series IDs for macro indicators
        self.fred_series = {
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

    def fetch_price_data(self) -> pd.DataFrame:
        """
        Fetch daily price data from Yahoo Finance for all tickers.

        Returns:
        --------
        pd.DataFrame
            DataFrame with adjusted close prices for all assets
        """
        print("Fetching price data from Yahoo Finance...")

        price_data = pd.DataFrame()

        for name, ticker in self.tickers.items():
            try:
                print(f"  Downloading {name} ({ticker})...")
                data = yf.download(ticker, start=self.start_date, end=self.end_date,
                                  progress=False, auto_adjust=True)

                if isinstance(data.columns, pd.MultiIndex):
                    price_data[name] = data['Close'].iloc[:, 0]
                else:
                    price_data[name] = data['Close']

            except Exception as e:
                print(f"  Warning: Could not download {name}: {e}")

        print(f"Price data fetched: {len(price_data)} rows, {len(price_data.columns)} assets")
        return price_data

    def fetch_macro_data(self) -> pd.DataFrame:
        """
        Fetch macroeconomic indicators from FRED API.

        Returns:
        --------
        pd.DataFrame
            DataFrame with macro indicators
        """
        if self.fred is None:
            print("Warning: FRED API key not provided. Skipping macro data.")
            return pd.DataFrame()

        print("Fetching macroeconomic data from FRED...")

        macro_data = pd.DataFrame()

        for series_id, description in self.fred_series.items():
            try:
                print(f"  Downloading {description} ({series_id})...")
                data = self.fred.get_series(series_id,
                                           observation_start=self.start_date,
                                           observation_end=self.end_date)
                macro_data[series_id] = data

            except Exception as e:
                print(f"  Warning: Could not download {series_id}: {e}")

        print(f"Macro data fetched: {len(macro_data)} rows, {len(macro_data.columns)} indicators")
        return macro_data

    def calculate_returns(self, prices: pd.DataFrame,
                         method: str = 'log') -> pd.DataFrame:
        """
        Calculate returns from price data.

        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        method : str
            'log' for log returns, 'simple' for simple returns

        Returns:
        --------
        pd.DataFrame
            Returns data
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns

    def align_and_merge(self, price_data: pd.DataFrame,
                       macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Align and merge price and macro data on common business days.
        Forward-fills macro data to match daily trading data.

        Parameters:
        -----------
        price_data : pd.DataFrame
            Daily price data
        macro_data : pd.DataFrame
            Macro indicator data (may be lower frequency)

        Returns:
        --------
        pd.DataFrame
            Merged dataset
        """
        print("Aligning and merging datasets...")

        # Start with price data index (business days)
        merged_data = price_data.copy()

        if not macro_data.empty:
            # Reindex macro data to match price data index and forward fill
            macro_aligned = macro_data.reindex(merged_data.index, method='ffill')

            # Merge with price data
            merged_data = pd.concat([merged_data, macro_aligned], axis=1)

        # Drop any rows with all NaN values
        merged_data = merged_data.dropna(how='all')

        print(f"Merged dataset: {len(merged_data)} rows, {len(merged_data.columns)} columns")
        return merged_data

    def calculate_macro_changes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate changes and growth rates for macro indicators.

        Parameters:
        -----------
        data : pd.DataFrame
            Merged data with macro indicators

        Returns:
        --------
        pd.DataFrame
            Data with additional macro change columns
        """
        result = data.copy()

        # Calculate changes in key macro variables
        if 'FEDFUNDS' in result.columns:
            result['FEDFUNDS_3M_CHANGE'] = result['FEDFUNDS'] - result['FEDFUNDS'].shift(63)
            result['FEDFUNDS_CHANGE'] = result['FEDFUNDS'].diff()

        if 'CPIAUCSL' in result.columns:
            # Year-over-year inflation rate
            result['INFLATION_YOY'] = (result['CPIAUCSL'] / result['CPIAUCSL'].shift(252) - 1) * 100

        if 'INDPRO' in result.columns:
            # Industrial production growth (YoY)
            result['INDPRO_YOY'] = (result['INDPRO'] / result['INDPRO'].shift(252) - 1) * 100

        if 'GDP' in result.columns:
            # GDP growth (YoY)
            result['GDP_YOY'] = (result['GDP'] / result['GDP'].shift(252) - 1) * 100

        return result

    def clean_data(self, data: pd.DataFrame,
                   max_missing_pct: float = 0.5) -> pd.DataFrame:
        """
        Clean the merged dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Raw merged data
        max_missing_pct : float
            Maximum percentage of missing values allowed per column

        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        print("Cleaning data...")

        # Remove columns with too many missing values
        missing_pct = data.isnull().sum() / len(data)
        cols_to_keep = missing_pct[missing_pct < max_missing_pct].index
        data_clean = data[cols_to_keep].copy()

        # Forward fill remaining missing values (up to 5 days)
        data_clean = data_clean.fillna(method='ffill', limit=5)

        # Drop any remaining rows with missing values in price columns
        price_cols = list(self.tickers.keys())
        data_clean = data_clean.dropna(subset=[col for col in price_cols if col in data_clean.columns])

        print(f"Cleaned data: {len(data_clean)} rows, {len(data_clean.columns)} columns")
        print(f"Date range: {data_clean.index.min()} to {data_clean.index.max()}")

        return data_clean

    def run_pipeline(self, save_to_csv: bool = True,
                    output_dir: str = '../data') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete data pipeline.

        Parameters:
        -----------
        save_to_csv : bool
            Whether to save results to CSV files
        output_dir : str
            Directory to save CSV files

        Returns:
        --------
        tuple
            (prices_df, merged_df) - Price data and merged data with macro indicators
        """
        print("="*60)
        print("RUNNING DATA PIPELINE")
        print("="*60)

        # 1. Fetch price data
        prices = self.fetch_price_data()

        # 2. Fetch macro data
        macro = self.fetch_macro_data()

        # 3. Calculate returns
        returns = self.calculate_returns(prices, method='log')
        returns.columns = [f'{col}_RETURN' for col in returns.columns]

        # 4. Merge price and macro data
        merged = self.align_and_merge(prices, macro)

        # 5. Add returns to merged data
        merged = pd.concat([merged, returns], axis=1)

        # 6. Calculate macro changes
        merged = self.calculate_macro_changes(merged)

        # 7. Clean data
        merged = self.clean_data(merged)

        # 8. Save to CSV if requested
        if save_to_csv:
            os.makedirs(output_dir, exist_ok=True)

            prices_file = os.path.join(output_dir, 'prices.csv')
            merged_file = os.path.join(output_dir, 'merged_data.csv')

            prices.to_csv(prices_file)
            merged.to_csv(merged_file)

            print(f"\nData saved to:")
            print(f"  - {prices_file}")
            print(f"  - {merged_file}")

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)

        return prices, merged

    def get_data_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset.

        Parameters:
        -----------
        data : pd.DataFrame
            Dataset to summarize

        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        summary = pd.DataFrame({
            'Count': data.count(),
            'Mean': data.mean(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Missing_Pct': (data.isnull().sum() / len(data) * 100)
        })

        return summary.round(4)


def main():
    """
    Example usage of the data pipeline.
    """
    # Initialize pipeline
    # Note: Set FRED_API_KEY environment variable or pass it here
    pipeline = DataPipeline(
        start_date="2015-01-01",
        end_date="2025-11-30"
    )

    # Run pipeline
    prices, merged_data = pipeline.run_pipeline(save_to_csv=True, output_dir='../data')

    # Print summary
    print("\n" + "="*60)
    print("PRICE DATA SUMMARY")
    print("="*60)
    print(pipeline.get_data_summary(prices))

    print("\n" + "="*60)
    print("RETURN DATA SUMMARY (Annualized %)")
    print("="*60)
    return_cols = [col for col in merged_data.columns if '_RETURN' in col]
    returns_summary = pipeline.get_data_summary(merged_data[return_cols]) * np.sqrt(252) * 100
    print(returns_summary[['Mean', 'Std']])


if __name__ == "__main__":
    main()
