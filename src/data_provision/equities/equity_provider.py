"""
Equity data provider using Databento for historical OHLCV data.

This provider fetches 1-minute equity data and stores it locally.
It can then aggregate this data into larger timeframes as requested.
"""

import asyncio
import databento as db
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from src.data_provision.base_provider import BaseDataProvider
from src.data_provision.ohlcv_data import OHLCVData


class EquityProvider(BaseDataProvider):
    """
    Provides equity OHLCV data using Databento.
    Fetches 1-minute data and aggregates to other intervals.
    """

    BASE_INTERVAL = '1m'
    DATASET = 'XNAS.ITCH'  # Example dataset, you can change this

    def __init__(self, api_key: Optional[str] = None, data_dir: str = "data/equities"):
        """
        Initialize the EquityProvider.

        Args:
            api_key: Your Databento API key. Defaults to DATABENTO_API_KEY env var.
            data_dir: Directory to store cached Parquet data.
        """
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY")
        if not self.api_key:
            raise ValueError("Databento API key not provided or found in DATABENTO_API_KEY environment variable.")
        
        self.client = db.Historical(self.api_key)
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)


    async def _get_base_data(self, symbol: str, start_date: datetime, end_date: datetime, interval: str, force_reload: bool) -> pd.DataFrame:
        """
        Loads or fetches the base 1-minute data for a given symbol and date range.
        """
        if not force_reload:
            local_df = await self._load_symbol_from_parquet(symbol, start_date, end_date, interval)
            if not local_df.empty:
                self.logger.info(f"Loaded {len(local_df)} records for {symbol} from local files.")
                return local_df

        self.logger.info(f"Fetching {symbol} from Databento (force_reload={force_reload}).")
        
        fetched_df = await self._fetch_from_databento(symbol, start_date, end_date, interval)
        
        if not fetched_df.empty:
            # Asynchronously save the newly fetched data
            asyncio.create_task(self.save_to_parquet(fetched_df.copy(), symbol, interval))

        return fetched_df

    async def _fetch_from_databento(self, symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
        """
        Fetches OHLCV data from the Databento API for a given symbol.
        """
        try:
            if start >= end:
                self.logger.warning(f"Start date {start} is after end date {end}. No data to fetch.")
                return pd.DataFrame()

            data = await self.client.timeseries.get_range_async(
                dataset=self.DATASET,
                symbols=[symbol],
                schema=f"ohlcv-{interval}",
                start=start,
                end=end
            )
            df = data.to_df()
            if df.empty:
                return pd.DataFrame()
            
            # Rename and format columns to be consistent
            df.reset_index(inplace=True)
            df.rename(columns={'ts_event': 'timestamp'}, inplace=True)
            df['timestamp'] = (df['timestamp'].astype('int64') // 1_000_000) # nanoseconds to milliseconds
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('datetime', inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol} from Databento: {e}")
            return pd.DataFrame()

    def _get_file_path(self, symbol: str, date: datetime, interval: str) -> Path:
        """Generate file path for storing data."""
        path = (
            self.data_dir / "ohlcv" / symbol /
            f"year={date.year}" / f"month={date.month:02d}" / f"day={date.day:02d}"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{interval}.parquet"

    async def save_to_parquet(self, df: pd.DataFrame, symbol: str, interval: str):
        """Saves a DataFrame of OHLCV data to partitioned Parquet files by day."""
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return

        for day, daily_data in df.groupby(df.index.date):
            if daily_data.empty:
                continue
            
            file_path = self._get_file_path(symbol, day, interval)
            try:
                # Reset index to save datetime as a column
                data_to_save = daily_data.reset_index()
                await asyncio.to_thread(
                    data_to_save.to_parquet, file_path, compression='snappy', index=False
                )
                self.logger.info(f"Saved {len(data_to_save)} records to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save data to {file_path}: {e}")

    async def _load_symbol_from_parquet(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """Loads data for a single symbol from local Parquet files within a date range."""
        all_dfs = []
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')

        for date in date_range:
            file_path = self._get_file_path(symbol, date, interval)
            if file_path.exists():
                try:
                    df = await asyncio.to_thread(pd.read_parquet, file_path)
                    all_dfs.append(df)
                except Exception as e:
                    self.logger.error(f"Error loading Parquet file {file_path}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], utc=True)
        combined_df = combined_df.set_index('datetime')

        mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
        return combined_df.loc[mask].sort_index()

    async def fetch_ohlcv(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[pd.Timestamp, datetime],
        end_date: Union[pd.Timestamp, datetime],
        interval: str,
        force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for one or more equity symbols.

        This method first loads cached 1-minute data, then fetches any missing
        data from Databento. Finally, it aggregates the 1-minute data to the
        requested interval.

        Args:
            symbols: A symbol or list of symbols to fetch.
            start_date: The start date for the data.
            end_date: The end date for the data.
            interval: The target frequency of the data (e.g., '5m', '1h', '1d').
            force_reload: If True, bypasses local cache and fetches from Databento.

        Returns:
            A dictionary where keys are symbols and values are pandas DataFrames
            with the OHLCV data aggregated to the requested interval.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Ensure dates are timezone-aware
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
            
        results = {}
        for symbol in symbols:
            # Always fetch and work with the base 1-minute interval
            base_data = await self._get_base_data(symbol, start_date, end_date, self.BASE_INTERVAL, force_reload)

            if base_data.empty:
                self.logger.warning(f"No 1-minute data found for {symbol} in the given date range.")
                results[symbol] = pd.DataFrame()
                continue
            
            if interval == self.BASE_INTERVAL:
                results[symbol] = base_data
            else:
                # Aggregate to the target interval if different from base
                results[symbol] = self._aggregate_ohlcv(base_data, interval)
        
        return results

    @staticmethod
    def _aggregate_ohlcv(df: pd.DataFrame, interval: str) -> pd.DataFrame:
        """
        Aggregates 1-minute OHLCV data to a larger interval.
        """
        if df.empty:
            return df
            
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample and apply aggregation
        resampled_df = df.resample(interval).agg(agg_rules)
        
        # Drop rows with no data
        return resampled_df.dropna(how='all')
        
    async def get_markets(self, exchange_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        self.logger.info("get_markets is not applicable for Databento provider.")
        return {}

    async def close(self):
        self.logger.info("Databento provider closed.")
        pass # Databento client doesn't require explicit closing

async def main():
    """Example usage of the EquityProvider."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # The provider will now automatically look for the .env file in your project root
    provider = EquityProvider()
    
    symbols = ['AAPL', 'MSFT']
    
    start_date = datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)
    end_date = datetime(2025, 1, 2, 14, 40, tzinfo=timezone.utc)

    try:
        # Fetch 15-minute aggregated data
        print(f"\nFetching 15-minute data for {', '.join(symbols)} from {start_date.date()} to {end_date.date()}...")
        data_15m = await provider.fetch_ohlcv(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval='1min' # Note: pandas uses 'min' for minutes
        )
        
        for symbol, df in data_15m.items():
            print(f"\n--- Data for {symbol} (15min) ---")
            if not df.empty:
                print(df.head())
                print(df.tail())
            else:
                print("No data returned.")

    finally:
        await provider.close()

if __name__ == "__main__":
    # To run this example, create a .env file in the root of your project with:
    # DATABENTO_API_KEY="YOUR_API_KEY"
    asyncio.run(main())

