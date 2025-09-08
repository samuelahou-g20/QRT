"""
Robust cryptocurrency data provider with multi-exchange support,
error handling, rate limiting, data validation, and optional symbol mapping.
"""

import asyncio
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
import logging
import time
import aiolimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data_provision.base_provider import BaseDataProvider
from src.data_provision.cryptos.circuit_breaker import CircuitBreaker


class CryptoProvider(BaseDataProvider):
    """
    Robust cryptocurrency data provider using CCXT with multiple exchanges,
    error handling, rate limiting, data validation, and optional symbol mapping.
    """
    
    # Exchange configurations with their rate limits (requests per minute)
    EXCHANGE_CONFIGS = {
        'binance': {'ratelimit': 1200, 'sandbox': False},
        'coinbase': {'ratelimit': 600, 'sandbox': False},
        'kraken': {'ratelimit': 60, 'sandbox': False},
        'ftx': {'ratelimit': 30, 'sandbox': False},  # If still available
        'okx': {'ratelimit': 600, 'sandbox': False},
        'bybit': {'ratelimit': 600, 'sandbox': False},
    }
    

    def __init__(self, 
                 exchanges: List[str] = None,
                 data_dir: str = "data/cryptos",
                 api_keys: Dict[str, Dict[str, str]] = None,
                 max_retries: int = 3,
                 rate_limit_buffer: float = 0.8,
                 use_symbol_mapping: bool = False,
                 symbol_config_path: Optional[Path] = None):
        """
        Initialize crypto data provider
        
        Args:
            exchanges: List of exchange names to use
            data_dir: Directory to store data
            api_keys: Dict of {exchange: {apiKey, secret, sandbox}}
            max_retries: Maximum retry attempts for failed requests
            rate_limit_buffer: Buffer factor for rate limiting (0.8 = use 80% of limit)
            use_symbol_mapping: Enable unified symbol mapping across exchanges
            symbol_config_path: Path to symbol configuration file (if using mapping)
        """
        self.exchanges = exchanges or ['binance', 'coinbase', 'kraken']
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.api_keys = api_keys or {}
        self.max_retries = max_retries
        self.use_symbol_mapping = use_symbol_mapping
        self.rate_limit_buffer = rate_limit_buffer
        
        # Initialize symbol manager if requested
        self.symbol_manager = None
        if use_symbol_mapping:
            try:
                from src.data_provision.cryptos.symbol_mapper import SymbolManager
                self.symbol_manager = SymbolManager(symbol_config_path)
                self.logger = logging.getLogger(__name__)
                self.logger.info("Symbol mapping enabled")
            except ImportError:
                self.logger = logging.getLogger(__name__)
                self.logger.warning("Symbol mapper module not found. Proceeding without symbol mapping.")
                self.use_symbol_mapping = False
        else:
            self.logger = logging.getLogger(__name__)
        
        # Initialize exchanges and rate limiters
        self.exchange_instances: Dict[str, ccxt_async.Exchange] = {}
        self.rate_limiters: Dict[str, aiolimiter.AsyncLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Symbol availability cache (only used with symbol mapping)
        if self.use_symbol_mapping:
            self.available_symbols_cache: Dict[str, Set[str]] = {}
            self.cache_timestamp: Dict[str, float] = {}
            self.cache_ttl = 3600  # 1 hour cache TTL
        
        # Initialize exchanges
        self._initialize_exchanges()
    

    def _initialize_exchanges(self):
        """Initialize exchange instances with rate limiters and circuit breakers"""
        for exchange_name in self.exchanges:
            if exchange_name not in self.EXCHANGE_CONFIGS:
                self.logger.warning(f"Unknown exchange: {exchange_name}")
                continue
                
            try:
                config = self.EXCHANGE_CONFIGS[exchange_name]
                
                # Create exchange instance
                exchange_class = getattr(ccxt_async, exchange_name)
                exchange_config = {
                    'enableRateLimit': True,
                    'rateLimit': 60000 / config['ratelimit'],  # Convert to milliseconds
                }
                
                # Add API keys if provided
                if exchange_name in self.api_keys:
                    exchange_config.update(self.api_keys[exchange_name])
                
                self.exchange_instances[exchange_name] = exchange_class(exchange_config)
                
                # Create rate limiter (requests per minute)
                rate_limit = int(config['ratelimit'] * self.rate_limit_buffer)  # Apply buffer
                self.rate_limiters[exchange_name] = aiolimiter.AsyncLimiter(
                    rate_limit, 60
                )
                
                # Create circuit breaker
                self.circuit_breakers[exchange_name] = CircuitBreaker()
                
                self.logger.info(f"Initialized {exchange_name} exchange")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {e}")
    

    def _convert_symbol(self, symbol: str, exchange_name: str, to_exchange: bool = True) -> str:
        """
        Convert symbol between unified and exchange format if symbol mapping is enabled
        
        Args:
            symbol: Symbol to convert
            exchange_name: Exchange name
            to_exchange: True to convert to exchange format, False for unified format
            
        Returns:
            Converted symbol or original if mapping not enabled
        """
        if not self.use_symbol_mapping or not self.symbol_manager:
            return symbol
        
        try:
            if to_exchange:
                # Convert unified to exchange format
                return self.symbol_manager.to_exchange(symbol, exchange_name)
            else:
                # Convert exchange to unified format
                from src.data_provision.cryptos.symbol_mapper import SymbolType
                return str(self.symbol_manager.to_unified(symbol, exchange_name, SymbolType.SPOT))
        except Exception as e:
            self.logger.debug(f"Symbol conversion failed for {symbol} on {exchange_name}: {e}")
            return symbol


    def _is_valid_ohlcv(self, candle: List) -> bool:
        """Validate a single candle."""
        try:
            if len(candle) < 6:
                return False
            
            timestamp, open_p, high, low, close, volume = candle[:6]
            
            # Convert to float for validation
            open_p, high, low, close, volume = map(float, [open_p, high, low, close, volume])
            
            # Check OHLC relationships
            if not (low <= open_p <= high and low <= close <= high and low <= high):
                return False
            
            # Check for negative values
            if any(v < 0 for v in [open_p, high, low, close, volume]):
                return False
            
            # Optional: Check timestamp is reasonable (not in future, not too old)
            now = int(datetime.now(timezone.utc).timestamp() * 1000)
            timestamp = int(timestamp)
            # Allow data up to 10 years old
            if timestamp > now or timestamp < (now - 10 * 365 * 24 * 60 * 60 * 1000):
                return False
                
            return True
        except (ValueError, TypeError, IndexError):
            return False


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    )
    async def _fetch_symbol_from_exchange(self, 
        exchange: ccxt_async.Exchange, 
        exchange_name: str, 
        symbol: str, 
        interval: str, 
        start_date: Union[pd.Timestamp, datetime], 
        end_date: Union[pd.Timestamp, datetime]
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol from a specific exchange.
        Returns a DataFrame directly instead of OHLCVData objects.
        """
        since = int(start_date.timestamp() * 1000)
        limit = 1000  # Default limit per request
        all_candles = []

        while since < end_date.timestamp() * 1000:
            circuit_breaker = self.circuit_breakers[exchange_name]
            if not circuit_breaker.can_execute():
                self.logger.warning(f"{exchange_name} circuit breaker is open for {symbol}")
                break

            try:
                exchange_symbol = self._convert_symbol(symbol, exchange_name, to_exchange=True)
                
                async with self.rate_limiters[exchange_name]:
                    raw_data = await exchange.fetch_ohlcv(exchange_symbol, interval, since, limit)

                if not raw_data:
                    break

                # Filter valid candles and collect them
                valid_candles = [c for c in raw_data if self._is_valid_ohlcv(c)]
                all_candles.extend(valid_candles)
                
                since = raw_data[-1][0] + exchange.parse_timeframe(interval) * 1000
                circuit_breaker.record_success()

            except Exception as e:
                circuit_breaker.record_failure()
                self.logger.error(f"Error fetching from {exchange_name} for {symbol}: {e}")
                break
        
        # Convert to DataFrame if we have data
        if not all_candles:
            return pd.DataFrame()
        
        # Create DataFrame directly from candles
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['exchange'] = exchange_name
        df['symbol'] = symbol
        df['interval'] = interval
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        return df


    def _get_file_path(self, symbol: str, date: datetime, exchange: str, 
                      interval: str, data_type: str = 'ohlcv') -> Path:
        """Generate file path for storing data"""
        # Clean symbol for filename
        clean_symbol = symbol.replace('/', '_').replace('-', '_')
        
        path = (self.data_dir / 
                f"{data_type}" /
                f"{exchange}" /
                f"{clean_symbol}" /
                f"year={date.year}" /
                f"month={date.month:02d}" /
                f"day={date.day:02d}")
        
        path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{interval}.parquet"
        return path / filename


    async def _load_symbol_from_parquet(self, symbol: str, start_date: datetime, end_date: datetime, interval: str) -> pd.DataFrame:
        """Loads data for a single symbol from local Parquet files within a date range."""
        all_dfs = []
        
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')

        for exchange_name in self.exchanges:
            for date in date_range:
                file_path = self._get_file_path(symbol, date, exchange_name, interval)
                if file_path.exists():
                    try:
                        df = pd.read_parquet(file_path)
                        # Add exchange, symbol, interval columns if missing (for backward compatibility)
                        if 'exchange' not in df.columns:
                            df['exchange'] = exchange_name
                        if 'symbol' not in df.columns:
                            df['symbol'] = symbol
                        if 'interval' not in df.columns:
                            df['interval'] = interval
                        all_dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Error loading Parquet file {file_path}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Ensure datetime column exists
        if 'datetime' not in combined_df.columns:
            combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms', utc=True)
        
        combined_df = combined_df.set_index('datetime')
        mask = (combined_df.index >= start_date) & (combined_df.index <= end_date)
        filtered_df = combined_df.loc[mask]
        
        # Drop duplicates across exchanges for the same timestamp, keeping the first entry
        final_df = filtered_df[~filtered_df.index.duplicated(keep='first')]
        return final_df.sort_index()


    async def save_to_parquet(self, data: pd.DataFrame):
        """Save OHLCV data to parquet files, partitioned by day."""
        if data.empty:
            return

        df_to_save = data.copy()
        
        # Ensure 'datetime' is the index for Grouper
        if 'datetime' in df_to_save.columns and df_to_save.index.name != 'datetime':
            df_to_save = df_to_save.set_index('datetime')
        elif df_to_save.index.name != 'datetime' and 'timestamp' in df_to_save.columns:
            df_to_save['datetime'] = pd.to_datetime(df_to_save['timestamp'], unit='ms', utc=True)
            df_to_save = df_to_save.set_index('datetime')
        
        # Group by exchange, symbol, interval
        for (exchange_name, symbol, interval), group in df_to_save.groupby(['exchange', 'symbol', 'interval']):
            if group.empty:
                continue

            # Group by day from the datetime index and save each day's data
            for date, daily_group in group.groupby(pd.Grouper(freq='D')):
                if daily_group.empty:
                    continue
                
                file_path = self._get_file_path(symbol, date, exchange_name, interval)
                
                try:
                    # Keep essential columns, drop redundant ones
                    columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    daily_group_to_save = daily_group[columns_to_save] if all(c in daily_group.columns for c in columns_to_save) else daily_group
                    
                    # Reset index to save datetime as a column
                    daily_group_to_save.reset_index().to_parquet(file_path, compression='snappy', index=False)
                    self.logger.info(f"Saved {len(daily_group)} records for {symbol} on {exchange_name} to {file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save data to {file_path}: {e}")
    

    async def get_available_symbols(self, exchange: str, force_refresh: bool = False) -> Set[str]:
        """
        Get available symbols for an exchange. Returns unified symbols if mapping is enabled.
        
        Args:
            exchange: Exchange name
            force_refresh: Force refresh of symbol cache
            
        Returns:
            Set of available symbol strings
        """
        if not self.use_symbol_mapping:
            # Without symbol mapping, just return the raw symbols from the exchange
            if exchange not in self.exchange_instances:
                return set()
            
            try:
                markets = await self.exchange_instances[exchange].load_markets()
                return set(markets.keys())
            except Exception as e:
                self.logger.error(f"Failed to get symbols from {exchange}: {e}")
                return set()
        
        # With symbol mapping, return unified symbols
        exchange = exchange.lower()
        
        # Check cache
        if not force_refresh and exchange in self.available_symbols_cache:
            cache_age = time.time() - self.cache_timestamp.get(exchange, 0)
            if cache_age < self.cache_ttl:
                return self.available_symbols_cache[exchange]
        
        if exchange not in self.exchange_instances:
            return set()
        
        try:
            # Load markets from exchange
            markets = await self.exchange_instances[exchange].load_markets()
            
            unified_symbols = set()
            for market_id, market in markets.items():
                try:
                    # Convert to unified format
                    unified = self._convert_symbol(market['symbol'], exchange, to_exchange=False)
                    unified_symbols.add(unified)
                except Exception as e:
                    self.logger.debug(f"Could not convert {market_id} from {exchange}: {e}")
            
            # Update cache
            self.available_symbols_cache[exchange] = unified_symbols
            self.cache_timestamp[exchange] = time.time()
            
            self.logger.info(f"Loaded {len(unified_symbols)} symbols from {exchange}")
            return unified_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get symbols from {exchange}: {e}")
            return set()
    
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all exchanges"""
        status = {
            'exchanges': {},
            'overall_healthy': True,
            'symbol_mapping_enabled': self.use_symbol_mapping
        }
        
        for exchange_name in self.exchange_instances.keys():
            circuit_breaker = self.circuit_breakers[exchange_name]
            exchange_status = {
                'circuit_breaker_state': circuit_breaker.state,
                'failure_count': circuit_breaker.failure_count,
                'healthy': circuit_breaker.state == 'closed'
            }
            status['exchanges'][exchange_name] = exchange_status
            
            if not exchange_status['healthy']:
                status['overall_healthy'] = False
        
        return status


    async def get_markets(self, exchange_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available markets from exchanges.
        If symbol mapping is enabled, includes unified symbol information.
        """
        if exchange_name:
            exchanges_to_query = [exchange_name]
        else:
            exchanges_to_query = list(self.exchange_instances.keys())
        
        markets = {}
        for name in exchanges_to_query:
            if name in self.exchange_instances:
                try:
                    exchange_markets = await self.exchange_instances[name].load_markets()
                    
                    # Optionally add unified symbol information
                    if self.use_symbol_mapping:
                        enhanced_markets = {}
                        for market_id, market in exchange_markets.items():
                            try:
                                unified = self._convert_symbol(market['symbol'], name, to_exchange=False)
                                market['unified_symbol'] = unified
                                enhanced_markets[market_id] = market
                            except:
                                enhanced_markets[market_id] = market
                        markets[name] = enhanced_markets
                    else:
                        markets[name] = exchange_markets
                        
                except Exception as e:
                    self.logger.error(f"Failed to get markets from {name}: {e}")
                    markets[name] = {}
        
        return markets


    async def fetch_ohlcv(self, 
        symbols: Union[str, List[str]], 
        start_date: Union[pd.Timestamp, datetime], 
        end_date: Union[pd.Timestamp, datetime], 
        interval: str,
        force_reload: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data from multiple exchanges for multiple symbols.

        Args:
            symbols: A symbol or list of symbols to fetch.
            start_date: The start date for the data.
            end_date: The end date for the data.
            interval: The frequency of the data (e.g., '1m', '1h', '1d').
            force_reload: If True, bypasses local cache and fetches from the exchange.

        Returns:
            A dictionary where keys are symbols and values are pandas DataFrames
            with the OHLCV data for that symbol.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        results: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            # 1. Try to load from local storage first if not forcing reload
            if not force_reload:
                local_df = await self._load_symbol_from_parquet(symbol, start_date, end_date, interval)
                if not local_df.empty:
                    self.logger.info(f"Loaded {len(local_df)} records for {symbol} from local Parquet files.")
                    results[symbol] = local_df
                    continue

            # 2. If no local data or force_reload is True, fetch from exchanges
            self.logger.info(f"Fetching {symbol} from exchanges (force_reload={force_reload}).")
            
            # Fetch from all exchanges in parallel
            tasks = []
            for exchange_name, exchange in self.exchange_instances.items():
                tasks.append(self._fetch_symbol_from_exchange(
                    exchange, exchange_name, symbol, interval, start_date, end_date
                ))
            
            # Gather results from all exchanges
            exchange_dfs = await asyncio.gather(*tasks)
            
            # Combine non-empty DataFrames
            valid_dfs = [df for df in exchange_dfs if not df.empty]
            
            if not valid_dfs:
                results[symbol] = pd.DataFrame()
                continue
            
            # Concatenate all exchange data
            combined_df = pd.concat(valid_dfs, ignore_index=True)
            
            # Set datetime index
            combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms', utc=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamp', 'exchange']).set_index('datetime')
            combined_df = combined_df.sort_index()
            
            # Save the newly fetched data
            await self.save_to_parquet(combined_df)
            
            results[symbol] = combined_df
        
        return results
    

    async def close(self):
        """Close all exchange connections"""
        # Save symbol configuration if using symbol mapping
        if self.use_symbol_mapping and self.symbol_manager:
            try:
                config_path = self.data_dir / 'symbol_config.json'
                self.symbol_manager.save_config(config_path)
                self.logger.info(f"Saved symbol configuration to {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to save symbol configuration: {e}")
        
        # Close all exchange connections
        for exchange in self.exchange_instances.values():
            await exchange.close()
        self.logger.info("Closed all exchange connections")


async def main():
    """Main function to test the CryptoProvider"""
    provider = CryptoProvider(exchanges=['binance', 'coinbase', 'kraken'], data_dir='data/cryptos')
    
    symbols = ['BTC/USDT']
    start_date = datetime(2025, 8, 5, 16, 00, tzinfo=timezone.utc)
    end_date = datetime(2025, 8, 5, 17, 40, tzinfo=timezone.utc)
    interval = '1m'
    force_reload = False
    
    data = await provider.fetch_ohlcv(symbols, start_date, end_date, interval, force_reload)
    print(data)
    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())