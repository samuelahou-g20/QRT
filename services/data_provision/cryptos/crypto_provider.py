"""
Robust cryptocurrency data provider with multi-exchange support,
error handling, rate limiting, data validation, and optional symbol mapping.
"""

import asyncio
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
import logging
import time
import json
from abc import ABC, abstractmethod
import aiofiles
import aiolimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import hashlib


@dataclass
class OHLCVData:
    """Standardized OHLCV data structure"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str
    symbol: str
    timeframe: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def validate(self) -> bool:
        """Basic validation for OHLCV data"""
        try:
            # Check for valid OHLC relationships
            if not (self.low <= self.open <= self.high and 
                self.low <= self.close <= self.high and
                self.low <= self.high):
                return False
            
            # Check for reasonable values
            if any(v < 0 for v in [self.open, self.high, self.low, self.close, self.volume]):
                return False
                
            # Check timestamp is reasonable (not in future, not too old)
            now = int(datetime.now(timezone.utc) * 1000)
            # Allow data up to 10 years old instead of just 1 year
            if self.timestamp > now or self.timestamp < (now - 10 * 365 * 24 * 60 * 60 * 1000):
                return False
                
            return True
        except Exception:
            return False


class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
    
    def can_execute(self) -> bool:
        if self.state == 'closed':
            return True
        elif self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half_open'
                return True
            return False
        else:  # half_open
            return True
    
    def record_success(self):
        self.failure_count = 0
        self.state = 'closed'
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


class BaseDataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def fetch_ohlcv(self, symbol: str, timeframe: str, 
                         since: Optional[int] = None, limit: Optional[int] = None) -> List[OHLCVData]:
        pass
    
    @abstractmethod
    async def get_markets(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def close(self):
        pass


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
                from services.data_provision.cryptos.symbol_mapper import SymbolManager
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
                from services.data_provision.cryptos.symbol_mapper import SymbolType
                return str(self.symbol_manager.to_unified(symbol, exchange_name, SymbolType.SPOT))
        except Exception as e:
            self.logger.debug(f"Symbol conversion failed for {symbol} on {exchange_name}: {e}")
            return symbol
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable))
    )
    async def _fetch_exchange_ohlcv(self, 
                                   exchange: ccxt_async.Exchange,
                                   exchange_name: str,
                                   symbol: str, 
                                   timeframe: str,
                                   since: Optional[int] = None,
                                   limit: Optional[int] = None) -> List[OHLCVData]:
        """
        Fetch OHLCV data from a specific exchange with error handling
        """
        circuit_breaker = self.circuit_breakers[exchange_name]
        
        if not circuit_breaker.can_execute():
            raise ccxt.ExchangeNotAvailable(f"{exchange_name} circuit breaker is open")
        
        try:
            # Convert symbol if mapping is enabled
            exchange_symbol = self._convert_symbol(symbol, exchange_name, to_exchange=True)
            
            if exchange_symbol != symbol:
                self.logger.debug(f"Converted {symbol} to {exchange_symbol} for {exchange_name}")
            
            # Apply rate limiting
            async with self.rate_limiters[exchange_name]:
                raw_data = await exchange.fetch_ohlcv(
                    symbol=exchange_symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
            
            # Convert to standardized format
            ohlcv_data = []
            for candle in raw_data:
                data = OHLCVData(
                    timestamp=int(candle[0]),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    exchange=exchange_name,
                    symbol=symbol,  # Store original/unified symbol
                    timeframe=timeframe
                )
                
                # Validate data
                if data.validate():
                    ohlcv_data.append(data)
                else:
                    self.logger.warning(f"Invalid OHLCV data from {exchange_name}: {data}")
            
            circuit_breaker.record_success()
            return ohlcv_data
            
        except Exception as e:
            circuit_breaker.record_failure()
            self.logger.error(f"Error fetching from {exchange_name}: {e}")
            raise
    
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
    
    async def find_common_symbols(self, min_exchanges: int = 2) -> Dict[str, List[str]]:
        """
        Find symbols available on multiple exchanges (useful for arbitrage).
        Returns unified symbols if mapping is enabled.
        
        Args:
            min_exchanges: Minimum number of exchanges a symbol must be on
            
        Returns:
            Dict mapping symbols to list of exchanges that support them
        """
        # Get symbols from all exchanges
        all_symbols: Dict[str, Set[str]] = {}
        
        tasks = []
        for exchange in self.exchange_instances.keys():
            tasks.append(self.get_available_symbols(exchange))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for exchange, result in zip(self.exchange_instances.keys(), results):
            if isinstance(result, set):
                all_symbols[exchange] = result
            else:
                self.logger.error(f"Failed to get symbols from {exchange}: {result}")
                all_symbols[exchange] = set()
        
        # Find common symbols
        symbol_exchanges: Dict[str, List[str]] = {}
        
        # Collect all unique symbols
        all_unique_symbols = set()
        for symbols in all_symbols.values():
            all_unique_symbols.update(symbols)
        
        # Check which exchanges have each symbol
        for symbol in all_unique_symbols:
            exchanges_with_symbol = []
            for exchange, symbols in all_symbols.items():
                if symbol in symbols:
                    exchanges_with_symbol.append(exchange)
            
            if len(exchanges_with_symbol) >= min_exchanges:
                symbol_exchanges[symbol] = exchanges_with_symbol
        
        return symbol_exchanges
    
    async def fetch_ohlcv(self, 
                         symbol: str, 
                         timeframe: str = '1m',
                         since: Optional[int] = None, 
                         limit: Optional[int] = None,
                         exchanges: Optional[List[str]] = None,
                         validate_availability: bool = False) -> Dict[str, List[OHLCVData]]:
        """
        Fetch OHLCV data from multiple exchanges
        
        Args:
            symbol: Trading symbol (unified format if mapping enabled, e.g., 'BTC/USDT')
            timeframe: Timeframe (1m, 5m, 1h, 1d, etc.)
            since: Timestamp in milliseconds
            limit: Number of candles to fetch
            exchanges: Specific exchanges to query
            validate_availability: Check if symbol is available before fetching (only with mapping)
            
        Returns:
            Dict mapping exchange names to OHLCV data lists
        """
        target_exchanges = exchanges or list(self.exchange_instances.keys())
        results = {}
        
        # Optionally validate symbol availability (only works with symbol mapping)
        if validate_availability and self.use_symbol_mapping:
            valid_exchanges = []
            for exchange_name in target_exchanges:
                available = await self.get_available_symbols(exchange_name)
                if symbol in available:
                    valid_exchanges.append(exchange_name)
                else:
                    self.logger.warning(f"{symbol} not available on {exchange_name}")
            target_exchanges = valid_exchanges
        
        tasks = []
        for exchange_name in target_exchanges:
            if exchange_name in self.exchange_instances:
                exchange = self.exchange_instances[exchange_name]
                task = self._fetch_exchange_ohlcv(
                    exchange, exchange_name, symbol, timeframe, since, limit
                )
                tasks.append((exchange_name, task))
        
        # Execute all requests concurrently
        for exchange_name, task in tasks:
            try:
                data = await task
                results[exchange_name] = data
                self.logger.info(f"Fetched {len(data)} candles from {exchange_name}")
            except Exception as e:
                self.logger.error(f"Failed to fetch from {exchange_name}: {e}")
                results[exchange_name] = []
        
        return results
    
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
    
    def _get_file_path(self, symbol: str, date: datetime, exchange: str, 
                      timeframe: str, data_type: str = 'ohlcv') -> Path:
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
        
        filename = f"{timeframe}_{data_type}.parquet"
        return path / filename
    
    async def save_to_parquet(self, 
                            data: Dict[str, List[OHLCVData]], 
                            symbol: str, 
                            timeframe: str,
                            date: Optional[datetime] = None):
        """Save OHLCV data to parquet files"""
        if not date:
            date = datetime.now()
        
        for exchange_name, ohlcv_list in data.items():
            if not ohlcv_list:
                continue
            
            # Convert to DataFrame
            df_data = [data.to_dict() for data in ohlcv_list]
            df = pd.DataFrame(df_data)
            
            if df.empty:
                continue
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

            if not date:
                date = datetime.now(timezone.utc)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Get file path
            file_path = self._get_file_path(symbol, date, exchange_name, timeframe)
            
            try:
                # Save to parquet with compression
                df.to_parquet(file_path, compression='snappy', index=False)
                self.logger.info(f"Saved {len(df)} records to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save data to {file_path}: {e}")
    
    async def fetch_and_store(self, 
                            symbols: List[str], 
                            timeframe: str = '1m',
                            days_back: int = 1) -> None:
        """
        Fetch data for multiple symbols and store to parquet files
        
        Args:
            symbols: List of symbols to fetch (unified format if mapping enabled)
            timeframe: Timeframe to fetch
            days_back: Number of days of historical data to fetch
        """
        since = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
        
        for symbol in symbols:
            try:
                self.logger.info(f"Fetching {symbol} data...")
                data = await self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000,
                    validate_availability=self.use_symbol_mapping
                )
                
                await self.save_to_parquet(data, symbol, timeframe)
                
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {e}")
            
            # Small delay between symbols to be respectful
            await asyncio.sleep(0.1)
    
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


# Example usage and utility functions
async def main():
    """Example usage of the CryptoProvider with and without symbol mapping"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Example 1: Traditional usage (without symbol mapping)")
    print("=" * 60)
    
    # Initialize provider without symbol mapping (backward compatible)
    provider_traditional = CryptoProvider(
        exchanges=['binance'],
        data_dir='data/crypto_traditional',
        use_symbol_mapping=False  # This is the default
    )
    
    try:
        # Must use exchange-specific format
        symbols = ['BTCUSDT']  # Binance format
        await provider_traditional.fetch_and_store(
            symbols=symbols,
            timeframe='1h',
            days_back=1
        )
        print(f"Traditional mode: Fetched data for {symbols}")
    finally:
        await provider_traditional.close()
    
    print("\n" + "=" * 60)
    print("Example 2: With unified symbol mapping")
    print("=" * 60)
    
    # Initialize provider with symbol mapping
    provider_unified = CryptoProvider(
        exchanges=['binance', 'coinbase', 'kraken'],
        data_dir='data/crypto_unified',
        use_symbol_mapping=True
    )
    
    try:
        # Can use unified format for all exchanges
        symbols = ['BTC/USDT', 'ETH/USDT']  # Unified format
        
        print(f"\nFetching {symbols} from all exchanges using unified format...")
        await provider_unified.fetch_and_store(
            symbols=symbols,
            timeframe='1h',
            days_back=1
        )
        print(f"Unified mode: Fetched data for {symbols}")
        
        # Find arbitrage opportunities
        print("\nFinding symbols available on multiple exchanges...")
        common = await provider_unified.find_common_symbols(min_exchanges=2)
        for symbol, exchanges in list(common.items())[:3]:
            print(f"  {symbol}: {', '.join(exchanges)}")
        
        # Check health status
        health = provider_unified.get_health_status()
        print(f"\nSystem health: {health['overall_healthy']}")
        print(f"Symbol mapping enabled: {health['symbol_mapping_enabled']}")
        
    finally:
        await provider_unified.close()


if __name__ == "__main__":
    asyncio.run(main())