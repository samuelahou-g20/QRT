"""
Comprehensive unit tests for the CryptoProvider class.
Tests data validation, error handling, storage, API interactions,
and the new symbol mapping functionality.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_provision.cryptos.crypto_provider import CryptoProvider
from src.data_provision.cryptos.circuit_breaker import CircuitBreaker


class TestDataValidation:
    """Test the OHLCV data validation functionality"""
    
    def test_valid_ohlcv_data(self):
        """Test that valid OHLCV data passes validation"""
        provider = CryptoProvider(exchanges=['binance'], data_dir=tempfile.mkdtemp())
        
        valid_candle = [
            int(datetime.now().timestamp() * 1000),
            100.0,  # open
            105.0,  # high
            95.0,   # low
            102.0,  # close
            1000.0  # volume
        ]
        assert provider._is_valid_ohlcv(valid_candle) is True
    
    def test_invalid_ohlc_relationship(self):
        """Test that invalid OHLC relationships fail validation"""
        provider = CryptoProvider(exchanges=['binance'], data_dir=tempfile.mkdtemp())
        
        # High lower than open
        invalid_candle = [
            int(datetime.now().timestamp() * 1000),
            100.0,  # open
            95.0,   # high (Invalid: high < open)
            90.0,   # low
            98.0,   # close
            1000.0  # volume
        ]
        assert provider._is_valid_ohlcv(invalid_candle) is False
    
    def test_negative_values(self):
        """Test that negative prices fail validation"""
        provider = CryptoProvider(exchanges=['binance'], data_dir=tempfile.mkdtemp())
        
        negative_candle = [
            int(datetime.now().timestamp() * 1000),
            -100.0,  # negative open
            105.0,   # high
            95.0,    # low
            102.0,   # close
            1000.0   # volume
        ]
        assert provider._is_valid_ohlcv(negative_candle) is False
    
    def test_future_timestamp(self):
        """Test that future timestamps fail validation"""
        provider = CryptoProvider(exchanges=['binance'], data_dir=tempfile.mkdtemp())
        
        future_time = int((datetime.now() + timedelta(days=1)).timestamp() * 1000)
        future_candle = [
            future_time,
            100.0,  # open
            105.0,  # high
            95.0,   # low
            102.0,  # close
            1000.0  # volume
        ]
        assert provider._is_valid_ohlcv(future_candle) is False
    
    def test_insufficient_data(self):
        """Test that candles with insufficient data fail validation"""
        provider = CryptoProvider(exchanges=['binance'], data_dir=tempfile.mkdtemp())
        
        # Less than 6 elements
        short_candle = [1640995200000, 100.0, 105.0, 95.0]
        assert provider._is_valid_ohlcv(short_candle) is False
    
    def test_invalid_data_types(self):
        """Test that invalid data types fail validation"""
        provider = CryptoProvider(exchanges=['binance'], data_dir=tempfile.mkdtemp())
        
        # String instead of number
        invalid_type_candle = [
            int(datetime.now().timestamp() * 1000),
            "not_a_number",  # invalid type
            105.0,
            95.0,
            102.0,
            1000.0
        ]
        assert provider._is_valid_ohlcv(invalid_type_candle) is False


class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_initial_state_closed(self):
        """Test that circuit breaker starts in closed state"""
        cb = CircuitBreaker()
        assert cb.state == 'closed'
        assert cb.can_execute() is True
    
    def test_failure_threshold(self):
        """Test that circuit breaker opens after failure threshold"""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record failures up to threshold
        for _ in range(3):
            cb.record_failure()
        
        assert cb.state == 'open'
        assert cb.can_execute() is False
    
    def test_timeout_recovery(self):
        """Test that circuit breaker moves to half-open after timeout"""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)  # 100ms timeout
        
        # Trigger circuit breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.state == 'open'
        
        # Wait for timeout
        import time
        time.sleep(0.2)
        
        assert cb.can_execute() is True
        assert cb.state == 'half_open'
    
    def test_success_resets_circuit_breaker(self):
        """Test that success resets the circuit breaker"""
        cb = CircuitBreaker()
        
        cb.record_failure()
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == 'closed'


class TestCryptoProviderTraditional:
    """Test the CryptoProvider without symbol mapping (backward compatibility)"""
    
    @pytest.fixture
    def temp_data_directory(self):
        """Create a temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def provider(self, temp_data_directory):
        """Create a CryptoProvider instance for testing (traditional mode)"""
        return CryptoProvider(
            exchanges=['binance', 'coinbase'],
            data_dir=temp_data_directory,
            api_keys={},
            use_symbol_mapping=False  # Traditional mode
        )
    
    def test_initialization_without_mapping(self, temp_data_directory):
        """Test proper initialization without symbol mapping"""
        provider = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_directory,
            use_symbol_mapping=False
        )
        
        assert 'binance' in provider.exchange_instances
        assert 'binance' in provider.rate_limiters
        assert 'binance' in provider.circuit_breakers
        assert provider.data_dir == Path(temp_data_directory)
        assert provider.use_symbol_mapping is False
        assert provider.symbol_manager is None
    
    def test_file_path_generation(self, provider):
        """Test that file paths are generated correctly"""
        test_date = datetime(2022, 1, 15)
        path = provider._get_file_path(
            symbol='BTC/USDT',
            date=test_date,
            exchange='binance',
            interval='1h'
        )
        
        expected_parts = [
            'ohlcv',
            'binance',
            'BTC_USDT',
            'year=2022',
            'month=01',
            'day=15',
            '1h.parquet'
        ]
        
        path_str = str(path)
        for part in expected_parts:
            assert part in path_str
    
    @pytest.mark.asyncio
    async def test_fetch_symbol_from_exchange_returns_dataframe(self, provider):
        """Test that _fetch_symbol_from_exchange returns a DataFrame"""
        mock_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],
            [1640998800000, 47200.0, 47400.0, 46900.0, 47100.0, 95.2]
        ]
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
        
        result = await provider._fetch_symbol_from_exchange(
            mock_exchange,
            'binance',
            'BTCUSDT',
            '1h',
            pd.Timestamp('2022-01-01', tz='UTC'),
            pd.Timestamp('2022-01-02', tz='UTC')
        )
        
        # Should return a DataFrame now, not a list
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'timestamp' in result.columns
        assert 'open' in result.columns
        assert 'exchange' in result.columns
        assert result['exchange'].iloc[0] == 'binance'
        assert result['symbol'].iloc[0] == 'BTCUSDT'
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_traditional_format(self, provider):
        """Test fetching with traditional exchange-specific formats"""
        mock_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],
            [1640998800000, 47200.0, 47400.0, 46900.0, 47100.0, 95.2]
        ]
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
        provider.exchange_instances['binance'] = mock_exchange
        provider.exchange_instances.pop('coinbase', None)

        # Use Binance-specific format
        result = await provider.fetch_ohlcv(
            symbols='BTCUSDT', 
            start_date=pd.Timestamp('2022-01-01', tz='UTC'), 
            end_date=pd.Timestamp('2022-01-02', tz='UTC'), 
            interval='1h'
        )
        
        assert 'BTCUSDT' in result
        df = result['BTCUSDT']
        assert len(df) == 2
        
        # Verify the exchange was called with the original symbol
        mock_exchange.fetch_ohlcv.assert_called()
        call_args = mock_exchange.fetch_ohlcv.call_args
        assert call_args[0][0] == 'BTCUSDT'
        
        assert df.iloc[0]['open'] == 47000.0
        assert df.iloc[0]['symbol'] == 'BTCUSDT'
    
    @pytest.mark.asyncio
    async def test_health_status_without_mapping(self, provider):
        """Test health status reporting shows mapping disabled"""
        status = provider.get_health_status()
        
        assert 'symbol_mapping_enabled' in status
        assert status['symbol_mapping_enabled'] is False
        assert 'exchanges' in status
        assert 'overall_healthy' in status


class TestCryptoProviderWithSymbolMapping:
    """Test the CryptoProvider with symbol mapping enabled"""
    
    @pytest.fixture
    def temp_data_directory(self):
        """Create a temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_symbol_manager(self):
        """Create a mock symbol manager"""
        manager = MagicMock()
        manager.to_exchange = MagicMock(side_effect=lambda s, e: {
            ('BTC/USDT', 'binance'): 'BTCUSDT',
            ('BTC/USDT', 'coinbase'): 'BTC-USD',
            ('BTC/USDT', 'kraken'): 'XBTUSDT',
            ('ETH/USDT', 'binance'): 'ETHUSDT',
            ('ETH/USDT', 'coinbase'): 'ETH-USD',
        }.get((s, e), s))
        
        manager.to_unified = MagicMock(side_effect=lambda s, e, t: {
            ('BTCUSDT', 'binance'): 'BTC/USDT',
            ('BTC-USD', 'coinbase'): 'BTC/USDT',
            ('XBTUSDT', 'kraken'): 'BTC/USDT',
            ('ETHUSDT', 'binance'): 'ETH/USDT',
            ('ETH-USD', 'coinbase'): 'ETH/USDT',
        }.get((s, e), s))
        
        return manager
    
    @pytest.fixture
    def provider_with_mapping(self, temp_data_directory, mock_symbol_manager):
        """Create a CryptoProvider with symbol mapping enabled"""
        with patch('src.data_provision.cryptos.symbol_mapper.SymbolManager') as mock_sm_class:
            mock_sm_class.return_value = mock_symbol_manager
            
            provider = CryptoProvider(
                exchanges=['binance', 'coinbase'],
                data_dir=temp_data_directory,
                use_symbol_mapping=True
            )
            provider.symbol_manager = mock_symbol_manager
            return provider
    
    def test_initialization_with_mapping(self, temp_data_directory):
        """Test initialization with symbol mapping enabled"""
        # Mock the symbol_mapper module import
        with patch('src.data_provision.cryptos.symbol_mapper.SymbolManager') as mock_sm:
            provider = CryptoProvider(
                exchanges=['binance'],
                data_dir=temp_data_directory,
                use_symbol_mapping=True
            )
            
            assert provider.use_symbol_mapping is True
            assert provider.symbol_manager is not None
            assert hasattr(provider, 'available_symbols_cache')
            assert hasattr(provider, 'cache_timestamp')
    
    @pytest.mark.asyncio
    async def test_symbol_conversion_to_exchange(self, provider_with_mapping):
        """Test symbol conversion from unified to exchange format"""
        # Test Binance conversion
        result = provider_with_mapping._convert_symbol('BTC/USDT', 'binance', to_exchange=True)
        assert result == 'BTCUSDT'
        
        # Test Coinbase conversion
        result = provider_with_mapping._convert_symbol('BTC/USDT', 'coinbase', to_exchange=True)
        assert result == 'BTC-USD'
        
        # Test Kraken conversion
        result = provider_with_mapping._convert_symbol('BTC/USDT', 'kraken', to_exchange=True)
        assert result == 'XBTUSDT'
    
    @pytest.mark.asyncio
    async def test_symbol_conversion_to_unified(self, provider_with_mapping):
        """Test symbol conversion from exchange to unified format"""
        # Mock the SymbolType import
        with patch('src.data_provision.cryptos.symbol_mapper.SymbolType'):
            # Test Binance conversion
            result = provider_with_mapping._convert_symbol('BTCUSDT', 'binance', to_exchange=False)
            assert result == 'BTC/USDT'
            
            # Test Coinbase conversion  
            result = provider_with_mapping._convert_symbol('BTC-USD', 'coinbase', to_exchange=False)
            assert result == 'BTC/USDT'
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_with_unified_symbols(self, provider_with_mapping):
        """Test fetching OHLCV data using unified symbols"""
        mock_data = [[1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5]]
        
        # Mock exchanges to return data
        for exchange_name in ['binance', 'coinbase']:
            mock_exchange = AsyncMock()
            mock_exchange.fetch_ohlcv.return_value = mock_data
            mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
            provider_with_mapping.exchange_instances[exchange_name] = mock_exchange
        
        # Fetch using unified symbol
        result = await provider_with_mapping.fetch_ohlcv(
            'BTC/USDT', 
            pd.Timestamp('2022-01-01', tz='UTC'), 
            pd.Timestamp('2022-01-01 01:00:00', tz='UTC'), 
            '1h'
        )
        
        start_ts = int(pd.Timestamp('2022-01-01', tz='UTC').timestamp() * 1000)
        
        # Check exchanges were called with correct converted symbols
        provider_with_mapping.exchange_instances['binance'].fetch_ohlcv.assert_called_once_with('BTCUSDT', '1h', start_ts, 1000)
        provider_with_mapping.exchange_instances['coinbase'].fetch_ohlcv.assert_called_once_with('BTC-USD', '1h', start_ts, 1000)

        # Check results
        assert 'BTC/USDT' in result
        df = result['BTC/USDT']
        assert len(df) > 0
        assert df.iloc[0]['symbol'] == 'BTC/USDT'
    
    @pytest.mark.asyncio
    async def test_get_available_symbols_with_mapping(self, provider_with_mapping):
        """Test getting available symbols returns unified format"""
        # Mock exchange markets
        mock_markets = {
            'BTCUSDT': {'symbol': 'BTC/USDT', 'id': 'BTCUSDT'},
            'ETHUSDT': {'symbol': 'ETH/USDT', 'id': 'ETHUSDT'},
        }
        
        mock_exchange = AsyncMock()
        mock_exchange.load_markets.return_value = mock_markets
        provider_with_mapping.exchange_instances['binance'] = mock_exchange
        
        # Get available symbols
        symbols = await provider_with_mapping.get_available_symbols('binance')
        
        # Should return unified symbols
        assert 'BTC/USDT' in symbols
        assert 'ETH/USDT' in symbols
        assert 'BTCUSDT' not in symbols  # Exchange format should not be in result
    
    @pytest.mark.asyncio
    async def test_save_and_load_from_parquet(self, provider_with_mapping):
        """Test that data is saved to and loaded from parquet correctly."""
        symbol = 'BTC/USDT'
        interval = '1h'
        start_date = pd.Timestamp('2022-01-01', tz='UTC')
        end_date = pd.Timestamp('2022-01-01 01:00:00', tz='UTC')
        
        # Create mock data as DataFrame for both exchanges
        mock_df_binance = pd.DataFrame({
            'timestamp': [int(start_date.timestamp() * 1000)],
            'open': [47000.0],
            'high': [47500.0],
            'low': [46800.0],
            'close': [47200.0],
            'volume': [100.5],
            'exchange': ['binance'],
            'symbol': [symbol],
            'interval': [interval],
            'datetime': [start_date]
        })
        
        mock_df_coinbase = pd.DataFrame({
            'timestamp': [int(start_date.timestamp() * 1000)],
            'open': [47010.0],
            'high': [47510.0],
            'low': [46810.0],
            'close': [47210.0],
            'volume': [101.5],
            'exchange': ['coinbase'],
            'symbol': [symbol],
            'interval': [interval],
            'datetime': [start_date]
        })

        # Mock _fetch_symbol_from_exchange to return DataFrame based on exchange
        with patch.object(provider_with_mapping, '_fetch_symbol_from_exchange', new_callable=AsyncMock) as mock_fetch:
            # Setup side effect to return different data for different exchanges
            def fetch_side_effect(exchange, exchange_name, *args, **kwargs):
                if exchange_name == 'binance':
                    return mock_df_binance
                elif exchange_name == 'coinbase':
                    return mock_df_coinbase
                return pd.DataFrame()
            
            mock_fetch.side_effect = fetch_side_effect
            
            # 1. Fetch and save data (will call for both exchanges)
            results_fetch = await provider_with_mapping.fetch_ohlcv(symbol, start_date, end_date, interval)
            
            # Give time for async save
            await asyncio.sleep(0.1)

            # Check that files were created for both exchanges
            for exchange_name in ['binance', 'coinbase']:
                expected_path = provider_with_mapping._get_file_path(symbol, start_date, exchange_name, interval)
                assert expected_path.exists(), f"File should exist for {exchange_name}"

            # Record initial call count (should be 2 - one for each exchange)
            initial_calls = mock_fetch.call_count
            assert initial_calls == 2, "Should have fetched from both exchanges"

            # 2. Now fetch again with force_reload=False, should load from files
            results_load = await provider_with_mapping.fetch_ohlcv(symbol, start_date, end_date, interval, force_reload=False)
            
            # Should not have made any additional fetch calls
            assert mock_fetch.call_count == initial_calls, "Should not fetch when loading from cache"
            
            # Data should be loaded
            assert symbol in results_load
            assert len(results_load[symbol]) > 0
            
            # 3. Test force_reload=True makes new fetch calls
            await provider_with_mapping.fetch_ohlcv(symbol, start_date, end_date, interval, force_reload=True)
            assert mock_fetch.call_count > initial_calls, "force_reload should trigger new fetches"

    @pytest.mark.asyncio
    async def test_save_symbol_config_on_close(self, provider_with_mapping):
        """Test that symbol configuration is saved on close"""
        await provider_with_mapping.close()
        
        # Check that save_config was called
        provider_with_mapping.symbol_manager.save_config.assert_called_once()
        
        # Check the path used
        call_args = provider_with_mapping.symbol_manager.save_config.call_args
        config_path = call_args[0][0]
        assert 'symbol_config.json' in str(config_path)
    
    def test_health_status_with_mapping(self, provider_with_mapping):
        """Test health status shows mapping enabled"""
        status = provider_with_mapping.get_health_status()
        
        assert status['symbol_mapping_enabled'] is True


class TestDataQuality:
    """Test data quality and validation functions"""
    
    def test_duplicate_timestamp_removal(self):
        """Test that duplicate timestamps are handled correctly"""
        data = [
            {'timestamp': 1640995200000, 'open': 100, 'high': 105, 'low': 95, 'close': 102, 'volume': 1000, 'exchange': 'binance'},
            {'timestamp': 1640995200000, 'open': 101, 'high': 106, 'low': 96, 'close': 103, 'volume': 1100, 'exchange': 'coinbase'},  # Same timestamp, different exchange
            {'timestamp': 1640998800000, 'open': 102, 'high': 107, 'low': 97, 'close': 104, 'volume': 1200, 'exchange': 'binance'}
        ]
        
        df = pd.DataFrame(data)
        # CryptoProvider keeps duplicates from different exchanges
        df_cleaned = df.drop_duplicates(subset=['timestamp', 'exchange'])
        
        assert len(df_cleaned) == 3  # All kept since different exchanges
    
    def test_data_sorting(self):
        """Test that data is properly sorted by timestamp"""
        data = [
            {'timestamp': 1640998800000, 'open': 102},  # Later timestamp first
            {'timestamp': 1640995200000, 'open': 100},  # Earlier timestamp
        ]
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df_sorted = df.set_index('datetime').sort_index()
        
        assert df_sorted.iloc[0]['timestamp'] == 1640995200000
        assert df_sorted.iloc[1]['timestamp'] == 1640998800000


class TestIntegrationScenarios:
    """Integration-style tests for real-world scenarios"""
    
    @pytest.fixture
    def temp_data_directory(self):
        """Create a temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            
    @pytest.mark.asyncio
    async def test_mixed_mode_providers(self, temp_data_directory):
        """Test that providers with and without mapping can coexist"""
        # Create provider without mapping
        provider_traditional = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_directory,
            use_symbol_mapping=False
        )
        
        # Create provider with mapping
        with patch('src.data_provision.cryptos.symbol_mapper.SymbolManager'):
            provider_unified = CryptoProvider(
                exchanges=['binance'],
                data_dir=temp_data_directory,
                use_symbol_mapping=True
            )
        
        assert provider_traditional.use_symbol_mapping is False
        assert provider_unified.use_symbol_mapping is True
        
        await provider_traditional.close()
        await provider_unified.close()
    
    @pytest.mark.asyncio
    async def test_fallback_when_symbol_mapper_missing(self, temp_data_directory):
        """Test graceful fallback when symbol_mapper module is not available"""
        # Simulate missing module
        with patch('src.data_provision.cryptos.symbol_mapper.SymbolManager', 
                   side_effect=ImportError("Module not found")):
            
            provider = CryptoProvider(
                exchanges=['binance'],
                data_dir=temp_data_directory,
                use_symbol_mapping=True  # Try to enable but module missing
            )
            
            # Should fall back to disabled
            assert provider.use_symbol_mapping is False
            assert provider.symbol_manager is None
            
            await provider.close()
    
    @pytest.mark.asyncio
    async def test_validation_filters_bad_data(self, temp_data_directory):
        """Test that invalid candles are filtered out during fetch"""
        provider = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_directory
        )
        
        # Mix of valid and invalid candles
        mock_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],  # Valid
            [1640998800000, -47200.0, 47400.0, 46900.0, 47100.0, 95.2],   # Invalid: negative open
            [1641002400000, 47100.0, 47300.0, 46700.0, 47000.0, 110.8],   # Valid
            [1641006000000, 47000.0, 46500.0, 46800.0, 47200.0, 100.5],   # Invalid: high < low
        ]
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
        
        df = await provider._fetch_symbol_from_exchange(
            mock_exchange,
            'binance',
            'BTC/USDT',
            '1h',
            pd.Timestamp('2022-01-01', tz='UTC'),
            pd.Timestamp('2022-01-02', tz='UTC')
        )
        
        # Should only have the 2 valid candles
        assert len(df) == 2
        assert df.iloc[0]['open'] == 47000.0
        assert df.iloc[1]['open'] == 47100.0
        
        await provider.close()


class TestPerformance:
    """Performance-related tests"""
    
    @pytest.fixture
    def temp_data_directory(self):
        """Create a temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.asyncio
    async def test_concurrent_exchange_calls(self, temp_data_directory):
        """Test that concurrent calls to multiple exchanges work efficiently"""
        provider = CryptoProvider(
            exchanges=['binance', 'coinbase'],
            data_dir=temp_data_directory
        )
        
        # Mock both exchanges
        for exchange_name in ['binance', 'coinbase']:
            mock_exchange = AsyncMock()
            mock_exchange.fetch_ohlcv.return_value = [
                [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5]
            ]
            mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
            provider.exchange_instances[exchange_name] = mock_exchange
        
        start_time = asyncio.get_event_loop().time()
        result = await provider.fetch_ohlcv(
            'BTC/USDT', 
            pd.Timestamp('2022-01-01', tz='UTC'), 
            pd.Timestamp('2022-01-02', tz='UTC'), 
            '1h'
        )
        end_time = asyncio.get_event_loop().time()
        
        # Should have data from both exchanges
        df = result['BTC/USDT']
        exchanges_in_data = df['exchange'].unique() if 'exchange' in df.columns else []
        assert len(exchanges_in_data) >= 1  # At least one exchange returned data
        
        # Concurrent calls should be fast
        elapsed = end_time - start_time
        assert elapsed < 2.0  # Should complete quickly with mocked data
        
        await provider.close()

    @pytest.mark.asyncio  
    async def test_empty_response_handling(self, temp_data_directory):
        """Test handling of empty responses from exchanges"""
        provider = CryptoProvider(
            exchanges=['binance', 'coinbase'],
            data_dir=temp_data_directory
        )
        
        # Mock exchanges to return empty data
        for exchange_name in ['binance', 'coinbase']:
            mock_exchange = AsyncMock()
            mock_exchange.fetch_ohlcv.return_value = []
            mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
            provider.exchange_instances[exchange_name] = mock_exchange
        
        result = await provider.fetch_ohlcv(
            'BTC/USDT',
            pd.Timestamp('2022-01-01', tz='UTC'),
            pd.Timestamp('2022-01-02', tz='UTC'),
            '1h'
        )
        
        # Should return empty DataFrame for the symbol
        assert 'BTC/USDT' in result
        assert result['BTC/USDT'].empty
        
        await provider.close()


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])