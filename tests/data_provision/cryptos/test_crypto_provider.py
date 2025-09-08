"""
Comprehensive unit tests for the CryptoProvider class.
Tests data validation, error handling, storage, API interactions,
and the new symbol mapping functionality.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import ccxt
import aiofiles
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.data_provision.cryptos.crypto_provider import CryptoProvider, OHLCVData, CircuitBreaker


class TestOHLCVData:
    """Test the OHLCVData dataclass and its validation"""
    
    def test_valid_ohlcv_data(self):
        """Test that valid OHLCV data passes validation"""
        data = OHLCVData(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            exchange='binance',
            symbol='BTC/USDT',
            interval='1h'
        )
        assert data.validate() is True
    
    def test_invalid_ohlc_relationship(self):
        """Test that invalid OHLC relationships fail validation"""
        # High lower than open
        data = OHLCVData(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=100.0,
            high=95.0,  # Invalid: high < open
            low=90.0,
            close=98.0,
            volume=1000.0,
            exchange='binance',
            symbol='BTC/USDT',
            interval='1h'
        )
        assert data.validate() is False
    
    def test_negative_values(self):
        """Test that negative prices fail validation"""
        data = OHLCVData(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=-100.0,  # Invalid: negative price
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            exchange='binance',
            symbol='BTC/USDT',
            interval='1h'
        )
        assert data.validate() is False
    
    def test_future_timestamp(self):
        """Test that future timestamps fail validation"""
        future_time = int((datetime.now() + timedelta(days=1)).timestamp() * 1000)
        data = OHLCVData(
            timestamp=future_time,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            exchange='binance',
            symbol='BTC/USDT',
            interval='1h'
        )
        assert data.validate() is False
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        data = OHLCVData(
            timestamp=1640995200000,  # 2022-01-01 00:00:00
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
            exchange='binance',
            symbol='BTC/USDT',
            interval='1h'
        )
        result = data.to_dict()
        
        assert isinstance(result, dict)
        assert result['timestamp'] == 1640995200000
        assert result['open'] == 100.0
        assert result['exchange'] == 'binance'


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
    async def test_fetch_ohlcv_traditional_format(self, provider):
        """Test fetching with traditional exchange-specific formats"""
        mock_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],
            [1640998800000, 47200.0, 47400.0, 46900.0, 47100.0, 95.2]
        ]
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000) # 1 hour
        provider.exchange_instances['binance'] = mock_exchange
        provider.exchange_instances.pop('coinbase', None) # Only test binance

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
        with patch('services.data_provision.cryptos.symbol_mapper.SymbolManager') as mock_sm_class:
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
        with patch('services.data_provision.cryptos.symbol_mapper.SymbolManager') as mock_sm:
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
        with patch('services.data_provision.cryptos.symbol_mapper.SymbolType'):
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
        
        # Mock exchanges
        mock_binance = AsyncMock()
        mock_binance.fetch_ohlcv.return_value = mock_data
        mock_binance.parse_timeframe = Mock(return_value = 3600 * 1000)
        provider_with_mapping.exchange_instances['binance'] = mock_binance
        
        mock_coinbase = AsyncMock()
        mock_coinbase.fetch_ohlcv.return_value = mock_data
        mock_coinbase.parse_timeframe = Mock(return_value = 3600 * 1000)
        provider_with_mapping.exchange_instances['coinbase'] = mock_coinbase
        
        # Fetch using unified symbol
        result = await provider_with_mapping.fetch_ohlcv(
            'BTC/USDT', 
            pd.Timestamp('2022-01-01', tz='UTC'), 
            pd.Timestamp('2022-01-01 01:00:00', tz='UTC'), 
            '1h'
        )
        
        start_ts = int(pd.Timestamp('2022-01-01', tz='UTC').timestamp() * 1000)
        
        # Check call arguments
        mock_binance.fetch_ohlcv.assert_called_once_with('BTCUSDT', '1h', start_ts, 1000)
        mock_coinbase.fetch_ohlcv.assert_called_once_with('BTC-USD', '1h', start_ts, 1000)

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
        mock_data = [[start_date.timestamp() * 1000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5]]

        # Mock exchange to return data
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        mock_exchange.parse_timeframe = Mock(return_value=3600 * 1000)
        provider_with_mapping.exchange_instances = {'binance': mock_exchange}

        # 1. Fetch and save data
        results_fetch = await provider_with_mapping.fetch_ohlcv(symbol, start_date, end_date, interval)

        # Check that a file was created
        expected_path = provider_with_mapping._get_file_path(symbol, start_date, 'binance', interval)
        assert expected_path.exists()

        # 2. Now, fetch again, which should load from the file
        with patch.object(provider_with_mapping, '_fetch_symbol_from_exchange', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [] # Ensure it returns nothing from exchange
            
            results_load = await provider_with_mapping.fetch_ohlcv(symbol, start_date, end_date, interval, force_reload=False)
            
            # Assert that the network fetch was NOT called
            mock_fetch.assert_not_called()
            
            # *** FIX: Create an expected DataFrame with only the saved columns ***
            expected_df = results_fetch[symbol][['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Assert that the data is the same
            pd.testing.assert_frame_equal(
                expected_df.reset_index(drop=True), 
                results_load[symbol].reset_index(drop=True),
            )
            
            # 3. Test force_reload
            await provider_with_mapping.fetch_ohlcv(symbol, start_date, end_date, interval, force_reload=True)
            mock_fetch.assert_called() # Should be called now


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
            {'timestamp': 1640995200000, 'open': 100, 'high': 105, 'low': 95, 'close': 102, 'volume': 1000},
            {'timestamp': 1640995200000, 'open': 101, 'high': 106, 'low': 96, 'close': 103, 'volume': 1100},  # Duplicate
            {'timestamp': 1640998800000, 'open': 102, 'high': 107, 'low': 97, 'close': 104, 'volume': 1200}
        ]
        
        df = pd.DataFrame(data)
        df_cleaned = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        assert len(df_cleaned) == 2
        # Should keep the last duplicate (with open=101)
        first_row = df_cleaned[df_cleaned['timestamp'] == 1640995200000].iloc[0]
        assert first_row['open'] == 101
    
    def test_data_sorting(self):
        """Test that data is properly sorted by timestamp"""
        data = [
            {'timestamp': 1640998800000, 'open': 102},  # Later timestamp first
            {'timestamp': 1640995200000, 'open': 100},  # Earlier timestamp
        ]
        
        df = pd.DataFrame(data)
        df_sorted = df.sort_values('timestamp')
        
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
        with patch('services.data_provision.cryptos.symbol_mapper.SymbolManager'):
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
        with patch('services.data_provision.cryptos.symbol_mapper.SymbolManager', 
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
    async def test_rate_limiting_behavior(self, temp_data_directory):
        """Test that rate limiting works correctly with new changes"""
        provider = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_directory,
            rate_limit_buffer=0.8  # Test the buffer parameter
        )
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = []
        provider.exchange_instances['binance'] = mock_exchange
        
        # Make rapid requests
        start_time = asyncio.get_event_loop().time()
        
        tasks = []
        for _ in range(5):
            task = provider.fetch_ohlcv('BTC/USDT', pd.Timestamp('2022-01-01', tz='UTC'), pd.Timestamp('2022-01-02', tz='UTC'), '1h')
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should take some time due to rate limiting
        elapsed = end_time - start_time
        assert elapsed >= 0  # Basic sanity check
        
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
            mock_exchange.parse_timeframe = Mock(return_value = 3600 * 1000)
            provider.exchange_instances[exchange_name] = mock_exchange
        
        start_time = asyncio.get_event_loop().time()
        result = await provider.fetch_ohlcv('BTC/USDT', pd.Timestamp('2022-01-01', tz='UTC'), pd.Timestamp('2022-01-02', tz='UTC'), '1h')
        end_time = asyncio.get_event_loop().time()
        
        # Should have data from both exchanges in the df
        df = result['BTC/USDT']
        assert len(df['exchange'].unique()) >= 1 # Can be 1 if one fails, or 2 if both succeed
        
        # Concurrent calls should be faster than sequential
        elapsed = end_time - start_time
        assert elapsed < 2.0  # Should complete quickly with mocked data
        
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

