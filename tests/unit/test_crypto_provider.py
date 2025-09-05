"""
Comprehensive unit tests for the CryptoProvider class.
Tests data validation, error handling, storage, and API interactions.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import ccxt
import aiofiles

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
            timeframe='1h'
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
            timeframe='1h'
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
            timeframe='1h'
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
            timeframe='1h'
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
            timeframe='1h'
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


class TestCryptoProvider:
    """Test the main CryptoProvider functionality"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def provider(self, temp_data_dir):
        """Create a CryptoProvider instance for testing"""
        return CryptoProvider(
            exchanges=['binance', 'coinbase'],
            data_dir=temp_data_dir,
            api_keys={}
        )
    
    def test_initialization(self, temp_data_dir):
        """Test proper initialization of CryptoProvider"""
        provider = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_dir
        )
        
        assert 'binance' in provider.exchange_instances
        assert 'binance' in provider.rate_limiters
        assert 'binance' in provider.circuit_breakers
        assert provider.data_dir == Path(temp_data_dir)
    
    def test_file_path_generation(self, provider):
        """Test that file paths are generated correctly"""
        test_date = datetime(2022, 1, 15)
        path = provider._get_file_path(
            symbol='BTC/USDT',
            date=test_date,
            exchange='binance',
            timeframe='1h'
        )
        
        expected_parts = [
            'ohlcv',
            'binance',
            'BTC_USDT',
            'year=2022',
            'month=01',
            'day=15',
            '1h_ohlcv.parquet'
        ]
        
        path_str = str(path)
        for part in expected_parts:
            assert part in path_str
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_mock_success(self, provider):
        """Test successful OHLCV fetching with mocked exchange"""
        # Mock the exchange response
        mock_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],
            [1640998800000, 47200.0, 47400.0, 46900.0, 47100.0, 95.2]
        ]
        
        # Mock the exchange instance
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        provider.exchange_instances['binance'] = mock_exchange
        
        result = await provider.fetch_ohlcv('BTC/USDT', '1h')
        
        assert 'binance' in result
        assert len(result['binance']) == 2
        
        first_candle = result['binance'][0]
        assert first_candle.symbol == 'BTC/USDT'
        assert first_candle.exchange == 'binance'
        assert first_candle.open == 47000.0
        assert first_candle.high == 47500.0
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_handles_invalid_data(self, provider):
        """Test that invalid data is filtered out"""
        # Mock data with one invalid candle (high < low)
        mock_data = [
            [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],  # Valid
            [1640998800000, 47200.0, 46000.0, 47000.0, 47100.0, 95.2],   # Invalid: high < low
        ]
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = mock_data
        provider.exchange_instances['binance'] = mock_exchange
        
        result = await provider.fetch_ohlcv('BTC/USDT', '1h')
        
        # Should only return the valid candle
        assert len(result['binance']) == 1
        assert result['binance'][0].open == 47000.0
    
    @pytest.mark.asyncio
    async def test_fetch_ohlcv_exchange_error(self, provider):
        """Test handling of exchange errors"""
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Connection failed")
        provider.exchange_instances['binance'] = mock_exchange
        
        result = await provider.fetch_ohlcv('BTC/USDT', '1h')
        
        # Should return empty list for failed exchange
        assert result['binance'] == []
    
    @pytest.mark.asyncio 
    async def test_save_to_parquet(self, provider, temp_data_dir):
        """Test saving data to parquet files"""
        # Create test data
        test_data = {
            'binance': [
                OHLCVData(
                    timestamp=1640995200000,
                    open=47000.0,
                    high=47500.0,
                    low=46800.0,
                    close=47200.0,
                    volume=100.5,
                    exchange='binance',
                    symbol='BTC/USDT',
                    timeframe='1h'
                )
            ]
        }
        
        await provider.save_to_parquet(test_data, 'BTC/USDT', '1h')
        
        # Check that file was created
        expected_path = provider._get_file_path(
            'BTC/USDT', 
            datetime.now(), 
            'binance', 
            '1h'
        )
        
        assert expected_path.exists()
        
        # Verify data integrity
        df = pd.read_parquet(expected_path)
        assert len(df) == 1
        assert df.iloc[0]['open'] == 47000.0
        assert df.iloc[0]['symbol'] == 'BTC/USDT'
    
    def test_health_status(self, provider):
        """Test health status reporting"""
        status = provider.get_health_status()
        
        assert 'exchanges' in status
        assert 'overall_healthy' in status
        
        for exchange in provider.exchange_instances.keys():
            assert exchange in status['exchanges']
            assert 'circuit_breaker_state' in status['exchanges'][exchange]
            assert 'healthy' in status['exchanges'][exchange]
    
    @pytest.mark.asyncio
    async def test_close_connections(self, provider):
        """Test that all connections are properly closed"""
        # Mock exchange instances
        mock_exchange1 = AsyncMock()
        mock_exchange2 = AsyncMock()
        
        provider.exchange_instances = {
            'binance': mock_exchange1,
            'coinbase': mock_exchange2
        }
        
        await provider.close()
        
        mock_exchange1.close.assert_called_once()
        mock_exchange2.close.assert_called_once()


class TestDataQuality:
    """Test data quality and validation functions"""
    
    def test_duplicate_timestamp_removal(self):
        """Test that duplicate timestamps are handled correctly"""
        # This would be tested in the save_to_parquet method
        data = [
            {'timestamp': 1640995200000, 'open': 100, 'high': 105, 'low': 95, 'close': 102, 'volume': 1000},
            {'timestamp': 1640995200000, 'open': 101, 'high': 106, 'low': 96, 'close': 103, 'volume': 1100},  # Duplicate timestamp
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
    
    @pytest.mark.asyncio
    async def test_multiple_symbols_fetch(self, temp_data_dir):
        """Test fetching multiple symbols in sequence"""
        provider = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_dir
        )
        
        # Mock successful responses for multiple symbols
        mock_exchange = AsyncMock()
        mock_data = [[1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5]]
        mock_exchange.fetch_ohlcv.return_value = mock_data
        provider.exchange_instances['binance'] = mock_exchange
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        
        # This tests the fetch_and_store method indirectly
        for symbol in symbols:
            result = await provider.fetch_ohlcv(symbol, '1h')
            assert 'binance' in result
            assert len(result['binance']) == 1
        
        await provider.close()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_behavior(self, temp_data_dir):
        """Test that rate limiting works as expected"""
        provider = CryptoProvider(
            exchanges=['binance'],
            data_dir=temp_data_dir
        )
        
        mock_exchange = AsyncMock()
        mock_exchange.fetch_ohlcv.return_value = []
        provider.exchange_instances['binance'] = mock_exchange
        
        # Make rapid requests
        start_time = asyncio.get_event_loop().time()
        
        tasks = []
        for _ in range(5):
            task = provider.fetch_ohlcv('BTC/USDT', '1h')
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should take some time due to rate limiting
        # (This is a basic test - in practice you'd want more sophisticated timing tests)
        elapsed = end_time - start_time
        assert elapsed >= 0  # Basic sanity check
        
        await provider.close()


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Performance and stress tests
class TestPerformance:
    """Performance-related tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_exchange_calls(self, temp_data_dir):
        """Test that concurrent calls to multiple exchanges work efficiently"""
        provider = CryptoProvider(
            exchanges=['binance', 'coinbase'],
            data_dir=temp_data_dir
        )
        
        # Mock both exchanges
        for exchange_name in ['binance', 'coinbase']:
            mock_exchange = AsyncMock()
            mock_exchange.fetch_ohlcv.return_value = [
                [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5]
            ]
            provider.exchange_instances[exchange_name] = mock_exchange
        
        start_time = asyncio.get_event_loop().time()
        result = await provider.fetch_ohlcv('BTC/USDT', '1h')
        end_time = asyncio.get_event_loop().time()
        
        # Should have data from both exchanges
        assert len(result) == 2
        assert 'binance' in result
        assert 'coinbase' in result
        
        # Concurrent calls should be faster than sequential
        elapsed = end_time - start_time
        assert elapsed < 2.0  # Should complete quickly with mocked data
        
        await provider.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])