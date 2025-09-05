# conftest.py - Pytest configuration and shared fixtures

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# Test data fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing"""
    return [
        [1640995200000, 47000.0, 47500.0, 46800.0, 47200.0, 100.5],  # 2022-01-01 00:00:00
        [1640998800000, 47200.0, 47400.0, 46900.0, 47100.0, 95.2],   # 2022-01-01 01:00:00
        [1641002400000, 47100.0, 47300.0, 46700.0, 47000.0, 110.8],  # 2022-01-01 02:00:00
    ]


@pytest.fixture
def invalid_ohlcv_data():
    """Invalid OHLCV data for testing error handling"""
    return [
        [1640995200000, 47000.0, 46500.0, 46800.0, 47200.0, 100.5],  # high < low
        [1640998800000, -47200.0, 47400.0, 46900.0, 47100.0, 95.2],  # negative open
        [1641002400000, 47100.0, 47300.0, 46700.0, 47000.0, -110.8], # negative volume
    ]


@pytest.fixture
def sample_market_data():
    """Sample market data structure"""
    return {
        'BTC/USDT': {
            'id': 'BTCUSDT',
            'symbol': 'BTC/USDT',
            'base': 'BTC',
            'quote': 'USDT',
            'active': True,
            'type': 'spot',
            'precision': {'amount': 8, 'price': 2},
            'limits': {
                'amount': {'min': 0.00001, 'max': 9000},
                'price': {'min': 0.01, 'max': 1000000}
            }
        },
        'ETH/USDT': {
            'id': 'ETHUSDT', 
            'symbol': 'ETH/USDT',
            'base': 'ETH',
            'quote': 'USDT',
            'active': True,
            'type': 'spot',
            'precision': {'amount': 8, 'price': 2},
            'limits': {
                'amount': {'min': 0.0001, 'max': 10000},
                'price': {'min': 0.01, 'max': 100000}
            }
        }
    }


@pytest.fixture(scope="session")
def event_loop():
    """Event loop for async tests"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_data_directory():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_exchange_config():
    """Mock exchange configuration"""
    return {
        'binance': {
            'ratelimit': 1200,
            'sandbox': True,
            'apiKey': 'test_key',
            'secret': 'test_secret'
        },
        'coinbase': {
            'ratelimit': 600, 
            'sandbox': True,
            'apiKey': 'test_key',
            'secret': 'test_secret',
            'passphrase': 'test_passphrase'
        }
    }


# Performance test markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "live_api: marks tests that require live API access")


# Utility functions for tests
class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_ohlcv_series(
        start_price: float = 50000.0,
        num_candles: int = 100,
        timeframe_minutes: int = 60,
        volatility: float = 0.02
    ) -> list:
        """
        Generate realistic OHLCV data series
        
        Args:
            start_price: Starting price
            num_candles: Number of candles to generate
            timeframe_minutes: Minutes between candles
            volatility: Price volatility (0.02 = 2%)
        """
        data = []
        current_price = start_price
        base_time = int(datetime(2022, 1, 1).timestamp() * 1000)
        
        for i in range(num_candles):
            timestamp = base_time + (i * timeframe_minutes * 60 * 1000)
            
            # Generate price movement
            change = np.random.normal(0, volatility)
            new_price = current_price * (1 + change)
            
            # Generate OHLC from price movement
            price_range = abs(change) * current_price
            high = max(current_price, new_price) + np.random.uniform(0, price_range * 0.5)
            low = min(current_price, new_price) - np.random.uniform(0, price_range * 0.5)
            
            # Ensure valid OHLC relationships
            high = max(high, current_price, new_price)
            low = min(low, current_price, new_price)
            
            volume = np.random.uniform(50, 200)
            
            data.append([
                timestamp,
                round(current_price, 2),
                round(high, 2),
                round(low, 2), 
                round(new_price, 2),
                round(volume, 4)
            ])
            
            current_price = new_price
        
        return data
    
    @staticmethod
    def create_test_parquet_file(file_path: Path, data: list):
        """Create a test parquet file with OHLCV data"""
        df_data = []
        for candle in data:
            df_data.append({
                'timestamp': candle[0],
                'open': candle[1], 
                'high': candle[2],
                'low': candle[3],
                'close': candle[4],
                'volume': candle[5],
                'datetime': pd.to_datetime(candle[0], unit='ms')
            })
        
        df = pd.DataFrame(df_data)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_path, compression='snappy', index=False)


# Custom assertions for financial data
class FinancialAssertions:
    """Custom assertions for financial data testing"""
    
    @staticmethod
    def assert_valid_ohlcv(ohlcv_data):
        """Assert that OHLCV data is valid"""
        assert ohlcv_data.low <= ohlcv_data.open <= ohlcv_data.high
        assert ohlcv_data.low <= ohlcv_data.close <= ohlcv_data.high  
        assert ohlcv_data.low <= ohlcv_data.high
        assert ohlcv_data.volume >= 0
        assert all(v >= 0 for v in [ohlcv_data.open, ohlcv_data.high, 
                                   ohlcv_data.low, ohlcv_data.close])
    
    @staticmethod
    def assert_increasing_timestamps(data_list):
        """Assert that timestamps are in increasing order"""
        timestamps = [d.timestamp for d in data_list]
        assert timestamps == sorted(timestamps), "Timestamps should be in increasing order"
    
    @staticmethod
    def assert_no_duplicate_timestamps(data_list):
        """Assert no duplicate timestamps"""
        timestamps = [d.timestamp for d in data_list]
        assert len(timestamps) == len(set(timestamps)), "No duplicate timestamps allowed"
    
    @staticmethod
    def assert_reasonable_price_movements(data_list, max_change_pct: float = 0.5):
        """Assert that price movements between candles are reasonable"""
        for i in range(1, len(data_list)):
            prev_close = data_list[i-1].close
            curr_open = data_list[i].open
            
            change_pct = abs(curr_open - prev_close) / prev_close
            assert change_pct <= max_change_pct, f"Price movement too large: {change_pct:.2%}"


# Mock helpers
class MockExchangeBuilder:
    """Builder pattern for creating mock exchanges"""
    
    def __init__(self):
        self.responses = {}
        self.side_effects = {}
        
    def with_ohlcv_response(self, symbol: str, timeframe: str, data: list):
        """Add OHLCV response for symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        self.responses[key] = data
        return self
        
    def with_error(self, symbol: str, timeframe: str, exception):
        """Add error response for symbol/timeframe"""
        key = f"{symbol}_{timeframe}"
        self.side_effects[key] = exception
        return self
    
    def build(self):
        """Build the mock exchange"""
        from unittest.mock import AsyncMock
        
        mock_exchange = AsyncMock()
        
        async def fetch_ohlcv_side_effect(symbol, timeframe, **kwargs):
            key = f"{symbol}_{timeframe}"
            
            if key in self.side_effects:
                raise self.side_effects[key]
                
            if key in self.responses:
                return self.responses[key]
                
            return []  # Default empty response
        
        mock_exchange.fetch_ohlcv.side_effect = fetch_ohlcv_side_effect
        return mock_exchange


PYTEST_INI_CONTENT = """
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    -q
    --strict-markers
    --strict-config
    --disable-warnings
testpaths = tests
markers =
    slow: marks tests as slow (deselect with -m "not slow")
    integration: marks tests as integration tests  
    live_api: marks tests that require live API access
    unit: marks tests as unit tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
asyncio_mode = auto
"""

# Save pytest.ini content to a file
def create_pytest_ini():
    """Create pytest.ini file with the configuration"""
    with open('pytest.ini', 'w') as f:
        f.write(PYTEST_INI_CONTENT)


if __name__ == "__main__":
    create_pytest_ini()
    print("Created pytest.ini configuration file")