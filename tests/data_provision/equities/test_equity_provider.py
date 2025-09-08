"""
Unit tests for the EquityProvider class.
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime, timezone
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_provision.equities.equity_provider import EquityProvider

# Mock the databento DbnStore object
class MockDbnStore:
    def __init__(self, data):
        self._data = data
    
    def to_df(self):
        return pd.DataFrame(self._data)

@pytest.fixture
def mock_databento_client():
    """Fixture to mock the databento historical client."""
    with patch('databento.Historical') as mock_historical:
        mock_client = MagicMock()
        mock_historical.return_value = mock_client
        yield mock_client

@pytest.fixture
def sample_1m_data():
    """Generates a sample 1-minute pandas DataFrame."""
    base_time = int(datetime(2023, 1, 1, 14, 30, tzinfo=timezone.utc).timestamp() * 1e9) # nanoseconds
    data = []
    for i in range(10): # 10 minutes of data
        ts = base_time + i * 60 * 1_000_000_000
        data.append({
            'ts_event': ts,
            'open': 150.0 + i * 0.1,
            'high': 150.5 + i * 0.1,
            'low': 149.5 + i * 0.1,
            'close': 150.2 + i * 0.1,
            'volume': 1000 + i * 50
        })
    return data

@pytest.fixture
def provider(tmp_path):
    """Fixture for an EquityProvider instance with a temporary data directory."""
    with patch.dict(os.environ, {"DATABENTO_API_KEY": "test-key"}):
        return EquityProvider(data_dir=str(tmp_path))

class TestEquityProvider:
    
    @pytest.mark.asyncio
    async def test_fetch_1m_data_from_api(self, provider, mock_databento_client, sample_1m_data):
        """Test fetching 1-minute data directly from the API."""
        symbol = 'AAPL'
        start_date = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 1, 14, 40, tzinfo=timezone.utc)

        # Mock the async API call
        mock_databento_client.timeseries.get_range_async = AsyncMock(return_value=MockDbnStore(sample_1m_data))
        
        results = await provider.fetch_ohlcv(symbol, start_date, end_date, '1m', force_reload=True)
        
        mock_databento_client.timeseries.get_range_async.assert_called_once()
        assert symbol in results
        assert not results[symbol].empty
        assert len(results[symbol]) == 10
        assert results[symbol].index.name == 'datetime'

    @pytest.mark.asyncio
    async def test_aggregation_from_1m_to_5m(self, provider, mock_databento_client, sample_1m_data):
        """Test aggregation of 1-minute data to 5-minute intervals."""
        symbol = 'AAPL'
        start_date = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 1, 14, 40, tzinfo=timezone.utc)
        
        mock_databento_client.timeseries.get_range_async = AsyncMock(return_value=MockDbnStore(sample_1m_data))

        results = await provider.fetch_ohlcv(symbol, start_date, end_date, '5min', force_reload=True)
        
        assert symbol in results
        df_5m = results[symbol]
        
        assert len(df_5m) == 2
        # Check first candle (14:30 to 14:34)
        assert df_5m.iloc[0]['open'] == 150.0
        assert df_5m.iloc[0]['high'] == 150.5 + 4 * 0.1
        assert df_5m.iloc[0]['low'] == 149.5
        assert df_5m.iloc[0]['close'] == 150.2 + 4 * 0.1
        assert df_5m.iloc[0]['volume'] == (1000 + 1025 + 1050 + 1075 + 1100)

    @pytest.mark.asyncio
    async def test_save_and_load_from_parquet(self, provider, mock_databento_client, sample_1m_data):
        """Test that data is saved to and loaded from parquet correctly."""
        symbol = 'MSFT'
        interval = '1m'
        start_date = datetime(2025, 1, 1, 14, 30, tzinfo=timezone.utc)
        end_date = datetime(2025, 1, 1, 14, 40, tzinfo=timezone.utc)
        
        mock_databento_client.timeseries.get_range_async = AsyncMock(return_value=MockDbnStore(sample_1m_data))

        # 1. Fetch and save data
        await provider.fetch_ohlcv(symbol, start_date, end_date, interval, force_reload=True)
        
        # Give some time for the async save task to complete
        await asyncio.sleep(0.1)

        # Check that a file was created
        expected_path = provider._get_file_path(symbol, start_date, interval)
        assert expected_path.exists()
        
        # 2. Now, fetch again, which should load from the file
        # To prove it loads from file, mock the client to return nothing
        mock_databento_client.timeseries.get_range_async.return_value = MockDbnStore([])
        
        results_load = await provider.fetch_ohlcv(symbol, start_date, end_date, interval, force_reload=False)
        
        assert symbol in results_load
        assert not results_load[symbol].empty
        assert len(results_load[symbol]) == 10
        assert results_load[symbol].iloc[0]['open'] == 150.0
