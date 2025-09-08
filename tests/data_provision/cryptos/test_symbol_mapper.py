"""
Unit tests for the symbol mapper module.
Tests symbol conversions, mappers for different exchanges, and the symbol manager.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_provision.cryptos.symbol_mapper import (
    UnifiedSymbol, SymbolType, SymbolManager,
    BinanceSymbolMapper, CoinbaseSymbolMapper, KrakenSymbolMapper,
    OKXSymbolMapper, BybitSymbolMapper, ExchangeSymbolMapper
)


class TestUnifiedSymbol:
    """Test the UnifiedSymbol dataclass"""
    
    def test_spot_symbol_creation(self):
        """Test creating a spot trading symbol"""
        symbol = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        
        assert symbol.base == 'BTC'
        assert symbol.quote == 'USDT'
        assert symbol.symbol_type == SymbolType.SPOT
        assert str(symbol) == 'BTC/USDT'
    
    def test_perpetual_symbol_creation(self):
        """Test creating a perpetual futures symbol"""
        symbol = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.PERPETUAL)
        
        assert str(symbol) == 'BTC/USDT:PERP'
    
    def test_futures_symbol_with_expiry(self):
        """Test creating a futures symbol with expiry date"""
        symbol = UnifiedSymbol(
            base='BTC', 
            quote='USDT', 
            symbol_type=SymbolType.FUTURES,
            expiry='20240329'
        )
        
        assert str(symbol) == 'BTC/USDT:20240329'
    
    def test_option_symbol(self):
        """Test creating an option symbol"""
        symbol = UnifiedSymbol(
            base='BTC',
            quote='USDT',
            symbol_type=SymbolType.OPTION,
            expiry='20240329',
            strike=50000.0,
            option_type='C'
        )
        
        assert str(symbol) == 'BTC/USDT:20240329:C50000.0'
    
    def test_from_string_spot(self):
        """Test parsing a spot symbol from string"""
        symbol = UnifiedSymbol.from_string('BTC/USDT')
        
        assert symbol.base == 'BTC'
        assert symbol.quote == 'USDT'
        assert symbol.symbol_type == SymbolType.SPOT
    
    def test_from_string_perpetual(self):
        """Test parsing a perpetual symbol from string"""
        symbol = UnifiedSymbol.from_string('BTC/USDT:PERP')
        
        assert symbol.base == 'BTC'
        assert symbol.quote == 'USDT'
        assert symbol.symbol_type == SymbolType.PERPETUAL
    
    def test_from_string_futures(self):
        """Test parsing a futures symbol from string"""
        symbol = UnifiedSymbol.from_string('BTC/USDT:20240329')
        
        assert symbol.base == 'BTC'
        assert symbol.quote == 'USDT'
        assert symbol.symbol_type == SymbolType.FUTURES
        assert symbol.expiry == '20240329'
    
    def test_from_string_invalid_format(self):
        """Test parsing invalid symbol format raises error"""
        with pytest.raises(ValueError):
            UnifiedSymbol.from_string('BTCUSDT')  # Missing separator
        
        with pytest.raises(ValueError):
            UnifiedSymbol.from_string('BTC')  # Missing quote
    
    def test_to_dict(self):
        """Test converting symbol to dictionary"""
        symbol = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        result = symbol.to_dict()
        
        assert result['base'] == 'BTC'
        assert result['quote'] == 'USDT'
        assert result['type'] == 'spot'
        assert result['unified'] == 'BTC/USDT'


class TestBinanceSymbolMapper:
    """Test Binance-specific symbol mapping"""
    
    def setup_method(self):
        self.mapper = BinanceSymbolMapper()
    
    def test_to_exchange_spot(self):
        """Test converting unified spot symbol to Binance format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTCUSDT'
    
    def test_to_exchange_perpetual(self):
        """Test converting unified perpetual to Binance format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.PERPETUAL)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTCUSDTPERP'
    
    def test_from_exchange_spot(self):
        """Test converting Binance spot symbol to unified format"""
        result = self.mapper.from_exchange('BTCUSDT', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'
        assert result.symbol_type == SymbolType.SPOT
    
    def test_from_exchange_perpetual(self):
        """Test converting Binance perpetual to unified format"""
        result = self.mapper.from_exchange('BTCUSDTPERP', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'
        assert result.symbol_type == SymbolType.PERPETUAL
    
    def test_from_exchange_various_quotes(self):
        """Test parsing symbols with different quote currencies"""
        # Test BUSD quote
        result = self.mapper.from_exchange('ETHBUSD')
        assert result.base == 'ETH'
        assert result.quote == 'BUSD'
        
        # Test BTC quote
        result = self.mapper.from_exchange('ETHBTC')
        assert result.base == 'ETH'
        assert result.quote == 'BTC'


class TestCoinbaseSymbolMapper:
    """Test Coinbase-specific symbol mapping"""
    
    def setup_method(self):
        self.mapper = CoinbaseSymbolMapper()
    
    def test_to_exchange_spot(self):
        """Test converting unified spot symbol to Coinbase format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTC-USD'  # Coinbase uses USD instead of USDT
    
    def test_to_exchange_usd_quote(self):
        """Test that USD quote is preserved"""
        unified = UnifiedSymbol(base='BTC', quote='USD', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTC-USD'
    
    def test_from_exchange_spot(self):
        """Test converting Coinbase symbol to unified format"""
        result = self.mapper.from_exchange('BTC-USD', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'  # Converted to USDT for consistency
        assert result.symbol_type == SymbolType.SPOT
    
    def test_from_exchange_invalid_format(self):
        """Test that invalid format raises error"""
        with pytest.raises(ValueError):
            self.mapper.from_exchange('BTCUSD')  # Missing dash
    
    def test_unsupported_symbol_type(self):
        """Test that non-spot symbols raise error"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.PERPETUAL)
        with pytest.raises(NotImplementedError):
            self.mapper.to_exchange(unified)


class TestKrakenSymbolMapper:
    """Test Kraken-specific symbol mapping"""
    
    def setup_method(self):
        self.mapper = KrakenSymbolMapper()
    
    def test_to_exchange_with_btc_mapping(self):
        """Test that BTC is converted to XBT for Kraken"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'XBTUSDT'
    
    def test_to_exchange_regular_symbol(self):
        """Test regular symbols without special mapping"""
        unified = UnifiedSymbol(base='ETH', quote='USDT', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'ETHUSDT'
    
    def test_to_exchange_perpetual(self):
        """Test perpetual symbol conversion"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.PERPETUAL)
        result = self.mapper.to_exchange(unified)
        assert result == 'PI_XBTUSDT'
    
    def test_from_exchange_with_xbt(self):
        """Test converting XBT back to BTC"""
        result = self.mapper.from_exchange('XBTUSDT', SymbolType.SPOT)
        
        assert result.base == 'BTC'  # XBT converted back to BTC
        assert result.quote == 'USDT'
    
    def test_from_exchange_perpetual(self):
        """Test parsing Kraken perpetual"""
        result = self.mapper.from_exchange('PI_XBTUSDT', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'
        assert result.symbol_type == SymbolType.PERPETUAL


class TestOKXSymbolMapper:
    """Test OKX-specific symbol mapping"""
    
    def setup_method(self):
        self.mapper = OKXSymbolMapper()
    
    def test_to_exchange_spot(self):
        """Test converting to OKX spot format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTC-USDT'
    
    def test_to_exchange_perpetual(self):
        """Test converting to OKX perpetual format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.PERPETUAL)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTC-USDT-SWAP'
    
    def test_from_exchange_spot(self):
        """Test parsing OKX spot symbol"""
        result = self.mapper.from_exchange('BTC-USDT', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'
        assert result.symbol_type == SymbolType.SPOT
    
    def test_from_exchange_swap(self):
        """Test parsing OKX swap/perpetual symbol"""
        result = self.mapper.from_exchange('BTC-USDT-SWAP', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'
        assert result.symbol_type == SymbolType.PERPETUAL


class TestBybitSymbolMapper:
    """Test Bybit-specific symbol mapping"""
    
    def setup_method(self):
        self.mapper = BybitSymbolMapper()
    
    def test_to_exchange_spot(self):
        """Test converting to Bybit spot format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTCUSDT'
    
    def test_to_exchange_perpetual(self):
        """Test converting to Bybit perpetual format"""
        unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.PERPETUAL)
        result = self.mapper.to_exchange(unified)
        assert result == 'BTCUSDT'  # Same as spot for linear perpetuals
    
    def test_from_exchange_spot(self):
        """Test parsing Bybit symbol"""
        result = self.mapper.from_exchange('BTCUSDT', SymbolType.SPOT)
        
        assert result.base == 'BTC'
        assert result.quote == 'USDT'
        assert result.symbol_type == SymbolType.SPOT


class TestSymbolManager:
    """Test the central symbol management system"""
    
    def setup_method(self):
        self.manager = SymbolManager()
    
    def test_normalize_base_quote_with_slash(self):
        """Test parsing symbol with slash separator"""
        base, quote = self.manager.normalize_base_quote('BTC/USDT')
        assert base == 'BTC'
        assert quote == 'USDT'
    
    def test_normalize_base_quote_with_dash(self):
        """Test parsing symbol with dash separator"""
        base, quote = self.manager.normalize_base_quote('BTC-USDT')
        assert base == 'BTC'
        assert quote == 'USDT'
    
    def test_normalize_base_quote_no_separator(self):
        """Test parsing symbol without separator"""
        base, quote = self.manager.normalize_base_quote('BTCUSDT')
        assert base == 'BTC'
        assert quote == 'USDT'
    
    def test_normalize_with_aliases(self):
        """Test that aliases are applied"""
        # WBTC should be converted to BTC
        base, quote = self.manager.normalize_base_quote('WBTC/USDT')
        assert base == 'BTC'
        assert quote == 'USDT'
    
    def test_to_unified_binance(self):
        """Test converting Binance symbol to unified"""
        unified = self.manager.to_unified('BTCUSDT', 'binance', SymbolType.SPOT)
        
        assert unified.base == 'BTC'
        assert unified.quote == 'USDT'
        assert unified.symbol_type == SymbolType.SPOT
    
    def test_to_unified_coinbase(self):
        """Test converting Coinbase symbol to unified"""
        unified = self.manager.to_unified('BTC-USD', 'coinbase', SymbolType.SPOT)
        
        assert unified.base == 'BTC'
        assert unified.quote == 'USDT'  # USD converted to USDT
    
    def test_to_exchange_binance(self):
        """Test converting unified symbol to Binance format"""
        result = self.manager.to_exchange('BTC/USDT', 'binance')
        assert result == 'BTCUSDT'
    
    def test_to_exchange_coinbase(self):
        """Test converting unified symbol to Coinbase format"""
        result = self.manager.to_exchange('BTC/USDT', 'coinbase')
        assert result == 'BTC-USD'
    
    def test_register_symbol(self):
        """Test registering a symbol mapping"""
        unified = UnifiedSymbol(base='TEST', quote='USDT', symbol_type=SymbolType.SPOT)
        self.manager.register_symbol(unified, 'binance', 'TESTUSDT')
        
        # Check it's in the registry
        assert 'TEST/USDT' in self.manager.registry
        assert self.manager.registry['TEST/USDT']['binance'] == 'TESTUSDT'
        
        # Check reverse registry
        assert ('binance', 'TESTUSDT') in self.manager.reverse_registry
        assert self.manager.reverse_registry[('binance', 'TESTUSDT')] == 'TEST/USDT'
    
    def test_get_all_exchange_symbols(self):
        """Test getting all exchange formats for a unified symbol"""
        result = self.manager.get_all_exchange_symbols('BTC/USDT')
        
        assert 'binance' in result
        assert 'coinbase' in result
        assert 'kraken' in result
        
        assert result['binance'] == 'BTCUSDT'
        assert result['coinbase'] == 'BTC-USD'
        assert result['kraken'] == 'XBTUSDT'
    
    def test_find_common_symbols(self):
        """Test finding symbols common to multiple exchanges"""
        # Register some test symbols
        btc_unified = UnifiedSymbol(base='BTC', quote='USDT', symbol_type=SymbolType.SPOT)
        self.manager.register_symbol(btc_unified, 'binance', 'BTCUSDT')
        self.manager.register_symbol(btc_unified, 'coinbase', 'BTC-USD')
        
        eth_unified = UnifiedSymbol(base='ETH', quote='USDT', symbol_type=SymbolType.SPOT)
        self.manager.register_symbol(eth_unified, 'binance', 'ETHUSDT')
        # ETH not registered for Coinbase
        
        # Find common symbols
        common = self.manager.find_common_symbols(['binance', 'coinbase'])
        
        assert 'BTC/USDT' in common  # Available on both
        assert 'ETH/USDT' not in common  # Only on Binance
    
    def test_validate_symbol_consistency(self):
        """Test symbol consistency validation (round-trip conversion)"""
        result = self.manager.validate_symbol_consistency('BTC/USDT')
        
        assert result['symbol'] == 'BTC/USDT'
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.json'
            
            # Register a custom symbol
            unified = UnifiedSymbol(base='TEST', quote='USDT', symbol_type=SymbolType.SPOT)
            self.manager.register_symbol(unified, 'binance', 'TESTUSDT')
            
            # Save config
            self.manager.save_config(config_path)
            assert config_path.exists()
            
            # Create new manager and load config
            new_manager = SymbolManager()
            new_manager.load_config(config_path)
            
            # Check that the custom symbol was loaded
            assert 'TEST/USDT' in new_manager.registry
            assert new_manager.registry['TEST/USDT']['binance'] == 'TESTUSDT'
    
    def test_unknown_exchange_handling(self):
        """Test handling of unknown exchanges"""
        with pytest.raises(ValueError):
            self.manager.to_exchange('BTC/USDT', 'unknown_exchange')
    
    def test_add_custom_mapper(self):
        """Test adding a custom exchange mapper"""
        # Create a mock mapper
        mock_mapper = Mock(spec=ExchangeSymbolMapper)
        mock_mapper.to_exchange.return_value = 'CUSTOM_FORMAT'
        
        # Add the mapper
        self.manager.add_custom_mapper('custom_exchange', mock_mapper)
        
        # Test it works
        result = self.manager.to_exchange('BTC/USDT', 'custom_exchange')
        assert result == 'CUSTOM_FORMAT'
        mock_mapper.to_exchange.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_symbol_formats(self):
        """Test handling of various invalid symbol formats"""
        manager = SymbolManager()
        
        # Test empty string
        with pytest.raises(ValueError):
            manager.normalize_base_quote('')
        
        # Test single character
        with pytest.raises(ValueError):
            manager.normalize_base_quote('B')
        
        # Test numbers only
        with pytest.raises(ValueError):
            manager.normalize_base_quote('123456')
    
    def test_case_insensitivity(self):
        """Test that symbol parsing is case-insensitive"""
        manager = SymbolManager()
        
        # Lowercase input
        base, quote = manager.normalize_base_quote('btc/usdt')
        assert base == 'BTC'
        assert quote == 'USDT'
        
        # Mixed case
        base, quote = manager.normalize_base_quote('Btc/uSdT')
        assert base == 'BTC'
        assert quote == 'USDT'
    
    def test_mapper_error_handling(self):
        """Test that mappers handle errors gracefully"""
        mapper = BinanceSymbolMapper()
        
        # Test with invalid symbol format
        with pytest.raises(ValueError):
            mapper.from_exchange('INVALID!!!SYMBOL', SymbolType.SPOT)
    
    def test_symbol_with_special_characters(self):
        """Test handling of symbols with special characters"""
        manager = SymbolManager()
        
        # Test with spaces (should be stripped)
        base, quote = manager.normalize_base_quote(' BTC / USDT ')
        assert base == 'BTC'
        assert quote == 'USDT'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])