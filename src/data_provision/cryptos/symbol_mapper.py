"""
Symbol mapping and standardization system for multi-exchange cryptocurrency data.
Handles the conversion between standardized symbols and exchange-specific formats.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from enum import Enum
import logging
from abc import ABC, abstractmethod


class SymbolType(Enum):
    """Types of trading symbols"""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"
    INDEX = "index"


@dataclass
class UnifiedSymbol:
    """
    Standardized symbol representation
    
    Examples:
        BTC/USDT (spot)
        BTC/USDT:PERP (perpetual)
        BTC/USDT:20240329 (futures with expiry)
        BTC/USDT:20240329:C50000 (call option)
    """
    base: str
    quote: str
    symbol_type: SymbolType = SymbolType.SPOT
    expiry: Optional[str] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # 'C' for call, 'P' for put
    
    def __str__(self) -> str:
        """Convert to standard string format"""
        symbol = f"{self.base}/{self.quote}"
        
        if self.symbol_type == SymbolType.PERPETUAL:
            symbol += ":PERP"
        elif self.symbol_type == SymbolType.FUTURES and self.expiry:
            symbol += f":{self.expiry}"
        elif self.symbol_type == SymbolType.OPTION:
            symbol += f":{self.expiry}:{self.option_type}{self.strike}"
            
        return symbol
    
    @classmethod
    def from_string(cls, symbol_str: str) -> 'UnifiedSymbol':
        """Parse a unified symbol string"""
        parts = symbol_str.split(':')
        base_quote = parts[0].split('/')
        
        if len(base_quote) != 2:
            raise ValueError(f"Invalid symbol format: {symbol_str}")
        
        base, quote = base_quote
        
        # Determine type from additional parts
        if len(parts) == 1:
            return cls(base=base, quote=quote, symbol_type=SymbolType.SPOT)
        elif len(parts) == 2:
            if parts[1] == 'PERP':
                return cls(base=base, quote=quote, symbol_type=SymbolType.PERPETUAL)
            else:
                return cls(base=base, quote=quote, symbol_type=SymbolType.FUTURES, expiry=parts[1])
        elif len(parts) == 3:
            # Option format
            option_match = re.match(r'([CP])(\d+(?:\.\d+)?)', parts[2])
            if option_match:
                option_type = option_match.group(1)
                strike = float(option_match.group(2))
                return cls(
                    base=base, quote=quote, 
                    symbol_type=SymbolType.OPTION,
                    expiry=parts[1], 
                    strike=strike, 
                    option_type=option_type
                )
        
        raise ValueError(f"Cannot parse symbol: {symbol_str}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'base': self.base,
            'quote': self.quote,
            'type': self.symbol_type.value,
            'expiry': self.expiry,
            'strike': self.strike,
            'option_type': self.option_type,
            'unified': str(self)
        }


class ExchangeSymbolMapper(ABC):
    """Abstract base class for exchange-specific symbol mappers"""
    
    @abstractmethod
    def to_exchange(self, unified: UnifiedSymbol) -> str:
        """Convert unified symbol to exchange format"""
        pass
    
    @abstractmethod
    def from_exchange(self, exchange_symbol: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert exchange symbol to unified format"""
        pass


class BinanceSymbolMapper(ExchangeSymbolMapper):
    """Binance-specific symbol mapping"""
    
    def to_exchange(self, unified: UnifiedSymbol) -> str:
        """
        Convert to Binance format
        Examples: BTC/USDT -> BTCUSDT, BTC/USDT:PERP -> BTCUSDTPERP
        """
        base = unified.base.upper()
        quote = unified.quote.upper()
        
        if unified.symbol_type == SymbolType.SPOT:
            return f"{base}{quote}"
        elif unified.symbol_type == SymbolType.PERPETUAL:
            return f"{base}{quote}PERP"
        elif unified.symbol_type == SymbolType.FUTURES:
            # Binance futures format: BTCUSDT_240329
            if unified.expiry:
                expiry = unified.expiry[2:]  # Remove '20' from year
                return f"{base}{quote}_{expiry}"
            return f"{base}{quote}"
        else:
            raise NotImplementedError(f"Binance doesn't support {unified.symbol_type}")
    
    def from_exchange(self, exchange_symbol: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert from Binance format to unified"""
        exchange_symbol = exchange_symbol.upper()
        
        # Handle perpetuals
        if exchange_symbol.endswith('PERP'):
            base_quote = exchange_symbol[:-4]
            # Common quote currencies
            for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB']:
                if base_quote.endswith(quote):
                    base = base_quote[:-len(quote)]
                    return UnifiedSymbol(base=base, quote=quote, symbol_type=SymbolType.PERPETUAL)
        
        # Handle futures with expiry
        if '_' in exchange_symbol:
            base_quote, expiry = exchange_symbol.split('_')
            expiry = '20' + expiry  # Add century
            for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']:
                if base_quote.endswith(quote):
                    base = base_quote[:-len(quote)]
                    return UnifiedSymbol(base=base, quote=quote, symbol_type=SymbolType.FUTURES, expiry=expiry)
        
        # Handle spot
        # Try common quote currencies
        for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB', 'USD', 'EUR', 'GBP']:
            if exchange_symbol.endswith(quote):
                base = exchange_symbol[:-len(quote)]
                return UnifiedSymbol(base=base, quote=quote, symbol_type=symbol_type)
        
        raise ValueError(f"Cannot parse Binance symbol: {exchange_symbol}")


class CoinbaseSymbolMapper(ExchangeSymbolMapper):
    """Coinbase-specific symbol mapping"""
    
    def to_exchange(self, unified: UnifiedSymbol) -> str:
        """
        Convert to Coinbase format
        Examples: BTC/USDT -> BTC-USDT, BTC/USD -> BTC-USD
        """
        base = unified.base.upper()
        
        # Coinbase uses USD instead of USDT
        quote = 'USD' if unified.quote.upper() == 'USDT' else unified.quote.upper()
        
        if unified.symbol_type == SymbolType.SPOT:
            return f"{base}-{quote}"
        else:
            raise NotImplementedError(f"Coinbase doesn't support {unified.symbol_type}")
    
    def from_exchange(self, exchange_symbol: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert from Coinbase format to unified"""
        if '-' not in exchange_symbol:
            raise ValueError(f"Invalid Coinbase symbol format: {exchange_symbol}")
        
        base, quote = exchange_symbol.upper().split('-')
        
        # Convert USD to USDT for consistency
        if quote == 'USD':
            quote = 'USDT'
        
        return UnifiedSymbol(base=base, quote=quote, symbol_type=SymbolType.SPOT)


class KrakenSymbolMapper(ExchangeSymbolMapper):
    """Kraken-specific symbol mapping"""
    
    # Kraken uses different ticker symbols
    KRAKEN_MAPPINGS = {
        'BTC': 'XBT',
        'DOGE': 'XDG',
    }
    
    REVERSE_MAPPINGS = {v: k for k, v in KRAKEN_MAPPINGS.items()}
    
    def to_exchange(self, unified: UnifiedSymbol) -> str:
        """Convert to Kraken format"""
        base = self.KRAKEN_MAPPINGS.get(unified.base.upper(), unified.base.upper())
        quote = self.KRAKEN_MAPPINGS.get(unified.quote.upper(), unified.quote.upper())
        
        if unified.symbol_type == SymbolType.SPOT:
            return f"{base}{quote}"
        elif unified.symbol_type == SymbolType.PERPETUAL:
            return f"PI_{base}{quote}"
        else:
            raise NotImplementedError(f"Kraken doesn't support {unified.symbol_type}")
    
    def from_exchange(self, exchange_symbol: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert from Kraken format to unified"""
        exchange_symbol = exchange_symbol.upper()
        
        # Handle perpetuals
        if exchange_symbol.startswith('PI_'):
            exchange_symbol = exchange_symbol[3:]
            symbol_type = SymbolType.PERPETUAL
        
        # Try to parse - Kraken uses various formats
        # Common quote currencies for Kraken
        for quote_kraken in ['USD', 'USDT', 'USDC', 'EUR', 'GBP', 'CAD', 'JPY', 'CHF', 'XBT']:
            if exchange_symbol.endswith(quote_kraken):
                base_kraken = exchange_symbol[:-len(quote_kraken)]
                
                # Convert Kraken symbols to standard
                base = self.REVERSE_MAPPINGS.get(base_kraken, base_kraken)
                quote = self.REVERSE_MAPPINGS.get(quote_kraken, quote_kraken)
                
                # Convert BTC to standard
                if quote == 'XBT':
                    quote = 'BTC'
                
                return UnifiedSymbol(base=base, quote=quote, symbol_type=symbol_type)
        
        raise ValueError(f"Cannot parse Kraken symbol: {exchange_symbol}")


class OKXSymbolMapper(ExchangeSymbolMapper):
    """OKX-specific symbol mapping"""
    
    def to_exchange(self, unified: UnifiedSymbol) -> str:
        """Convert to OKX format"""
        base = unified.base.upper()
        quote = unified.quote.upper()
        
        if unified.symbol_type == SymbolType.SPOT:
            return f"{base}-{quote}"
        elif unified.symbol_type == SymbolType.PERPETUAL:
            return f"{base}-{quote}-SWAP"
        else:
            raise NotImplementedError(f"OKX doesn't support {unified.symbol_type}")
    
    def from_exchange(self, exchange_symbol: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert from OKX format to unified"""
        parts = exchange_symbol.upper().split('-')
        
        if len(parts) == 2:
            return UnifiedSymbol(base=parts[0], quote=parts[1], symbol_type=SymbolType.SPOT)
        elif len(parts) == 3 and parts[2] == 'SWAP':
            return UnifiedSymbol(base=parts[0], quote=parts[1], symbol_type=SymbolType.PERPETUAL)
        
        raise ValueError(f"Cannot parse OKX symbol: {exchange_symbol}")


class BybitSymbolMapper(ExchangeSymbolMapper):
    """Bybit-specific symbol mapping"""
    
    def to_exchange(self, unified: UnifiedSymbol) -> str:
        """Convert to Bybit format"""
        base = unified.base.upper()
        quote = unified.quote.upper()
        
        if unified.symbol_type in [SymbolType.SPOT, SymbolType.PERPETUAL]:
            return f"{base}{quote}"
        else:
            raise NotImplementedError(f"Bybit doesn't support {unified.symbol_type}")
    
    def from_exchange(self, exchange_symbol: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert from Bybit format to unified"""
        exchange_symbol = exchange_symbol.upper()
        
        # Try spot/perpetual format
        for quote in ['USDT', 'USDC', 'USD', 'BTC', 'ETH']:
            if exchange_symbol.endswith(quote):
                base = exchange_symbol[:-len(quote)]
                return UnifiedSymbol(base=base, quote=quote, symbol_type=symbol_type)
        
        raise ValueError(f"Cannot parse Bybit symbol: {exchange_symbol}")


class SymbolManager:
    """
    Central symbol management system that handles all symbol conversions
    and maintains a symbol registry.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchange mappers
        self.mappers: Dict[str, ExchangeSymbolMapper] = {
            'binance': BinanceSymbolMapper(),
            'coinbase': CoinbaseSymbolMapper(),
            'kraken': KrakenSymbolMapper(),
            'okx': OKXSymbolMapper(),
            'bybit': BybitSymbolMapper(),
        }
        
        # Symbol registry: unified -> {exchange: exchange_symbol}
        self.registry: Dict[str, Dict[str, str]] = {}
        
        # Reverse registry: (exchange, exchange_symbol) -> unified
        self.reverse_registry: Dict[Tuple[str, str], str] = {}
        
        # Aliases for common variations
        self.aliases: Dict[str, str] = {
            'WBTC': 'BTC',  # Wrapped BTC
            'WETH': 'ETH',  # Wrapped ETH
            'UST': 'USDT',  # Common confusion
        }
        
        # Load custom mappings if provided
        if config_file and config_file.exists():
            self.load_config(config_file)
    
    def add_custom_mapper(self, exchange: str, mapper: ExchangeSymbolMapper):
        """Add a custom exchange mapper"""
        self.mappers[exchange.lower()] = mapper
        self.logger.info(f"Added custom mapper for {exchange}")
    
    def normalize_base_quote(self, symbol: str) -> Tuple[str, str]:
        """
        Normalize and split a symbol into base and quote currencies
        Handles various formats like BTC/USDT, BTC-USDT, BTCUSDT
        """
        # Remove spaces and convert to uppercase
        symbol = symbol.strip().upper().replace(' ', '')
        
        # Try different separators
        for separator in ['/', '-', '_', ':']:
            if separator in symbol:
                parts = symbol.split(separator)
                if len(parts) >= 2:
                    base = self.aliases.get(parts[0], parts[0])
                    quote = self.aliases.get(parts[1], parts[1])
                    return base, quote
        
        # Try to parse without separator (e.g., BTCUSDT)
        # Common quote currencies to try
        quote_currencies = ['USDT', 'USDC', 'BUSD', 'USD', 'BTC', 'ETH', 'BNB', 'EUR', 'GBP']
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                base = self.aliases.get(base, base)
                quote = self.aliases.get(quote, quote)
                return base, quote
        
        raise ValueError(f"Cannot parse symbol: {symbol}")
    
    def to_unified(self, symbol: str, exchange: str, symbol_type: SymbolType = SymbolType.SPOT) -> UnifiedSymbol:
        """Convert any exchange symbol to unified format"""
        exchange = exchange.lower()
        
        # Check reverse registry first
        registry_key = (exchange, symbol.upper())
        if registry_key in self.reverse_registry:
            return UnifiedSymbol.from_string(self.reverse_registry[registry_key])
        
        # Use exchange-specific mapper
        if exchange in self.mappers:
            try:
                unified = self.mappers[exchange].from_exchange(symbol, symbol_type)
                # Add to registry
                self.register_symbol(unified, exchange, symbol)
                return unified
            except Exception as e:
                self.logger.warning(f"Failed to parse {symbol} from {exchange}: {e}")
        
        # Fallback: try generic parsing
        try:
            base, quote = self.normalize_base_quote(symbol)
            unified = UnifiedSymbol(base=base, quote=quote, symbol_type=symbol_type)
            self.register_symbol(unified, exchange, symbol)
            return unified
        except Exception as e:
            raise ValueError(f"Cannot convert {symbol} from {exchange} to unified format: {e}")
    
    def to_exchange(self, unified_symbol: str, exchange: str) -> str:
        """Convert unified symbol to exchange-specific format"""
        exchange = exchange.lower()
        
        # Check registry first
        if unified_symbol in self.registry and exchange in self.registry[unified_symbol]:
            return self.registry[unified_symbol][exchange]
        
        # Parse unified symbol
        unified = UnifiedSymbol.from_string(unified_symbol)
        
        # Use exchange-specific mapper
        if exchange in self.mappers:
            try:
                exchange_symbol = self.mappers[exchange].to_exchange(unified)
                # Add to registry
                self.register_symbol(unified, exchange, exchange_symbol)
                return exchange_symbol
            except Exception as e:
                self.logger.warning(f"Failed to convert {unified_symbol} to {exchange} format: {e}")
                raise
        else:
            raise ValueError(f"No mapper available for exchange: {exchange}")
    
    def register_symbol(self, unified: UnifiedSymbol, exchange: str, exchange_symbol: str):
        """Register a symbol mapping"""
        unified_str = str(unified)
        
        if unified_str not in self.registry:
            self.registry[unified_str] = {}
        
        self.registry[unified_str][exchange.lower()] = exchange_symbol.upper()
        self.reverse_registry[(exchange.lower(), exchange_symbol.upper())] = unified_str
    
    def get_all_exchange_symbols(self, unified_symbol: str) -> Dict[str, str]:
        """Get all known exchange symbols for a unified symbol"""
        result = {}
        
        for exchange in self.mappers.keys():
            try:
                exchange_symbol = self.to_exchange(unified_symbol, exchange)
                result[exchange] = exchange_symbol
            except Exception as e:
                self.logger.debug(f"Cannot convert {unified_symbol} for {exchange}: {e}")
        
        return result
    
    def find_common_symbols(self, exchanges: List[str]) -> Set[str]:
        """Find symbols that are available on all specified exchanges"""
        if not exchanges:
            return set()
        
        exchanges = [e.lower() for e in exchanges]
        common_symbols = set()
        
        for unified_symbol in self.registry.keys():
            exchange_symbols = self.registry[unified_symbol]
            if all(exchange in exchange_symbols for exchange in exchanges):
                common_symbols.add(unified_symbol)
        
        return common_symbols
    
    def save_config(self, file_path: Path):
        """Save current registry to a JSON file"""
        config = {
            'registry': self.registry,
            'aliases': self.aliases
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        self.logger.info(f"Saved symbol configuration to {file_path}")
    
    def load_config(self, file_path: Path):
        """Load registry from a JSON file"""
        with open(file_path, 'r') as f:
            config = json.load(f)
        
        self.registry.update(config.get('registry', {}))
        self.aliases.update(config.get('aliases', {}))
        
        # Rebuild reverse registry
        for unified_symbol, exchange_map in self.registry.items():
            for exchange, exchange_symbol in exchange_map.items():
                self.reverse_registry[(exchange, exchange_symbol)] = unified_symbol
        
        self.logger.info(f"Loaded symbol configuration from {file_path}")
    
    def validate_symbol_consistency(self, unified_symbol: str) -> Dict[str, Any]:
        """
        Validate that symbol conversions are consistent (round-trip test)
        """
        results = {
            'symbol': unified_symbol,
            'valid': True,
            'errors': []
        }
        
        unified = UnifiedSymbol.from_string(unified_symbol)
        
        for exchange in self.mappers.keys():
            try:
                # Convert to exchange format
                exchange_symbol = self.to_exchange(unified_symbol, exchange)
                
                # Convert back to unified
                unified_back = self.to_unified(exchange_symbol, exchange, unified.symbol_type)
                
                # Check if they match
                if str(unified_back) != unified_symbol:
                    results['valid'] = False
                    results['errors'].append({
                        'exchange': exchange,
                        'original': unified_symbol,
                        'converted': exchange_symbol,
                        'back_converted': str(unified_back)
                    })
            except Exception as e:
                results['errors'].append({
                    'exchange': exchange,
                    'error': str(e)
                })
        
        return results


# Convenience functions
def create_symbol_manager() -> SymbolManager:
    """Create a pre-configured symbol manager"""
    manager = SymbolManager()
    
    # Add some common symbol mappings
    common_symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
        'DOGE/USDT', 'MATIC/USDT', 'SOL/USDT', 'DOT/USDT', 'AVAX/USDT',
        'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'ETC/USDT'
    ]
    
    # Pre-register common symbols
    for symbol_str in common_symbols:
        unified = UnifiedSymbol.from_string(symbol_str)
        for exchange in manager.mappers.keys():
            try:
                exchange_symbol = manager.mappers[exchange].to_exchange(unified)
                manager.register_symbol(unified, exchange, exchange_symbol)
            except:
                pass  # Some exchanges might not support all symbols
    
    return manager


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create symbol manager
    manager = create_symbol_manager()
    
    # Test conversions
    test_symbols = [
        ('BTC/USDT', 'binance'),
        ('ETH/USDT', 'coinbase'),
        ('BTC/USDT', 'kraken'),
    ]
    
    print("Symbol Conversion Examples:")
    print("-" * 50)
    
    for unified_symbol, exchange in test_symbols:
        try:
            exchange_symbol = manager.to_exchange(unified_symbol, exchange)
            back_to_unified = manager.to_unified(exchange_symbol, exchange)
            
            print(f"Unified: {unified_symbol}")
            print(f"  -> {exchange}: {exchange_symbol}")
            print(f"  -> Back to unified: {back_to_unified}")
            print()
        except Exception as e:
            print(f"Error converting {unified_symbol} for {exchange}: {e}")
            print()
    
    # Show all exchange formats for a symbol
    print("All exchange formats for BTC/USDT:")
    all_formats = manager.get_all_exchange_symbols('BTC/USDT')
    for exchange, symbol in all_formats.items():
        print(f"  {exchange}: {symbol}")
    
    # Validate consistency
    print("\nValidation Results:")
    validation = manager.validate_symbol_consistency('BTC/USDT')
    print(f"  Symbol: {validation['symbol']}")
    print(f"  Valid: {validation['valid']}")
    if validation['errors']:
        print("  Errors:", validation['errors'])