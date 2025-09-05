from dataclasses import dataclass, asdict
from typing import Dict
from datetime import datetime, timezone


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
    interval: str
    
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
            now = int(datetime.now(timezone.utc).timestamp() * 1000)
            # Allow data up to 10 years old instead of just 1 year
            if self.timestamp > now or self.timestamp < (now - 10 * 365 * 24 * 60 * 60 * 1000):
                return False
                
            return True
        except Exception:
            return False