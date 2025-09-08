from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd


class BaseDataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    async def fetch_ohlcv(self,
                         symbols: Union[str, List[str]],
                         start_date: Union[pd.Timestamp, datetime],
                         end_date: Union[pd.Timestamp, datetime],
                         interval: str,
                         force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        pass
    
    @abstractmethod
    async def get_markets(self, exchange_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def close(self):
        pass