from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict
import pandas as pd


@dataclass
class Signal:
    direction: str          # "BUY" | "SELL" | "NONE"
    confidence: float       # 0.0-1.0 weighted confluence score
    entry: float
    sl: float
    tp: float
    strategy_name: str
    timeframe_votes: Dict[str, str] = field(default_factory=dict)
    atr_5m: float = 0.0


class BaseStrategy(ABC):
    name: str = "Base"
    weight: float = 1.0
    suspended_until: Optional[date] = None
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    def is_active(self, today: date) -> bool:
        if self.suspended_until and today <= self.suspended_until:
            return False
        return True

    def _safe_last(self, series: pd.Series, default=float("nan")):
        s = series.dropna()
        return float(s.iloc[-1]) if len(s) > 0 else default

    @abstractmethod
    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal: ...

    def _no_signal(self, entry: float = 0.0) -> Signal:
        return Signal("NONE", 0.0, entry, 0.0, 0.0, self.name)
