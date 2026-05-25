from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Dict, List
import pandas as pd


@dataclass
class Signal:
    direction: str          # "BUY" | "SELL" | "NONE"
    confidence: float       # 0.0–1.0 weighted confluence score
    entry: float
    sl: float
    tp: float
    strategy_name: str
    timeframe_votes: Dict[str, str] = field(default_factory=dict)
    atr_5m: float = 0.0
    regime: str = "TRANSITIONING"
    session: str = "Unknown"


class BaseStrategy(ABC):
    name: str = "Base"
    weight: float = 1.0
    suspended_until: Optional[date] = None

    # Which regimes this strategy is allowed to trade in.
    regime_preferences: List[str] = ["TRENDING", "RANGING", "TRANSITIONING"]

    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # Injected by main.py after construction so strategies can call get_multiplier()
    session_tracker = None   # SessionTracker | None

    def is_active(self, today: date) -> bool:
        if self.suspended_until and today <= self.suspended_until:
            return False
        return True

    def regime_ok(self, regime: str) -> bool:
        return regime in self.regime_preferences

    def session_multiplier(self, session: str) -> float:
        if self.session_tracker is None:
            return 1.0
        return self.session_tracker.get_multiplier(self.name, session)

    def _min_bars(self) -> int:
        return 50

    def _safe_last(self, series: pd.Series, default=float("nan")):
        s = series.dropna()
        return float(s.iloc[-1]) if len(s) > 0 else default

    @abstractmethod
    def generate_signal(self,
                        tf1m:  pd.DataFrame,
                        tf5m:  pd.DataFrame,
                        tf15m: pd.DataFrame,
                        timestamp: pd.Timestamp) -> Signal:
        ...

    def _no_signal(self, entry: float = 0.0) -> Signal:
        return Signal("NONE", 0.0, entry, 0.0, 0.0, self.name)
