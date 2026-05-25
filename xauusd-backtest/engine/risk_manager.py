import uuid
from datetime import datetime
from .trade import Trade
from strategies.base import Signal
import config


def _session_label(hour: int) -> str:
    for name, (start, end) in config.SESSIONS.items():
        if start <= hour < end:
            return name
    return "Unknown"


class RiskManager:
    def __init__(self,
                 risk_pct: float = config.RISK_PER_TRADE,
                 spread:   float = config.SPREAD_USD):
        self.risk_pct = risk_pct
        self.spread   = spread

    def build_trade(self, signal: Signal, equity: float, bar: object) -> Trade:
        entry   = self._apply_spread(signal.entry, signal.direction)
        sl, tp  = self._adjust_sl_tp(signal, entry)
        sl_dist = abs(entry - sl)
        size    = self._calc_size(config.INITIAL_CAPITAL, sl_dist)

        ts   = bar.name if hasattr(bar, "name") else datetime.utcnow()
        hour = ts.hour if hasattr(ts, "hour") else 0

        return Trade(
            id               = str(uuid.uuid4())[:8],
            strategy         = signal.strategy_name,
            direction        = signal.direction,
            entry_time       = ts,
            entry_price      = entry,
            sl               = sl,
            tp               = tp,
            size_oz          = size,
            session          = _session_label(hour),
            confluence_score = signal.confidence,
            atr_at_entry     = signal.atr_5m,
            pattern_type     = getattr(signal, "pattern_type", ""),
            regime           = getattr(signal, "regime", "TRANSITIONING"),
            sl_distance      = sl_dist,   # stored for breakeven trigger
        )

    def _apply_spread(self, price: float, direction: str) -> float:
        half = self.spread / 2
        return price + half if direction == "BUY" else price - half

    def _adjust_sl_tp(self, signal: Signal, entry: float):
        sl_dist = abs(signal.entry - signal.sl)
        tp_dist = abs(signal.tp    - signal.entry)
        if signal.direction == "BUY":
            return entry - sl_dist, entry + tp_dist
        return entry + sl_dist, entry - tp_dist

    def _calc_size(self, capital: float, sl_dist: float) -> float:
        if sl_dist < 0.01:
            return 0.01
        risk_usd = capital * self.risk_pct
        return max(round(risk_usd / sl_dist, 4), 0.01)
