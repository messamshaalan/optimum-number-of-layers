import uuid
from datetime import datetime
from .trade import Trade
from strategies.base import Signal
import config


def _session(hour):
    for name,(s,e) in config.SESSIONS.items():
        if s<=hour<e: return name
    return "Unknown"


class RiskManager:
    def __init__(self, risk_pct=config.RISK_PER_TRADE, spread=config.SPREAD_USD):
        self.risk_pct = risk_pct; self.spread = spread

    def build_trade(self, signal, equity, bar):
        entry = self._spread(signal.entry, signal.direction)
        sl, tp = self._adjust(signal, entry)
        size = self._size(config.INITIAL_CAPITAL, entry, sl)  # fixed dollar risk
        ts = bar.name if hasattr(bar, "name") else datetime.utcnow()
        return Trade(
            id=str(uuid.uuid4())[:8], strategy=signal.strategy_name,
            direction=signal.direction, entry_time=ts, entry_price=entry,
            sl=sl, tp=tp, size_oz=size,
            session=_session(ts.hour if hasattr(ts,"hour") else 0),
            confluence_score=signal.confidence, atr_at_entry=signal.atr_5m,
        )

    def _spread(self, p, d): h=self.spread/2; return p+h if d=="BUY" else p-h

    def _adjust(self, sig, entry):
        sd=abs(sig.entry-sig.sl); td=abs(sig.tp-sig.entry)
        return (entry-sd,entry+td) if sig.direction=="BUY" else (entry+sd,entry-td)

    def _size(self, equity, entry, sl):
        d=abs(entry-sl)
        return max(round(equity*self.risk_pct/d, 4), 0.01) if d>0.01 else 0.01
