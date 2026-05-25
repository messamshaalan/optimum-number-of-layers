"""London Open Breakout: 06-08 UTC range, trade breakout after 08 UTC."""
import pandas as pd
from .base import BaseStrategy, Signal
from indicators import atr
import config

_RANGE_START, _RANGE_END, _BRK = 6, 8, 0.30


class LondonBreakoutStrategy(BaseStrategy):
    name = "London_Breakout"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < 20: return self._no_signal()
        ts = pd.Timestamp(timestamp); hour = ts.hour
        if not (8 <= hour < 14): return self._no_signal(float(tf1m["Close"].iloc[-1]))
        entry = float(tf1m["Close"].iloc[-1]); today = ts.normalize()
        mask = (tf5m.index >= today+pd.Timedelta(hours=_RANGE_START)) & (tf5m.index < today+pd.Timedelta(hours=_RANGE_END))
        rb = tf5m[mask]
        if len(rb) < 4: return self._no_signal(entry)
        rh,rl = float(rb["High"].max()), float(rb["Low"].min())
        if rh-rl < 0.50: return self._no_signal(entry)
        if entry > rh+_BRK: votes={"15M":"BUY","5M":"BUY","1M":"BUY"}; d="BUY"
        elif entry < rl-_BRK: votes={"15M":"SELL","5M":"SELL","1M":"SELL"}; d="SELL"
        else: return self._no_signal(entry)
        atr5=float(atr(tf5m["High"],tf5m["Low"],tf5m["Close"],config.ATR_PERIOD).iloc[-1])
        if d=="BUY": sl=rl-0.10; tp=entry+(entry-sl)*config.RISK_REWARD
        else: sl=rh+0.10; tp=entry-(sl-entry)*config.RISK_REWARD
        return Signal(d, 0.82*min(self.weight,1.0), entry, sl, tp, self.name, votes, atr5)
