import pandas as pd
from .base import BaseStrategy, Signal
from indicators import ema, atr
import config


class EMACrossoverStrategy(BaseStrategy):
    """EMA 8/21 crossover with EMA 50 trend filter."""
    name = "EMA_Crossover"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < 60 or len(tf15m) < 60:
            return self._no_signal()
        entry = float(tf1m["Close"].iloc[-1])
        votes = {}
        for label, df in [("15M", tf15m), ("5M", tf5m), ("1M", tf1m)]:
            c = df["Close"]
            fast  = ema(c, config.EMA_FAST)
            slow  = ema(c, config.EMA_SLOW)
            trend = ema(c, config.EMA_TREND)
            f_now, f_prev = float(fast.iloc[-1]), float(fast.iloc[-2])
            s_now, s_prev = float(slow.iloc[-1]), float(slow.iloc[-2])
            t_now         = float(trend.iloc[-1])
            cross_up   = f_prev <= s_prev and f_now > s_now and entry > t_now
            cross_down = f_prev >= s_prev and f_now < s_now and entry < t_now
            votes[label] = "BUY" if cross_up else "SELL" if cross_down else "NONE"
        direction = _dom(votes)
        if direction == "NONE": return self._no_signal(entry)
        atr5 = float(atr(tf5m["High"], tf5m["Low"], tf5m["Close"], config.ATR_PERIOD).iloc[-1])
        sl, tp = _sltp(entry, direction, atr5)
        return Signal(direction, _conf(votes, direction, self.weight), entry, sl, tp, self.name, votes, atr5)


def _dom(v):
    b = sum(1 for x in v.values() if x=="BUY")
    s = sum(1 for x in v.values() if x=="SELL")
    return "BUY" if b>s and b>=2 else "SELL" if s>b and s>=2 else "NONE"

def _sltp(e, d, a):
    if d=="BUY": return e-config.SL_ATR_MULT*a, e+config.TP_ATR_MULT*a
    return e+config.SL_ATR_MULT*a, e-config.TP_ATR_MULT*a

def _conf(v, d, w): return min(0.5+0.25*sum(1 for x in v.values() if x==d), 1.0)*min(w,1.0)
