import pandas as pd
from .base import BaseStrategy, Signal
from indicators import rsi, ema, atr
import config


class RSIStrategy(BaseStrategy):
    """RSI <30 BUY, >70 SELL, confirmed by EMA200 trend."""
    name = "RSI"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < 30: return self._no_signal()
        entry = float(tf1m["Close"].iloc[-1])
        votes = {}
        if len(tf15m) >= 200:
            e200 = ema(tf15m["Close"], config.EMA_LONG)
            votes["15M"] = "BUY" if entry > float(e200.iloc[-1]) else "SELL"
        else:
            votes["15M"] = "NONE"
        rsi5 = rsi(tf5m["Close"], config.RSI_PERIOD)
        r5, r5p = float(rsi5.iloc[-1]), float(rsi5.iloc[-2]) if len(rsi5)>1 else float(rsi5.iloc[-1])
        votes["5M"] = "BUY" if r5p<30 and r5>=30 else "SELL" if r5p>70 and r5<=70 else "NONE"
        rsi1 = rsi(tf1m["Close"], config.RSI_PERIOD)
        r1 = float(rsi1.iloc[-1])
        votes["1M"] = "BUY" if r1<35 else "SELL" if r1>65 else "NONE"
        direction = _dom(votes)
        if direction == "NONE": return self._no_signal(entry)
        atr5 = float(atr(tf5m["High"], tf5m["Low"], tf5m["Close"], config.ATR_PERIOD).iloc[-1])
        sl, tp = _sltp(entry, direction, atr5)
        return Signal(direction, _conf(votes,direction,self.weight), entry, sl, tp, self.name, votes, atr5)


def _dom(v):
    b=sum(1 for x in v.values() if x=="BUY"); s=sum(1 for x in v.values() if x=="SELL")
    return "BUY" if b>=2 else "SELL" if s>=2 else "NONE"
def _sltp(e,d,a):
    return (e-config.SL_ATR_MULT*a, e+config.TP_ATR_MULT*a) if d=="BUY" else (e+config.SL_ATR_MULT*a, e-config.TP_ATR_MULT*a)
def _conf(v,d,w): return min(0.45+0.2*sum(1 for x in v.values() if x==d),1.0)*min(w,1.0)
