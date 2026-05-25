"""Fibonacci 61.8% Retracement Strategy."""
import pandas as pd
from .base import BaseStrategy, Signal
from indicators import fibonacci_levels, ema, atr
import config

_FIB, _TOL = 0.618, 0.003


class FibonacciStrategy(BaseStrategy):
    name = "Fibonacci"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf15m)<30 or len(tf5m)<20: return self._no_signal()
        entry=float(tf1m["Close"].iloc[-1])
        w15=tf15m.tail(30)
        sh,sl=float(w15["High"].max()),float(w15["Low"].min())
        if sh-sl<3.0: return self._no_signal(entry)
        fib618=fibonacci_levels(sh,sl)[_FIB]
        if abs(entry-fib618)/(sh-sl)>_TOL: return self._no_signal(entry)
        e50=ema(tf15m["Close"],config.EMA_TREND)
        up=entry>float(e50.iloc[-1]); dn=entry<float(e50.iloc[-1])
        hi=w15["High"].idxmax(); lo=w15["Low"].idxmin()
        if up and lo>hi: d="BUY"
        elif dn and hi>lo: d="SELL"
        else: return self._no_signal(entry)
        atr5=float(atr(tf5m["High"],tf5m["Low"],tf5m["Close"],config.ATR_PERIOD).iloc[-1])
        votes={"15M":d,"5M":d,"1M":d}
        sl_p=entry-config.SL_ATR_MULT*atr5 if d=="BUY" else entry+config.SL_ATR_MULT*atr5
        tp_p=entry+config.TP_ATR_MULT*atr5 if d=="BUY" else entry-config.TP_ATR_MULT*atr5
        return Signal(d,0.72*min(self.weight,1.0),entry,sl_p,tp_p,self.name,votes,atr5)
