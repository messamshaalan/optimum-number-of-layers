"""Support/Resistance Bounce Strategy."""
import pandas as pd
from .base import BaseStrategy, Signal
from indicators import swing_levels, atr, detect_pin_bar
import config

_PROX = 1.50


class SRBounceStrategy(BaseStrategy):
    name = "SR_Bounce"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf15m)<50 or len(tf5m)<20: return self._no_signal()
        entry = float(tf1m["Close"].iloc[-1])
        s15,r15=swing_levels(tf15m,15); s5,r5=swing_levels(tf5m,10)
        ns=any(abs(entry-s)<=_PROX for s in s15+s5)
        nr=any(abs(entry-r)<=_PROX for r in r15+r5)
        if not ns and not nr: return self._no_signal(entry)
        p5=detect_pin_bar(tf5m); p1=detect_pin_bar(tf1m)
        hp5=bool(p5.iloc[-1]) or bool(p5.iloc[-2]); hp1=bool(p1.iloc[-1])
        atr5=float(atr(tf5m["High"],tf5m["Low"],tf5m["Close"],config.ATR_PERIOD).iloc[-1])
        d="NONE"
        if ns and (hp5 or hp1): d="BUY"
        elif nr and (hp5 or hp1): d="SELL"
        if d=="NONE": return self._no_signal(entry)
        votes={"15M":d,"5M":d,"1M":d}
        sl=entry-config.SL_ATR_MULT*atr5 if d=="BUY" else entry+config.SL_ATR_MULT*atr5
        tp=entry+config.TP_ATR_MULT*atr5 if d=="BUY" else entry-config.TP_ATR_MULT*atr5
        conf=(0.55+(0.15 if hp5 and hp1 else 0.0))*min(self.weight,1.0)
        return Signal(d,conf,entry,sl,tp,self.name,votes,atr5)
