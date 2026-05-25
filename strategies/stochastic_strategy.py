"""Stochastic (14,3,3) cross in OB/OS + EMA trend filter."""
import pandas as pd
from .base import BaseStrategy, Signal
from indicators import stochastic, ema, atr
import config

_OB,_OS=80,20


class StochasticStrategy(BaseStrategy):
    name="Stochastic"

    def generate_signal(self,tf1m,tf5m,tf15m,timestamp) -> Signal:
        if len(tf5m)<25: return self._no_signal()
        entry=float(tf1m["Close"].iloc[-1]); votes={}
        if len(tf15m)>=55:
            votes["15M"]="BUY" if entry>float(ema(tf15m["Close"],config.EMA_TREND).iloc[-1]) else "SELL"
        else: votes["15M"]="NONE"
        k5,d5=stochastic(tf5m["High"],tf5m["Low"],tf5m["Close"],config.STOCH_K,config.STOCH_D,config.STOCH_S)
        k5n,k5p,d5n=float(k5.iloc[-1]),float(k5.iloc[-2]),float(d5.iloc[-1])
        votes["5M"]="BUY" if k5p<_OS and k5n>=_OS and k5n>d5n else "SELL" if k5p>_OB and k5n<=_OB and k5n<d5n else "NONE"
        k1,d1=stochastic(tf1m["High"],tf1m["Low"],tf1m["Close"],config.STOCH_K,config.STOCH_D,config.STOCH_S)
        k1n,k1p,d1n,d1p=float(k1.iloc[-1]),float(k1.iloc[-2]),float(d1.iloc[-1]),float(d1.iloc[-2])
        votes["1M"]="BUY" if k1p<=d1p and k1n>d1n else "SELL" if k1p>=d1p and k1n<d1n else "NONE"
        d=_dom(votes)
        if d=="NONE": return self._no_signal(entry)
        atr5=float(atr(tf5m["High"],tf5m["Low"],tf5m["Close"],config.ATR_PERIOD).iloc[-1])
        sl=entry-config.SL_ATR_MULT*atr5 if d=="BUY" else entry+config.SL_ATR_MULT*atr5
        tp=entry+config.TP_ATR_MULT*atr5 if d=="BUY" else entry-config.TP_ATR_MULT*atr5
        return Signal(d,_conf(votes,d,self.weight),entry,sl,tp,self.name,votes,atr5)

def _dom(v):
    b=sum(1 for x in v.values() if x=="BUY"); s=sum(1 for x in v.values() if x=="SELL")
    return "BUY" if b>=2 else "SELL" if s>=2 else "NONE"
def _conf(v,d,w): return min(0.45+0.2*sum(1 for x in v.values() if x==d),1.0)*min(w,1.0)
