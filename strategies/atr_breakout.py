"""ATR Channel Breakout: price breaks EMA20 +/- 2xATR channel."""
import pandas as pd
from .base import BaseStrategy, Signal
from indicators import ema, atr, adx
import config

_ADX_MIN=20


class ATRBreakoutStrategy(BaseStrategy):
    name="ATR_Breakout"

    def generate_signal(self,tf1m,tf5m,tf15m,timestamp) -> Signal:
        if len(tf5m)<25: return self._no_signal()
        entry=float(tf1m["Close"].iloc[-1]); votes={}
        for label,df in [("15M",tf15m),("5M",tf5m)]:
            if len(df)<20: votes[label]="NONE"; continue
            e20=ema(df["Close"],20); a14=atr(df["High"],df["Low"],df["Close"],14)
            e20v,a14v=float(e20.iloc[-1]),float(a14.iloc[-1])
            c,cp=float(df["Close"].iloc[-1]),float(df["Close"].iloc[-2])
            adx_v=float(adx(df["High"],df["Low"],df["Close"]).iloc[-1]) if len(df)>=28 else _ADX_MIN+1
            upper,lower=e20v+2*a14v,e20v-2*a14v
            if adx_v>_ADX_MIN and c>upper and cp<=upper: votes[label]="BUY"
            elif adx_v>_ADX_MIN and c<lower and cp>=lower: votes[label]="SELL"
            else: votes[label]="NONE"
        e1=ema(tf1m["Close"],20); c1=float(tf1m["Close"].iloc[-1])
        votes["1M"]="BUY" if c1>float(e1.iloc[-1]) else "SELL" if c1<float(e1.iloc[-1]) else "NONE"
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
