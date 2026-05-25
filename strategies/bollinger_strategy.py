import pandas as pd
from .base import BaseStrategy, Signal
from indicators import bollinger, atr
import config


class BollingerStrategy(BaseStrategy):
    """Bollinger Band mean reversion with band expansion filter."""
    name = "Bollinger"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < config.BB_PERIOD+5: return self._no_signal()
        entry = float(tf1m["Close"].iloc[-1])
        votes = {}
        for label, df in [("15M",tf15m),("5M",tf5m),("1M",tf1m)]:
            if len(df) < config.BB_PERIOD+5: votes[label]="NONE"; continue
            upper,mid,lower,bw=bollinger(df["Close"],config.BB_PERIOD,config.BB_STD)
            u,m,l=float(upper.iloc[-1]),float(mid.iloc[-1]),float(lower.iloc[-1])
            bw_now=float(bw.iloc[-1]); bw_avg=float(bw.rolling(20).mean().iloc[-1])
            c=float(df["Close"].iloc[-1]); expanding=bw_now>bw_avg*0.8
            votes[label]="BUY" if c<=l and expanding else "SELL" if c>=u and expanding else "NONE"
        direction=_dom(votes)
        if direction=="NONE": return self._no_signal(entry)
        atr5=float(atr(tf5m["High"],tf5m["Low"],tf5m["Close"],config.ATR_PERIOD).iloc[-1])
        sl,tp=_sltp(entry,direction,atr5)
        return Signal(direction,_conf(votes,direction,self.weight),entry,sl,tp,self.name,votes,atr5)


def _dom(v):
    b=sum(1 for x in v.values() if x=="BUY"); s=sum(1 for x in v.values() if x=="SELL")
    return "BUY" if b>=2 else "SELL" if s>=2 else "NONE"
def _sltp(e,d,a):
    return (e-config.SL_ATR_MULT*a,e+config.TP_ATR_MULT*a) if d=="BUY" else (e+config.SL_ATR_MULT*a,e-config.TP_ATR_MULT*a)
def _conf(v,d,w): return min(0.45+0.2*sum(1 for x in v.values() if x==d),1.0)*min(w,1.0)
