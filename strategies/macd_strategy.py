import pandas as pd
from .base import BaseStrategy, Signal
from indicators import macd, atr
import config


class MACDStrategy(BaseStrategy):
    """MACD signal-line cross + histogram momentum."""
    name = "MACD"

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < 40: return self._no_signal()
        entry = float(tf1m["Close"].iloc[-1])
        votes = {}
        for label, df in [("15M", tf15m), ("5M", tf5m)]:
            if len(df) < 35: votes[label]="NONE"; continue
            m,s,h = macd(df["Close"])
            mn,sn,hn = float(m.iloc[-1]),float(s.iloc[-1]),float(h.iloc[-1])
            mp,sp = float(m.iloc[-2]),float(s.iloc[-2])
            cross_up   = mp<=sp and mn>sn and mn<0
            cross_down = mp>=sp and mn<sn and mn>0
            if cross_up and hn>0: votes[label]="BUY"
            elif cross_down and hn<0: votes[label]="SELL"
            else:
                h3=float(h.iloc[-3]) if len(h)>=3 else 0
                votes[label]="BUY" if mn>0 and hn>h3 else "SELL" if mn<0 and hn<h3 else "NONE"
        m1,s1,h1=macd(tf1m["Close"])
        h1n,h1p=float(h1.iloc[-1]),float(h1.iloc[-2])
        votes["1M"]="BUY" if h1n>h1p and h1n>0 else "SELL" if h1n<h1p and h1n<0 else "NONE"
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
