"""Price Action: pin bars + engulfing candles with EMA50 trend filter."""
import pandas as pd
from .base import BaseStrategy, Signal
from indicators import detect_pin_bar, detect_engulfing, ema, atr
import config


class PriceActionStrategy(BaseStrategy):
    name="Price_Action"

    def generate_signal(self,tf1m,tf5m,tf15m,timestamp) -> Signal:
        if len(tf5m)<10 or len(tf15m)<10: return self._no_signal()
        entry=float(tf1m["Close"].iloc[-1]); votes={}
        for label,df in [("15M",tf15m),("5M",tf5m),("1M",tf1m)]:
            if len(df)<5: votes[label]="NONE"; continue
            pin=detect_pin_bar(df); eng=detect_engulfing(df)
            c,o=df["Close"],df["Open"]
            bp=any(pin.iloc[-2:]) and any(c.iloc[-2:]>o.iloc[-2:])
            brp=any(pin.iloc[-2:]) and any(c.iloc[-2:]<o.iloc[-2:])
            be=any(eng.iloc[-2:]) and float(c.iloc[-1])>float(o.iloc[-1])
            bre=any(eng.iloc[-2:]) and float(c.iloc[-1])<float(o.iloc[-1])
            votes[label]="BUY" if bp or be else "SELL" if brp or bre else "NONE"
        if len(tf15m)>=55:
            e50=ema(tf15m["Close"],config.EMA_TREND)
            trend="BUY" if entry>float(e50.iloc[-1]) else "SELL"
            if votes.get("15M")!=trend: votes["15M"]="NONE"
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
