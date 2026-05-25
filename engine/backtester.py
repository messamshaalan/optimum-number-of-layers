"""Optimised backtester: signal checks every 5 minutes, 1M for trade management."""
import pandas as pd
from datetime import date
from typing import List, Optional
from strategies.base import BaseStrategy, Signal
from .portfolio import Portfolio
from .risk_manager import RiskManager
from learning.daily_review import DailyReview
from learning.pattern_log import PatternLog
import config

_WARMUP_5M=80; _COOLDOWN_MIN=30; _WIN_1M=80; _WIN_5M=60; _WIN_15M=50


class Backtester:
    def __init__(self,df_1m,df_5m,df_15m,strategies,portfolio,risk_mgr,reviewer,pattern_log):
        self.df_1m=df_1m; self.df_5m=df_5m; self.df_15m=df_15m
        self.strategies=strategies; self.portfolio=portfolio
        self.risk_mgr=risk_mgr; self.reviewer=reviewer; self.pattern_log=pattern_log

    def run(self, start, end, label="Backtest"):
        days=self._trading_days(start,end)
        print(f"\n[{label}] {len(days)} trading days from {start} to {end}")
        for i,day in enumerate(days):
            self._run_day(day)
            if (i+1)%50==0:
                eq=self.portfolio.equity; ret=self.portfolio.total_return_pct()
                print(f"  Day {i+1}/{len(days)} | {day} | Equity: ${eq:,.0f} ({ret:+.1f}%) | Trades: {len(self.portfolio.closed_trades)}")
        print(f"\n[{label}] Completed. Final equity: ${self.portfolio.equity:,.2f} ({self.portfolio.total_return_pct():+.1f}%)")

    @staticmethod
    def _trading_days(start,end): return pd.bdate_range(start=start,end=end).date.tolist()

    def _run_day(self, day):
        ts=pd.Timestamp(day); nxt=ts+pd.Timedelta(days=1)
        b1=self.df_1m.loc[ts:nxt-pd.Timedelta(minutes=1)]
        b5=self.df_5m.loc[ts:nxt-pd.Timedelta(minutes=1)]
        if len(b1)<60 or len(b5)<10: return
        g5s=self.df_5m.index.searchsorted(ts); last_e=None; dtc=0
        b1l=list(b1.iterrows()); b5l=list(b5.iterrows()); ptr=0
        for s5i,(ts5,bar5) in enumerate(b5l):
            g5=g5s+s5i
            while ptr<len(b1l):
                ts1,bar1=b1l[ptr]
                if ts1>ts5: break
                self.portfolio.update_open_trades(bar1); ptr+=1
            if dtc>=config.MAX_DAILY_TRADES or not self.portfolio.can_open(): continue
            if g5<_WARMUP_5M: continue
            if last_e and (ts5-last_e).total_seconds()<_COOLDOWN_MIN*60: continue
            g15=_ae(self.df_15m,ts5); g1=_ae(self.df_1m,ts5)
            if g15<30: continue
            w1=self.df_1m.iloc[max(0,g1-_WIN_1M):g1+1]
            w5=self.df_5m.iloc[max(0,g5-_WIN_5M):g5+1]
            w15=self.df_15m.iloc[max(0,g15-_WIN_15M):g15+1]
            sigs=[]
            for s in self.strategies:
                if not s.is_active(day): continue
                try: sig=s.generate_signal(w1,w5,w15,ts5)
                except: continue
                if sig.direction!="NONE": sigs.append(sig)
            if not sigs: continue
            best=_best(sigs,self.strategies)
            if best is None or best.confidence<config.CONFLUENCE_THRESHOLD: continue
            self.portfolio.open_trade(self.risk_mgr.build_trade(best,self.portfolio.equity,bar5))
            last_e=ts5; dtc+=1
        while ptr<len(b1l): _,bar1=b1l[ptr]; self.portfolio.update_open_trades(bar1); ptr+=1
        if b1l: self.portfolio.close_all_eod(b1l[-1][1])
        result=self.reviewer.analyze(day,self.portfolio.today_trades(),self.strategies)
        self.pattern_log.update_weights(self.strategies,result)
        self.pattern_log.append_log(day,result)
        self.portfolio.end_of_day(day)


def _ae(df,ts):
    pos=df.index.searchsorted(ts,side="right")-1; return max(pos,0)

def _best(sigs,strats):
    wm={s.name:s.weight for s in strats}; bs=ss=0.0
    for sig in sigs:
        w=wm.get(sig.strategy_name,1.0)
        if sig.direction=="BUY": bs+=sig.confidence*w
        elif sig.direction=="SELL": ss+=sig.confidence*w
    total=bs+ss
    if not total: return None
    if bs>ss:
        best=max((s for s in sigs if s.direction=="BUY"),key=lambda x:x.confidence)
        best.confidence=bs/total
    else:
        best=max((s for s in sigs if s.direction=="SELL"),key=lambda x:x.confidence)
        best.confidence=ss/total
    return best
