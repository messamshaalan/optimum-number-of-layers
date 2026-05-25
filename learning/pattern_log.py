"""Cumulative strategy weight adjustment and daily observation logging."""
import json, math
from collections import defaultdict, deque
from datetime import date, timedelta
from typing import List, Dict, Deque
from strategies.base import BaseStrategy
from .daily_review import DailyResult
import config

_LOG_PATH = config.RESULTS_DIR / "daily_learning_log.json"


class PatternLog:
    def __init__(self):
        self._rolling: Dict[str,Deque[float]] = defaultdict(lambda: deque(maxlen=config.ROLLING_WINDOW))
        self._consec_losses: Dict[str,int] = defaultdict(int)
        self._consec_wins:   Dict[str,int] = defaultdict(int)
        self._log: List[dict] = []

    def update_weights(self, strategies, result):
        today=result.trading_date
        for strat in strategies:
            sr=result.strategy_results.get(strat.name)
            day_pnl=sr.total_pnl if sr else 0.0
            self._rolling[strat.name].append(day_pnl)
            if sr:
                if sr.wins>sr.losses+sr.eod_closes: self._consec_wins[strat.name]+=1; self._consec_losses[strat.name]=0
                elif sr.losses>sr.wins: self._consec_losses[strat.name]+=1; self._consec_wins[strat.name]=0
            window=list(self._rolling[strat.name]); sharpe=_sharpe(window)
            if len(window)>=5:
                if sharpe>0.5: strat.weight=min(strat.weight*config.WEIGHT_BOOST,config.MAX_WEIGHT)
                elif sharpe<0.0: strat.weight=max(strat.weight*config.WEIGHT_CUT,config.MIN_WEIGHT)
            if self._consec_losses[strat.name]>=config.SUSPEND_AFTER_LOSSES:
                strat.suspended_until=today+timedelta(days=config.SUSPEND_DAYS)
                self._consec_losses[strat.name]=0

    def append_log(self, day, result):
        self._log.append({
            "date":str(day),"total_trades":result.total_trades,
            "wins":result.wins,"losses":result.losses,"eod_closes":result.eod_closes,
            "total_pnl":round(result.total_pnl,2),"win_rate":round(result.win_rate,4),
            "session_pnl":{k:round(v,2) for k,v in result.session_pnl.items()},
            "strategy_results":{
                n:{"wins":s.wins,"losses":s.losses,"total_pnl":round(s.total_pnl,2),"win_rate":round(s.win_rate,4)}
                for n,s in result.strategy_results.items()},
            "observations":result.observations,
        })

    def save(self):
        with open(_LOG_PATH,"w") as f: json.dump(self._log,f,indent=2)
        print(f"[learning] Saved {len(self._log)} daily log entries -> {_LOG_PATH}")

    def get_log(self): return self._log


def _sharpe(returns):
    if len(returns)<2: return 0.0
    mean=sum(returns)/len(returns)
    var=sum((r-mean)**2 for r in returns)/(len(returns)-1)
    std=math.sqrt(var) if var>0 else 1e-9
    return mean/std
