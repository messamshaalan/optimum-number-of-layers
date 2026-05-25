import pandas as pd
from datetime import datetime, date
from typing import List, Dict
from .trade import Trade
import config


class Portfolio:
    def __init__(self, initial_capital=config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: Dict[date, float] = {}
        self._today_trades: List[Trade] = []

    def can_open(self): return len(self.open_trades) < config.MAX_CONCURRENT_TRADES

    def open_trade(self, trade):
        if self.can_open():
            self.open_trades.append(trade)
            self._today_trades.append(trade)

    def update_open_trades(self, bar):
        still_open = []
        for t in self.open_trades:
            h, l, ts = float(bar["High"]), float(bar["Low"]), bar.name
            if t.direction == "BUY":
                if l <= t.sl: t.close(ts, t.sl, "SL"); self._finalise(t)
                elif h >= t.tp: t.close(ts, t.tp, "TP"); self._finalise(t)
                else: still_open.append(t)
            else:
                if h >= t.sl: t.close(ts, t.sl, "SL"); self._finalise(t)
                elif l <= t.tp: t.close(ts, t.tp, "TP"); self._finalise(t)
                else: still_open.append(t)
        self.open_trades = still_open

    def close_all_eod(self, last_bar):
        price, ts = float(last_bar["Close"]), last_bar.name
        for t in list(self.open_trades): t.close(ts, price, "EOD"); self._finalise(t)
        self.open_trades = []

    def _finalise(self, trade): self.equity += trade.pnl_usd; self.closed_trades.append(trade)

    def end_of_day(self, d):
        self.equity_curve[d] = round(self.equity, 2)
        self._today_trades = []

    def today_trades(self): return list(self._today_trades)

    def total_return_pct(self):
        return (self.equity - self.initial_capital) / self.initial_capital * 100

    def max_drawdown(self):
        if not self.equity_curve: return 0.0
        vals = list(self.equity_curve.values()); peak = vals[0]; max_dd = 0.0
        for v in vals: peak=max(peak,v); dd=(peak-v)/peak; max_dd=max(max_dd,dd)
        return max_dd * 100

    def to_equity_series(self): return pd.Series(self.equity_curve)
