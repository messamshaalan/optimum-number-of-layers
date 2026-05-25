"""Main backtesting engine — optimised daily loop with multi-timeframe analysis.

Signal checks happen on each 5-minute bar close (not every minute).
1-minute bars (or 10-second tick bars) are used only for trade management.
Indicator windows are kept short (50-80 bars) for performance.

df_tick (optional): 10-second Heston bars for high-accuracy SL/TP management.
If provided, trade management runs at 10-second resolution instead of 1-minute.
"""

import pandas as pd
from datetime import date
from typing import List, Optional

from strategies.base import BaseStrategy, Signal
from .portfolio import Portfolio
from .risk_manager import RiskManager
from learning.daily_review import DailyReview
from learning.pattern_log import PatternLog
import config

_WARMUP_5M    = 80
_COOLDOWN_MIN = 30
_WIN_1M  = 80
_WIN_5M  = 60
_WIN_15M = 50


class Backtester:
    def __init__(self,
                 df_1m:       pd.DataFrame,
                 df_5m:       pd.DataFrame,
                 df_15m:      pd.DataFrame,
                 strategies:  List[BaseStrategy],
                 portfolio:   Portfolio,
                 risk_mgr:    RiskManager,
                 reviewer:    DailyReview,
                 pattern_log: PatternLog,
                 df_tick:     Optional[pd.DataFrame] = None):
        self.df_1m        = df_1m
        self.df_5m        = df_5m
        self.df_15m       = df_15m
        self.strategies   = strategies
        self.portfolio    = portfolio
        self.risk_mgr     = risk_mgr
        self.reviewer     = reviewer
        self.pattern_log  = pattern_log
        self.df_mgmt      = df_tick if df_tick is not None else df_1m
        self._tick_mode   = df_tick is not None

    def run(self, start: date, end: date, label: str = "Backtest") -> None:
        days = self._trading_days(start, end)
        print(f"\n[{label}] {len(days)} trading days from {start} to {end}")
        for i, day in enumerate(days):
            self._run_day(day)
            if (i + 1) % 50 == 0:
                eq  = self.portfolio.equity
                ret = self.portfolio.total_return_pct()
                print(f"  Day {i+1}/{len(days)} | {day} | "
                      f"Equity: ${eq:,.0f} ({ret:+.1f}%) | "
                      f"Trades: {len(self.portfolio.closed_trades)}")
        print(f"\n[{label}] Completed. Final equity: ${self.portfolio.equity:,.2f} "
              f"({self.portfolio.total_return_pct():+.1f}%)")

    @staticmethod
    def _trading_days(start: date, end: date):
        return pd.bdate_range(start=start, end=end).date.tolist()

    def _run_day(self, day: date) -> None:
        day_ts  = pd.Timestamp(day)
        next_ts = day_ts + pd.Timedelta(days=1)

        bars_1m = self.df_1m.loc[day_ts:next_ts - pd.Timedelta(minutes=1)]
        bars_5m = self.df_5m.loc[day_ts:next_ts - pd.Timedelta(minutes=1)]

        if len(bars_1m) < 60 or len(bars_5m) < 10:
            return

        if self._tick_mode:
            bars_mgmt = self.df_mgmt.loc[day_ts:next_ts - pd.Timedelta(seconds=9)]
        else:
            bars_mgmt = bars_1m

        g5_start = self.df_5m.index.searchsorted(day_ts)
        last_entry_ts: Optional[pd.Timestamp] = None
        daily_trade_count = 0
        bar_mgmt_iter = list(bars_mgmt.iterrows())
        bar_5m_list   = list(bars_5m.iterrows())
        bar_1m_iter   = list(bars_1m.iterrows())
        ptr_mgmt = 0

        for s5_i, (ts5, bar5) in enumerate(bar_5m_list):
            g5 = g5_start + s5_i
            while ptr_mgmt < len(bar_mgmt_iter):
                ts_m, bar_m = bar_mgmt_iter[ptr_mgmt]
                if ts_m > ts5:
                    break
                self.portfolio.update_open_trades(bar_m)
                ptr_mgmt += 1

            if daily_trade_count >= config.MAX_DAILY_TRADES:
                continue
            if not self.portfolio.can_open():
                continue
            if g5 < _WARMUP_5M:
                continue
            if (last_entry_ts is not None and
                    (ts5 - last_entry_ts).total_seconds() < _COOLDOWN_MIN * 60):
                continue

            g15 = _aligned_end(self.df_15m, ts5)
            g1  = _aligned_end(self.df_1m,  ts5)
            if g15 < 30:
                continue

            win_1m  = self.df_1m.iloc[max(0, g1  - _WIN_1M):  g1  + 1]
            win_5m  = self.df_5m.iloc[max(0, g5  - _WIN_5M):  g5  + 1]
            win_15m = self.df_15m.iloc[max(0, g15 - _WIN_15M): g15 + 1]

            signals: List[Signal] = []
            for strat in self.strategies:
                if not strat.is_active(day):
                    continue
                try:
                    sig = strat.generate_signal(win_1m, win_5m, win_15m, ts5)
                except Exception:
                    continue
                if sig.direction != "NONE":
                    signals.append(sig)

            if not signals:
                continue

            best = _best_signal(signals, self.strategies)
            if best is None or best.confidence < config.CONFLUENCE_THRESHOLD:
                continue

            trade = self.risk_mgr.build_trade(best, self.portfolio.equity, bar5)
            self.portfolio.open_trade(trade)
            last_entry_ts = ts5
            daily_trade_count += 1

        while ptr_mgmt < len(bar_mgmt_iter):
            _, bar_m = bar_mgmt_iter[ptr_mgmt]
            self.portfolio.update_open_trades(bar_m)
            ptr_mgmt += 1

        eod_bars = bar_mgmt_iter if self._tick_mode else bar_1m_iter
        if eod_bars:
            self.portfolio.close_all_eod(eod_bars[-1][1])

        today_trades = self.portfolio.today_trades()
        result = self.reviewer.analyze(day, today_trades, self.strategies)
        self.pattern_log.update_weights(self.strategies, result, today_trades)
        self.pattern_log.append_log(day, result)
        self.portfolio.end_of_day(day)


def _aligned_end(df: pd.DataFrame, ts: pd.Timestamp) -> int:
    pos = df.index.searchsorted(ts, side="right") - 1
    return max(pos, 0)


def _best_signal(signals: List[Signal],
                 strategies: List[BaseStrategy]) -> Optional[Signal]:
    weight_map = {s.name: s.weight for s in strategies}
    buy_score = sell_score = 0.0
    for sig in signals:
        w = weight_map.get(sig.strategy_name, 1.0)
        if sig.direction == "BUY":
            buy_score  += sig.confidence * w
        elif sig.direction == "SELL":
            sell_score += sig.confidence * w
    total = buy_score + sell_score
    if total == 0:
        return None
    if buy_score > sell_score:
        best = max((s for s in signals if s.direction == "BUY"), key=lambda x: x.confidence)
        best.confidence = buy_score / total
    else:
        best = max((s for s in signals if s.direction == "SELL"), key=lambda x: x.confidence)
        best.confidence = sell_score / total
    return best
