import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Optional
from .trade import Trade
import config


class Portfolio:
    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.equity          = initial_capital
        self.cash            = initial_capital
        self.open_trades: List[Trade]   = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: Dict[date, float] = {}
        self._today_trades: List[Trade] = []

    # ------------------------------------------------------------------ #
    def can_open(self) -> bool:
        return len(self.open_trades) < config.MAX_CONCURRENT_TRADES

    def open_trade(self, trade: Trade) -> None:
        if not self.can_open():
            return
        self.open_trades.append(trade)
        self._today_trades.append(trade)

    # ------------------------------------------------------------------ #
    def update_open_trades(self, bar: pd.Series) -> None:
        """Check SL/TP against current bar and apply breakeven stop logic."""
        still_open = []
        for t in self.open_trades:
            h, l = float(bar["High"]), float(bar["Low"])
            ts   = bar.name

            if t.direction == "BUY":
                if l <= t.sl:
                    t.close(ts, t.sl, "SL")
                    self._finalise(t)
                elif h >= t.tp:
                    t.close(ts, t.tp, "TP")
                    self._finalise(t)
                else:
                    # Breakeven: move SL to entry after price moves 1R in favour
                    if not t.breakeven_moved and t.sl_distance > 0:
                        if h >= t.entry_price + t.sl_distance:
                            t.sl = max(t.sl, t.entry_price)
                            t.breakeven_moved = True
                    still_open.append(t)
            else:  # SELL
                if h >= t.sl:
                    t.close(ts, t.sl, "SL")
                    self._finalise(t)
                elif l <= t.tp:
                    t.close(ts, t.tp, "TP")
                    self._finalise(t)
                else:
                    if not t.breakeven_moved and t.sl_distance > 0:
                        if l <= t.entry_price - t.sl_distance:
                            t.sl = min(t.sl, t.entry_price)
                            t.breakeven_moved = True
                    still_open.append(t)

        self.open_trades = still_open

    def close_all_eod(self, last_bar: pd.Series) -> None:
        price = float(last_bar["Close"])
        ts    = last_bar.name
        for t in list(self.open_trades):
            t.close(ts, price, "EOD")
            self._finalise(t)
        self.open_trades = []

    def _finalise(self, trade: Trade) -> None:
        self.equity += trade.pnl_usd
        self.closed_trades.append(trade)

    # ------------------------------------------------------------------ #
    def end_of_day(self, trading_date: date) -> None:
        self.equity_curve[trading_date] = round(self.equity, 2)
        self._today_trades = []

    def today_trades(self) -> List[Trade]:
        return list(self._today_trades)

    # ------------------------------------------------------------------ #
    def total_return_pct(self) -> float:
        return (self.equity - self.initial_capital) / self.initial_capital * 100

    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        vals = list(self.equity_curve.values())
        peak = vals[0]
        max_dd = 0.0
        for v in vals:
            peak = max(peak, v)
            dd   = (peak - v) / peak
            max_dd = max(max_dd, dd)
        return max_dd * 100

    def to_equity_series(self) -> pd.Series:
        return pd.Series(self.equity_curve)
