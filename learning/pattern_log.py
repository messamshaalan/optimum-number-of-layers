"""Cumulative strategy weight adjustment and daily observation logging.

Enhancement: integrates SessionTracker so each trade's session/outcome is
recorded and strategies can query per-session win rates at signal time.
"""

import json
import math
from collections import defaultdict, deque
from datetime import date, timedelta
from typing import List, Dict, Deque, Optional

from strategies.base import BaseStrategy
from .daily_review import DailyResult
from .session_tracker import SessionTracker
import config

_LOG_PATH = config.RESULTS_DIR / "daily_learning_log.json"


class PatternLog:
    def __init__(self, session_tracker: Optional[SessionTracker] = None):
        self._rolling: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=config.ROLLING_WINDOW)
        )
        self._consec_losses: Dict[str, int] = defaultdict(int)
        self._consec_wins:   Dict[str, int] = defaultdict(int)
        self._log: List[dict] = []
        self.session_tracker = session_tracker

    def update_weights(self,
                       strategies: List[BaseStrategy],
                       result: DailyResult,
                       trades: list = None) -> None:
        today = result.trading_date

        if self.session_tracker and trades:
            for t in trades:
                if not t.is_open and t.exit_reason in ("TP", "SL"):
                    won = (t.exit_reason == "TP")
                    self.session_tracker.record(t.strategy, t.session, won)

        for strat in strategies:
            sr      = result.strategy_results.get(strat.name)
            day_pnl = sr.total_pnl if sr else 0.0
            self._rolling[strat.name].append(day_pnl)

            if sr:
                if sr.wins > sr.losses + sr.eod_closes:
                    self._consec_wins[strat.name]   += 1
                    self._consec_losses[strat.name]  = 0
                elif sr.losses > sr.wins:
                    self._consec_losses[strat.name] += 1
                    self._consec_wins[strat.name]    = 0

            window = list(self._rolling[strat.name])
            sharpe = _rolling_sharpe(window)

            old_weight = strat.weight
            if len(window) >= 5:
                if sharpe > 0.5:
                    strat.weight = min(strat.weight * config.WEIGHT_BOOST, config.MAX_WEIGHT)
                elif sharpe < 0.0:
                    strat.weight = max(strat.weight * config.WEIGHT_CUT, config.MIN_WEIGHT)

            if self._consec_losses[strat.name] >= config.SUSPEND_AFTER_LOSSES:
                strat.suspended_until = today + timedelta(days=config.SUSPEND_DAYS)
                self._consec_losses[strat.name] = 0
                self._log_weight_change(today, strat.name, old_weight, strat.weight,
                                        f"SUSPENDED for {config.SUSPEND_DAYS} days after "
                                        f"{config.SUSPEND_AFTER_LOSSES} consecutive losses")

    def append_log(self, day: date, result: DailyResult) -> None:
        entry = {
            "date":         str(day),
            "total_trades": result.total_trades,
            "wins":         result.wins,
            "losses":       result.losses,
            "eod_closes":   result.eod_closes,
            "total_pnl":    round(result.total_pnl, 2),
            "win_rate":     round(result.win_rate, 4),
            "session_pnl":  {k: round(v, 2) for k, v in result.session_pnl.items()},
            "strategy_results": {
                name: {
                    "wins":      sr.wins,
                    "losses":    sr.losses,
                    "total_pnl": round(sr.total_pnl, 2),
                    "win_rate":  round(sr.win_rate, 4),
                }
                for name, sr in result.strategy_results.items()
            },
            "observations": result.observations,
        }
        self._log.append(entry)

    def _log_weight_change(self, day: date, strategy: str,
                           old: float, new: float, reason: str) -> None:
        if self._log and self._log[-1]["date"] == str(day):
            self._log[-1].setdefault("weight_changes", []).append({
                "strategy": strategy, "old": round(old, 3),
                "new": round(new, 3), "reason": reason,
            })

    def save(self) -> None:
        with open(_LOG_PATH, "w") as f:
            json.dump(self._log, f, indent=2)
        print(f"[learning] Saved {len(self._log)} daily log entries → {_LOG_PATH}")

    def get_log(self) -> List[dict]:
        return self._log

    def get_weight_history(self,
                           strategies: List[BaseStrategy]) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {s.name: [1.0] for s in strategies}
        for entry in self._log:
            for wc in entry.get("weight_changes", []):
                history[wc["strategy"]].append(wc["new"])
        return history


def _rolling_sharpe(returns: List[float], rf: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var  = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std  = math.sqrt(var) if var > 0 else 1e-9
    return (mean - rf) / std
