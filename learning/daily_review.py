"""Post-day trade analysis: per-strategy stats, session performance, observations."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, TYPE_CHECKING
from strategies.base import BaseStrategy

if TYPE_CHECKING:
    from engine.trade import Trade


@dataclass
class StrategyDayResult:
    name: str
    wins: int = 0
    losses: int = 0
    eod_closes: int = 0
    total_pnl: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0

    @property
    def total(self): return self.wins + self.losses + self.eod_closes
    @property
    def win_rate(self): return self.wins / self.total if self.total else 0.0
    @property
    def profit_factor(self):
        gross_profit = sum(1 for _ in range(self.wins))   # placeholder
        return 0.0


@dataclass
class DailyResult:
    trading_date: date
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    eod_closes: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    strategy_results: Dict[str, StrategyDayResult] = field(default_factory=dict)
    session_pnl: Dict[str, float] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    equity_end: float = 0.0


class DailyReview:
    def analyze(self, trading_date: date,
                trades: List[Trade],
                strategies: List[BaseStrategy]) -> DailyResult:
        result = DailyResult(trading_date=trading_date)
        if not trades:
            return result

        closed = [t for t in trades if not t.is_open]
        result.total_trades = len(closed)

        # Aggregate per strategy
        sr: Dict[str, StrategyDayResult] = {}
        for t in closed:
            if t.strategy not in sr:
                sr[t.strategy] = StrategyDayResult(name=t.strategy)
            s = sr[t.strategy]
            s.total_pnl += t.pnl_usd
            r = t.r_multiple or 0.0
            if t.exit_reason == "TP":
                s.wins += 1
            elif t.exit_reason == "SL":
                s.losses += 1
            else:
                s.eod_closes += 1
            s.best_r  = max(s.best_r, r)
            s.worst_r = min(s.worst_r, r)

        result.strategy_results = sr

        # Totals
        result.wins       = sum(1 for t in closed if t.exit_reason == "TP")
        result.losses     = sum(1 for t in closed if t.exit_reason == "SL")
        result.eod_closes = sum(1 for t in closed if t.exit_reason == "EOD")
        result.total_pnl  = sum(t.pnl_usd for t in closed)
        result.win_rate   = result.wins / result.total_trades if result.total_trades else 0.0

        # Session breakdown
        session_pnl: Dict[str, float] = {}
        for t in closed:
            session_pnl[t.session] = session_pnl.get(t.session, 0.0) + t.pnl_usd
        result.session_pnl = session_pnl

        # Generate observations
        result.observations = self._observe(result, strategies, closed)
        return result

    @staticmethod
    def _observe(result: DailyResult,
                 strategies: List[BaseStrategy],
                 trades: List[Trade]) -> List[str]:
        obs = []

        if result.total_trades == 0:
            obs.append("No trades today.")
            return obs

        # Overall day summary
        outcome = "profitable" if result.total_pnl > 0 else "losing"
        obs.append(
            f"Day result: {outcome} | PnL ${result.total_pnl:+.2f} | "
            f"Win rate {result.win_rate:.0%} ({result.wins}W/{result.losses}L/{result.eod_closes}EOD)"
        )

        # Best and worst strategy today
        if result.strategy_results:
            best_s  = max(result.strategy_results.values(), key=lambda x: x.total_pnl)
            worst_s = min(result.strategy_results.values(), key=lambda x: x.total_pnl)
            if best_s.total > 0:
                obs.append(
                    f"Best strategy: {best_s.name} "
                    f"(PnL ${best_s.total_pnl:+.2f}, {best_s.wins}W/{best_s.losses}L)"
                )
            if worst_s.total > 0 and worst_s.name != best_s.name:
                obs.append(
                    f"Worst strategy: {worst_s.name} "
                    f"(PnL ${worst_s.total_pnl:+.2f}, {worst_s.wins}W/{worst_s.losses}L)"
                )

        # Session insight
        if result.session_pnl:
            best_session  = max(result.session_pnl, key=result.session_pnl.get)
            worst_session = min(result.session_pnl, key=result.session_pnl.get)
            obs.append(f"Best session: {best_session} (${result.session_pnl[best_session]:+.2f})")
            if worst_session != best_session:
                obs.append(
                    f"Worst session: {worst_session} "
                    f"(${result.session_pnl[worst_session]:+.2f}) — reduce exposure"
                )

        # Pattern warnings
        sl_streak = _count_sl_streak(trades)
        if sl_streak >= 3:
            obs.append(
                f"Warning: {sl_streak} consecutive SL hits — "
                "market may be ranging or strategy trading against trend"
            )

        high_conf_losers = [t for t in trades
                            if t.exit_reason == "SL" and t.confluence_score >= 0.80]
        if high_conf_losers:
            obs.append(
                f"{len(high_conf_losers)} high-confidence signal(s) hit SL — "
                "review confluence scoring; potential false breakouts"
            )

        eod_profitable = [t for t in trades
                          if t.exit_reason == "EOD" and (t.pnl_usd or 0) > 0]
        if eod_profitable:
            obs.append(
                f"{len(eod_profitable)} EOD-closed trade(s) were profitable — "
                "consider extending hold time for winning trades"
            )

        return obs


def _count_sl_streak(trades: List[Trade]) -> int:
    streak = 0
    for t in reversed(trades):
        if t.exit_reason == "SL":
            streak += 1
        else:
            break
    return streak
