"""Post-day trade analysis and observation generation."""
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict
from engine.trade import Trade
from strategies.base import BaseStrategy


@dataclass
class StrategyDayResult:
    name: str; wins:int=0; losses:int=0; eod_closes:int=0
    total_pnl:float=0.0; best_r:float=0.0; worst_r:float=0.0
    @property
    def total(self): return self.wins+self.losses+self.eod_closes
    @property
    def win_rate(self): return self.wins/self.total if self.total else 0.0


@dataclass
class DailyResult:
    trading_date: date; total_trades:int=0; wins:int=0; losses:int=0
    eod_closes:int=0; total_pnl:float=0.0; win_rate:float=0.0
    strategy_results: Dict[str,StrategyDayResult] = field(default_factory=dict)
    session_pnl: Dict[str,float] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    equity_end: float = 0.0


class DailyReview:
    def analyze(self, trading_date, trades, strategies) -> DailyResult:
        result = DailyResult(trading_date=trading_date)
        closed = [t for t in trades if not t.is_open]
        if not closed: return result
        result.total_trades = len(closed)
        sr: Dict[str,StrategyDayResult] = {}
        for t in closed:
            if t.strategy not in sr: sr[t.strategy]=StrategyDayResult(name=t.strategy)
            s=sr[t.strategy]; s.total_pnl+=t.pnl_usd; r=t.r_multiple or 0.0
            if t.exit_reason=="TP": s.wins+=1
            elif t.exit_reason=="SL": s.losses+=1
            else: s.eod_closes+=1
            s.best_r=max(s.best_r,r); s.worst_r=min(s.worst_r,r)
        result.strategy_results=sr
        result.wins=sum(1 for t in closed if t.exit_reason=="TP")
        result.losses=sum(1 for t in closed if t.exit_reason=="SL")
        result.eod_closes=sum(1 for t in closed if t.exit_reason=="EOD")
        result.total_pnl=sum(t.pnl_usd for t in closed)
        result.win_rate=result.wins/result.total_trades if result.total_trades else 0.0
        sp: Dict[str,float]={}
        for t in closed: sp[t.session]=sp.get(t.session,0.0)+t.pnl_usd
        result.session_pnl=sp
        result.observations=self._observe(result,strategies,closed)
        return result

    @staticmethod
    def _observe(result,strategies,trades):
        obs=[]
        if not result.total_trades: obs.append("No trades today."); return obs
        out="profitable" if result.total_pnl>0 else "losing"
        obs.append(f"Day result: {out} | PnL ${result.total_pnl:+.2f} | Win rate {result.win_rate:.0%} ({result.wins}W/{result.losses}L/{result.eod_closes}EOD)")
        if result.strategy_results:
            bs=max(result.strategy_results.values(),key=lambda x:x.total_pnl)
            ws=min(result.strategy_results.values(),key=lambda x:x.total_pnl)
            if bs.total>0: obs.append(f"Best strategy: {bs.name} (PnL ${bs.total_pnl:+.2f}, {bs.wins}W/{bs.losses}L)")
            if ws.total>0 and ws.name!=bs.name: obs.append(f"Worst strategy: {ws.name} (PnL ${ws.total_pnl:+.2f})")
        if result.session_pnl:
            bss=max(result.session_pnl,key=result.session_pnl.get)
            wss=min(result.session_pnl,key=result.session_pnl.get)
            obs.append(f"Best session: {bss} (${result.session_pnl[bss]:+.2f})")
            if wss!=bss: obs.append(f"Worst session: {wss} (${result.session_pnl[wss]:+.2f}) -- reduce exposure")
        sl_streak=sum(1 for t in reversed(trades) if t.exit_reason=="SL") if trades else 0
        if sl_streak>=3: obs.append(f"Warning: {sl_streak} consecutive SL hits -- market may be ranging")
        hcl=[t for t in trades if t.exit_reason=="SL" and t.confluence_score>=0.80]
        if hcl: obs.append(f"{len(hcl)} high-confidence signal(s) hit SL -- review confluence; possible false breakouts")
        return obs
