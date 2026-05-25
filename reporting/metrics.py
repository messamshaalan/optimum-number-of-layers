"""Performance metrics: Sharpe, Sortino, profit factor, win rate, expectancy."""
import math
from typing import List, Dict, Any
from engine.trade import Trade


def compute_metrics(trades, initial_capital):
    closed=[t for t in trades if not t.is_open and t.pnl_usd is not None]
    if not closed: return {}
    pnls=[t.pnl_usd for t in closed]
    r_muls=[t.r_multiple or 0.0 for t in closed]
    wins=[t for t in closed if t.exit_reason=="TP"]
    losses=[t for t in closed if t.exit_reason=="SL"]
    eods=[t for t in closed if t.exit_reason=="EOD"]
    gp=sum(p for p in pnls if p>0); gl=abs(sum(p for p in pnls if p<0))
    total_pnl=sum(pnls); total_ret=total_pnl/initial_capital*100
    pf=gp/gl if gl>0 else float("inf")
    wr=len(wins)/len(closed) if closed else 0.0
    avg_r=sum(r_muls)/len(r_muls) if r_muls else 0.0
    dp=_daily(closed)
    by_strat: Dict[str,Dict]={}
    for t in closed:
        s=by_strat.setdefault(t.strategy,{"trades":0,"wins":0,"losses":0,"eod":0,"total_pnl":0.0,"pnls":[],"r_muls":[]})
        s["trades"]+=1; s["total_pnl"]+=t.pnl_usd; s["pnls"].append(t.pnl_usd); s["r_muls"].append(t.r_multiple or 0.0)
        if t.exit_reason=="TP": s["wins"]+=1
        elif t.exit_reason=="SL": s["losses"]+=1
        else: s["eod"]+=1
    ss={}
    for name,s in by_strat.items():
        sgp=sum(p for p in s["pnls"] if p>0); sgl=abs(sum(p for p in s["pnls"] if p<0))
        ss[name]={"trades":s["trades"],"wins":s["wins"],"losses":s["losses"],"eod":s["eod"],
                  "win_rate":s["wins"]/s["trades"] if s["trades"] else 0.0,
                  "profit_factor":sgp/sgl if sgl>0 else (float("inf") if sgp>0 else 0.0),
                  "total_pnl":round(s["total_pnl"],2),
                  "avg_r":sum(s["r_muls"])/len(s["r_muls"]) if s["r_muls"] else 0.0,
                  "sharpe":_sharpe(_daily([t for t in closed if t.strategy==name]))}
    session_stats={}
    for t in closed:
        sv=session_stats.setdefault(t.session,{"trades":0,"wins":0,"total_pnl":0.0})
        sv["trades"]+=1; sv["total_pnl"]+=t.pnl_usd
        if t.exit_reason=="TP": sv["wins"]+=1
    return {"total_trades":len(closed),"wins":len(wins),"losses":len(losses),"eod_closes":len(eods),
            "win_rate":wr,"profit_factor":pf,"total_pnl":round(total_pnl,2),"total_return":round(total_ret,2),
            "gross_profit":round(gp,2),"gross_loss":round(gl,2),"avg_r":round(avg_r,3),"expectancy_r":round(avg_r,3),
            "sharpe":round(_sharpe(dp),3),"sortino":round(_sortino(dp),3),
            "strategy_stats":ss,"session_stats":session_stats}


def _daily(trades):
    from collections import defaultdict
    by_day=defaultdict(float)
    for t in trades:
        d=t.entry_time.date() if hasattr(t.entry_time,"date") else t.entry_time
        by_day[d]+=t.pnl_usd
    return list(by_day.values())

def _sharpe(dp,rf=0.0):
    if len(dp)<2: return 0.0
    n=len(dp); mean=sum(dp)/n; var=sum((r-mean)**2 for r in dp)/(n-1)
    return (mean-rf)/math.sqrt(var)*math.sqrt(252) if var>0 else 0.0

def _sortino(dp,rf=0.0):
    if len(dp)<2: return 0.0
    mean=sum(dp)/len(dp); downs=[r for r in dp if r<rf]
    if not downs: return float("inf")
    dv=sum((r-rf)**2 for r in downs)/len(downs)
    return (mean-rf)/math.sqrt(dv)*math.sqrt(252) if dv>0 else float("inf")
