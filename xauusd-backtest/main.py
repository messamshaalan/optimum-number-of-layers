"""
XAUUSD Professional Day Trading Backtester — Enhanced Edition
=============================================================
Phase 1: Backtest 2016-2019 with 10 strategies, adaptive learning,
         session-aware weight tracking, and regime-filtered signals.
Phase 2: Forward test 2020 with best strategy.
Phase 3: Price Action deep backtest using Heston stochastic-volatility
         tick data (10-second bars) for accurate SL/TP management.
Produces: results/backtest_report.html
"""

import sys
import time
import copy
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from data import download_gold_daily, generate_intraday, resample_ohlcv
from data import generate_tick_intraday, resample_from_tick
from strategies import ALL_STRATEGIES
from engine import Backtester, Portfolio, RiskManager
from learning import DailyReview, PatternLog, SessionTracker
from reporting import compute_metrics, build_report


def main():
    t0 = time.time()
    print("=" * 60)
    print("  XAUUSD Day Trading Backtester  [Enhanced]")
    print("  Backtest: 2016–2019 | Forward Test: 2020")
    print("  + Session learning | Regime filter | Tick-level PA backtest")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Data — standard GBM bars for multi-strategy backtest
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading & generating price data …")
    daily_df = download_gold_daily(config.DATA_START, config.DATA_END)
    print(f"      Daily bars: {len(daily_df)} "
          f"({daily_df.index[0].date()} → {daily_df.index[-1].date()})")

    print("      Generating 1-minute intraday bars (GBM, ~30 s) …")
    df_1m  = generate_intraday(daily_df, seed=42)
    df_5m  = resample_ohlcv(df_1m, "5min")
    df_15m = resample_ohlcv(df_1m, "15min")
    print(f"      1M: {len(df_1m):,} | 5M: {len(df_5m):,} | 15M: {len(df_15m):,}")

    # ------------------------------------------------------------------ #
    # 2. Create shared SessionTracker — injected into all strategies
    # ------------------------------------------------------------------ #
    session_tracker = SessionTracker()
    bt_strategies   = ALL_STRATEGIES
    for strat in bt_strategies:
        strat.session_tracker = session_tracker

    # ------------------------------------------------------------------ #
    # 3. Backtest 2016–2019
    # ------------------------------------------------------------------ #
    print("\n[2/5] Running backtest 2016–2019 …")
    bt_portfolio   = Portfolio(config.INITIAL_CAPITAL)
    bt_risk_mgr    = RiskManager()
    bt_reviewer    = DailyReview()
    bt_pattern_log = PatternLog(session_tracker=session_tracker)

    bt = Backtester(
        df_1m=df_1m, df_5m=df_5m, df_15m=df_15m,
        strategies=bt_strategies,
        portfolio=bt_portfolio,
        risk_mgr=bt_risk_mgr,
        reviewer=bt_reviewer,
        pattern_log=bt_pattern_log,
    )
    bt.run(config.BACKTEST_START, config.BACKTEST_END, label="Backtest")
    bt_pattern_log.save()

    # ------------------------------------------------------------------ #
    # 4. Identify best strategy
    # ------------------------------------------------------------------ #
    bt_metrics = compute_metrics(bt_portfolio.closed_trades, config.INITIAL_CAPITAL)
    stats = bt_metrics.get("strategy_stats", {})

    if stats:
        eligible = {k: v for k, v in stats.items() if v["trades"] >= 5}
        best_strategy = max(eligible or stats, key=lambda k: (eligible or stats)[k]["total_pnl"])
    else:
        best_strategy = bt_strategies[0].name

    print(f"\n[3/5] Best strategy identified: {best_strategy}")
    _print_strategy_table(stats, best_strategy)
    _print_session_stats(session_tracker, best_strategy)
    _print_pattern_stats(session_tracker)

    # ------------------------------------------------------------------ #
    # 5. Forward test 2020 — best strategy with learned session weights
    # ------------------------------------------------------------------ #
    print(f"\n[4/5] Forward test 2020 with {best_strategy} …")

    best_strat_obj   = next((s for s in bt_strategies if s.name == best_strategy),
                            bt_strategies[0])
    best_strat_clone = copy.deepcopy(best_strat_obj)
    best_strat_clone.suspended_until  = None
    best_strat_clone.session_tracker  = session_tracker   # carry learned session stats

    fw_portfolio   = Portfolio(config.INITIAL_CAPITAL)
    fw_risk_mgr    = RiskManager()
    fw_reviewer    = DailyReview()
    fw_pattern_log = PatternLog(session_tracker=session_tracker)

    fw = Backtester(
        df_1m=df_1m, df_5m=df_5m, df_15m=df_15m,
        strategies=[best_strat_clone],
        portfolio=fw_portfolio,
        risk_mgr=fw_risk_mgr,
        reviewer=fw_reviewer,
        pattern_log=fw_pattern_log,
    )
    fw.run(config.FORWARD_START, config.FORWARD_END, label="Forward Test")
    fw_metrics = compute_metrics(fw_portfolio.closed_trades, config.INITIAL_CAPITAL)

    # ------------------------------------------------------------------ #
    # 6. Price Action deep backtest — Heston tick data
    # ------------------------------------------------------------------ #
    if best_strategy == "Price_Action":
        print("\n[5/5] Price Action deep backtest with Heston tick data …")
        print("      Generating 10-second Heston bars (~90 s) …")
        df_10s    = generate_tick_intraday(daily_df, seed=42)
        df_1m_heston  = resample_from_tick(df_10s, "1min")
        df_5m_heston  = resample_from_tick(df_10s, "5min")
        df_15m_heston = resample_from_tick(df_10s, "15min")
        print(f"      10s: {len(df_10s):,} | 1M: {len(df_1m_heston):,} "
              f"| 5M: {len(df_5m_heston):,} | 15M: {len(df_15m_heston):,}")

        pa_tracker = SessionTracker()
        pa_strat   = copy.deepcopy(best_strat_obj)
        pa_strat.weight           = 1.0   # reset — let it re-learn on Heston data
        pa_strat.suspended_until  = None
        pa_strat.session_tracker  = pa_tracker

        pa_portfolio   = Portfolio(config.INITIAL_CAPITAL)
        pa_risk_mgr    = RiskManager()
        pa_reviewer    = DailyReview()
        pa_pattern_log = PatternLog(session_tracker=pa_tracker)

        pa_bt = Backtester(
            df_1m=df_1m_heston, df_5m=df_5m_heston, df_15m=df_15m_heston,
            strategies=[pa_strat],
            portfolio=pa_portfolio,
            risk_mgr=pa_risk_mgr,
            reviewer=pa_reviewer,
            pattern_log=pa_pattern_log,
            df_tick=df_10s,   # 10-second bars for SL/TP management
        )
        pa_bt.run(config.BACKTEST_START, config.BACKTEST_END,
                  label="PA Tick Backtest")
        pa_pattern_log.save()

        pa_metrics = compute_metrics(pa_portfolio.closed_trades, config.INITIAL_CAPITAL)
        print("\n  [Tick Backtest] Session performance:")
        _print_session_stats(pa_tracker, "Price_Action")
        _print_pattern_stats(pa_tracker)

        _save_trades_csv(pa_portfolio.closed_trades,
                         config.RESULTS_DIR / "pa_tick_trades.csv")
        _save_tick_metrics(pa_metrics, pa_portfolio)
    else:
        print(f"\n[5/5] Skipping tick deep backtest (best strategy is {best_strategy})")
        pa_portfolio = None

    # ------------------------------------------------------------------ #
    # Save CSVs & build report
    # ------------------------------------------------------------------ #
    _save_trades_csv(bt_portfolio.closed_trades + fw_portfolio.closed_trades)
    _save_strategy_csv(stats, bt_portfolio, fw_metrics, best_strategy)

    print("\n[HTML] Building report …")
    build_report(
        bt_portfolio=bt_portfolio,
        fw_portfolio=fw_portfolio,
        daily_log=bt_pattern_log.get_log(),
        best_strategy=best_strategy,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Backtest equity:      ${bt_portfolio.equity:>10,.2f}  "
          f"({bt_portfolio.total_return_pct():+.1f}%)")
    print(f"  Forward test equity:  ${fw_portfolio.equity:>10,.2f}  "
          f"({fw_portfolio.total_return_pct():+.1f}%)")
    if pa_portfolio:
        print(f"  PA tick backtest:     ${pa_portfolio.equity:>10,.2f}  "
              f"({pa_portfolio.total_return_pct():+.1f}%)")
    print(f"  Best strategy:        {best_strategy}")
    print(f"\n  Report  → {config.RESULTS_DIR}/backtest_report.html")
    print(f"  Trades  → {config.RESULTS_DIR}/daily_trades.csv")
    print("=" * 60)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _print_strategy_table(stats: dict, best: str) -> None:
    print(f"\n  {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'PF':>6} "
          f"{'P&L':>10} {'Sharpe':>8}")
    print("  " + "-" * 68)
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["total_pnl"]):
        marker = " ★" if name == best else ""
        print(f"  {name:<25} {s['trades']:>7} {s['win_rate']:>7.1%} "
              f"{min(s['profit_factor'], 99):>6.2f} {s['total_pnl']:>10,.2f} "
              f"{s['sharpe']:>8.3f}{marker}")


def _print_session_stats(tracker: SessionTracker, strategy: str) -> None:
    stats = tracker.get_stats().get(strategy, {})
    if not stats:
        return
    print(f"\n  Session performance for {strategy}:")
    print(f"  {'Session':<14} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'Multiplier':>11}")
    print("  " + "-" * 48)
    for sess, d in sorted(stats.items(), key=lambda x: -x[1]["win_rate"]):
        mult = tracker.get_multiplier(strategy, sess)
        print(f"  {sess:<14} {d['total']:>7} {d['wins']:>6} "
              f"{d['win_rate']:>7.1%} {mult:>11.2f}×")


def _print_pattern_stats(tracker: SessionTracker) -> None:
    stats = tracker.get_pattern_stats()
    if not stats:
        return
    print(f"\n  Pattern type performance:")
    print(f"  {'Pattern':<12} {'Session':<14} {'Regime':<15} {'Trades':>7} {'Wins':>6} {'Win%':>7}")
    print("  " + "-" * 64)
    for ptype in sorted(stats):
        for sess in sorted(stats[ptype]):
            for regime, d in sorted(stats[ptype][sess].items()):
                if d["total"] < 3:
                    continue
                print(f"  {ptype:<12} {sess:<14} {regime:<15} "
                      f"{d['total']:>7} {d['wins']:>6} {d['win_rate']:>7.1%}")


def _save_trades_csv(trades, path=None) -> None:
    if path is None:
        path = config.RESULTS_DIR / "daily_trades.csv"
    if not trades:
        return
    fields = list(trades[0].to_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trades:
            writer.writerow(t.to_dict())
    print(f"[output] Trades CSV → {path} ({len(trades)} rows)")


def _save_strategy_csv(stats, bt_portfolio, fw_metrics, best) -> None:
    path = config.RESULTS_DIR / "strategy_performance.csv"
    rows = []
    for name, s in stats.items():
        rows.append({
            "strategy":      name,
            "is_best":       name == best,
            "trades":        s["trades"],
            "wins":          s["wins"],
            "losses":        s["losses"],
            "win_rate":      round(s["win_rate"], 4),
            "profit_factor": round(s["profit_factor"], 3),
            "total_pnl":     s["total_pnl"],
            "avg_r":         round(s["avg_r"], 4),
            "sharpe":        round(s["sharpe"], 3),
        })
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[output] Strategy CSV → {path}")


def _save_tick_metrics(metrics: dict, portfolio: Portfolio) -> None:
    path = config.RESULTS_DIR / "pa_tick_metrics.csv"
    row  = {
        "final_equity":    round(portfolio.equity, 2),
        "total_return_pct": round(portfolio.total_return_pct(), 2),
        "max_drawdown_pct": round(portfolio.max_drawdown(), 2),
        "total_trades":    metrics.get("total_trades", 0),
        "win_rate":        round(metrics.get("win_rate", 0), 4),
        "profit_factor":   round(metrics.get("profit_factor", 0), 3),
        "sharpe":          round(metrics.get("sharpe", 0), 3),
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"[output] PA tick metrics → {path}")


if __name__ == "__main__":
    main()
