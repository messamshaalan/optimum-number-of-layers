"""
XAUUSD Professional Day Trading Backtester
==========================================
Backtest 2016-2019 with 10 strategies + daily adaptive learning.
Forward test 2020 with the best-performing strategy.
Produces: results/backtest_report.html
"""

import sys
import time
import copy
import pandas as pd
import csv
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import config
from data import download_gold_daily, generate_intraday, resample_ohlcv
from strategies import ALL_STRATEGIES
from engine import Backtester, Portfolio, RiskManager
from learning import DailyReview, PatternLog
from reporting import compute_metrics, build_report


def main():
    t0 = time.time()
    print("=" * 60)
    print("  XAUUSD Day Trading Backtester")
    print("  Backtest: 2016–2019 | Forward Test: 2020")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading & generating price data …")
    daily_df = download_gold_daily(config.DATA_START, config.DATA_END)

    print(f"      Daily bars: {len(daily_df)} ({daily_df.index[0].date()} → {daily_df.index[-1].date()})")
    print("      Generating 1-minute intraday bars (this takes ~30 seconds) …")

    df_1m  = generate_intraday(daily_df, seed=42)
    df_5m  = resample_ohlcv(df_1m, "5min")
    df_15m = resample_ohlcv(df_1m, "15min")

    print(f"      1M bars: {len(df_1m):,} | 5M bars: {len(df_5m):,} | 15M bars: {len(df_15m):,}")

    # ------------------------------------------------------------------ #
    # 2. Backtest 2016–2019
    # ------------------------------------------------------------------ #
    print("\n[2/4] Running backtest 2016–2019 …")
    bt_strategies  = ALL_STRATEGIES
    bt_portfolio   = Portfolio(config.INITIAL_CAPITAL)
    bt_risk_mgr    = RiskManager()
    bt_reviewer    = DailyReview()
    bt_pattern_log = PatternLog()

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
    # 3. Identify best strategy
    # ------------------------------------------------------------------ #
    bt_metrics = compute_metrics(bt_portfolio.closed_trades, config.INITIAL_CAPITAL)
    stats = bt_metrics.get("strategy_stats", {})

    if stats:
        # Best by total P&L (min 5 trades)
        eligible = {k: v for k, v in stats.items() if v["trades"] >= 5}
        if eligible:
            best_strategy = max(eligible, key=lambda k: eligible[k]["total_pnl"])
        else:
            best_strategy = max(stats, key=lambda k: stats[k]["total_pnl"])
    else:
        best_strategy = bt_strategies[0].name

    print(f"\n[3/4] Best strategy identified: {best_strategy}")
    _print_strategy_table(stats, best_strategy)

    # ------------------------------------------------------------------ #
    # 4. Forward test 2020 — best strategy only
    # ------------------------------------------------------------------ #
    print(f"\n[4/4] Forward test 2020 with {best_strategy} …")

    # Find the strategy object; reset its weight to final learned value
    best_strat_obj = next((s for s in bt_strategies if s.name == best_strategy), bt_strategies[0])
    best_strat_clone = copy.deepcopy(best_strat_obj)
    best_strat_clone.suspended_until = None   # ensure not suspended

    fw_strategies  = [best_strat_clone]
    fw_portfolio   = Portfolio(config.INITIAL_CAPITAL)
    fw_risk_mgr    = RiskManager()
    fw_reviewer    = DailyReview()
    fw_pattern_log = PatternLog()

    fw = Backtester(
        df_1m=df_1m, df_5m=df_5m, df_15m=df_15m,
        strategies=fw_strategies,
        portfolio=fw_portfolio,
        risk_mgr=fw_risk_mgr,
        reviewer=fw_reviewer,
        pattern_log=fw_pattern_log,
    )
    fw.run(config.FORWARD_START, config.FORWARD_END, label="Forward Test")
    fw_metrics = compute_metrics(fw_portfolio.closed_trades, config.INITIAL_CAPITAL)

    # ------------------------------------------------------------------ #
    # 5. Save CSVs
    # ------------------------------------------------------------------ #
    _save_trades_csv(bt_portfolio.closed_trades + fw_portfolio.closed_trades)
    _save_strategy_csv(stats, bt_portfolio, fw_metrics, best_strategy)

    # ------------------------------------------------------------------ #
    # 6. Build HTML report
    # ------------------------------------------------------------------ #
    print("\n[5/5] Building HTML report …")
    build_report(
        bt_portfolio=bt_portfolio,
        fw_portfolio=fw_portfolio,
        daily_log=bt_pattern_log.get_log(),
        best_strategy=best_strategy,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Backtest equity:    ${bt_portfolio.equity:>10,.2f}  "
          f"({bt_portfolio.total_return_pct():+.1f}%)")
    print(f"  Forward test equity:${fw_portfolio.equity:>10,.2f}  "
          f"({fw_portfolio.total_return_pct():+.1f}%)")
    print(f"  Best strategy:      {best_strategy}")
    print(f"\n  Report → {config.RESULTS_DIR}/backtest_report.html")
    print(f"  Trades → {config.RESULTS_DIR}/daily_trades.csv")
    print("=" * 60)


# ------------------------------------------------------------------ #

def _print_strategy_table(stats: dict, best: str) -> None:
    print(f"\n  {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'PF':>6} {'P&L':>10} {'Sharpe':>8}")
    print("  " + "-" * 68)
    for name, s in sorted(stats.items(), key=lambda x: -x[1]["total_pnl"]):
        marker = " ★" if name == best else ""
        print(f"  {name:<25} {s['trades']:>7} {s['win_rate']:>7.1%} "
              f"{min(s['profit_factor'], 99):>6.2f} {s['total_pnl']:>10,.2f} "
              f"{s['sharpe']:>8.3f}{marker}")


def _save_trades_csv(trades) -> None:
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


if __name__ == "__main__":
    main()
