from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Capital & risk
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01        # 1% of INITIAL capital per trade (fixed dollar risk)
MAX_CONCURRENT_TRADES = 3
MAX_DAILY_TRADES = 5         # maximum trades placed per day
RISK_REWARD = 2.0            # min R:R ratio
SL_ATR_MULT = 1.5            # SL = 1.5 × ATR(5M)
TP_ATR_MULT = 3.0            # TP = 3.0 × ATR(5M)
SPREAD_USD = 0.30            # $0.30/oz spread

# Date ranges
BACKTEST_START = date(2016, 1, 1)
BACKTEST_END   = date(2019, 12, 31)
FORWARD_START  = date(2020, 1, 1)
FORWARD_END    = date(2020, 12, 31)
DATA_START     = "2015-10-01"   # extra warm-up data
DATA_END       = "2020-12-31"

# Signal confluence
CONFLUENCE_THRESHOLD = 0.72   # minimum weighted-vote score to enter (strict)
HIGH_CONF_THRESHOLD  = 0.88   # above this → 1.5× position size

# Indicator defaults
EMA_FAST   = 8
EMA_SLOW   = 21
EMA_TREND  = 50
EMA_LONG   = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD  = 20
BB_STD     = 2.0
STOCH_K    = 14
STOCH_D    = 3
STOCH_S    = 3

# Session hours (UTC)
SESSIONS = {
    "Asia":       (0, 7),
    "LondonOpen": (7, 9),
    "London":     (9, 12),
    "Overlap":    (12, 14),
    "NY":         (14, 17),
    "Rollover":   (17, 24),
}
SESSION_VOL_MULT = {
    "Asia":       0.3,
    "LondonOpen": 1.8,
    "London":     1.4,
    "Overlap":    1.6,
    "NY":         1.2,
    "Rollover":   0.4,
}

# Learning parameters
ROLLING_WINDOW = 20          # days for rolling Sharpe
WEIGHT_BOOST   = 1.05        # multiply weight when performing well
WEIGHT_CUT     = 0.90        # multiply weight when underperforming
MIN_WEIGHT     = 0.10
MAX_WEIGHT     = 2.00
SUSPEND_AFTER_LOSSES = 10    # consecutive losses before suspension
SUSPEND_DAYS   = 5           # suspension length

# Regime detection
ADX_TREND_THRESHOLD = 25.0   # ADX above this → TRENDING
ADX_RANGE_THRESHOLD = 20.0   # ADX below this → RANGING

# Session-aware learning
SESSION_MIN_SAMPLES = 15     # trades required before session multiplier activates

# Tick generator
TICK_FREQ = "10s"            # resolution of Heston tick simulation

# Instrument
TICKER = "GC=F"              # Gold futures (COMEX) — very close to spot XAUUSD
