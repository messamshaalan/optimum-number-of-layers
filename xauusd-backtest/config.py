from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Capital & risk
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01
MAX_CONCURRENT_TRADES = 3
MAX_DAILY_TRADES_TOTAL = 15    # global daily safety cap across all sessions
RISK_REWARD = 2.0
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
SPREAD_USD = 0.30

# Date ranges
BACKTEST_START = date(2016, 1, 1)
BACKTEST_END   = date(2019, 12, 31)
FORWARD_START  = date(2020, 1, 1)
FORWARD_END    = date(2020, 12, 31)
DATA_START     = "2015-10-01"
DATA_END       = "2020-12-31"

# Signal confluence
CONFLUENCE_THRESHOLD = 0.72
HIGH_CONF_THRESHOLD  = 0.88

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

# Session-specific trading rules (learned from backtest data)
SESSION_MAX_TRADES = {
    "Asia":       8,     # best win rate — expanded cap
    "LondonOpen": 2,     # volatile — strict limit
    "London":     3,
    "Overlap":    3,
    "NY":         0,     # blocked: 27.7% win rate, below 1:2 R:R breakeven
    "Rollover":   0,     # blocked: 1.2% win rate
}
SESSION_COOLDOWN_MIN = {
    "Asia":       15,    # shorter — clean trends, more opportunities
    "LondonOpen": 45,    # longer — false breakouts common
    "London":     30,
    "Overlap":    30,
    "NY":         9999,
    "Rollover":   9999,
}
# Session-specific TP multipliers (base, before regime adjustment)
SESSION_TP_MULT = {
    "Asia":       3.5,   # let winners run — clean directional moves
    "LondonOpen": 2.0,   # take profits early — reversals common
    "London":     2.5,
    "Overlap":    3.0,
    "NY":         2.0,
    "Rollover":   2.0,
}

# Price Action specific session rules
PA_BLOCKED_SESSIONS = {"NY", "Rollover"}
PA_STRICT_SESSIONS  = {"LondonOpen"}    # requires 3/3 TF agreement

# Learning parameters
ROLLING_WINDOW = 20
WEIGHT_BOOST   = 1.05
WEIGHT_CUT     = 0.90
MIN_WEIGHT     = 0.10
MAX_WEIGHT     = 2.00
SUSPEND_AFTER_LOSSES = 10
SUSPEND_DAYS   = 5

# Regime detection
ADX_TREND_THRESHOLD = 25.0
ADX_RANGE_THRESHOLD = 20.0

# Session-aware learning
SESSION_MIN_SAMPLES = 15

# Tick generator
TICK_FREQ = "10s"

# Instrument
TICKER = "GC=F"
