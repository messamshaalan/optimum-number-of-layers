"""Heston stochastic-volatility tick generator.

Generates 10-second OHLCV bars anchored to real daily OHLCV.

Improvements over the simple GBM generator:
  - Stochastic volatility (Heston model): vol is itself mean-reverting + random
  - Leverage effect (rho < 0): price drops correlate with vol spikes, as in gold
  - Volatility clustering: long quiet periods followed by bursts
  - 6x finer resolution (10 s vs 1 min): SL/TP hit times are far more accurate

The 10-second bars are aggregated to 1 M / 5 M / 15 M with standard
OHLCV resampling, exactly like the regular generator.  The signal quality
improvement comes from the Heston price paths, not the raw bar count.
"""

import numpy as np
import pandas as pd
import config

_BARS_PER_DAY = 8_640          # 24 h × 3 600 s ÷ 10 s per bar

# Session blocks in 10-second bar indices (must cover 0–8640)
_SESSION_BLOCKS_10S = [
    ("Asia",        0,      2_520),   # 00:00–07:00 UTC
    ("LondonOpen",  2_520,  3_240),   # 07:00–09:00 UTC
    ("London",      3_240,  4_320),   # 09:00–12:00 UTC
    ("Overlap",     4_320,  5_040),   # 12:00–14:00 UTC
    ("NY",          5_040,  6_120),   # 14:00–17:00 UTC
    ("Rollover",    6_120,  8_640),   # 17:00–24:00 UTC
]


def _build_vol_profile() -> np.ndarray:
    m = np.zeros(_BARS_PER_DAY)
    for name, start, end in _SESSION_BLOCKS_10S:
        m[start:end] = config.SESSION_VOL_MULT[name]
    return m


_VOL_PROFILE = _build_vol_profile()


def _heston_path(open_: float, high: float, low: float, close: float,
                 rng: np.random.Generator,
                 kappa:   float = 2.0,    # vol mean-reversion speed
                 theta:   float = 0.0324, # long-run variance (≈18% annual vol)
                 sigma_v: float = 0.35,   # vol-of-vol
                 rho:     float = -0.50,  # price-vol correlation (negative = leverage effect)
                 ) -> np.ndarray:
    """Simulate 8 641 price points using the Heston SV model."""
    n  = _BARS_PER_DAY
    dt = 10.0 / (252.0 * 86_400.0)   # 10 s expressed in trading-year units

    # Correlated Brownian motions via Cholesky
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    dW_s = z1
    dW_v = rho * z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * z2

    # Initial variance estimated from day's range
    day_range_pct = max((high - low) / open_, 0.002)
    v0 = min((day_range_pct / 4.0 * np.sqrt(252.0)) ** 2, 0.25)

    # --- Variance process (Euler-Maruyama with full truncation) ---
    v    = np.empty(n + 1)
    v[0] = v0
    for i in range(n):
        v_pos    = max(v[i], 1e-9)
        v[i + 1] = (v_pos
                    + kappa * (theta - v_pos) * dt
                    + sigma_v * np.sqrt(v_pos * dt) * dW_v[i])
        v[i + 1] = max(v[i + 1], 1e-9)

    # --- Log-price process ---
    log_drift_per_step = np.log(close / open_) / n
    log_price = np.zeros(n + 1)
    for i in range(n):
        vol_i = np.sqrt(max(v[i], 1e-9) * dt) * _VOL_PROFILE[i]
        log_price[i + 1] = log_price[i] + log_drift_per_step + vol_i * dW_s[i]

    path = open_ * np.exp(log_price)

    # --- Stretch path so min/max match daily Low/High ---
    p_min, p_max = path.min(), path.max()
    if p_max > p_min:
        path = low + (path - p_min) * (high - low) / (p_max - p_min)

    # Enforce daily open/close anchors
    path[0]  = open_
    path[-1] = close
    return path


def _path_to_10s_bars(path: np.ndarray,
                      timestamps: pd.DatetimeIndex,
                      volume: float,
                      rng: np.random.Generator) -> pd.DataFrame:
    n      = len(timestamps)
    opens  = path[:n]
    closes = path[1:n + 1]

    # Micro-spread: bid-ask noise at tick level
    micro  = rng.uniform(0.0, 0.03, n)
    highs  = np.maximum(opens, closes) + micro
    lows   = np.minimum(opens, closes) - micro

    bar_vol = np.abs(closes - opens)
    total   = bar_vol.sum()
    volumes = (bar_vol / total * volume) if total > 0 else np.full(n, volume / n)

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=timestamps,
    )


def generate_tick_intraday(daily_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return 10-second OHLCV bars for all days in daily_df (Heston model)."""
    rng    = np.random.default_rng(seed)
    frames = []

    for date_idx, row in daily_df.iterrows():
        day_ts = pd.Timestamp(date_idx).normalize().replace(hour=0, minute=0, second=0)
        ts_idx = pd.date_range(start=day_ts, periods=_BARS_PER_DAY, freq="10s")

        path = _heston_path(
            float(row["Open"]), float(row["High"]),
            float(row["Low"]),  float(row["Close"]),
            rng,
        )
        vol = float(row.get("Volume", 100_000))
        frames.append(_path_to_10s_bars(path, ts_idx, vol, rng))

    df = pd.concat(frames).sort_index()
    df.index.name = "timestamp"
    return df


def resample_from_tick(df_tick: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate 10-second bars to any coarser timeframe."""
    return df_tick.resample(freq).agg({
        "Open":   "first",
        "High":   "max",
        "Low":    "min",
        "Close":  "last",
        "Volume": "sum",
    }).dropna()
