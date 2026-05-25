"""Generate realistic 1-minute intraday OHLCV from daily bars.

Each daily bar is decomposed into 1440 one-minute bars using Geometric Brownian
Motion with session-specific volatility scaling. The path is stretched so its
min/max exactly match the daily Low/High, and endpoints are blended to match
the daily Open/Close.
"""

import numpy as np
import pandas as pd

import config

_SESSION_BLOCKS = [
    ("Asia",       0,    420),
    ("LondonOpen", 420,  540),
    ("London",     540,  720),
    ("Overlap",    720,  840),
    ("NY",         840,  1020),
    ("Rollover",   1020, 1440),
]


def _vol_multipliers() -> np.ndarray:
    mults = np.zeros(1440)
    for name, start, end in _SESSION_BLOCKS:
        mults[start:end] = config.SESSION_VOL_MULT[name]
    return mults


_VOL_MULT = _vol_multipliers()


def _generate_day_path(open_: float, high: float, low: float,
                       close: float, rng: np.random.Generator) -> np.ndarray:
    n = 1440
    drift = (np.log(close / open_)) / n

    day_range_pct = max((high - low) / open_, 0.002)
    base_sigma    = (day_range_pct / 4) * np.sqrt(252)
    per_min_sigma = base_sigma / np.sqrt(252 * 1440)

    sigmas   = per_min_sigma * _VOL_MULT
    shocks   = rng.normal(0, 1, n)
    log_rets = drift + sigmas * shocks

    path = open_ * np.exp(np.cumsum(np.concatenate([[0], log_rets])))

    path_min, path_max = path.min(), path.max()
    if path_max > path_min:
        path = low + (path - path_min) * (high - low) / (path_max - path_min)

    path[0]  = open_
    path[-1] = close
    return path


def _path_to_ohlcv(path: np.ndarray, timestamps: pd.DatetimeIndex,
                   volume: float, rng: np.random.Generator) -> pd.DataFrame:
    n = len(timestamps)
    opens   = path[:n]
    closes  = path[1:n+1]
    highs   = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.0005, n))
    lows    = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.0005, n))
    bar_vol = np.abs(closes - opens)
    bar_vol = bar_vol / bar_vol.sum() if bar_vol.sum() > 0 else np.ones(n) / n
    volumes = bar_vol * volume
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
    }, index=timestamps)


def generate_intraday(daily_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return a 1-minute OHLCV DataFrame for all days in daily_df."""
    rng = np.random.default_rng(seed)
    frames = []
    for date, row in daily_df.iterrows():
        date = pd.Timestamp(date).normalize()
        start_ts   = date.replace(hour=0, minute=0, second=0)
        timestamps = pd.date_range(start=start_ts, periods=1440, freq="1min")
        path = _generate_day_path(
            float(row["Open"]), float(row["High"]),
            float(row["Low"]),  float(row["Close"]), rng,
        )
        vol = float(row.get("Volume", 100_000))
        frames.append(_path_to_ohlcv(path, timestamps, vol, rng))
    df = pd.concat(frames).sort_index()
    df.index.name = "timestamp"
    return df


def resample_ohlcv(df_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1-minute bars to the given pandas offset alias."""
    return df_1m.resample(freq).agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
