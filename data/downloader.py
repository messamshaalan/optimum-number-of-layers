"""Download and cache daily OHLCV for gold (GC=F / XAUUSD proxy)."""

import numpy as np
import pandas as pd
from pathlib import Path

import config

_CACHE = config.CACHE_DIR / "gold_daily.csv"


def download_gold_daily(start: str = config.DATA_START,
                        end: str   = config.DATA_END) -> pd.DataFrame:
    """Return daily OHLCV DataFrame indexed by date. Uses cache when available."""
    if _CACHE.exists():
        df = pd.read_csv(_CACHE, index_col=0, parse_dates=True)
        need_start = pd.Timestamp(start)
        need_end   = pd.Timestamp(end)
        if df.index.min() <= need_start and df.index.max() >= need_end:
            print(f"[data] Loaded {len(df)} daily bars from cache.")
            return df

    print(f"[data] Downloading {config.TICKER} daily data {start} -> {end} ...")
    try:
        import yfinance as yf
        raw = yf.download(config.TICKER, start=start, end=end,
                          interval="1d", progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("yfinance returned empty data")
        df = _clean(raw)
        print(f"[data] Downloaded {len(df)} daily bars.")
    except Exception as exc:
        print(f"[data] Download failed ({exc}). Generating synthetic fallback data.")
        df = _synthetic_fallback(start, end)

    df.to_csv(_CACHE)
    return df


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df[df["Close"] > 0].dropna()
    return df.sort_index()


def _synthetic_fallback(start: str, end: str) -> pd.DataFrame:
    """Geometric Brownian Motion daily bars seeded from known gold statistics."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, end=end, freq="B")
    n = len(dates)

    annual_vol  = 0.18
    annual_ret  = 0.06
    dt          = 1 / 252
    sigma       = annual_vol * np.sqrt(dt)
    mu          = (annual_ret - 0.5 * annual_vol**2) * dt

    S0 = 1060.0
    log_returns = rng.normal(mu, sigma, n)
    closes = S0 * np.exp(np.cumsum(log_returns))

    daily_range = closes * rng.uniform(0.004, 0.012, n)
    highs  = closes + daily_range * rng.uniform(0.3, 0.7, n)
    lows   = closes - daily_range * rng.uniform(0.3, 0.7, n)
    opens  = np.roll(closes, 1)
    opens[0] = S0
    volumes = rng.integers(50_000, 200_000, n).astype(float)

    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": closes, "Volume": volumes,
    }, index=dates)
