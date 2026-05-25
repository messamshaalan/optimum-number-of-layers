"""Pure pandas/numpy technical indicators -- no TA-Lib dependency."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = ema(series, fast) - ema(series, slow)
    s = ema(m, signal)
    return m, s, m - s


def bollinger(series: pd.Series, period: int = 20,
              std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    bw    = (upper - lower) / mid
    return upper, mid, lower, bw


def atr(high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, d: int = 3,
               smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    raw_k = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw_k.rolling(smooth).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def adx(high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 14) -> pd.Series:
    up   = high.diff()
    down = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_val   = atr(high, low, close, period)
    plus_di  = 100 * pd.Series(plus_dm,  index=close.index).ewm(com=period-1, adjust=False).mean() / tr_val
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(com=period-1, adjust=False).mean() / tr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()


def swing_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[List[float], List[float]]:
    if len(df) < lookback * 2 + 1:
        return [], []
    highs = df["High"].values
    lows  = df["Low"].values
    supports, resistances = [], []
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i - lookback: i + lookback + 1]):
            resistances.append(highs[i])
        if lows[i]  == min(lows[i  - lookback: i + lookback + 1]):
            supports.append(lows[i])
    return supports, resistances


def fibonacci_levels(swing_high: float, swing_low: float) -> Dict[float, float]:
    rng = swing_high - swing_low
    return {
        0.000: swing_high,
        0.236: swing_high - 0.236 * rng,
        0.382: swing_high - 0.382 * rng,
        0.500: swing_high - 0.500 * rng,
        0.618: swing_high - 0.618 * rng,
        0.786: swing_high - 0.786 * rng,
        1.000: swing_low,
    }


def detect_pin_bar(df: pd.DataFrame) -> pd.Series:
    body   = (df["Close"] - df["Open"]).abs()
    upper  = df["High"] - df[["Open", "Close"]].max(axis=1)
    lower  = df[["Open", "Close"]].min(axis=1) - df["Low"]
    total_range = df["High"] - df["Low"]
    is_pin = (
        ((upper >= 2 * body) | (lower >= 2 * body))
        & (total_range > 0)
        & (body < 0.4 * total_range)
    )
    return is_pin.fillna(False)


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    curr_high = df[["Open", "Close"]].max(axis=1)
    curr_low  = df[["Open", "Close"]].min(axis=1)
    prev_high = df[["Open", "Close"]].max(axis=1).shift(1)
    prev_low  = df[["Open", "Close"]].min(axis=1).shift(1)
    bullish = (df["Close"] > df["Open"]) & (curr_high > prev_high) & (curr_low < prev_low)
    bearish = (df["Close"] < df["Open"]) & (curr_high > prev_high) & (curr_low < prev_low)
    return (bullish | bearish).fillna(False)
