"""Market regime classification using ADX.

TRENDING      : ADX > 25  — directional move, momentum strategies preferred
RANGING       : ADX < 20  — mean-reversion strategies preferred
TRANSITIONING : 20–25     — reduce conviction; smaller position sizing
"""

import pandas as pd
from indicators.technical import adx

TRENDING      = "TRENDING"
RANGING       = "RANGING"
TRANSITIONING = "TRANSITIONING"

_ADX_TREND = 25.0
_ADX_RANGE = 20.0


def classify_regime(df_15m: pd.DataFrame, period: int = 14) -> str:
    """Classify market regime from a 15-minute OHLCV window."""
    if len(df_15m) < period * 2 + 1:
        return TRANSITIONING
    val = adx(df_15m["High"], df_15m["Low"], df_15m["Close"], period).iloc[-1]
    if pd.isna(val):
        return TRANSITIONING
    if val > _ADX_TREND:
        return TRENDING
    if val < _ADX_RANGE:
        return RANGING
    return TRANSITIONING
