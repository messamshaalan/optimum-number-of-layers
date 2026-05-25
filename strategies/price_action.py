"""Price Action Strategy — Enhanced with regime filter, session multiplier,
and pattern quality scoring.

Signal logic (unchanged core):
  - Detect pin bars and engulfing candles on all three timeframes
  - Require ≥ 2 of 3 timeframes to agree on direction
  - Apply EMA 50 trend filter on 15 M

Enhancements:
  - Regime filter: wider TP in trending markets, tighter in ranging
  - Session multiplier: confidence scaled by historical per-session win rate
  - Pattern quality score: rewards strong wicks/bodies, penalises marginal patterns
  - Transitioning regime: 10 % confidence haircut (lower conviction)
"""

import pandas as pd
from .base import BaseStrategy, Signal
from indicators.technical import detect_pin_bar, detect_engulfing, ema, atr
from indicators.regime import classify_regime, TRENDING, RANGING, TRANSITIONING
import config


class PriceActionStrategy(BaseStrategy):
    name = "Price_Action"
    # Trades in all regimes but adjusts TP and confidence accordingly
    regime_preferences = [TRENDING, RANGING, TRANSITIONING]

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < 10 or len(tf15m) < 10:
            return self._no_signal()

        entry = float(tf1m["Close"].iloc[-1])

        # --- Regime & session ---
        regime  = classify_regime(tf15m)
        session = _hour_to_session(timestamp.hour)

        # --- Multi-timeframe pattern voting ---
        votes   = {}
        quality = {}

        for label, df in [("15M", tf15m), ("5M", tf5m), ("1M", tf1m)]:
            if len(df) < 5:
                votes[label]   = "NONE"
                quality[label] = 0.0
                continue

            pin = detect_pin_bar(df)
            eng = detect_engulfing(df)
            c   = df["Close"]
            o   = df["Open"]

            has_bull_pin = any(pin.iloc[-2:]) and any(c.iloc[-2:] > o.iloc[-2:])
            has_bear_pin = any(pin.iloc[-2:]) and any(c.iloc[-2:] < o.iloc[-2:])
            has_bull_eng = any(eng.iloc[-2:]) and float(c.iloc[-1]) > float(o.iloc[-1])
            has_bear_eng = any(eng.iloc[-2:]) and float(c.iloc[-1]) < float(o.iloc[-1])

            qual = _pattern_quality(df)

            if has_bull_pin or has_bull_eng:
                votes[label]   = "BUY"
                quality[label] = qual
            elif has_bear_pin or has_bear_eng:
                votes[label]   = "SELL"
                quality[label] = qual
            else:
                votes[label]   = "NONE"
                quality[label] = 0.0

        # --- EMA 50 trend filter on 15 M ---
        if len(tf15m) >= 55:
            e50   = ema(tf15m["Close"], config.EMA_TREND)
            trend = "BUY" if entry > float(e50.iloc[-1]) else "SELL"
            if votes.get("15M") != trend:
                votes["15M"]   = "NONE"
                quality["15M"] = 0.0

        direction = _dominant(votes)
        if direction == "NONE":
            return self._no_signal(entry)

        atr5   = float(atr(tf5m["High"], tf5m["Low"], tf5m["Close"],
                           config.ATR_PERIOD).iloc[-1])
        sl, tp = _sl_tp(entry, direction, atr5, regime)
        conf   = _confidence(votes, direction, self.weight, quality, regime)

        # Session multiplier from learned history
        conf = min(conf * self.session_multiplier(session), 1.0)

        sig         = Signal(direction, conf, entry, sl, tp, self.name, votes, atr5)
        sig.regime  = regime
        sig.session = session
        return sig


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _dominant(votes: dict) -> str:
    buy  = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")
    if buy  >= 2: return "BUY"
    if sell >= 2: return "SELL"
    return "NONE"


def _sl_tp(entry: float, direction: str, atr5: float, regime: str):
    sl_mult = config.SL_ATR_MULT
    # Trending: let profits run (+20 % TP). Ranging: tighter TP (-10 %).
    tp_mult = config.TP_ATR_MULT * (1.2 if regime == TRENDING
                                    else 0.9 if regime == RANGING
                                    else 1.0)
    if direction == "BUY":
        return entry - sl_mult * atr5, entry + tp_mult * atr5
    return entry + sl_mult * atr5, entry - tp_mult * atr5


def _confidence(votes: dict, direction: str, weight: float,
                quality: dict, regime: str) -> float:
    agree    = sum(1 for v in votes.values() if v == direction)
    avg_qual = (sum(quality.get(k, 0.0) for k, v in votes.items() if v == direction)
                / max(agree, 1))

    base = min(0.45 + 0.20 * agree, 1.0)
    base = min(base + 0.10 * avg_qual, 1.0)   # quality bonus ≤ +0.10

    # Reduce conviction in transitioning regime
    if regime == TRANSITIONING:
        base *= 0.90

    return base * min(weight, 1.0)


def _pattern_quality(df: pd.DataFrame) -> float:
    """Score 0–1: 1.0 for ideal pin bar or engulfing, lower for mediocre patterns."""
    if len(df) < 1:
        return 0.5
    bar   = df.iloc[-1]
    total = bar["High"] - bar["Low"]
    if total < 1e-6:
        return 0.0
    body     = abs(bar["Close"] - bar["Open"])
    body_pct = body / total
    if body_pct < 0.30:
        # Pin bar: reward tiny body with large wick
        return 1.0 - body_pct / 0.30 * 0.25
    if body_pct > 0.60:
        # Engulfing: reward large body
        return (body_pct - 0.60) / 0.40
    return 0.30   # mediocre


def _hour_to_session(hour: int) -> str:
    for name, (start, end) in config.SESSIONS.items():
        if start <= hour < end:
            return name
    return "Rollover"
