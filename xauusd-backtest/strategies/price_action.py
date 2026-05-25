"""Price Action Strategy — fully enhanced.

Improvements implemented from backtest learning:
  1. Hard block NY + Rollover (below breakeven win rate)
  2. Require 3/3 TF agreement in LondonOpen (reduce false breakouts)
  3. Session-specific TP targets (Asia 3.5×, LondonOpen 2.0×, etc.)
  4. Regime-adjusted TP (trending +20%, ranging -10%)
  5. Session multiplier from learned win-rate history
  6. Pattern quality scoring (rewards strong wicks/clean bodies)
  7. Pattern type logging (pin_bar / engulfing / both) for tracking
"""

import pandas as pd
from .base import BaseStrategy, Signal
from indicators.technical import detect_pin_bar, detect_engulfing, ema, atr
from indicators.regime import classify_regime, TRENDING, RANGING, TRANSITIONING
import config


class PriceActionStrategy(BaseStrategy):
    name = "Price_Action"
    regime_preferences = [TRENDING, RANGING, TRANSITIONING]

    def generate_signal(self, tf1m, tf5m, tf15m, timestamp) -> Signal:
        if len(tf5m) < 10 or len(tf15m) < 10:
            return self._no_signal()

        entry   = float(tf1m["Close"].iloc[-1])
        regime  = classify_regime(tf15m)
        session = _hour_to_session(timestamp.hour)

        # ---- 1. Hard block bad sessions ----
        if session in config.PA_BLOCKED_SESSIONS:
            return self._no_signal(entry)

        # ---- 2. Multi-timeframe pattern voting ----
        votes        = {}
        quality      = {}
        pattern_tags = {}

        for label, df in [("15M", tf15m), ("5M", tf5m), ("1M", tf1m)]:
            if len(df) < 5:
                votes[label]        = "NONE"
                quality[label]      = 0.0
                pattern_tags[label] = ""
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
                if has_bull_pin and has_bull_eng:
                    pattern_tags[label] = "both"
                elif has_bull_pin:
                    pattern_tags[label] = "pin_bar"
                else:
                    pattern_tags[label] = "engulfing"
            elif has_bear_pin or has_bear_eng:
                votes[label]   = "SELL"
                quality[label] = qual
                if has_bear_pin and has_bear_eng:
                    pattern_tags[label] = "both"
                elif has_bear_pin:
                    pattern_tags[label] = "pin_bar"
                else:
                    pattern_tags[label] = "engulfing"
            else:
                votes[label]        = "NONE"
                quality[label]      = 0.0
                pattern_tags[label] = ""

        # ---- 3. EMA50 trend filter on 15M ----
        if len(tf15m) >= 55:
            e50   = ema(tf15m["Close"], config.EMA_TREND)
            trend = "BUY" if entry > float(e50.iloc[-1]) else "SELL"
            if votes.get("15M") != trend:
                votes["15M"]        = "NONE"
                quality["15M"]      = 0.0
                pattern_tags["15M"] = ""

        direction = _dominant(votes)
        if direction == "NONE":
            return self._no_signal(entry)

        # ---- 4. Strict 3/3 agreement in LondonOpen ----
        if session in config.PA_STRICT_SESSIONS:
            agree_count = sum(1 for v in votes.values() if v == direction)
            if agree_count < 3:
                return self._no_signal(entry)

        # ---- 5. Pattern type summary (for logging) ----
        agreeing_tags = [pattern_tags[k] for k, v in votes.items() if v == direction]
        all_tags = set(t for t in agreeing_tags if t)
        if len(all_tags) > 1:
            pattern_type = "both"
        elif all_tags:
            pattern_type = all_tags.pop()
        else:
            pattern_type = ""

        # ---- 6. SL/TP with session + regime ----
        atr5   = float(atr(tf5m["High"], tf5m["Low"], tf5m["Close"],
                           config.ATR_PERIOD).iloc[-1])
        sl, tp = _sl_tp(entry, direction, atr5, regime, session)

        # ---- 7. Confidence ----
        conf = _confidence(votes, direction, quality, regime)
        conf = min(conf * self.session_multiplier(session), 1.0)

        sig              = Signal(direction, conf, entry, sl, tp, self.name, votes, atr5)
        sig.regime       = regime
        sig.session      = session
        sig.pattern_type = pattern_type
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


def _sl_tp(entry: float, direction: str, atr5: float, regime: str, session: str):
    sl_mult  = config.SL_ATR_MULT
    sess_tp  = config.SESSION_TP_MULT.get(session, config.TP_ATR_MULT)
    # Regime modifier on top of session-specific TP
    reg_mult = 1.2 if regime == TRENDING else 0.9 if regime == RANGING else 1.0
    tp_mult  = sess_tp * reg_mult
    if direction == "BUY":
        return entry - sl_mult * atr5, entry + tp_mult * atr5
    return entry + sl_mult * atr5, entry - tp_mult * atr5


def _confidence(votes: dict, direction: str, quality: dict, regime: str) -> float:
    agree    = sum(1 for v in votes.values() if v == direction)
    avg_qual = (sum(quality.get(k, 0.0) for k, v in votes.items() if v == direction)
                / max(agree, 1))
    base = min(0.45 + 0.20 * agree, 1.0)
    base = min(base + 0.10 * avg_qual, 1.0)
    if regime == TRANSITIONING:
        base *= 0.90
    # Confidence reflects signal quality (TF agreement, pattern quality, regime).
    # Strategy weight is used in multi-strategy ensemble weighting, not here.
    return base


def _pattern_quality(df: pd.DataFrame) -> float:
    if len(df) < 1:
        return 0.5
    bar   = df.iloc[-1]
    total = bar["High"] - bar["Low"]
    if total < 1e-6:
        return 0.0
    body     = abs(bar["Close"] - bar["Open"])
    body_pct = body / total
    if body_pct < 0.30:
        return 1.0 - body_pct / 0.30 * 0.25
    if body_pct > 0.60:
        return (body_pct - 0.60) / 0.40
    return 0.30


def _hour_to_session(hour: int) -> str:
    for name, (start, end) in config.SESSIONS.items():
        if start <= hour < end:
            return name
    return "Rollover"
