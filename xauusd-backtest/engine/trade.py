from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    id: str
    strategy: str
    direction: str
    entry_time: datetime
    entry_price: float
    sl: float
    tp: float
    size_oz: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float]  = None
    pnl_usd: Optional[float]     = None
    r_multiple: Optional[float]  = None
    exit_reason: Optional[str]   = None   # "TP" | "SL" | "EOD" | "BE" (breakeven)
    session: str                 = "Unknown"
    confluence_score: float      = 0.0
    atr_at_entry: float          = 0.0
    pattern_type: str            = ""     # "pin_bar" | "engulfing" | "both" | ""
    regime: str                  = "TRANSITIONING"
    sl_distance: float           = 0.0   # initial |entry - sl|, used for breakeven calc
    breakeven_moved: bool        = False  # True once SL has been moved to entry

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def risk_usd(self) -> float:
        return abs(self.entry_price - self.sl) * self.size_oz

    def close(self, exit_time: datetime, exit_price: float, reason: str) -> None:
        self.exit_time   = exit_time
        self.exit_price  = exit_price
        self.exit_reason = reason
        if self.direction == "BUY":
            self.pnl_usd = (exit_price - self.entry_price) * self.size_oz
        else:
            self.pnl_usd = (self.entry_price - exit_price) * self.size_oz
        risk = abs(self.entry_price - (self.entry_price - self.sl_distance)) * self.size_oz
        if risk <= 0:
            risk = abs(self.entry_price - self.sl) * self.size_oz
        self.r_multiple = self.pnl_usd / risk if risk > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "strategy":        self.strategy,
            "direction":       self.direction,
            "entry_time":      str(self.entry_time),
            "entry_price":     round(self.entry_price, 2),
            "sl":              round(self.sl, 2),
            "tp":              round(self.tp, 2),
            "size_oz":         round(self.size_oz, 4),
            "exit_time":       str(self.exit_time) if self.exit_time else "",
            "exit_price":      round(self.exit_price, 2) if self.exit_price else "",
            "pnl_usd":         round(self.pnl_usd, 2) if self.pnl_usd is not None else "",
            "r_multiple":      round(self.r_multiple, 3) if self.r_multiple is not None else "",
            "exit_reason":     self.exit_reason or "",
            "session":         self.session,
            "regime":          self.regime,
            "pattern_type":    self.pattern_type,
            "confluence_score": round(self.confluence_score, 3),
            "breakeven_moved": self.breakeven_moved,
        }
