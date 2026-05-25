from .base import Signal, BaseStrategy
from .ema_crossover import EMACrossoverStrategy
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .bollinger_strategy import BollingerStrategy
from .london_breakout import LondonBreakoutStrategy
from .sr_bounce import SRBounceStrategy
from .fibonacci_strategy import FibonacciStrategy
from .stochastic_strategy import StochasticStrategy
from .atr_breakout import ATRBreakoutStrategy
from .price_action import PriceActionStrategy

ALL_STRATEGIES = [
    EMACrossoverStrategy(),
    RSIStrategy(),
    MACDStrategy(),
    BollingerStrategy(),
    LondonBreakoutStrategy(),
    SRBounceStrategy(),
    FibonacciStrategy(),
    StochasticStrategy(),
    ATRBreakoutStrategy(),
    PriceActionStrategy(),
]

__all__ = [
    "Signal", "BaseStrategy", "ALL_STRATEGIES",
    "EMACrossoverStrategy", "RSIStrategy", "MACDStrategy",
    "BollingerStrategy", "LondonBreakoutStrategy", "SRBounceStrategy",
    "FibonacciStrategy", "StochasticStrategy", "ATRBreakoutStrategy",
    "PriceActionStrategy",
]
