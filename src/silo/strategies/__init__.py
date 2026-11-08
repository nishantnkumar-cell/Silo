from silo.strategies.base import Strategy
from silo.strategies.classic import (
    BollingerFade,
    BuyAndHold,
    DonchianBreakout,
    Momentum12_1,
    Rsi2MeanReversion,
    SmaCross,
    VolTarget,
)

REGISTRY = {
    "buy_and_hold": BuyAndHold,
    "sma_cross": SmaCross,
    "donchian": DonchianBreakout,
    "rsi2": Rsi2MeanReversion,
    "bollinger": BollingerFade,
    "momentum": Momentum12_1,
}

__all__ = [
    "REGISTRY",
    "Strategy",
    "BuyAndHold",
    "SmaCross",
    "DonchianBreakout",
    "Rsi2MeanReversion",
    "BollingerFade",
    "Momentum12_1",
    "VolTarget",
]
