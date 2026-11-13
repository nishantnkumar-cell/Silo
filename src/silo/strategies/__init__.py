from silo.strategies.base import Strategy
from silo.strategies.classic import (
    BollingerFade,
    BuyAndHold,
    DonchianBreakout,
    Momentum12_1,
    Rsi2MeanReversion,
    SmaCross,
    TrendQuality,
    TrendVolTarget,
    VolTarget,
)
from silo.strategies.random_entry import RandomEntry

REGISTRY = {
    "buy_and_hold": BuyAndHold,
    "sma_cross": SmaCross,
    "donchian": DonchianBreakout,
    "rsi2": Rsi2MeanReversion,
    "bollinger": BollingerFade,
    "momentum": Momentum12_1,
    "uk_trend": TrendQuality,
    "tsmom": TrendVolTarget,
    "random": RandomEntry,
}

from silo.agents.blend import Blend, SiloAgent  # noqa: E402 - these build members from REGISTRY
from silo.agents.contestants import DebateDesk, StylePanel  # noqa: E402

REGISTRY.update({"desk": DebateDesk, "panel": StylePanel, "blend": Blend, "silo_agent": SiloAgent})

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
    "TrendQuality",
    "TrendVolTarget",
    "RandomEntry",
]
