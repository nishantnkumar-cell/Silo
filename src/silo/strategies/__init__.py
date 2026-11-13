from silo.strategies.base import Strategy
from silo.strategies.classic import (
    BollingerFade,
    BuyAndHold,
    DonchianBreakout,
    Momentum12_1,
    Rsi2MeanReversion,
    SmaCross,
    TrendQuality,
    VolTarget,
)
from silo.strategies.crossasset import MacroTilt, SectorMomentum
from silo.strategies.news import DirectorBuys, NewsMomentum, ResultsDrift
from silo.strategies.random_entry import RandomEntry

REGISTRY = {
    "buy_and_hold": BuyAndHold,
    "sma_cross": SmaCross,
    "donchian": DonchianBreakout,
    "rsi2": Rsi2MeanReversion,
    "bollinger": BollingerFade,
    "momentum": Momentum12_1,
    "uk_trend": TrendQuality,
    "sector_mom": SectorMomentum,
    "macro_tilt": MacroTilt,
    "news_mom": NewsMomentum,
    "results_drift": ResultsDrift,
    "director_buys": DirectorBuys,
    "random": RandomEntry,
}

from silo.agents.blend import Blend, SiloAgent  # noqa: E402 - these build members from REGISTRY
from silo.agents.contestants import DebateDesk, StylePanel  # noqa: E402
from silo.agents.informed import FullDesk, FullPanel, InformedDesk, InformedPanel, NewsDesk, NewsPanel  # noqa: E402

REGISTRY.update({"desk": DebateDesk, "panel": StylePanel, "blend": Blend, "silo_agent": SiloAgent,
                 "panel_macro": InformedPanel, "desk_macro": InformedDesk, "panel_news": NewsPanel,
                 "desk_news": NewsDesk, "panel_full": FullPanel, "desk_full": FullDesk})

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
    "RandomEntry",
]
