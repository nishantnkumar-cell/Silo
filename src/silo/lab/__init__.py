"""Phase 1 backtest lab: honest evaluation of a strategy.

Walk-forward parameter selection, a locked holdout, a random-entry control,
deflated Sharpe ratio and bootstrap intervals, and an HTML report.
"""

from silo.lab.controls import RandomEntry, random_control
from silo.lab.holdout import HoldoutLog
from silo.lab.stats import bootstrap_sharpe_ci, deflated_sharpe, probabilistic_sharpe
from silo.lab.study import LabResult, run_study
from silo.lab.walkforward import WalkForwardResult, walk_forward

__all__ = [
    "HoldoutLog",
    "LabResult",
    "RandomEntry",
    "WalkForwardResult",
    "bootstrap_sharpe_ci",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "random_control",
    "run_study",
    "walk_forward",
]
