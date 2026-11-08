"""Silo: a research-only AI trading lab.

Silo backtests and paper-trades strategies. It never places orders with a
real brokerage and refuses to load configuration that points at one.
"""

from silo import safety as _safety

_safety.check_environment()

__version__ = "0.1.0"
