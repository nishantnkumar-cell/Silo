"""Luck checks for Sharpe ratios.

* Probabilistic Sharpe ratio (PSR): probability the true Sharpe exceeds a
  benchmark, given sample length, skew and fat tails.
* Deflated Sharpe ratio (DSR): PSR against the Sharpe you would expect from
  the best of N tries on pure noise, so trying many variants is penalised.
  (Bailey and Lopez de Prado, 2014.)
* Block-bootstrap confidence interval for the annualised Sharpe.

All Sharpe inputs here are per-period (daily), not annualised, unless named.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


def sharpe(returns: pd.Series | np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def probabilistic_sharpe(returns: pd.Series | np.ndarray, benchmark_sr: float = 0.0) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return float("nan")
    sr = sharpe(r)
    z = (r - r.mean()) / r.std(ddof=0) if r.std(ddof=0) > 0 else np.zeros_like(r)
    skew = float((z**3).mean())
    kurt = float((z**4).mean())  # Pearson kurtosis, 3 for a normal
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return float(_N.cdf((sr - benchmark_sr) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(trial_sharpes: list[float]) -> float:
    """Sharpe the best of N noise-only trials would reach by luck alone."""
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    sd = float(np.std(trial_sharpes, ddof=1))
    return sd * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n) + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n * math.e))
    )


def deflated_sharpe(returns: pd.Series | np.ndarray, trial_sharpes: list[float]) -> float:
    """Probability the strategy's Sharpe beats the best-of-N luck threshold."""
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes))


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    n_boot: int = 1000,
    block: int = 20,
    alpha: float = 0.05,
    seed: int = 0,
    periods: int = TRADING_DAYS,
) -> tuple[float, float]:
    """Circular block bootstrap interval for the annualised Sharpe."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n] % n
    samples = r[idx]
    sd = samples.std(axis=1, ddof=1)
    srs = np.where(sd > 0, samples.mean(axis=1) / np.where(sd > 0, sd, 1), 0.0) * math.sqrt(periods)
    lo, hi = np.quantile(srs, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))
