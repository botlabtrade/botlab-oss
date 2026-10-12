"""Overfitting-aware statistics for a series of per-trade returns.

All Sharpe ratios here are *per trade* (or per period) and are NOT annualised:
``SR = mean(r) / std(r, ddof=1)``.

References
----------
* Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier" (PSR, MinTRL).
* Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio" (DSR, False Strategy Theorem).
* Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of Backtest Overfitting" (PBO / CSCV).
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

_N = NormalDist()
norm_cdf = _N.cdf
norm_ppf = _N.inv_cdf

EULER_GAMMA = 0.5772156649015329


def _clean(r: Iterable[float]) -> np.ndarray:
    return np.asarray([x for x in r if x is not None and math.isfinite(x)], dtype=float)


def sharpe_stats(r: Iterable[float]) -> Tuple[Optional[float], float, float, int]:
    """Return ``(SR, skew, kurtosis, n)``.

    Kurtosis is the plain (non-excess) kurtosis, so a normal distribution gives 3.
    SR is ``None`` when ``n < 2`` or the standard deviation is zero.
    """
    a = _clean(r)
    n = len(a)
    if n < 2:
        return None, 0.0, 3.0, n
    sd = a.std(ddof=1)
    if sd <= 0:
        return None, 0.0, 3.0, n
    m = a.mean()
    z = (a - m) / a.std(ddof=0)
    return float(m / sd), float((z ** 3).mean()), float((z ** 4).mean()), n


def _den(sr: float, skew: float, kurt: float) -> float:
    return max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-6)


def psr_from_stats(sr: Optional[float], n: int, skew: float = 0.0, kurt: float = 3.0,
                   sr_star: float = 0.0) -> Optional[float]:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star) given the observed SR, n, skew and kurtosis.

    ``PSR = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))``
    """
    if sr is None or n < 2:
        return None
    z = (sr - sr_star) * math.sqrt(max(n - 1, 1)) / math.sqrt(_den(sr, skew, kurt))
    return float(norm_cdf(z))


def psr(r: Iterable[float], sr_star: float = 0.0) -> Optional[float]:
    sr, sk, ku, n = sharpe_stats(r)
    return psr_from_stats(sr, n, sk, ku, sr_star)


def expected_max_sr(n_trials: int, var_sr: float, mean_sr: float = 0.0) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` strategies with zero true skill.

    ``SR0 = mean + sqrt(V) * ((1 - g) Z^-1(1 - 1/N) + g Z^-1(1 - 1/(N e)))`` (False Strategy Theorem).
    """
    n_trials = max(int(n_trials), 1)
    if n_trials == 1:
        return float(mean_sr)
    return float(mean_sr + math.sqrt(max(var_sr, 0.0)) * (
        (1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e))))


def null_var_sr(n: int) -> float:
    """Variance of the estimated per-trade SR when the true SR is 0. Used when the spread of SRs
    across your trials is unknown."""
    return 1.0 / max(n - 1, 1)


def dsr_from_stats(sr: Optional[float], n: int, skew: float, kurt: float, n_trials: int,
                   var_sr: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(DSR, SR0)``. DSR = PSR evaluated against the best SR expected from luck alone."""
    if sr is None or n < 2:
        return None, None
    v = null_var_sr(n) if var_sr is None else var_sr
    sr0 = expected_max_sr(n_trials, v)
    return psr_from_stats(sr, n, skew, kurt, sr0), sr0


def dsr(r: Iterable[float], n_trials: int, var_sr: Optional[float] = None
        ) -> Tuple[Optional[float], Optional[float]]:
    """Deflated Sharpe Ratio for ``r`` after ``n_trials`` strategies/variants were tried."""
    sr, sk, ku, n = sharpe_stats(r)
    return dsr_from_stats(sr, n, sk, ku, n_trials, var_sr)


def min_track_record(r: Optional[Iterable[float]] = None, sr_star: float = 0.0, prob: float = 0.95,
                     sr: Optional[float] = None, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Minimum Track Record Length: number of trades needed for PSR(sr_star) >= prob.

    ``MinTRL = 1 + (1 - g3 SR + (g4 - 1)/4 SR^2) (Z_prob / (SR - SR*))^2``. Returns ``inf`` if SR <= SR*.
    """
    if r is not None:
        sr, skew, kurt, _ = sharpe_stats(r)
    if sr is None or sr <= sr_star:
        return math.inf
    return float(1.0 + _den(sr, skew, kurt) * (norm_ppf(prob) / (sr - sr_star)) ** 2)


def min_track_record_dsr(sr: Optional[float], n_trials: int, skew: float = 0.0, kurt: float = 3.0,
                         prob: float = 0.95) -> float:
    """Trades needed for DSR >= prob when V[SR] = 1/(n-1) (so SR0 shrinks as n grows)."""
    if sr is None or sr <= 0:
        return math.inf
    k = expected_max_sr(n_trials, 1.0)
    return float(1.0 + ((norm_ppf(prob) * math.sqrt(_den(sr, skew, kurt)) + k) / sr) ** 2)


def bootstrap_ci(r: Iterable[float], conf: float = 0.90, n_boot: int = 5000, seed: int = 12345
                 ) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap confidence interval of the mean. Fixed seed -> reproducible."""
    a = _clean(r)
    if len(a) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    q = (1 - conf) / 2
    return float(np.quantile(means, q)), float(np.quantile(means, 1 - q))


def pbo_cscv(M: Sequence[Sequence[float]], S: int = 8, metric=None) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

    ``M`` is a (T time slices x N strategies) matrix of returns. Rows are split into ``S`` contiguous
    blocks; each combination of S/2 blocks is in-sample, the rest out-of-sample. PBO is the share of
    combinations where the in-sample winner ranks in the bottom half out-of-sample.
    """
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if S % 2 or S < 2:
        raise ValueError("S must be even and >= 2")
    if N < 2 or T < S:
        raise ValueError("need >= 2 strategies and T >= S")

    def _sharpe(X):
        sd = X.std(axis=0, ddof=1)
        return X.mean(axis=0) / np.where(sd > 0, sd, np.inf)

    metric = metric or _sharpe
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for is_idx in combinations(range(S), S // 2):
        oos_idx = [i for i in range(S) if i not in is_idx]
        m_is = metric(M[np.concatenate([blocks[i] for i in is_idx])])
        m_oos = metric(M[np.concatenate([blocks[i] for i in oos_idx])])
        best = int(np.argmax(m_is))
        rank = float((m_oos < m_oos[best]).sum() + 0.5 * ((m_oos == m_oos[best]).sum() - 1) + 1)
        w = rank / (N + 1)
        logits.append(math.log(w / (1 - w)))
    la = np.asarray(logits)
    return dict(pbo=float((la <= 0).mean()), logits=la.tolist(), n_combos=len(la),
                median_logit=float(np.median(la)))


def max_drawdown(r: Iterable[float]) -> float:
    """Max drawdown of the cumulative sum of ``r`` (starting from 0), in the same unit as ``r``."""
    peak = cum = dd = 0.0
    for x in r:
        cum += x
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd
