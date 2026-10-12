import math

import numpy as np
import pytest

from honest_backtest import stats as S


def test_sharpe_stats_basic():
    sr, sk, ku, n = S.sharpe_stats([1.0, -1.0, 1.0, -1.0, 2.0])
    a = np.array([1.0, -1.0, 1.0, -1.0, 2.0])
    assert n == 5
    assert sr == pytest.approx(a.mean() / a.std(ddof=1))
    assert S.sharpe_stats([1.0])[0] is None
    assert S.sharpe_stats([0.5, 0.5, 0.5])[0] is None          # zero variance


def test_psr_symmetric_and_monotone():
    # SR == SR* -> PSR = 0.5
    assert S.psr_from_stats(0.1, 200, 0.0, 3.0, sr_star=0.1) == pytest.approx(0.5)
    # more trades with the same SR -> more confidence
    assert S.psr_from_stats(0.1, 400) > S.psr_from_stats(0.1, 100) > 0.5
    # negative SR -> PSR(0) < 0.5
    assert S.psr_from_stats(-0.1, 100) < 0.5
    # normal case closed form: Phi(SR sqrt(n-1) / sqrt(1 + SR^2/2))
    sr, n = 0.1, 101
    expect = S.norm_cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + sr * sr / 2))
    assert S.psr_from_stats(sr, n) == pytest.approx(expect)


def test_expected_max_sr_grows_with_trials():
    v = 1 / 99
    assert S.expected_max_sr(1, v) == 0.0
    vals = [S.expected_max_sr(k, v) for k in (2, 10, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # sanity: for N=100 trials and V=1, SR0 is about 2.5 standard deviations
    assert 2.3 < S.expected_max_sr(100, 1.0) < 2.7


def test_dsr_equals_psr_for_one_trial_and_drops_with_more():
    rng = np.random.default_rng(0)
    r = rng.normal(0.1, 1.0, 400)
    d1, sr0_1 = S.dsr(r, 1)
    assert sr0_1 == 0.0
    assert d1 == pytest.approx(S.psr(r))
    d100, sr0_100 = S.dsr(r, 100)
    assert sr0_100 > 0 and d100 < d1


def test_dsr_noise_strategy_rarely_passes():
    """Best of 50 pure-noise strategies looks good on PSR but DSR deflates it."""
    rng = np.random.default_rng(42)
    trials = [rng.normal(0.0, 1.0, 300) for _ in range(50)]
    best = max(trials, key=lambda x: x.mean() / x.std(ddof=1))
    assert S.psr(best) > 0.9                     # looks "significant" alone
    d, _ = S.dsr(best, 50)
    assert d < 0.95                              # but not after accounting for 50 trials


def test_min_track_record():
    assert S.min_track_record(sr=0.0) == math.inf
    assert S.min_track_record(sr=-0.2) == math.inf
    m = S.min_track_record(sr=0.1)
    # normal case: 1 + (1 + SR^2/2) (1.645/0.1)^2 ~ 272
    assert 265 < m < 280
    # at exactly MinTRL trades, PSR(0) ~ 0.95
    assert S.psr_from_stats(0.1, int(math.ceil(m))) == pytest.approx(0.95, abs=0.002)
    assert S.min_track_record_dsr(0.1, 60) > m


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    rng = np.random.default_rng(3)
    r = rng.normal(0.2, 1.0, 500)
    lo, hi = S.bootstrap_ci(r)
    assert lo < r.mean() < hi
    assert (lo, hi) == S.bootstrap_ci(r)
    assert S.bootstrap_ci([1.0]) == (None, None)


def test_pbo_detects_noise():
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, (240, 20))
    assert S.pbo_cscv(noise, S=8)["pbo"] > 0.3     # selection on noise is often overfit
    skill = noise.copy()
    skill[:, 0] += 0.8                             # one genuinely better strategy
    assert S.pbo_cscv(skill, S=8)["pbo"] < 0.1


def test_max_drawdown():
    assert S.max_drawdown([1, 1, -3, 1, -1, 4]) == 3
    assert S.max_drawdown([1, 2, 3]) == 0
    assert S.max_drawdown([-1, -1]) == 2
