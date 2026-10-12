import numpy as np
import pandas as pd

from honest_backtest.lookahead import (check_htf_rows, closed_upto, lookahead_analysis, sample_indices,
                                       signal_lookahead, warmup_analysis)

H = 3_600_000


def prices(n=400, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"start": np.arange(n) * H, "open": c, "high": c * 1.002, "low": c * 0.998, "close": c})


def test_clean_indicators_pass():
    df = prices()
    idx = sample_indices(50, len(df) - 1, 30)
    for fn in (lambda d: d["close"].ewm(span=20, adjust=False).mean(),
               lambda d: d["close"].rolling(14).mean(),
               lambda d: d["close"].pct_change().shift(1),
               lambda d: pd.DataFrame({"a": d["close"].diff(), "b": d["close"] > d["close"].rolling(10).max().shift(1)})):
        r = lookahead_analysis(fn, df, idx)
        assert r["n"] == len(idx) and r["mismatch"] == 0, r


def test_leaky_indicators_fail():
    df = prices()
    idx = sample_indices(50, len(df) - 20, 30)
    leaks = (lambda d: d["close"].rolling(21, center=True, min_periods=1).mean(),   # centred window
             lambda d: d["close"].shift(-1),                                       # next bar
             lambda d: (d["close"] - d["close"].mean()) / d["close"].std(),        # full-sample z-score
             lambda d: d["close"].bfill())                                         # harmless here: no NaN
    expected = [True, True, True, False]
    for fn, should_fail in zip(leaks, expected):
        r = lookahead_analysis(fn, df, idx)
        assert (r["mismatch"] > 0) is should_fail, r


def test_signal_lookahead_pipeline():
    df = prices()
    full = {int(i): int(df["close"].iloc[i] > df["close"].iloc[max(0, i - 5):i + 6].mean()) for i in range(10, 300)}
    honest = lambda i: int(df["close"].iloc[i] > df["close"].iloc[max(0, i - 5):i + 1].mean())
    r = signal_lookahead(full, honest, range(10, 300))
    assert r["mismatch"] > 0                       # the "full" pipeline peeked 5 bars ahead


def test_warmup_drift():
    df = prices(800)
    w = warmup_analysis(lambda d: float(d["close"].ewm(span=200, adjust=False).mean().iloc[-1]),
                        df, range(600, 800, 20), [50, 600])
    assert w["w50_vs_w600"]["fields"]["value"]["max_rel"] > 0.001    # short warm-up -> different EMA
    w2 = warmup_analysis(lambda d: float(d["close"].iloc[-10:].mean()), df, range(600, 800, 20), [50, 600])
    assert w2["w50_vs_w600"]["fields"]["value"]["max_rel"] < 1e-12  # finite window -> no drift


def test_multi_timeframe_rows():
    ltf = prices(48)                               # 1h candles
    h4 = ltf.assign(g=ltf["start"] // (4 * H)).groupby("g").agg(
        start=("start", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last")).reset_index(drop=True)
    t = 10 * H                                     # evaluating at 10:00, 4h candle 08:00-12:00 still open
    assert len(closed_upto(h4, 4 * H, t)) == 2     # 00-04 and 04-08 closed
    issues = check_htf_rows(h4, 4 * H, t, ltf, H)
    assert any("after t" in x["why"] for x in issues)        # full 08-12 candle leaks 10:00-12:00 data
    ok = closed_upto(h4, 4 * H, t)
    assert check_htf_rows(ok, 4 * H, t, ltf, H) == []
