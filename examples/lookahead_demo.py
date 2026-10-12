"""Look-ahead audit demo on a synthetic random-walk price series.

    python examples/lookahead_demo.py

An EMA only uses the past -> 0 mismatches.
A centred rolling mean (a classic bug) uses future bars -> mismatches.
"""
import numpy as np
import pandas as pd

from honest_backtest.lookahead import lookahead_analysis, sample_indices, warmup_analysis


def make_prices(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"start": np.arange(n) * 3_600_000, "close": close})


def ema_signal(df):
    return (df["close"] > df["close"].ewm(span=50, adjust=False).mean()).astype(int)


def leaky_signal(df):
    return (df["close"] > df["close"].rolling(21, center=True, min_periods=1).mean()).astype(int)


if __name__ == "__main__":
    df = make_prices()
    idx = sample_indices(100, len(df) - 1, 40)
    for name, fn in (("EMA(50) trend", ema_signal), ("centred rolling mean", leaky_signal)):
        r = lookahead_analysis(fn, df, idx)
        print(f"{name:22} checked {r['n']:3} bars, mismatches {r['mismatch']:3}"
              + ("   <-- LOOK-AHEAD" if r["mismatch"] else ""))
    w = warmup_analysis(lambda d: float(d["close"].ewm(span=200, adjust=False).mean().iloc[-1]),
                        df, range(500, 600, 10), [100, 400])
    f = w["w100_vs_w400"]["fields"]["value"]
    print(f"EMA(200) with 100 vs 400 bars of warm-up: median drift {f['med_rel']:.4%}, max {f['max_rel']:.4%}")
