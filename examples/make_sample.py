"""Generate examples/sample_trades.csv - SYNTHETIC trades (random numbers, not a real strategy).

    python examples/make_sample.py

420 trades over 20 made-up symbols, both sides, a small positive edge (~+0.1R per trade before luck).
Useful to see how the same trade list looks with --trials 1 versus --trials 60.
"""
import csv
import datetime as dt
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def make(path=os.path.join(HERE, "sample_trades.csv"), n=420, seed=31, p_win=0.45):
    rng = random.Random(seed)
    syms = [f"SYN{i:02d}USDT" for i in range(1, 21)]
    t = dt.datetime(2025, 1, 2, 8, 0)
    rows = []
    for _ in range(n):
        t += dt.timedelta(hours=rng.uniform(8, 64))
        win = rng.random() < p_win
        r = rng.gauss(1.5, 0.25) if win else rng.gauss(-1.0, 0.12)
        rows.append(dict(closed=t.strftime("%Y-%m-%d %H:%M:%S"), symbol=rng.choice(syms),
                         side=rng.choice(["long", "short"]), R=round(r, 4)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["closed", "symbol", "side", "R"])
        w.writeheader()
        w.writerows(rows)
    return path


if __name__ == "__main__":
    print("wrote", make())
