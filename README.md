# honest_backtest

**Overfitting-aware checks for trading strategies.** Six readiness gates, the Deflated Sharpe Ratio,
a transparent cost model, a look-ahead audit and a silent-death monitor - in one small Python package
with two dependencies (numpy, pandas).

This is the checking code behind [BOTLAB](https://botlab.org.uk), an "honest AI quant lab" where AI agents
build crypto trading strategies, test them with real costs and statistical gates, and publicly retire the
ones that fail. It is not a strategy, not a signal service and not financial advice.

## Why

Most trading bots you see published are overfit. The usual story:

1. Someone tries 50, 500 or 5,000 parameter sets.
2. They publish the best one, with a beautiful equity curve.
3. They don't mention the other 4,999.

The best of many random strategies *always* looks good. The Deflated Sharpe Ratio (Bailey & Lopez de
Prado, 2014) asks the right question: *is this result better than the best one you would expect from luck,
given how many things you tried?*

The other classic ways a backtest lies are also covered:

| Failure | What catches it here |
|---|---|
| Too few trades | G1: sample size >= max(MinTRL, 300) |
| Picked the best of many trials | G2: DSR with your honest trial count |
| Edge already decayed | G3: last 100 / last 30 trades |
| Profit from 1-2 coins or 1-2 lucky days | G4: concentration limits |
| One side (long or short) quietly bleeding | G5: side-leak check |
| Unbearable drawdown | G6: max drawdown < 15R |
| Fees, slippage and funding ignored | `CostModel` |
| Indicator peeks at future bars | `lookahead_analysis` |
| Bot silently stopped running | `PulseMonitor` |

For scale: in our own lab, out of 100 strategy "books" tracked on 2026-10-11 (mostly paper-trading
ledgers), **none** passed all six gates with the 300-trade floor. That is the expected outcome, and the point.

## Quickstart

```bash
pip install numpy pandas          # Python >= 3.9
git clone https://github.com/botlabtrade/botlab-oss && cd botlab-oss
python -m honest_backtest check examples/sample_trades.csv --trials 1
python -m honest_backtest check examples/sample_trades.csv --trials 60
```

`examples/sample_trades.csv` is **synthetic** (random numbers from `examples/make_sample.py`): 420 trades,
mean +0.128R. Scored as a single idea (`--trials 1`) it passes all six gates. Scored honestly as the best
of 60 tried variants, the luck-only Sharpe (0.115) is higher than the observed one (0.101), DSR drops to 0.39
and G2 fails:

```
DSR              0.389   (trials N = 60)
[FAIL] G2  significance: PSR(0) >= 0.95 and DSR >= 0.5 (strict target DSR >= 0.95)
         PSR(0)=0.98 (>= 0.95) · DSR=0.39 (>= 0.5) after 60 trial(s), luck-only SR0=0.115 vs SR=0.101
STATUS: NOT_READY   (not passed: G2)
```

Same trades, different honesty about how they were found.

### Your own trades

A CSV with one row per closed trade. Either give `R` directly (profit / amount risked, **after costs**):

```csv
closed,symbol,side,R
2025-01-03 14:00:00,BTCUSDT,long,1.42
2025-01-04 09:00:00,ETHUSDT,short,-1.03
```

or give prices and let the cost model compute R:

```csv
closed,symbol,side,entry,exit,stop,hold_hours
2025-01-03 14:00:00,BTCUSDT,long,100,103,98,12
```

```bash
python -m honest_backtest check my_trades.csv --trials 40 \
    --fee 0.00055 --slippage 0.001 --funding-8h 0.0001   # cost flags apply only when R is computed
python -m honest_backtest check my_trades.csv --trials 40 --json
python -m honest_backtest check my_trades.csv --trials 40 --strict --fail-unless-ready   # for CI
```

**`--trials` is the number that matters most.** Count every strategy, parameter set, filter, timeframe,
coin list and prompt change you tried before arriving at this one, including the ones you deleted.
If you are unsure, round up.

## The six gates

All thresholds live in `GateConfig` and are meant to be fixed **before** you look at results.

| Gate | Passes when | Why |
|---|---|---|
| **G1 sample size** | n >= max(MinTRL, 300). MinTRL = trades needed for PSR(0) = 0.95 given the observed Sharpe, skew and kurtosis. Infinite (fail) if mean R <= 0. | A few dozen trades can't separate skill from luck. |
| **G2 significance** | PSR(0) >= 0.95 **and** DSR >= 0.5 (`--strict`: DSR >= 0.95) | PSR: probability the true Sharpe > 0. DSR: same, but against the best Sharpe expected from N skill-less trials. 0.5 is a staging level; 0.95 is the real target. |
| **G3 recent** | mean R of last 100 trades > 0 and of last 30 > -0.1 | Edges decay. A strategy that made its money a year ago and loses now is not ready. |
| **G4 concentration** | top 3 symbols < 50% of positive R (summed per symbol) **and** best 2 days < 50% of total R | "Profitable" often means one coin pumped or one day was lucky. |
| **G5 side leak** | if both sides traded: no side with n >= 50 and mean R < 0 | A long-only bull market can hide a short side that loses steadily. |
| **G6 drawdown** | max drawdown of cumulative R < 15R | If you can't sit through it, you won't. |

Status: `READY_FOR_MICRO_LIVE` = all six pass (a reason to *consider* a small, monitored live test).
`WATCH` = G2 and G3 pass but something else does not (usually sample size). `NOT_READY` = the rest.
A gate that can't be scored (no symbol / time / side column) shows `----` and blocks `READY`.

## Library use

```python
from honest_backtest import Trade, evaluate, format_report, dsr, psr, min_track_record, CostModel

trades = [Trade(r=1.2, symbol="BTCUSDT", side="long", closed="2025-01-03 14:00"), ...]
print(format_report(evaluate(trades, n_trials=40)))

d, sr0 = dsr(r_list, n_trials=40)          # deflated Sharpe and the luck-only benchmark
cm = CostModel(taker_fee=0.00055, slippage=0.001, funding_8h=0.0001)
cm.r_multiple("long", entry=100, exit_=102, stop=98, hold_hours=16)
```

Modules:

* `honest_backtest.stats` - `sharpe_stats`, `psr`, `dsr`, `expected_max_sr`, `min_track_record`,
  `min_track_record_dsr`, `bootstrap_ci`, `pbo_cscv` (Probability of Backtest Overfitting), `max_drawdown`.
* `honest_backtest.gates` - `GateConfig`, `Trade`, `evaluate`, `format_report`.
* `honest_backtest.costs` - `CostModel` (taker fee per side, slippage per side, funding x hold time),
  `orderbook_slippage` (walk a book for your notional), and an **optional** `fetch_bybit_orderbook`
  (public endpoint, no key; the only network call in the package).
* `honest_backtest.lookahead` - see below.
* `honest_backtest.pulse` - see below.

### Cost model

```
cost = 2 x taker_fee + 2 x slippage + funding_8h x hold_hours / 8   (funding sign: longs pay positive funding)
R    = (side x (exit/entry - 1) - cost) / (|entry - stop| / entry)
```

Costs are a fixed fraction of notional, so they are a much larger fraction of a *tight* stop. A 0.3% stop
pays the same 0.31% round trip as a 3% stop - about 1R versus 0.1R. This is the main reason most
short-timeframe strategies die after costs.

### Look-ahead audit

```python
from honest_backtest.lookahead import lookahead_analysis, sample_indices

res = lookahead_analysis(my_indicator, df, sample_indices(200, len(df) - 1, 50))
assert res["mismatch"] == 0, res["examples"]
```

It computes your function once on the full history and again bar by bar on truncated history; any
difference means the function used data from after the bar (centred windows, `shift(-k)`, full-sample
normalisation, resampling without shifting...). Also: `warmup_analysis` (does the value depend on how much
history you load?), `closed_upto` and `check_htf_rows` (multi-timeframe: was that 4h candle really closed?).
Run `python examples/lookahead_demo.py` for a demo.

### Silent-death monitor

```python
from honest_backtest.pulse import PulseMonitor, file_fresh, windows_tasks_check

mon = PulseMonitor(
    [file_fresh("bots/trend_bot/logs", max_age_min=45, name="trend_bot"),
     windows_tasks_check(name_pattern="^Bot")],          # Windows Task Scheduler, optional
    state_path="pulse_state.json", confirm=2,
    notify=lambda text: print(text),                       # plug in Telegram/Slack/e-mail yourself
)
mon.loop(period_s=300)
```

A problem must be seen twice in a row before `notify` fires; recovery is reported once. No endpoints,
credentials or chat ids are built in - you pass callbacks.

## Honest limitations

* **The gates can't see what you didn't count.** DSR is only as good as `--trials`. Under-count and it
  flatters you.
* **Trades are treated as independent.** Overlapping positions, partial exits and grid fills are
  correlated; PSR/DSR will then be too optimistic. Aggregate to daily R first if in doubt.
* **The null variance is a simplification.** DSR uses V[SR] = 1/(n-1) unless you pass the variance of
  Sharpe ratios across your trials (`stats.dsr(r, n, var_sr=...)`).
* **Passing is not proof.** A pass means "worth a small forward test with a kill switch", nothing more.
  Regimes change; a backtest is one path of history.
* **The cost model is simple.** Flat slippage, average funding, no market impact beyond one book snapshot,
  no latency, no partial fills, no exchange outages. Use your exchange's actual fee tier.
* **The look-ahead audit samples bars.** Zero mismatches on 50 sampled bars is evidence, not proof. It
  can't detect leaks that live outside the function you pass (for example, a future-dated data file).
* **G4/G5/G6 thresholds are judgement calls** (50%, 50 trades, 15R). They are fixed in advance on purpose;
  change them only before looking at results.
* Not investment advice. No warranty (see LICENSE).

## Tests

```bash
pip install pytest
python -m pytest -q
```

## License

MIT - see [LICENSE](LICENSE). Contributions welcome - see [CONTRIBUTING.md](CONTRIBUTING.md).

Live results, retired bots and a browser version of this checker: **https://botlab.org.uk**
