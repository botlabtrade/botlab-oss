# Contributing

Thanks for helping make backtests more honest.

* **Bugs:** open an issue with a minimal CSV (synthetic data please) and the command you ran.
* **Pull requests:** keep them small, add a test in `tests/`, and run `python -m pytest -q` before pushing.
* **New checks** are welcome if they catch a real way backtests mislead. Please cite the method
  (paper, book or well-known tool) in the docstring.
* **Thresholds:** changing a default in `GateConfig` needs a reason in the PR description. Defaults are
  deliberately conservative.
* **Out of scope:** strategies, signals, "profitable settings", exchange order placement, anything that
  needs API keys.
* **Never commit** API keys, account data or real trade histories you are not allowed to share.
* Style: standard library + numpy/pandas only, type hints where they help, plain English docstrings.

By contributing you agree that your contribution is released under the MIT License.
