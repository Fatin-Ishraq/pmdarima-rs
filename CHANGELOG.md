# Changelog

## 0.1.0 — unreleased

First public release.

- `auto_arima`, `ARIMA`, `AutoARIMA`, `Pipeline`, the preprocessing
  transformers, the model-selection splitters and the unit-root and
  seasonality tests, all matching `pmdarima` 2.1.1's API. A test reads
  `pmdarima`'s own `__all__` for each module and fails on any missing name.
- The Kalman filter, its initialisation and the likelihood-plus-gradient
  objective are in Rust; everything that runs once per model stays in Python.
  `auto_arima` end to end is 9–30× faster depending on the series, selecting
  the same order on all ten datasets `pmdarima` ships.
- `install()` aliases `pmdarima` to this package on `sys.meta_path`, for code
  you cannot edit.
- `summary()` reproduces `statsmodels`' table, including the Ljung-Box,
  Jarque-Bera and heteroskedasticity block, with `as_text` / `as_html` /
  `as_latex` / `as_csv`.
- Wheels for Linux (x86\_64, aarch64), macOS (x86\_64, arm64) and Windows
  (x64), covering Python 3.10 through 3.14 from one `abi3` wheel per platform.
