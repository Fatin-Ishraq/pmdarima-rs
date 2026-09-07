<div align="center">

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/pmdarima-rs/main/assets/banner.svg" alt="pmdarima-rs" width="820">

[![CI](https://github.com/Fatin-Ishraq/pmdarima-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatin-Ishraq/pmdarima-rs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pmdarima-rs.svg?color=fb923c)](https://pypi.org/project/pmdarima-rs/)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-5eead4.svg)](https://pypi.org/project/pmdarima-rs/)
[![License](https://img.shields.io/badge/license-MIT-94a3b8.svg)](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/LICENSE)

**A drop-in replacement for [`pmdarima`](https://github.com/alkaline-ml/pmdarima)** —
the same `auto_arima`, the same `ARIMA`, the same arguments and the same
selected orders, with the Kalman filter rewritten in Rust.

</div>

---

## Install

```bash
pip install pmdarima-rs
```

Prebuilt wheels for Linux, macOS and Windows, **Python 3.10 through 3.14**
from a single `abi3` wheel per platform. No compiler needed, no Rust toolchain
needed.

## Switch

Change one import.

```diff
- import pmdarima as pm
+ import pmdarima_rs as pm
```

That is the whole migration. If you cannot edit the code that imports
`pmdarima` — someone else's library, a notebook you were handed, a vendored
script — [alias it instead](#for-code-you-cannot-edit).

## Use it

Everything works the way it does in `pmdarima`, because it is the same API.

```python
import pmdarima_rs as pm

y = pm.datasets.load_wineind()
model = pm.auto_arima(y, seasonal=True, m=12, trace=True)

forecast, ci = model.predict(n_periods=12, return_conf_int=True)
print(model.summary())
```

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/pmdarima-rs/main/assets/quickstart.svg" alt="auto_arima searching 21 candidate models on the wineind dataset and finishing in 0.7 seconds" width="620">

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/pmdarima-rs/main/assets/forecast.png" alt="wineind observed series and a 24-period forecast with a 95% interval" width="820">

The `summary()` table is `statsmodels`' own layout, and every figure in it —
coefficients, standard errors, information criteria, the Ljung-Box and
Jarque-Bera block — is checked against `statsmodels` in the test suite.

<details>
<summary><b>See the full summary output</b></summary>

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/pmdarima-rs/main/assets/summary.svg" alt="SARIMAX results table" width="760">

</details>

## Speed

<img src="https://raw.githubusercontent.com/Fatin-Ishraq/pmdarima-rs/main/assets/speedup.svg" alt="auto_arima speedup by dataset, 8.9x to 29.5x, same order selected on all ten" width="880">

| workload | `pmdarima` | `pmdarima-rs` | |
|---|---:|---:|---|
| `auto_arima` on all 10 bundled datasets | 58.7 s | 3.7 s | **16.0×**, 10/10 identical orders |
| fitting 6 known specifications | 6.94 s | 0.35 s | **20.0×**, identical AIC |
| 40 seasonal series, one model each | 17.2 min | 1.8 min | **9.6×** |
| one likelihood evaluation | 0.7–25 ms | 0.02–3.7 ms | **5.4× – 39.6×** |

**Every row is checked for agreement before it is timed**, so a fast wrong
answer cannot appear in the table. `pmdarima`'s filter is already compiled —
it is Cython — so this is not "Python versus native"; the wins are
algorithmic, and [docs/DESIGN.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/DESIGN.md) says what they are.

These are one machine's numbers and they are noisy: repeat runs move the
totals by 15–20% and individual rows by more. What holds across runs is the
order of magnitude and the agreement. Every measurement, and how to reproduce
it, is in [docs/BENCHMARKS.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/BENCHMARKS.md).

## Accuracy

The claim is not "similar results".

| what | agreement |
|---|---|
| loglikelihood, 120 fuzzed specifications | worst relative error **1.7e-9** |
| loglikelihood on series containing NaN | **1e-10** |
| `SARIMAX.start_params`, 27 specifications | **exact** |
| ADF / KPSS / PP / CH / OCSB / `ndiffs` / `nsdiffs` | **1,920 checks, 0 mismatches** |
| `auto_arima` order on the 10 real datasets | **10/10 identical** |

The tests are differential, not golden-file: they run `pmdarima` and this
package on the same input and compare. Where the two genuinely differ — three
places where we are *more* accurate, and a handful where a hard threshold
makes the choice a coin flip — it is measured and written down in
[docs/CORRECTNESS.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/CORRECTNESS.md) rather than smoothed over.

## For code you cannot edit

```python
import pmdarima_rs
pmdarima_rs.install()      # before the first `import pmdarima`

import pmdarima            # now resolves to pmdarima_rs
```

`install()` puts a finder on `sys.meta_path`, so every submodule resolves too,
including ones nothing has imported yet: `import pmdarima.arima.utils` or
`from pmdarima.datasets.wineind import load_wineind` gets the very same module
object this package exposes. `isinstance` checks and pickles therefore still
work across the alias.

One attribute is deliberately not passed through: `pmdarima.__version__`
reports the `pmdarima` API level this package implements (`2.1.1`), because
the code you cannot edit is exactly the code likely to gate on it. This
package's own version stays available as `pmdarima.__pmdarima_rs_version__`
and as `pmdarima_rs.__version__`.

## What is included

All of it. A test reads `pmdarima`'s own `__all__` for each module and fails
on any name this package does not provide.

| | |
|---|---|
| **Models** | `ARIMA`, `AutoARIMA`, `auto_arima`, `StepwiseContext`, `Pipeline` |
| **Unit-root / seasonality** | `ADFTest`, `KPSSTest`, `PPTest`, `CHTest`, `OCSBTest`, `ndiffs`, `nsdiffs` |
| **Preprocessing** | `FourierFeaturizer`, `BoxCoxEndogTransformer`, `LogEndogTransformer`, `DateFeaturizer` |
| **Model selection** | `RollingForecastCV`, `SlidingWindowForecastCV`, `cross_val_score`, `cross_val_predict`, `cross_validate`, `train_test_split` |
| **Utils** | `acf`, `pacf`, `diff`, `diff_inv`, `c`, `decompose`, `tsdisplay`, `plot_acf`, `plot_pacf`, `autocorr_plot` |
| **Datasets** | all 11 — `wineind`, `airpassengers`, `ausbeer`, `austres`, `heartrate`, `lynx`, `woolyrnq`, `sunspots`, `taylor`, `gasoline`, `msft` |

The estimators are real scikit-learn estimators when scikit-learn is
installed, and use a local fallback base when it is not — so `clone`,
`get_params`, `set_output` and metadata routing all work, without making
scikit-learn a dependency.

## Dependencies

Three, against `pmdarima`'s eight.

| | `pmdarima` | `pmdarima-rs` |
|---|---|---|
| required | numpy, pandas, scipy, **statsmodels, scikit-learn, joblib, Cython, setuptools** | numpy, pandas, scipy |
| optional | — | matplotlib (`[plot]`), scikit-learn |

`matplotlib` is imported lazily inside the plotting helpers, so it only has to
be there if you call them.

## Limitations

- `method` accepts the same nine solver names `statsmodels` does and rejects
  anything else with the same `ValueError`, but only `'lbfgs'` is implemented;
  the other eight warn and fall back to it. It is `pmdarima`'s default and the
  only one its own `auto_arima` uses.
- Five `SARIMAX` options raise `NotImplementedError` rather than being
  accepted and quietly ignored: `simple_differencing`, `measurement_error`,
  `time_varying_regression`, `mle_regression=False` and `use_exact_diffuse`.
  Each of them changes the model, so honouring the argument by ignoring it
  would report a different model's numbers under your specification. Every
  other `SARIMAX` keyword is either implemented or genuinely makes no
  difference to the likelihood, and an unrecognised one is a `TypeError`, as
  it is in `statsmodels`.
- Order selection agrees with `pmdarima` on real data (10/10) but can differ
  on series whose fitted MA roots sit on `auto_arima`'s 0.99 rejection
  threshold, where two optimisers agreeing to nine digits still land on
  opposite sides of a hard cutoff. It is measured in the benchmark rather than
  asserted away; details in [docs/CORRECTNESS.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/CORRECTNESS.md).
- The stationary initial covariance is solved by squaring, which is far
  cheaper than the `k² × k²` factorisation it replaces but is still the
  largest fixed cost per likelihood evaluation — roughly a sixth of a seasonal
  call at `m = 12`. An `O(r²)` recursion using the ARMA
  autocovariances would remove most of that; it is the clearest remaining
  headroom and is not implemented.

## Development

```bash
git clone https://github.com/Fatin-Ishraq/pmdarima-rs
cd pmdarima-rs
pip install maturin
maturin develop --release

pip install "pmdarima>=2.1.1" statsmodels pytest   # the reference to test against
pytest tests/ -q
cargo test --lib
python bench/bench.py
```

The test suite skips its differential tests when `pmdarima` is not installed,
so it still runs without the reference — it just checks less.

## Documentation

- [docs/DESIGN.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/DESIGN.md) — where the time went and what replaced it,
  what is compiled and what is not
- [docs/CORRECTNESS.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/CORRECTNESS.md) — what is verified, how, and every
  known difference
- [docs/BENCHMARKS.md](https://github.com/Fatin-Ishraq/pmdarima-rs/blob/main/docs/BENCHMARKS.md) — every measurement, and how to
  reproduce it
- `pmdarima`'s own [documentation](https://alkaline-ml.com/pmdarima/) applies
  unchanged

## Licence

MIT. `pmdarima` is MIT (Taylor G. Smith and contributors); `statsmodels` is
BSD-3. Portions of this package are ports of both, as noted in the module
docstrings.
