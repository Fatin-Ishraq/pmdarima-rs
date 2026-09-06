# pmdarima-rs

[![CI](https://github.com/Fatin-Ishraq/pmdarima-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Fatin-Ishraq/pmdarima-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg)](https://pypi.org/project/pmdarima-rs/)

Fast, drop-in ARIMA and auto-ARIMA for Python, powered by Rust.

```diff
- import pmdarima as pm
+ import pmdarima_rs as pm
```

That is the whole migration. Same classes, same arguments, same selected
orders.

| workload | `pmdarima` | `pmdarima-rs` | |
|---|---:|---:|---|
| `auto_arima` on all 10 bundled datasets | 96.2 s | 5.0 s | **19.4x**, 10/10 identical orders |
| fitting 6 known specifications | 5.17 s | 0.20 s | **26.0x**, identical AIC |
| 40 seasonal series, one model each | 18.6 min | 1.8 min | **10.3x** |
| one likelihood evaluation | 0.5-17 ms | 0.02-3.4 ms | **4.2x - 31.7x** |

Verified before it is timed: every benchmark row checks agreement first, so a
fast wrong answer cannot appear in the table.

```bash
pip install pmdarima-rs
```

Supports **Python 3.10 through 3.14** from a single `abi3` wheel per platform.

It also has a shorter dependency list than the library it replaces: numpy,
pandas and scipy. `pmdarima` additionally requires `statsmodels`,
`scikit-learn`, `joblib`, `Cython` and `setuptools`; none of those are needed
here. `matplotlib` is imported lazily inside the plotting helpers, so it stays
optional (`pip install pmdarima-rs[plot]`).

## Why it is faster

`pmdarima` wraps `statsmodels`' SARIMAX, and a profile of one seasonal
`auto_arima` fit — 55 seconds of wall clock for a single 600-point series —
puts essentially all of it in two places:

| where | cost | why |
|---|---|---|
| `kalman_filter._filter` | 12,387 calls, **9.8 ms each** | a **dense** filter: `T P T'` costs `O(k³)` |
| `scipy._numdiff._dense_difference` | 1,241 gradients, **12,193 likelihood calls** | no analytic gradient, so each gradient is `n_params + 1` filter passes |

The filter is already compiled — it is Cython — so this is not "Python versus
native". The wins had to be algorithmic.

**The transition matrix is almost entirely structure.** A SARIMAX transition
is a companion matrix (a superdiagonal identity plus one column of AR
coefficients), a handful of ones for the differencing states, a 0/1 design
vector, and a single-column selection matrix. Held as sparse rows, `T P T'`
costs `O(k²)` instead of `O(k³)`. For a seasonal model with `m = 12` the state
dimension is around 27, so that is most of the arithmetic gone before any
micro-optimisation.

**The observation is univariate.** `Z P Z'` is a scalar, so the matrix
inversion at the heart of a general filter becomes one divide and the Kalman
gain is a vector rather than a matrix.

**The initial covariance does not need a `k² × k²` solve.** The stationary
block solves a discrete Lyapunov equation; `statsmodels` reaches for a general
solver that forms and factors the Kronecker system — the reason
`_initialize_state` is 12% of the profile. Squaring converges quadratically
and never leaves `k × k`. The stationary initial *mean* has a closed form for a
companion matrix, so it is `O(r)` rather than a general `O(r³)` solve.

**The gradient comes back in one call.** `statsmodels` asks `scipy` for
`approx_grad=True`, so every gradient is `n_params + 1` separate Python round
trips into the likelihood. Here the whole gradient is computed inside Rust,
with the perturbations spread across cores and the GIL released.

**The steady state is exploited.** For a time-invariant system the Riccati
recursion converges, after which `P`, `F` and the gain stop changing;
`statsmodels` detects this and freezes them, and so do we — which is required
for the two to agree, and independently drops every step after convergence
from `O(k²)` to `O(k)`.

## Benchmarks

All figures below come from `python bench/bench.py` on an AMD Ryzen 5 5600G
(6 cores / 12 threads), Python 3.14.3, `pmdarima` 2.1.1, numpy 2.5.2. **Every
row is checked for agreement before it is timed**, so a fast wrong answer
cannot appear in these tables.

### One likelihood evaluation (the inner loop)

| n | order | seasonal | k | statsmodels | pmdarima-rs | speedup |
|---|---|---|---:|---:|---:|---:|
| 200 | (1, 1, 1) | (0, 0, 0, 0) | 3 | 0.530 ms | 0.017 ms | **31.7x** |
| 600 | (2, 1, 2) | (0, 0, 0, 0) | 4 | 1.645 ms | 0.102 ms | **16.2x** |
| 600 | (1, 1, 1) | (1, 1, 1, 12) | 27 | 5.220 ms | 0.861 ms | **6.1x** |
| 600 | (2, 1, 2) | (2, 0, 2, 12) | 28 | 4.959 ms | 1.195 ms | **4.2x** |
| 2000 | (2, 1, 2) | (2, 0, 2, 12) | 28 | 16.834 ms | 3.404 ms | **4.9x** |

### Fitting a single known specification

| series | order | seasonal | pmdarima | pmdarima-rs | speedup | d(AIC) |
|---|---|---|---:|---:|---:|---:|
| wineind | (2, 1, 1) | (0, 0, 0, 0) | 0.178 s | 0.013 s | **13.7x** | -0.000 |
| wineind | (0, 1, 1) | (0, 1, 1, 12) | 0.168 s | 0.013 s | **13.3x** | +0.000 |
| airpassengers | (2, 1, 1) | (0, 1, 0, 12) | 0.500 s | 0.038 s | **13.0x** | -0.000 |
| ausbeer | (2, 1, 1) | (1, 1, 2, 4) | 0.891 s | 0.046 s | **19.5x** | -0.000 |
| sunspots | (3, 1, 2) | (0, 0, 0, 0) | 1.995 s | 0.041 s | **48.4x** | +0.000 |
| taylor | (5, 0, 1) | (0, 0, 0, 0) | 1.443 s | 0.048 s | **29.9x** | -0.000 |
| **total** | | | **5.17 s** | **0.20 s** | **26.0x** | |

### `auto_arima` on every dataset pmdarima ships

| dataset | n | m | order | pmdarima | pmdarima-rs | speedup | same? |
|---|---:|---:|---|---:|---:|---:|---|
| wineind | 176 | 12 | (0, 1, 2)(0, 1, 1, 12) | 8.12 s | 0.85 s | **9.5x** | yes |
| airpassengers | 144 | 12 | (2, 1, 1)(0, 1, 0, 12) | 8.15 s | 0.54 s | **15.2x** | yes |
| ausbeer | 211 | 4 | (2, 1, 1)(1, 1, 2, 4) | 12.33 s | 0.69 s | **18.0x** | yes |
| austres | 89 | 4 | (0, 2, 1)(1, 0, 0, 4) | 1.82 s | 0.13 s | **13.8x** | yes |
| heartrate | 150 | 1 | (0, 2, 1) | 1.73 s | 0.13 s | **13.1x** | yes |
| lynx | 114 | 1 | (2, 0, 0) | 1.09 s | 0.12 s | **9.3x** | yes |
| woolyrnq | 119 | 4 | (3, 1, 2)(2, 0, 1, 4) | 23.53 s | 1.27 s | **18.5x** | yes |
| sunspots | 1200 | 1 | (3, 1, 2) | 21.25 s | 0.61 s | **35.1x** | yes |
| taylor | 1200 | 1 | (5, 0, 1) | 14.76 s | 0.48 s | **30.9x** | yes |
| gasoline | 745 | 1 | (0, 1, 1) | 3.37 s | 0.15 s | **22.6x** | yes |
| **total** | | | | **96.2 s** | **5.0 s** | **19.4x** | **10/10** |

### 40 independent series, `auto_arima` on each (m=12, n=180)

| metric | pmdarima | pmdarima-rs |
|---|---:|---:|
| wall clock | 1114.5 s | 107.9 s |
| per series | 27862 ms | 2697 ms |
| **speedup** | | **10.3x** |

- identical order selected: **31/40**
- AIC of our selected model vs theirs: median -0.00, better on 9, worse on 8

The last table is the workload people actually run — one model per SKU — and
it is also the honest one. On synthetic seasonal series the two searches pick
the same order 31 times in 40. Where they differ, our AIC is better on 9 and
worse on 8, median 0.00: the disagreement is a coin flip, not a degradation.

The reason is the near-non-invertibility cutoff. `auto_arima` discards any
candidate whose fitted inverse roots exceed 0.99, and seasonal ARIMA fits
routinely land at 0.98-0.999. Two optimisers that agree on the likelihood to
nine digits can still land on opposite sides of a hard threshold, and then
the searches take different paths. On the ten real datasets — which is what
users actually fit — this does not happen: **10/10 identical orders, identical
AIC.**


## Correctness

The claim is not "similar results". Every number below is checked against
`pmdarima` or `statsmodels` on the same input, and the tests are differential
rather than golden-file.

- **The likelihood matches evaluation for evaluation**, not just at the
  optimum: 120 fuzzed specifications across random orders, lengths and
  admissible parameter draws, worst relative error **1.7e-9**. A fit that
  happens to land in the right place can hide a filter that is wrong
  everywhere else, so the filter is compared directly.
- **Starting values are exact.** `SARIMAX.start_params` is reproduced
  including its two-stage conditional-sum-of-squares regression and every
  fallback — worst relative difference **0.0** across 27 specifications. This
  matters more than it looks: ARIMA likelihoods are not concave, so a
  different starting point can reach a different optimum and select a
  different order.
- **The unit-root and seasonality machinery is exact.** ADF, KPSS,
  Phillips-Perron, Canova-Hansen, OCSB, `ndiffs` and `nsdiffs`: **1,920
  differential checks, zero mismatches**.
- **The public API is complete, and a test proves it.** For each module the
  suite reads `pmdarima`'s own `__all__` and fails on any name we do not
  provide. A drop-in missing one symbol is not a drop-in; it is a library that
  breaks on the line you did not think to check.

Where the two libraries genuinely differ, the difference is measured and
written down rather than smoothed over.

### We are more accurate in three places

**Fourier terms.** `pmdarima` computes these in single precision, so its error
grows with the time index. At `t = 174` with `m = 12, k = 4` the argument is an
exact multiple of `2π`, so the sine must be zero: `pmdarima` returns `1.0e-5`,
we return `6e-14`.

**`sigma2`'s standard error.** Fitting `(2,1,1)` to `wineind` puts `sigma2`
near `2.9e7`. `statsmodels` differentiates its per-observation loglikelihood
with a step that does not scale with the parameter, so at that magnitude the
difference is pure round-off and the reported standard error comes back as
`1.1e-4`. Recomputing the same outer-product-of-gradients estimator from
`statsmodels`' *own* `loglikeobs` with a scaled step gives `3.79e6`, and
complex step gives `3.68e6`. Both agree with us.

**Fitted optima, sometimes.** For `(1,1,1)(1,0,1,12)` on `wineind`, both
optimisers exhaust `maxiter=50` still climbing and ours ends 2.0 loglike
higher — AIC 3380.5 against 3384.5. The test suite asserts our optimum is
never materially *worse*, and separately asserts that each library's
likelihood evaluated at the *other's* fitted parameters agrees to `1e-8`,
which is the actual correctness property.

### One place agreement is limited, and why

`statsmodels` stands in for an infinite prior variance on the differencing
states with a large finite one (`1e6`). Forming `P − M M'/F` when `P` carries
entries of `1e6` and the answer is of order `sigma2` cancels away roughly
`log10(1e6 / sigma2)` significant digits. Both libraries pay that; they simply
round it differently. Agreement is at machine precision for realistic
`sigma2`, and degrades predictably as `sigma2` shrinks — a property of the
model specification, not of either implementation, and one that lives far
outside the region any optimiser visits. A test pins the behaviour so a real
regression cannot hide inside it.

## A deliberately *worse* optimiser, by default

This package can fit a strictly better model than `pmdarima`: with central
differences, a longer L-BFGS memory and restarts, it reached a higher
likelihood on 15 of 27 fits and an identical one on 11, mean gain +7.0
loglike.

That is **not** the default, and the reason is worth stating.

`auto_arima` discards any candidate whose fitted inverse roots exceed 0.99 as
near-non-invertible. ARIMA likelihoods frequently have their maximum *at* that
boundary, so a better gradient climbs closer to it and gets more candidates
discarded — which changes which order is selected. Measured on one series,
restarting bought 0.75 loglike, moved an MA inverse root from 0.9874 to
0.9998, and ended the search on a model **24 AIC worse**. Climbing further up
a ridge the caller is going to reject is not an improvement.

So the default reproduces `statsmodels`' optimiser configuration — forward
differences with `epsilon=1e-5`, `m=10`, no restarts — and the stronger
settings are available per fit, for when you are estimating one specification
you have already chosen:

```python
from pmdarima_rs import _fit
from pmdarima_rs._ssm import Spec

res = _fit.fit(Spec((2, 1, 1), (1, 0, 1, 12), "c"), y,
               restarts=3, m=20, epsilon=None)   # central differences
```

## What is compiled, and what is not

The compiled surface is deliberately small. Only the Kalman filter and the
objective built on it run thousands of times per fitted model, so only those
are in Rust. The order search, the unit-root tests, the estimator API and the
preprocessing all run *once* per model and stay in Python, where fidelity to
`pmdarima` is easy to see and to test.

Those Python parts are faithful ports of the `pmdarima` originals (MIT, Taylor
G. Smith et al.), which this package is also licensed under. The stepwise walk
in particular *is* the algorithm — which neighbours are tried, in which order,
and when the walk stops — so it is reproduced rather than reinterpreted.

Two pieces had no Python original to port and are written out directly here:
`C_canova_hansen_sd_test` (the Bartlett-weighted long-run covariance of the
score contributions) and a small OLS providing `params`, `tvalues`, `aic` and
`bic`, so the seasonality tests do not pull `statsmodels` back into a package
whose point is not to need it.

## For code you cannot edit

```python
import pmdarima_rs
pmdarima_rs.install()      # before the first `import pmdarima`

import pmdarima            # now resolves to pmdarima_rs
```

## Limitations

- `method` is accepted for compatibility but only `'lbfgs'` is implemented;
  other solvers fall back to it. It is `pmdarima`'s default and the only one
  its own `auto_arima` uses.
- `predict_in_sample(dynamic=True)` is accepted and ignored, as it is in
  `pmdarima` when confidence intervals are requested.
- Order selection agrees with `pmdarima` on real data (10/10). It can differ
  on series whose fitted MA roots sit on the 0.99 rejection threshold, where
  two optimisers agreeing on the likelihood to nine digits still land on
  opposite sides of a hard cutoff. On 40 synthetic seasonal series the orders
  agree 31 times; where they differ our AIC is better on 9 and worse on 8, so
  the disagreement is a coin flip rather than a degradation. It is measured in
  the benchmark rather than asserted away.
- The stationary initial covariance is solved by squaring, which is far
  cheaper than the `k² × k²` factorisation it replaces but is still the
  largest fixed cost per likelihood evaluation - about 0.17 ms of a 1.06 ms
  seasonal call at `m = 12`. An `O(r²)` recursion using the ARMA
  autocovariances would remove most of that; it is the clearest remaining
  headroom and is not implemented.

## Licence

MIT. `pmdarima` is MIT (Taylor G. Smith and contributors); `statsmodels` is
BSD-3. Portions of this package are ports of both, as noted in the module
docstrings.
