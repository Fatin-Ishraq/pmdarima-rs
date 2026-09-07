# Why it is faster

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

## What is compiled, and what is not

The compiled surface is deliberately small. Only the Kalman filter and the
objective built on it run thousands of times per fitted model, so only those
are in Rust. The order search, the unit-root tests, the estimator API and the
preprocessing all run *once* per model and stay in Python, where fidelity to
`pmdarima` is easy to see and to test.

```
python/pmdarima_rs/        the whole public API, ported from pmdarima
├── arima/                 ARIMA, AutoARIMA, auto_arima, the stepwise walk
├── preprocessing/         Fourier, BoxCox, LogEndog, DateFeaturizer
├── model_selection/       splitters, cross_val_score, cross_val_predict
├── datasets/              the same 11 datasets pmdarima ships
└── _fit.py                the optimiser loop

src/                       the only compiled code, ~2,000 lines
├── ssm.rs                 the SARIMAX state space
├── filter.rs              the univariate Kalman filter, and initialisation
├── objective.rs           likelihood + gradient in a single call
├── poly.rs                lag polynomials and the reparametrisation
└── lib.rs                 the pyo3 boundary
```

Those Python parts are faithful ports of the `pmdarima` originals (MIT, Taylor
G. Smith et al.), which this package is also licensed under. The stepwise walk
in particular *is* the algorithm — which neighbours are tried, in which order,
and when the walk stops — so it is reproduced rather than reinterpreted.

Two pieces had no Python original to port and are written out directly here:
`C_canova_hansen_sd_test` (the Bartlett-weighted long-run covariance of the
score contributions) and a small OLS providing `params`, `tvalues`, `aic` and
`bic`, so the seasonality tests do not pull `statsmodels` back into a package
whose point is not to need it.

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

One deviation from `statsmodels` is deliberate and always on: if L-BFGS-B
stops at a point *worse* than the starting values it was given, the starting
values are kept. `statsmodels` reports wherever the optimiser stopped, and on a
badly conditioned likelihood that can be below where it began. Returning a
strictly worse fit than one already computed is a bug however faithfully it
copies the reference. On Linux and macOS this fires often enough to matter
(1.4 AIC on one `wineind` specification, and a different selected order on
`austres`); on Windows it fires on none of 105 fits.
