# Correctness

The claim is not "similar results". Every number below is checked against
`pmdarima` or `statsmodels` on the same input, and the tests are differential
rather than golden-file: they run both libraries and compare, so a change in
either one shows up as a failure rather than as a quietly stale fixture.

```bash
pip install "pmdarima-rs[test]"
pytest tests/ -q
```

## What is checked

- **The likelihood matches evaluation for evaluation**, not just at the
  optimum: 120 fuzzed specifications across random orders, lengths and
  admissible parameter draws, worst relative error **1.7e-9**. A fit that
  happens to land in the right place can hide a filter that is wrong
  everywhere else, so the filter is compared directly.
- **Series with holes match too.** A missing observation knocks the filter out
  of its steady state — with nothing to correct against, `P` leaves the fixed
  point the Riccati recursion had settled on and has to find it again — and
  `statsmodels` drops its converged flag there for exactly that reason.
  Reproducing that brings a NaN-bearing series back to **1e-10** agreement;
  keeping the frozen covariance instead is wrong by several percent of the
  loglikelihood, which is the kind of error that quietly changes a selected
  order.
- **Starting values are exact.** `SARIMAX.start_params` is reproduced
  including its two-stage conditional-sum-of-squares regression and every
  fallback — worst relative difference **0.0** across 27 specifications, short
  series and the too-few-observations fallback included. This matters more
  than it looks: ARIMA likelihoods are not concave, so a different starting
  point can reach a different optimum and select a different order.
- **The unit-root and seasonality machinery is exact.** ADF, KPSS,
  Phillips-Perron, Canova-Hansen, OCSB, `ndiffs` and `nsdiffs`: **1,920
  differential checks, zero mismatches**.
- **The public API is complete, and a test proves it.** For each module the
  suite reads `pmdarima`'s own `__all__` and fails on any name we do not
  provide. A drop-in missing one symbol is not a drop-in; it is a library that
  breaks on the line you did not think to check.
- **`summary()` matches statsmodels' numbers**, including the Ljung-Box,
  Jarque-Bera and heteroskedasticity block and the `as_text` / `as_html` /
  `as_latex` / `as_csv` renderings.

Where the two libraries genuinely differ, the difference is measured and
written down rather than smoothed over.

## We are more accurate in three places

**Fourier terms.** `pmdarima` computes these in single precision, so its error
grows with the time index. At `t = 174` with `m = 12, k = 4` the argument is an
exact multiple of `2π`, so the sine must be zero: `pmdarima` returns `1.0e-5`,
we return `6e-14`.

**Standard errors, because of `sigma2`.** Fitting `(2,1,1)` to `wineind` puts
`sigma2` near `2.9e7`. `statsmodels` differentiates its per-observation
loglikelihood with a step that does not scale with the parameter, so at that
magnitude the difference is pure round-off and the reported standard error
comes back as `1.1e-4`. Recomputing the same outer-product-of-gradients
estimator from `statsmodels`' *own* `loglikeobs` with a scaled step gives
`3.79e6`, and complex step gives `3.68e6`. Both agree with us.

That one bad row contaminates the whole covariance: `inv(G'G)` mixes the
columns, so **every** standard error differs, not only `sigma2`'s — by around
1–3% on `(2,1,1)`, and by up to 22% on a seasonal `(1,0,1)(1,0,1,4)` where
`G'G` is badly conditioned to begin with. The `z`, `P>|z|` and `conf_int`
columns of `summary()` move with them. Our score is taken by a five-point
central difference with a step scaled to each parameter, which is the closest
real-arithmetic stand-in for the complex step `statsmodels` uses.

**Fitted optima, sometimes — but not dependably.** For `(1,1,1)(1,0,1,12)` on
`wineind` this machine reports AIC 3380.5 for us against 3384.5 for
`pmdarima`. Do not read that as a general result. Neither optimiser reaches
the optimum on that spec at `maxiter=50` — it is near 3334, and both stop
above 3380 still climbing. From there the landing point is chaotic: the two
start from identical parameters and evaluate an identical likelihood, but 50
L-BFGS iterations with differently-rounded finite-difference gradients
diverge, and the LAPACK behind `pinv` differs by platform. Across CI the
reference lands anywhere from 3339.6 to 3384.5 on that one spec while we sit
at 3383, so on some machines it wins and on others we do.

What *is* dependable, and is what the test suite asserts on every platform, is
that each library's likelihood evaluated at the *other's* fitted parameters
agrees to `1e-8`. That is the correctness property; which of two
under-converged climbs stopped higher is not one. Where you want the optimum
rather than the reference's stopping point, `restarts=3` reaches 3334 and
converges — see [DESIGN.md](DESIGN.md) for why that is not the default.

## One place agreement is limited, and why

`statsmodels` stands in for an infinite prior variance on the differencing
states with a large finite one (`1e6`). Forming `P − M M'/F` when `P` carries
entries of `1e6` and the answer is of order `sigma2` cancels away roughly
`log10(1e6 / sigma2)` significant digits. Both libraries pay that; they simply
round it differently. Agreement is at machine precision for realistic
`sigma2`, and degrades predictably as `sigma2` shrinks — a property of the
model specification, not of either implementation, and one that lives far
outside the region any optimiser visits. A test pins the behaviour so a real
regression cannot hide inside it.

Because that loss is a property of the specification, the differential tests
set their tolerance from the specification rather than using one flat number.
Over 400 admissible parameter draws each: a model with no differencing agrees
to 2.5e-13 at worst (on the 26-state `(2,0,2)(2,0,2,12)`, where the round-off
is accumulation rather than cancellation), and one with thirteen diffuse
states to 1.7e-9. A flat `1e-9` bound sat between those two and held only
until a `statsmodels` release rounded differently, which is exactly the kind
of test that fails without telling you anything.

## Known differences

Everything in this list is deliberate and tested; none of it is a to-do.

- **Optimiser landing points.** Where the searches differ, both libraries
  evaluate the *other's* parameters to the same loglikelihood, so it is the
  stopping point rather than the filter.
- **Order selection on borderline series.** `auto_arima` discards a candidate
  whose fitted inverse roots exceed 0.99, and seasonal fits routinely land at
  0.98–0.999. Two optimisers agreeing on the likelihood to nine digits can
  still land on opposite sides of a hard cutoff, and then the searches take
  different paths. On the ten real datasets this does not happen (10/10
  identical); on 40 synthetic seasonal series the orders agree 31 times, and
  where they differ our AIC is better on 9 and worse on 8 — a coin flip, not a
  degradation.

  The reference's own selection is not platform-invariant either. On `austres`
  both libraries fit `(2,2,2)(1,0,1,4)` to a worst inverse root of 0.9911,
  exceed the cutoff, discard it and settle on `(0,2,1)(1,0,0,4)` at AIC
  651.95 — but on one Windows and BLAS combination the reference lands at
  0.9899 instead, keeps the model and reports 650.27. Nothing separates those
  runs but rounding in the third decimal of a root. Our own selection was
  stable across all nine CI platforms.
- **`method='nm'` and the other seven** warn and fall back to `lbfgs`; an
  invalid name raises the same `ValueError` `statsmodels` does.
- **Error-message wording** differs where the text embeds a class repr
  (`pmdarima_rs.` vs `pmdarima.`), or where `pmdarima`'s own message has a
  typo (`{'trend', 'null'}` where it means `'level'`) or an uninterpolated
  `'%s'`.
- **Phillips-Perron on 3-to-5-point series**, where the statistic's
  denominator is analytically zero and the two libraries round the
  cancellation differently.
- **`update()` keeps the pandas index**, where `pmdarima` drops it and returns
  an ndarray — the same call returns a `Series` before an update there, so
  this is the consistent behaviour.
