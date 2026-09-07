# Benchmarks

Reproduce all of it with:

```bash
pip install "pmdarima-rs[test]"
python bench/bench.py            # add --quick to skip the 20-minute last table
```

**Every row is checked for agreement before it is timed.** `bench.py` compares
the two libraries' answers first and only then reports a duration, so a fast
wrong answer cannot appear in these tables. Where the two can legitimately
disagree — the optimiser reaching a different point on a surface that is not
concave — the check is that our answer is not worse, and the difference is
printed rather than hidden.

Hardware and versions for the numbers below: AMD Ryzen 5 5600G (6 cores / 12
threads), Python 3.14.3, `pmdarima` 2.1.1, numpy 2.5.2, Windows 11.

**How much these move.** Every row is one timing run of a workload that takes
seconds, so the ratios are noisy. Three runs of this file on the same idle
machine put the `auto_arima` total at 16.0×, 15.5× and 19.4×, and individual
single-fit rows swing by a factor of two — the `wineind (2,1,1)` row came out
at 7.3×, 13.7× and 18.4×. What does *not* move is the shape: the inner-loop
ratio is largest for long non-seasonal series and smallest for short seasonal
ones, and the selected orders were identical on all ten datasets every time.
Read the individual rows as "about this much", not as measurements.

<!-- BENCH:BEGIN -->
## One likelihood evaluation (the inner loop)

| n | order | seasonal | k | statsmodels | pmdarima-rs | speedup |
|---|---|---|---:|---:|---:|---:|
| 200 | (1, 1, 1) | (0, 0, 0, 0) | 3 | 0.665 ms | 0.017 ms | **39.6x** |
| 600 | (2, 1, 2) | (0, 0, 0, 0) | 4 | 1.916 ms | 0.056 ms | **34.0x** |
| 600 | (1, 1, 1) | (1, 1, 1, 12) | 27 | 8.376 ms | 0.966 ms | **8.7x** |
| 600 | (2, 1, 2) | (2, 0, 2, 12) | 28 | 7.372 ms | 1.354 ms | **5.4x** |
| 2000 | (2, 1, 2) | (2, 0, 2, 12) | 28 | 25.257 ms | 3.651 ms | **6.9x** |

## Fitting a single known specification

| series | order | seasonal | pmdarima | pmdarima-rs | speedup | d(AIC) |
|---|---|---|---:|---:|---:|---:|
| wineind | (2, 1, 1) | (0, 0, 0, 0) | 0.401 s | 0.055 s | **7.3x** | -0.000 |
| wineind | (0, 1, 1) | (0, 1, 1, 12) | 0.251 s | 0.023 s | **11.1x** | +0.000 |
| airpassengers | (2, 1, 1) | (0, 1, 0, 12) | 0.820 s | 0.060 s | **13.7x** | -0.000 |
| ausbeer | (2, 1, 1) | (1, 1, 2, 4) | 1.555 s | 0.088 s | **17.7x** | -0.000 |
| sunspots | (3, 1, 2) | (0, 0, 0, 0) | 2.277 s | 0.069 s | **32.8x** | +0.000 |
| taylor | (5, 0, 1) | (0, 0, 0, 0) | 1.635 s | 0.052 s | **31.2x** | -0.000 |
| **total** | | | **6.94 s** | **0.35 s** | **20.0x** | |

## `auto_arima` on every dataset pmdarima ships

| dataset | n | m | order | pmdarima | pmdarima-rs | speedup | same? |
|---|---:|---:|---|---:|---:|---:|---|
| wineind | 176 | 12 | (0, 1, 2)(0, 1, 1, 12) | 8.11 s | 0.69 s | **11.7x** | yes |
| airpassengers | 144 | 12 | (2, 1, 1)(0, 1, 0, 12) | 4.02 s | 0.41 s | **9.7x** | yes |
| ausbeer | 211 | 4 | (2, 1, 1)(1, 1, 2, 4) | 7.48 s | 0.54 s | **13.9x** | yes |
| austres | 89 | 4 | (0, 2, 1)(1, 0, 0, 4) | 1.18 s | 0.11 s | **11.1x** | yes |
| heartrate | 150 | 1 | (0, 2, 1) | 1.06 s | 0.10 s | **10.2x** | yes |
| lynx | 114 | 1 | (2, 0, 0) | 0.65 s | 0.07 s | **8.9x** | yes |
| woolyrnq | 119 | 4 | (3, 1, 2)(2, 0, 1, 4) | 15.01 s | 0.90 s | **16.6x** | yes |
| sunspots | 1200 | 1 | (3, 1, 2) | 11.12 s | 0.44 s | **25.1x** | yes |
| taylor | 1200 | 1 | (5, 0, 1) | 8.32 s | 0.28 s | **29.5x** | yes |
| gasoline | 745 | 1 | (0, 1, 1) | 1.77 s | 0.11 s | **16.3x** | yes |
| **total** | | | | **58.7 s** | **3.7 s** | **16.0x** | **10/10** |

## 40 independent series, `auto_arima` on each (m=12, n=180)

| metric | pmdarima | pmdarima-rs |
|---|---:|---:|
| wall clock | 1033.7 s | 108.1 s |
| per series | 25843 ms | 2702 ms |
| **speedup** | | **9.6x** |

- identical order selected: **31/40**
- AIC of our selected model vs theirs: median -0.00, better on 9, worse on 8
<!-- BENCH:END -->

## Reading the last table

Forty independent seasonal series with one model fitted to each is the
workload people actually run — one model per SKU, per store, per sensor — and
it is also the honest one.

On synthetic seasonal series the two searches pick the same order 31 times in
40. Where they differ, our AIC is better on 9 and worse on 8, median 0.00: the
disagreement is a coin flip, not a degradation.

The reason is the near-non-invertibility cutoff. `auto_arima` discards any
candidate whose fitted inverse roots exceed 0.99, and seasonal ARIMA fits
routinely land at 0.98–0.999. Two optimisers that agree on the likelihood to
nine digits can still land on opposite sides of a hard threshold, and from
there the two searches take different paths. On the ten real datasets — which
is what users actually fit — this does not happen: identical orders,
identical AIC.

There is more on this, including the one dataset where the *reference's* own
selection is not stable across platforms, in [CORRECTNESS.md](CORRECTNESS.md).

## Why the ratios vary so much

The speedup is largest where `pmdarima` spends the most time in the two places
a profile finds it — the dense Kalman filter and the finite-difference
gradient — so it grows with the number of likelihood evaluations the search
makes and shrinks as the state dimension grows relative to the series length.
A long non-seasonal series like `sunspots` is the best case; a short seasonal
one at `m = 12`, where the state dimension is 27 and there are only 176
observations to amortise the fixed initialisation cost over, is the worst.

[DESIGN.md](DESIGN.md) has the profile and what replaced each line of it.
