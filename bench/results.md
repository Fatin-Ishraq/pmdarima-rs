C:\Users\Fatin\Downloads\blank\pmdarima-rs\bench\bench.py:87: EstimationWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
  p = ref.start_params
C:\Users\Fatin\Downloads\blank\pmdarima-rs\bench\bench.py:87: EstimationWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
  p = ref.start_params
# pmdarima-rs benchmarks

- pmdarima 2.1.1, pmdarima-rs 0.1.0
- Python 3.14.3, numpy 2.5.2

Every row is verified for agreement before it is timed.

## One likelihood evaluation (the inner loop)

| n | order | seasonal | k | statsmodels | pmdarima-rs | speedup |
|---|---|---|---:|---:|---:|---:|
| 200 | (1, 1, 1) | (0, 0, 0, 0) | 3 | 0.530 ms | 0.017 ms | **31.7x** |
| 600 | (2, 1, 2) | (0, 0, 0, 0) | 4 | 1.645 ms | 0.102 ms | **16.2x** |
| 600 | (1, 1, 1) | (1, 1, 1, 12) | 27 | 5.220 ms | 0.861 ms | **6.1x** |
| 600 | (2, 1, 2) | (2, 0, 2, 12) | 28 | 4.959 ms | 1.195 ms | **4.2x** |
| 2000 | (2, 1, 2) | (2, 0, 2, 12) | 28 | 16.834 ms | 3.404 ms | **4.9x** |

## Fitting a single known specification

| series | order | seasonal | pmdarima | pmdarima-rs | speedup | d(AIC) |
|---|---|---|---:|---:|---:|---:|
| wineind | (2, 1, 1) | (0, 0, 0, 0) | 0.178 s | 0.013 s | **13.7x** | -0.000 |
| wineind | (0, 1, 1) | (0, 1, 1, 12) | 0.168 s | 0.013 s | **13.3x** | +0.000 |
| airpassengers | (2, 1, 1) | (0, 1, 0, 12) | 0.500 s | 0.038 s | **13.0x** | -0.000 |
| ausbeer | (2, 1, 1) | (1, 1, 2, 4) | 0.891 s | 0.046 s | **19.5x** | -0.000 |
| sunspots | (3, 1, 2) | (0, 0, 0, 0) | 1.995 s | 0.041 s | **48.4x** | +0.000 |
| taylor | (5, 0, 1) | (0, 0, 0, 0) | 1.443 s | 0.048 s | **29.9x** | -0.000 |
| **total** | | | **5.17 s** | **0.20 s** | **26.0x** | |

## `auto_arima` on every dataset pmdarima ships

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

## 40 independent series, `auto_arima` on each (m=12, n=180)

| metric | pmdarima | pmdarima-rs |
|---|---:|---:|
| wall clock | 1114.5 s | 107.9 s |
| per series | 27862 ms | 2697 ms |
| **speedup** | | **10.3x** |

- identical order selected: **31/40**
- AIC of our selected model vs theirs: median -0.00, better on 9, worse on 8
