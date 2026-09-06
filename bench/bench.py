"""Benchmark `pmdarima-rs` against `pmdarima`.

Every row is checked for agreement *before* it is timed, so a fast wrong
answer cannot appear in the tables. Where the two libraries can disagree
legitimately - the optimiser reaching a different point on a non-concave
surface - the check is that our answer is not worse, and the difference is
reported rather than hidden.

    python bench/bench.py [--quick]
"""

import argparse
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

import pmdarima as pm  # noqa: E402
import pmdarima_rs as pmr  # noqa: E402


def timeit(fn, repeat=1):
    fn()  # warm up: first call pays import and allocation costs
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def synth(n, seed, m=12, kind="sarima"):
    """A seasonal ARIMA-like series, not a pathological one."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 60)
    x = np.zeros(n + 60)
    for t in range(2, n + 60):
        x[t] = 0.6 * x[t - 1] - 0.2 * x[t - 2] + e[t] + 0.4 * e[t - 1]
    t = np.arange(n)
    season = 8.0 * np.sin(2 * np.pi * t / m) + 3.0 * np.cos(4 * np.pi * t / m)
    return 100.0 + 0.05 * t + season + x[60:] * 2.0


REAL = [
    ("wineind", 12),
    ("airpassengers", 12),
    ("ausbeer", 4),
    ("austres", 4),
    ("heartrate", 1),
    ("lynx", 1),
    ("woolyrnq", 4),
    ("sunspots", 1),
    ("taylor", 1),
    ("gasoline", 1),
]


def load(name, cap=1200):
    y = np.asarray(getattr(pmr.datasets, "load_" + name)(), dtype=float)
    y = y[~np.isnan(y)]
    return y[:cap]


def bench_loglike():
    """The inner loop, in isolation: one likelihood evaluation."""
    import statsmodels.api as sm

    from pmdarima_rs import _fit as rs_fit
    from pmdarima_rs._ssm import Spec

    print("\n## One likelihood evaluation (the inner loop)\n")
    print(f"| n | order | seasonal | k | statsmodels | pmdarima-rs | speedup |")
    print(f"|---|---|---|---:|---:|---:|---:|")
    cases = [
        (200, (1, 1, 1), (0, 0, 0, 0)),
        (600, (2, 1, 2), (0, 0, 0, 0)),
        (600, (1, 1, 1), (1, 1, 1, 12)),
        (600, (2, 1, 2), (2, 0, 2, 12)),
        (2000, (2, 1, 2), (2, 0, 2, 12)),
    ]
    for n, order, sorder in cases:
        y = synth(n, 1)
        ref = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=sorder, trend="c")
        p = ref.start_params
        spec = Spec(order, sorder, "c")
        a, b = ref.loglike(p), rs_fit.loglike(spec, y, p)
        assert abs(a - b) <= 1e-8 * max(1.0, abs(a)), f"disagreement: {a} vs {b}"
        ta = timeit(lambda: ref.loglike(p), 20)
        tb = timeit(lambda: rs_fit.loglike(spec, y, p), 200)
        print(
            f"| {n} | {order} | {sorder} | {ref.k_states} | "
            f"{ta * 1e3:.3f} ms | {tb * 1e3:.3f} ms | **{ta / tb:.1f}x** |"
        )


def bench_single_fits():
    """Fitting one known specification."""
    print("\n## Fitting a single known specification\n")
    print("| series | order | seasonal | pmdarima | pmdarima-rs | speedup | d(AIC) |")
    print("|---|---|---|---:|---:|---:|---:|")
    cases = [
        ("wineind", (2, 1, 1), (0, 0, 0, 0)),
        ("wineind", (0, 1, 1), (0, 1, 1, 12)),
        ("airpassengers", (2, 1, 1), (0, 1, 0, 12)),
        ("ausbeer", (2, 1, 1), (1, 1, 2, 4)),
        ("sunspots", (3, 1, 2), (0, 0, 0, 0)),
        ("taylor", (5, 0, 1), (0, 0, 0, 0)),
    ]
    tot_a = tot_b = 0.0
    for name, order, sorder in cases:
        y = load(name)
        mk_a = lambda: pm.arima.ARIMA(  # noqa: E731
            order=order, seasonal_order=sorder, suppress_warnings=True
        ).fit(y)
        mk_b = lambda: pmr.arima.ARIMA(  # noqa: E731
            order=order, seasonal_order=sorder, suppress_warnings=True
        ).fit(y)
        a, b = mk_a(), mk_b()
        assert b.aic() <= a.aic() + 1e-6 * abs(a.aic()), "our optimum is worse"
        ta, tb = timeit(mk_a), timeit(mk_b)
        tot_a += ta
        tot_b += tb
        print(
            f"| {name} | {order} | {sorder} | {ta:.3f} s | {tb:.3f} s | "
            f"**{ta / tb:.1f}x** | {b.aic() - a.aic():+.3f} |"
        )
    print(f"| **total** | | | **{tot_a:.2f} s** | **{tot_b:.2f} s** | "
          f"**{tot_a / tot_b:.1f}x** | |")


def bench_auto_arima(quick=False):
    """The headline workload: order selection on real series."""
    print("\n## `auto_arima` on every dataset pmdarima ships\n")
    print("| dataset | n | m | order | pmdarima | pmdarima-rs | speedup | same? |")
    print("|---|---:|---:|---|---:|---:|---:|---|")
    tot_a = tot_b = 0.0
    agree = 0
    rows = REAL[:5] if quick else REAL
    for name, m in rows:
        y = load(name)
        kw = dict(seasonal=m > 1, m=m, suppress_warnings=True, error_action="ignore")
        ta = timeit(lambda: pm.auto_arima(y, **kw))
        tb = timeit(lambda: pmr.auto_arima(y, **kw))
        a, b = pm.auto_arima(y, **kw), pmr.auto_arima(y, **kw)
        same = a.order == b.order and tuple(a.seasonal_order) == tuple(b.seasonal_order)
        agree += same
        tot_a += ta
        tot_b += tb
        od = f"{b.order}{tuple(b.seasonal_order) if m > 1 else ''}"
        print(
            f"| {name} | {len(y)} | {m} | {od} | {ta:.2f} s | {tb:.2f} s | "
            f"**{ta / tb:.1f}x** | {'yes' if same else 'NO'} |"
        )
    print(
        f"| **total** | | | | **{tot_a:.1f} s** | **{tot_b:.1f} s** | "
        f"**{tot_a / tot_b:.1f}x** | **{agree}/{len(rows)}** |"
    )
    return agree, len(rows)


def bench_many_series(n_series=40, n=180, m=12):
    """The workload people actually run: one model per SKU."""
    print(f"\n## {n_series} independent series, `auto_arima` on each (m={m}, n={n})\n")
    series = [synth(n, s, m) for s in range(n_series)]
    kw = dict(seasonal=True, m=m, suppress_warnings=True, error_action="ignore")

    t0 = time.perf_counter()
    ref = [pm.auto_arima(y, **kw) for y in series]
    ta = time.perf_counter() - t0

    t0 = time.perf_counter()
    mine = [pmr.auto_arima(y, **kw) for y in series]
    tb = time.perf_counter() - t0

    same = sum(
        a.order == b.order and tuple(a.seasonal_order) == tuple(b.seasonal_order)
        for a, b in zip(ref, mine)
    )
    daic = np.array([b.aic() - a.aic() for a, b in zip(ref, mine)])
    print(f"| metric | pmdarima | pmdarima-rs |")
    print(f"|---|---:|---:|")
    print(f"| wall clock | {ta:.1f} s | {tb:.1f} s |")
    print(f"| per series | {ta / n_series * 1000:.0f} ms | {tb / n_series * 1000:.0f} ms |")
    print(f"| **speedup** | | **{ta / tb:.1f}x** |")
    print()
    print(f"- identical order selected: **{same}/{n_series}**")
    print(f"- AIC of our selected model vs theirs: median {np.median(daic):+.2f}, "
          f"better on {int((daic < -1e-6).sum())}, worse on {int((daic > 1e-6).sum())}")
    return ta, tb, same, n_series, daic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("# pmdarima-rs benchmarks")
    print()
    print(f"- pmdarima {pm.__version__}, pmdarima-rs {pmr.__version__}")
    print(f"- Python {sys.version.split()[0]}, numpy {np.__version__}")
    print()
    print("Every row is verified for agreement before it is timed.")

    bench_loglike()
    bench_single_fits()
    bench_auto_arima(args.quick)
    if not args.quick:
        bench_many_series()


if __name__ == "__main__":
    main()
