"""Differential tests for the Kalman filter loglikelihood.

The whole package rests on this one number: `auto_arima` selects orders by
comparing information criteria, and every information criterion is a function
of the loglikelihood. If this matches `statsmodels` then order selection can
only differ through the optimiser, which is tested separately.

These compare against `statsmodels` evaluation-for-evaluation rather than only
comparing final fits, because a fit that happens to land in the same place can
hide a filter that is wrong everywhere else.
"""

import zlib

import numpy as np
import pytest

sm = pytest.importorskip("statsmodels.api")
from pmdarima_rs import _pmdarima_rs as R  # noqa: E402

TREND_POWERS = {None: [], "c": [0], "t": [1], "ct": [0, 1]}

# (order, seasonal_order, trend)
SPECS = [
    ((0, 0, 0), (0, 0, 0, 0), "c"),
    ((1, 0, 0), (0, 0, 0, 0), None),
    ((1, 0, 0), (0, 0, 0, 0), "c"),
    ((0, 0, 1), (0, 0, 0, 0), "c"),
    ((2, 0, 2), (0, 0, 0, 0), "c"),
    ((1, 1, 0), (0, 0, 0, 0), None),
    ((0, 1, 1), (0, 0, 0, 0), "c"),
    ((2, 1, 1), (0, 0, 0, 0), "c"),
    ((3, 2, 2), (0, 0, 0, 0), "c"),
    ((1, 0, 0), (1, 0, 0, 4), "c"),
    ((0, 1, 1), (0, 1, 1, 4), None),
    ((2, 1, 1), (1, 1, 1, 4), "c"),
    ((1, 1, 1), (1, 1, 1, 12), "c"),
    ((0, 1, 1), (0, 1, 1, 12), None),
    ((2, 0, 2), (2, 0, 2, 12), "c"),
    ((1, 0, 1), (1, 1, 1, 12), "ct"),
    ((1, 0, 0), (0, 0, 0, 0), "ct"),
    ((1, 0, 0), (0, 0, 0, 0), "t"),
    ((2, 1, 2), (2, 0, 2, 12), "c"),
]


def seed_for(spec):
    """A seed that does not move between processes.

    `hash()` is salted per interpreter for anything containing a string, so
    seeding from it made every trend-bearing specification draw a different
    parameter set on each run - a fuzz test wearing a unit test's clothes.
    `test_fuzz_random_orders` is where the fuzzing belongs; this file's job is
    to check the same points every time.
    """
    return zlib.crc32(repr(spec).encode())


def tolerance(order, sorder):
    """How close the two implementations can get for this specification.

    `statsmodels` stands in for an infinite prior variance on the differencing
    states with a large finite one (`1e6`), and forming `P - M M'/F` against
    that cancels away roughly `log10(1e6 / sigma2)` significant digits - see
    `test_approximate_diffuse_conditioning`, which measures exactly this.

    So the achievable agreement depends on the specification, and a single
    flat bound is either too loose to catch anything on the well-conditioned
    half or too tight to hold on the other - which is what a flat `1e-9` here
    turned out to be, passing only until the reference's rounding shifted
    under it.

    With no differencing there is no diffuse block and the two track each
    other to within round-off accumulated over the state and the sample; the
    worst of those, over 400 admissible draws per specification, is 2.5e-13,
    on the 26-state `(2,0,2)(2,0,2,12)`. With `d + D*m` diffuse states they
    cannot do nearly that well: 1.7e-9 with thirteen of them. The second bound
    is the one `test_fuzz_random_orders` already uses.
    """
    return 1e-12 if order[1] + sorder[1] * sorder[3] == 0 else 1e-8


def make_series(rng, n):
    t = np.arange(n)
    return (
        10.0
        + 0.02 * t
        + 3.0 * np.sin(2 * np.pi * t / 12)
        + np.cumsum(rng.standard_normal(n)) * 0.3
        + rng.standard_normal(n) * 0.5
    )


def draw_params(rng, mod):
    """Draw a parameter vector that statsmodels considers admissible.

    Sampling in the unconstrained space and pushing it through
    `transform_params` guarantees stationarity and invertibility, so the
    comparison is never contaminated by one library rejecting a point the
    other accepts.

    The innovation variance is kept away from zero deliberately. Both
    implementations initialise the differencing states with a large finite
    variance (`1e6`), so a `sigma2` many orders of magnitude below that makes
    the covariance update cancel catastrophically and costs both libraries
    about `log10(1e6 / sigma2)` digits - see
    `test_approximate_diffuse_conditioning`, which measures exactly that. A
    fitted model never sits there, so testing there would only measure
    floating-point luck.
    """
    u = rng.standard_normal(len(mod.param_names)) * 0.5
    if not mod.k_params == 0:
        # last free parameter is sigma2, entering as u**2
        u[-1] = np.sign(u[-1] or 1.0) * (0.4 + abs(u[-1]))
    return mod.transform_params(u)


@pytest.mark.parametrize("order,sorder,trend", SPECS)
def test_matches_statsmodels(order, sorder, trend):
    rng = np.random.default_rng(seed_for((order, sorder, trend)))
    y = make_series(rng, 160)
    mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=sorder, trend=trend)
    tol = tolerance(order, sorder)
    for _ in range(12):
        p = draw_params(rng, mod)
        ref = mod.loglike(p)
        got = R.loglike(y, p, order, sorder, TREND_POWERS[trend])
        assert np.isfinite(got)
        assert abs(got - ref) <= tol * max(1.0, abs(ref)), (
            f"{order} {sorder} {trend}: ref={ref!r} got={got!r}"
        )


@pytest.mark.parametrize("order,sorder,trend", SPECS)
def test_state_space_dimensions_match(order, sorder, trend):
    rng = np.random.default_rng(0)
    y = make_series(rng, 120)
    mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=sorder, trend=trend)
    k, r, kd, burn, npar = R.spec_dims(order, sorder, TREND_POWERS[trend])
    assert k == mod.k_states
    assert r == mod._k_order
    assert kd == mod._k_states_diff
    assert burn == mod.loglikelihood_burn
    assert npar == len(mod.param_names)


@pytest.mark.parametrize("n", [30, 75, 200, 500])
def test_matches_across_series_lengths(n):
    rng = np.random.default_rng(n)
    y = make_series(rng, n)
    order, sorder, trend = (1, 1, 1), (1, 1, 1, 12), "c"
    if n < 2 * 12 + 5:
        pytest.skip("series too short for a seasonal model")
    mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=sorder, trend=trend)
    for _ in range(6):
        p = draw_params(rng, mod)
        ref = mod.loglike(p)
        got = R.loglike(y, p, order, sorder, [0])
        assert abs(got - ref) <= tolerance(order, sorder) * max(1.0, abs(ref))


def test_fuzz_random_orders():
    """Random orders, random admissible parameters, random data."""
    rng = np.random.default_rng(20260906)
    worst = 0.0
    checked = 0
    for _ in range(120):
        p_ = int(rng.integers(0, 4))
        d_ = int(rng.integers(0, 3))
        q_ = int(rng.integers(0, 4))
        m = int(rng.choice([0, 4, 12]))
        if m:
            P_ = int(rng.integers(0, 2))
            D_ = int(rng.integers(0, 2))
            Q_ = int(rng.integers(0, 2))
        else:
            P_ = D_ = Q_ = 0
        trend = str(rng.choice(["c", "n"]))
        trend = None if trend == "n" else "c"
        order, sorder = (p_, d_, q_), (P_, D_, Q_, m)
        n = int(rng.integers(60, 220))
        y = make_series(rng, n)
        try:
            mod = sm.tsa.statespace.SARIMAX(
                y, order=order, seasonal_order=sorder, trend=trend
            )
        except Exception:
            continue
        if mod.k_states == 0:
            continue
        par = draw_params(rng, mod)
        ref = mod.loglike(par)
        if not np.isfinite(ref):
            continue
        got = R.loglike(y, par, order, sorder, TREND_POWERS[trend])
        rel = abs(got - ref) / max(1.0, abs(ref))
        worst = max(worst, rel)
        checked += 1
        assert rel <= 1e-8, f"{order} {sorder} {trend} n={n}: ref={ref} got={got}"
    assert checked > 80, f"only {checked} cases actually ran"
    print(f"\nfuzzed {checked} specifications, worst relative error {worst:.2e}")


def test_approximate_diffuse_conditioning():
    """Document the one place agreement is limited, and why.

    `statsmodels` initialises diffuse states with a large *finite* variance
    (`1e6`) as a stand-in for infinity. Forming `P - M M'/F` when `P` carries
    entries of `1e6` and the answer is of order `sigma2` cancels away roughly
    `log10(1e6 / sigma2)` significant digits. Both libraries pay that; they
    simply round the cancellation differently, so their agreement degrades
    predictably as `sigma2` shrinks.

    This is a property of the model specification, not of either
    implementation, and it lives far outside the region any optimiser visits.
    The test pins the behaviour so a genuine regression cannot hide inside it.
    """
    rng = np.random.default_rng(3)
    n = 120
    t = np.arange(n)
    y = 10.0 + 0.02 * t + np.cumsum(rng.standard_normal(n)) * 0.3
    order, sorder = (1, 1, 0), (0, 0, 0, 0)
    mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=sorder, trend=None)

    # Well-conditioned: agreement is at machine precision.
    for s2 in (1e2, 1e0):
        p = np.array([0.4, s2])
        ref = mod.loglike(p)
        got = R.loglike(y, p, order, sorder, [])
        assert abs(got - ref) <= 1e-13 * abs(ref), f"sigma2={s2}"

    # Ill-conditioned: still close, but no longer to machine precision.
    for s2 in (1e-6, 1e-8):
        p = np.array([0.4, s2])
        ref = mod.loglike(p)
        got = R.loglike(y, p, order, sorder, [])
        assert abs(got - ref) <= 1e-4 * abs(ref), f"sigma2={s2}"


def test_steady_state_short_circuit_matches():
    """The filter must freeze at the same period statsmodels freezes at.

    For a time-invariant system the Riccati recursion reaches a fixed point,
    and statsmodels stops updating the covariance once successive predicted
    covariances differ by less than `tolerance`. That is an approximation:
    with `tolerance=0` statsmodels agrees with a fully-converged filter far
    more tightly. Reproducing the short-circuit is what keeps the two
    libraries agreeing on the *same* answer, and it is also why the filter
    becomes O(k) per step after convergence.
    """
    rng = np.random.default_rng(11)
    y = make_series(rng, 300)
    order, sorder = (0, 1, 1), (0, 1, 1, 4)
    mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=sorder, trend=None)
    assert mod.ssm.time_invariant, "this test needs a time-invariant system"
    for _ in range(8):
        p = draw_params(rng, mod)
        ref = mod.loglike(p)
        got = R.loglike(y, p, order, sorder, [])
        assert abs(got - ref) <= 1e-9 * max(1.0, abs(ref))
