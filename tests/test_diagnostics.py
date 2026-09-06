"""Differential tests for the unit-root and seasonality machinery.

These decide `d` and `D`. A disagreement here is not a rounding difference,
it is a different model, so the bar is exact equality of the selected order
and near-exact equality of the underlying p-values.
"""

import numpy as np
import pytest

pm = pytest.importorskip("pmdarima")

from pmdarima_rs.arima import (  # noqa: E402
    ADFTest,
    CHTest,
    KPSSTest,
    OCSBTest,
    PPTest,
    ndiffs,
    nsdiffs,
)
from pmdarima_rs.utils.array import diff, diff_inv  # noqa: E402

KINDS = ("random_walk", "stationary", "seasonal", "seasonal_trend")


def make(kind, n, seed, m=12):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if kind == "random_walk":
        return np.cumsum(rng.standard_normal(n)) * 0.5 + 10
    if kind == "stationary":
        return rng.standard_normal(n) * 2 + 5
    if kind == "seasonal":
        return 10 + 3 * np.sin(2 * np.pi * t / m) + rng.standard_normal(n) * 0.4
    return (
        10.0
        + 0.05 * t
        + 5 * np.sin(2 * np.pi * t / m)
        + np.cumsum(rng.standard_normal(n)) * 0.4
    )


ALL_SERIES = [
    (kind, n, seed, m)
    for kind in KINDS
    for m in (4, 7, 12)
    for n in (60, 80, 150, 300)
    for seed in (1, 2, 3)
]


@pytest.mark.parametrize("kind,n,seed,m", ALL_SERIES)
def test_stationarity_tests_match(kind, n, seed, m):
    y = make(kind, n, seed, m)
    pairs = [
        (ADFTest(), pm.arima.ADFTest()),
        (KPSSTest(), pm.arima.KPSSTest()),
        (PPTest(), pm.arima.PPTest()),
        (KPSSTest(null="trend"), pm.arima.KPSSTest(null="trend")),
        (KPSSTest(lshort=False), pm.arima.KPSSTest(lshort=False)),
        (PPTest(lshort=False), pm.arima.PPTest(lshort=False)),
        (ADFTest(k=3), pm.arima.ADFTest(k=3)),
    ]
    for mine, ref in pairs:
        gp, gd = mine.should_diff(y)
        rp, rd = ref.should_diff(y)
        assert gd == rd, f"{type(mine).__name__}: decision differs"
        assert abs(gp - rp) < 1e-10, f"{type(mine).__name__}: p-value differs"


@pytest.mark.parametrize("kind,n,seed,m", ALL_SERIES)
def test_seasonal_tests_match(kind, n, seed, m):
    y = make(kind, n, seed, m)
    for mine, ref in (
        (CHTest(m), pm.arima.CHTest(m)),
        (OCSBTest(m), pm.arima.OCSBTest(m)),
        (OCSBTest(m, lag_method="bic"), pm.arima.OCSBTest(m, lag_method="bic")),
        (OCSBTest(m, lag_method="aicc"), pm.arima.OCSBTest(m, lag_method="aicc")),
        (OCSBTest(m, lag_method="fixed"), pm.arima.OCSBTest(m, lag_method="fixed")),
    ):
        try:
            expected = ref.estimate_seasonal_differencing_term(y)
        except Exception as exc:  # the reference itself refuses this input
            with pytest.raises(type(exc)):
                mine.estimate_seasonal_differencing_term(y)
            continue
        assert mine.estimate_seasonal_differencing_term(y) == expected


@pytest.mark.parametrize("kind,n,seed,m", ALL_SERIES)
def test_ndiffs_nsdiffs_match(kind, n, seed, m):
    y = make(kind, n, seed, m)
    for test in ("kpss", "adf", "pp"):
        assert ndiffs(y, test=test) == pm.arima.ndiffs(y, test=test)
    for test in ("ocsb", "ch"):
        assert nsdiffs(y, m, test=test) == pm.arima.nsdiffs(y, m, test=test)


@pytest.mark.parametrize("lag", [1, 2, 12])
@pytest.mark.parametrize("d", [1, 2])
def test_diff_and_diff_inv_match(lag, d):
    y = make("seasonal_trend", 80, 1)
    assert np.allclose(diff(y, lag, d), pm.utils.diff(y, lag, d))
    assert np.allclose(diff_inv(y, lag, d), pm.utils.diff_inv(y, lag, d))


def test_ocsb_singular_lag_falls_back_like_pmdarima():
    """A regression test for a real difference that cost real time.

    `OCSBTest` can select a lag of zero, which makes the regressor all zeros.
    `statsmodels`' QR path raises `LinAlgError` there, and `pmdarima` catches
    it and reuses the best regression it already had. An implementation that
    solves the singular system instead - `lstsq` returns a minimum-norm
    answer quite happily - sails past that point and fails later, somewhere
    unrelated. This pins the fallback.
    """
    y = make("seasonal_trend", 80, 1, 12)
    assert OCSBTest(12).estimate_seasonal_differencing_term(y) == (
        pm.arima.OCSBTest(12).estimate_seasonal_differencing_term(y)
    )
