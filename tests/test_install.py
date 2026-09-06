"""Tests that need no reference implementation.

Everything else in this suite is differential against `pmdarima`. These are
the checks that must hold even where `pmdarima` cannot be installed - which
includes Python 3.14, where it currently has no wheel for some platforms, and
which is exactly the situation a user reaching for this package may be in.
"""

import numpy as np
import pytest

import pmdarima_rs as pmr


def test_imports_and_reports_a_version():
    assert isinstance(pmr.__version__, str)
    assert pmr.__version__.count(".") >= 1


def test_core_symbols_exist():
    for name in pmr.__all__:
        assert hasattr(pmr, name), name


def test_fit_predict_round_trip():
    y = pmr.datasets.load_wineind()
    model = pmr.arima.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(y)
    preds = np.asarray(model.predict(n_periods=12), dtype=float)
    assert preds.shape == (12,)
    assert np.all(np.isfinite(preds))
    assert np.isfinite(model.aic())


def test_auto_arima_runs_without_a_reference():
    y = pmr.datasets.load_wineind()
    model = pmr.auto_arima(
        y, seasonal=True, m=12, suppress_warnings=True, error_action="ignore"
    )
    assert len(model.order) == 3
    assert np.all(np.isfinite(np.asarray(model.predict(n_periods=6), dtype=float)))


def test_pandas_series_in_series_out():
    """A `Series` in must give a `Series` out, with a continued index."""
    import pandas as pd

    y = pd.Series(
        np.asarray(pmr.datasets.load_wineind(), dtype=float),
        index=pd.date_range("1980-01-01", periods=176, freq="ME"),
    )
    model = pmr.arima.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(y)
    preds = model.predict(n_periods=6)
    assert isinstance(preds, pd.Series)
    assert len(preds) == 6
    assert preds.index[0] > y.index[-1]


def test_missing_values_are_handled():
    """NaNs are legitimate input: the filter skips the update and propagates.

    Rejecting them would remove a capability the reference has, and silently
    dropping them would shift every subsequent lag.
    """
    y = np.asarray(pmr.datasets.load_wineind(), dtype=float).copy()
    y[50] = np.nan
    y[51] = np.nan
    model = pmr.arima.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(y)
    assert np.isfinite(model.res_.loglike)
    assert np.all(np.isfinite(np.asarray(model.predict(n_periods=6), dtype=float)))


def test_constant_series_short_circuits():
    y = np.full(60, 7.0)
    with pytest.warns(UserWarning):
        model = pmr.auto_arima(y, seasonal=False, suppress_warnings=False)
    assert model.order == (0, 0, 0)


@pytest.mark.parametrize("trend", [None, "c", "t", "ct"])
def test_every_trend_fits_and_forecasts(trend):
    y = pmr.datasets.load_wineind()
    model = pmr.arima.ARIMA(
        order=(1, 1, 1), trend=trend, with_intercept=False, suppress_warnings=True
    ).fit(y)
    preds = np.asarray(model.predict(n_periods=6), dtype=float)
    assert np.all(np.isfinite(preds))


def test_exogenous_regressors_round_trip():
    y = np.asarray(pmr.datasets.load_wineind(), dtype=float)
    n = len(y)
    rng = np.random.default_rng(0)
    X = np.column_stack([np.arange(n) / n, rng.standard_normal(n)])
    model = pmr.arima.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(y, X=X)
    Xf = np.column_stack([np.arange(n, n + 6) / n, rng.standard_normal(6)])
    preds = np.asarray(model.predict(n_periods=6, X=Xf), dtype=float)
    assert preds.shape == (6,)
    assert np.all(np.isfinite(preds))
    with pytest.raises(ValueError):
        model.predict(n_periods=6)  # fitted with exog, so exog is required


def test_forecast_variance_grows_with_horizon():
    """A sanity property no reference is needed to state."""
    y = pmr.datasets.load_wineind()
    model = pmr.arima.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(y)
    _, conf = model.predict(n_periods=24, return_conf_int=True)
    width = conf[:, 1] - conf[:, 0]
    assert np.all(np.diff(width) > -1e-9), "prediction intervals must not narrow"


def test_repeated_fits_are_deterministic():
    y = pmr.datasets.load_wineind()
    a = pmr.arima.ARIMA(order=(2, 1, 1), suppress_warnings=True).fit(y)
    b = pmr.arima.ARIMA(order=(2, 1, 1), suppress_warnings=True).fit(y)
    assert np.array_equal(a.params(), b.params())


def test_parallel_gradient_matches_serial():
    """Threading must not change the answer."""
    from pmdarima_rs import _fit as rs_fit
    from pmdarima_rs._ssm import Spec

    y = np.asarray(pmr.datasets.load_wineind(), dtype=float)
    spec = Spec((2, 1, 1), (1, 0, 1, 12), "c")
    a = rs_fit.fit(spec, y, parallel=False)
    b = rs_fit.fit(spec, y, parallel=True)
    assert abs(a.loglike - b.loglike) < 1e-10
    assert np.allclose(a.params, b.params, atol=1e-10)


def test_restarts_never_lower_the_likelihood():
    """The stronger optimiser is opt-in, and must only ever help."""
    from pmdarima_rs import _fit as rs_fit
    from pmdarima_rs._ssm import Spec

    y = np.asarray(pmr.datasets.load_wineind(), dtype=float)
    for order, sorder in (((2, 1, 1), (0, 0, 0, 0)), ((1, 1, 1), (1, 0, 1, 12))):
        spec = Spec(order, sorder, "c")
        base = rs_fit.fit(spec, y)
        strong = rs_fit.fit(spec, y, restarts=3, m=20, epsilon=None)
        assert strong.loglike >= base.loglike - 1e-8
