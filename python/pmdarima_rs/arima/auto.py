"""`auto_arima`: pick d and D by test, then search for p, q, P, Q.

A port of `pmdarima.arima.auto` (MIT, Taylor G. Smith et al.). The order of
operations here is the algorithm - which differencing test runs on which
series, when the intercept is decided, which short-circuits fire for constant
data - so it is reproduced rather than reinterpreted.
"""

import time
import warnings

import numpy as np

from ..base import BaseARIMA
from ..compat.sklearn import NotFittedError
from ..compat.statsmodels import check_seasonal_order
from ..utils.array import check_endog, check_exog, diff, is_iterable
from . import _auto_solvers as solvers
from . import _validation as val
from .utils import is_constant, ndiffs, nsdiffs

__all__ = ["auto_arima", "AutoARIMA", "StepwiseContext"]

from ._context import StepwiseContext  # noqa: E402,F401  (re-exported)

VALID_CRITERIA = val.VALID_CRITERIA


def _return_wrapper(fits, return_all, start, trace):
    """Return the best fit, or all of them, as `pmdarima` shapes it."""
    if not is_iterable(fits):
        fits = [fits]
    if trace:
        print("Total fit time: %.3f seconds" % (time.time() - start))
    if not return_all:
        return fits[0]
    # `pmdarima` hands back a tuple here, and callers unpack and index it.
    return tuple(fits)


def auto_arima(
    y,
    X=None,
    start_p=2,
    d=None,
    start_q=2,
    max_p=5,
    max_d=2,
    max_q=5,
    start_P=1,
    D=None,
    start_Q=1,
    max_P=2,
    max_D=1,
    max_Q=2,
    max_order=5,
    m=1,
    seasonal=True,
    stationary=False,
    information_criterion="aic",
    alpha=0.05,
    test="kpss",
    seasonal_test="ocsb",
    stepwise=True,
    n_jobs=1,
    start_params=None,
    trend=None,
    method="lbfgs",
    maxiter=50,
    offset_test_args=None,
    seasonal_test_args=None,
    suppress_warnings=True,
    error_action="trace",
    trace=False,
    random=False,
    random_state=None,
    n_fits=10,
    return_valid_fits=False,
    out_of_sample_size=0,
    scoring="mse",
    scoring_args=None,
    with_intercept="auto",
    sarimax_kwargs=None,
    **fit_args,
):
    """Automatically discover the optimal ARIMA order for a series."""
    import functools

    offset_test_args = val.check_kwargs(offset_test_args)
    seasonal_test_args = val.check_kwargs(seasonal_test_args)
    scoring_args = val.check_kwargs(scoring_args)
    sarimax_kwargs = val.check_kwargs(sarimax_kwargs)

    m = val.check_m(m, seasonal)
    trace = val.check_trace(trace)
    n_jobs = val.check_n_jobs(stepwise, n_jobs)

    start_p, max_p = val.check_start_max_values(start_p, max_p, "p")
    start_q, max_q = val.check_start_max_values(start_q, max_q, "q")
    start_P, max_P = val.check_start_max_values(start_P, max_P, "P")
    start_Q, max_Q = val.check_start_max_values(start_Q, max_Q, "Q")

    for _d, _max_d in ((d, max_d), (D, max_D)):
        if _max_d < 0:
            raise ValueError("max_d & max_D must be positive integers (>= 0)")
        if _d is not None and _d < 0:
            raise ValueError("d & D must be None or a positive integer (>= 0)")
    if random and n_fits < 0:
        raise ValueError("n_fits must be a positive integer for a random search")
    actions = {"warn", "raise", "ignore", "trace", None}
    if error_action not in actions:
        raise ValueError(
            "error_action must be one of %r, but got %r" % (actions, error_action)
        )

    start = time.time()
    y = check_endog(y, copy=False, preserve_series=True)
    n_samples = y.shape[0]
    if X is not None:
        X = check_exog(X, force_all_finite=False, copy=False)

    fit_partial = functools.partial(
        solvers._fit_candidate_model,
        start_params=start_params,
        trend=trend,
        method=method,
        maxiter=maxiter,
        fit_params=fit_args,
        suppress_warnings=suppress_warnings,
        trace=trace,
        error_action=error_action,
        scoring=scoring,
        out_of_sample_size=out_of_sample_size,
        scoring_args=scoring_args,
        information_criterion=information_criterion,
    )

    if is_constant(y):
        warnings.warn(
            "Input time-series is completely constant; returning a (0, 0, 0) ARMA."
        )
        return _return_wrapper(
            solvers._sort_and_filter_fits(
                fit_partial(
                    y,
                    X=X,
                    order=(0, 0, 0),
                    seasonal_order=(0, 0, 0, 0),
                    with_intercept=val.auto_intercept(with_intercept, False),
                    **sarimax_kwargs,
                )
            ),
            return_valid_fits,
            start,
            trace,
        )

    information_criterion = val.check_information_criterion(
        information_criterion, out_of_sample_size
    )

    max_p = int(min(max_p, np.floor(n_samples / 3)))
    max_q = int(min(max_q, np.floor(n_samples / 3)))
    start_p = min(start_p, max_p)
    start_q = min(start_q, max_q)

    if not seasonal:
        D = m = -1

    xx = np.asarray(y, dtype=float).copy()
    if X is not None:
        Xa = np.asarray(X, dtype=float)
        design = np.hstack([np.ones((Xa.shape[0], 1)), Xa])
        beta, *_ = np.linalg.lstsq(design, xx, rcond=None)
        xx = xx - design.dot(beta)

    if stationary:
        d = D = 0
    if m == 1:
        D = max_P = max_Q = 0
    elif D is None:
        D = nsdiffs(xx, m=m, test=seasonal_test, max_D=max_D, **seasonal_test_args)
        if D > 0 and X is not None:
            diffxreg = diff(np.asarray(X, dtype=float), differences=D, lag=m)
            if np.apply_along_axis(is_constant, arr=diffxreg, axis=0).any():
                D -= 1

    dx = diff(xx, differences=D, lag=m) if D > 0 else xx
    if dx.shape[0] == 0:
        raise ValueError(
            f"The seasonal differencing order, D={D}, was too large for your "
            "time series, and after differencing, there are no samples "
            "remaining in your data. Try a smaller value for D, or if you "
            "didn't set D to begin with, try setting it explicitly. This can "
            "also occur in seasonal settings when m is too large."
        )

    if X is not None:
        Xa = np.asarray(X, dtype=float)
        diffxreg = diff(Xa, differences=D, lag=m) if D > 0 else Xa
    else:
        diffxreg = None

    if d is None:
        d = ndiffs(dx, test=test, alpha=alpha, max_d=max_d, **offset_test_args)
        if d > 0 and X is not None:
            diffxreg = diff(diffxreg, differences=d, lag=1)
            if np.apply_along_axis(is_constant, arr=diffxreg, axis=0).any():
                d -= 1

    if not suppress_warnings:
        val.warn_for_D(d=d, D=D)

    if d > 0:
        dx = diff(dx, differences=d, lag=1)

    if is_constant(dx):
        ssn = (0, 0, 0, 0) if not seasonal else check_seasonal_order((0, D, 0, m))
        if D > 0 and d == 0:
            with_intercept = val.auto_intercept(with_intercept, True)
        elif D > 0 and d > 0:
            pass
        elif d == 2:
            pass
        elif d < 2:
            with_intercept = val.auto_intercept(with_intercept, True)
        else:
            raise ValueError(
                "data follow a simple polynomial and are not suitable for "
                "ARIMA modeling"
            )
        return _return_wrapper(
            solvers._sort_and_filter_fits(
                fit_partial(
                    y,
                    X=X,
                    order=(0, d, 0),
                    seasonal_order=ssn,
                    with_intercept=with_intercept,
                    **sarimax_kwargs,
                )
            ),
            return_valid_fits,
            start,
            trace,
        )

    if m > 1:
        if max_P > 0:
            max_p = min(max_p, m - 1)
        if max_Q > 0:
            max_q = min(max_q, m - 1)

    if with_intercept == "auto":
        with_intercept = (d + D) in (0, 1)

    if not stepwise:
        if max_order is None:
            max_order = np.inf
        elif max_order < 0:
            raise ValueError("max_order must be None or a positive integer (>= 0)")
        search = solvers._RandomFitWrapper(
            y=y,
            X=X,
            fit_partial=fit_partial,
            d=d,
            D=D,
            m=m,
            max_order=max_order,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            random=random,
            random_state=random_state,
            n_fits=n_fits,
            n_jobs=n_jobs,
            seasonal=seasonal,
            trace=trace,
            with_intercept=with_intercept,
            sarimax_kwargs=sarimax_kwargs,
        )
    else:
        if n_samples < 10:
            start_p = min(start_p, 1)
            start_q = min(start_q, 1)
            start_P = start_Q = 0

        search = solvers._StepwiseFitWrapper(
            y,
            X=X,
            start_params=start_params,
            trend=trend,
            method=method,
            maxiter=maxiter,
            fit_params=fit_args,
            suppress_warnings=suppress_warnings,
            trace=trace,
            error_action=error_action,
            out_of_sample_size=out_of_sample_size,
            scoring=scoring,
            scoring_args=scoring_args,
            p=min(start_p, max_p),
            d=d,
            q=min(start_q, max_q),
            P=min(start_P, max_P),
            D=D,
            Q=min(start_Q, max_Q),
            m=m,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            seasonal=seasonal,
            information_criterion=information_criterion,
            with_intercept=with_intercept,
            **sarimax_kwargs,
        )

    sorted_res = search.solve()
    return _return_wrapper(sorted_res, return_valid_fits, start, trace)


class AutoARIMA(BaseARIMA):
    """An estimator wrapping :func:`auto_arima`.

    Every `auto_arima` argument is a constructor argument, so the estimator
    can be cloned, grid-searched and introspected the way `pmdarima`'s can.
    """

    def __init__(
        self,
        start_p=2,
        d=None,
        start_q=2,
        max_p=5,
        max_d=2,
        max_q=5,
        start_P=1,
        D=None,
        start_Q=1,
        max_P=2,
        max_D=1,
        max_Q=2,
        max_order=5,
        m=1,
        seasonal=True,
        stationary=False,
        information_criterion="aic",
        alpha=0.05,
        test="kpss",
        seasonal_test="ocsb",
        stepwise=True,
        n_jobs=1,
        start_params=None,
        trend=None,
        method="lbfgs",
        maxiter=50,
        offset_test_args=None,
        seasonal_test_args=None,
        suppress_warnings=True,
        error_action="trace",
        trace=False,
        random=False,
        random_state=None,
        n_fits=10,
        out_of_sample_size=0,
        scoring="mse",
        scoring_args=None,
        with_intercept="auto",
        **kwargs,
    ):
        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.max_order = max_order
        self.m = m
        self.seasonal = seasonal
        self.stationary = stationary
        self.information_criterion = information_criterion
        self.alpha = alpha
        self.test = test
        self.seasonal_test = seasonal_test
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.maxiter = maxiter
        self.offset_test_args = offset_test_args
        self.seasonal_test_args = seasonal_test_args
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.trace = trace
        self.random = random
        self.random_state = random_state
        self.n_fits = n_fits
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.with_intercept = with_intercept

        for deprecated in ("solver", "transparams"):
            if kwargs.pop(deprecated, None):
                warnings.warn(
                    "%s has been deprecated and will be removed in a future "
                    "version." % deprecated,
                    DeprecationWarning,
                )
        self.kwargs = kwargs

    _AUTO_PARAMS = (
        "start_p", "d", "start_q", "max_p", "max_d", "max_q", "start_P", "D",
        "start_Q", "max_P", "max_D", "max_Q", "max_order", "m", "seasonal",
        "stationary", "information_criterion", "alpha", "test",
        "seasonal_test", "stepwise", "n_jobs", "start_params", "trend",
        "method", "maxiter", "offset_test_args", "seasonal_test_args",
        "suppress_warnings", "error_action", "trace", "random",
        "random_state", "n_fits", "out_of_sample_size", "scoring",
        "scoring_args", "with_intercept",
    )

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._AUTO_PARAMS}

    def set_params(self, **params):
        for k, v in params.items():
            if k in self._AUTO_PARAMS:
                setattr(self, k, v)
            else:
                self.kwargs[k] = v
        return self

    def fit(self, y, X=None, **fit_args):
        """Fit an `auto_arima` search and keep the winning model."""
        sarimax_kwargs = {} if not self.kwargs else self.kwargs
        self.model_ = auto_arima(
            y,
            X=X,
            **{k: getattr(self, k) for k in self._AUTO_PARAMS},
            return_valid_fits=False,
            sarimax_kwargs=sarimax_kwargs,
            **fit_args,
        )
        return self

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise NotFittedError("Model has not been fit!")

    def predict(self, n_periods=10, X=None, return_conf_int=False, alpha=0.05):
        self._check_fitted()
        return self.model_.predict(
            n_periods=n_periods,
            X=X,
            return_conf_int=return_conf_int,
            alpha=alpha,
        )

    def predict_in_sample(
        self,
        X=None,
        start=None,
        end=None,
        dynamic=False,
        return_conf_int=False,
        alpha=0.05,
        typ="levels",
    ):
        self._check_fitted()
        return self.model_.predict_in_sample(
            X=X,
            start=start,
            end=end,
            dynamic=dynamic,
            return_conf_int=return_conf_int,
            alpha=alpha,
        )

    def update(self, y, X=None, maxiter=None, **kwargs):
        self._check_fitted()
        self.model_.update(y, X=X, maxiter=maxiter, **kwargs)
        return self

    def summary(self):
        self._check_fitted()
        return self.model_.summary()
