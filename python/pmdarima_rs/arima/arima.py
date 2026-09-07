"""The `ARIMA` estimator.

A drop-in for `pmdarima.arima.ARIMA`: same constructor arguments, same
methods, same return types, including the pandas index behaviour that decides
whether you get an array or a `Series` back.

The only compiled part is what runs thousands of times - the likelihood and
its gradient. Everything visible here runs once per fit.
"""

import datetime
import warnings

import numpy as np
import pandas as pd

from .. import _fit as _fitmod
from .._ssm import Spec
from ..base import BaseARIMA
from ..compat.sklearn import NotFittedError, check_is_fitted
from ..utils.array import check_endog, check_exog, is_iterable
from ..utils.metaestimators import if_has_delegate
from ._summary import Summary, break_variance, jarque_bera, ljung_box

# `pmdarima` exposes `ARMAtoMA` from this module as well as from `approx`.
from .approx import ARMAtoMA  # noqa: F401  (re-exported)

__all__ = ["ARIMA", "ARMAtoMA"]


def _mse(y_true, y_pred, **kw):
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def _mae(y_true, y_pred, **kw):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def _smape(y_true, y_pred, **kw):
    from ..metrics import smape

    return float(smape(y_true, y_pred))


VALID_SCORING = {
    "mse": _mse,
    "mean_squared_error": _mse,
    "mae": _mae,
    "mean_absolute_error": _mae,
    "smape": _smape,
}


def _get_scoring(scoring):
    if callable(scoring):
        return scoring
    if isinstance(scoring, str):
        try:
            return VALID_SCORING[scoring]
        except KeyError:
            raise ValueError(
                "'%s' is not a valid scoring method." % scoring
            ) from None
    raise TypeError(
        "`metric` must be a valid scoring method, or a callable, but got "
        "type=%s" % type(scoring)
    )


#: The optimisers `statsmodels` accepts. Only `lbfgs` is implemented here -
#: it is pmdarima's default and the only one `auto_arima` selects - but a name
#: outside this set is a caller error and has to be rejected, not warned about.
VALID_METHODS = (
    "newton",
    "nm",
    "bfgs",
    "lbfgs",
    "powell",
    "cg",
    "ncg",
    "basinhopping",
    "minimize",
)


def _norm_ppf(q):
    """Inverse standard normal CDF, so scipy.stats is not imported per call."""
    from scipy.special import ndtri

    return float(ndtri(q))


#: Options `statsmodels` accepts that leave the likelihood this package
#: computes unchanged, so they can be honoured or safely ignored.
_SARIMAX_SUPPORTED = frozenset(
    {
        "enforce_stationarity",
        "enforce_invertibility",
        "concentrate_scale",
        "trend_offset",
        "hamilton_representation",
        "validate_specification",
        "mle_regression",
    }
)

#: Options that change the model. This package does not implement them, and
#: accepting them quietly would report a different model's numbers under the
#: caller's specification - so they are refused by name instead.
_SARIMAX_UNSUPPORTED = {
    "simple_differencing": (
        "differencing the series before filtering, which changes the "
        "likelihood, the sample size and the scale predictions come back on"
    ),
    "measurement_error": "an additional observation-error variance parameter",
    "time_varying_regression": "regression coefficients that evolve as states",
    "use_exact_diffuse": "the exact diffuse initialisation",
}


def _validate_sarimax_kwargs(kwargs):
    """Reject what is unknown, and what is known but not implemented."""
    unknown = sorted(
        k
        for k in kwargs
        if k not in _SARIMAX_SUPPORTED and k not in _SARIMAX_UNSUPPORTED
    )
    if unknown:
        raise TypeError(f"Unknown keyword arguments: {unknown}.")
    for key, what in _SARIMAX_UNSUPPORTED.items():
        if kwargs.get(key):
            raise NotImplementedError(
                f"{key}=True is not implemented in pmdarima-rs. It selects "
                f"{what}, which this package's filter does not model. Use "
                f"`pmdarima` for this specification rather than a silently "
                f"different one."
            )
    if kwargs.get("mle_regression", True) is False:
        raise NotImplementedError(
            "mle_regression=False is not implemented in pmdarima-rs. It puts "
            "the regression coefficients in the state vector rather than "
            "estimating them by maximum likelihood."
        )


def _supported_index(index, n):
    """Whether `statsmodels` would accept this index, and how to extend it.

    statsmodels accepts a default integer range, or a dated index that has a
    frequency. Anything else - an integer index that does not start at zero, a
    string index - makes it refuse to build a date index at all, and
    `pmdarima` propagates that refusal.
    """
    if index is None:
        return True
    if isinstance(index, pd.RangeIndex):
        return index.start == 0 and index.step == 1
    if isinstance(index, pd.PeriodIndex):
        return index.freq is not None
    if isinstance(index, pd.DatetimeIndex):
        return index.freq is not None or pd.infer_freq(index) is not None
    values = np.asarray(index)
    if values.dtype.kind in "iu":
        return bool(np.array_equal(values, np.arange(n)))
    return False


class ARIMA(BaseARIMA):
    """An ARIMA/SARIMAX estimator with a `pmdarima`-compatible interface."""

    def __init__(
        self,
        order,
        seasonal_order=(0, 0, 0, 0),
        start_params=None,
        method="lbfgs",
        maxiter=50,
        suppress_warnings=False,
        out_of_sample_size=0,
        scoring="mse",
        scoring_args=None,
        trend=None,
        with_intercept=True,
        **sarimax_kwargs,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.start_params = start_params
        self.method = method
        self.maxiter = maxiter
        self.suppress_warnings = suppress_warnings
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.trend = trend
        self.with_intercept = with_intercept
        for deprecated_key, still_in_use in (
            ("disp", True),
            ("callback", True),
            ("transparams", False),
            ("solver", False),
        ):
            if sarimax_kwargs.pop(deprecated_key, None):
                msg = (
                    f"'{deprecated_key}' is deprecated in the ARIMA constructor "
                    "and will be removed in a future release."
                )
                if still_in_use:
                    msg += " Pass via **fit_kwargs instead"
                warnings.warn(msg, DeprecationWarning)
        _validate_sarimax_kwargs(sarimax_kwargs)
        self.sarimax_kwargs = sarimax_kwargs

    # ---------------------------------------------------------------- sklearn
    _ctor_params = (
        "maxiter",
        "method",
        "order",
        "out_of_sample_size",
        "scoring",
        "scoring_args",
        "seasonal_order",
        "start_params",
        "suppress_warnings",
        "trend",
        "with_intercept",
    )

    _ctor_defaults = {
        "seasonal_order": (0, 0, 0, 0),
        "start_params": None,
        "method": "lbfgs",
        "maxiter": 50,
        "suppress_warnings": False,
        "out_of_sample_size": 0,
        "scoring": "mse",
        "scoring_args": None,
        "trend": None,
        "with_intercept": True,
    }

    def get_params(self, deep=True):
        """The constructor arguments, as scikit-learn expects them.

        `pmdarima` leaves the `**sarimax_kwargs` out of this, and `clone`
        relies on that, so they are left out here too.
        """
        return {k: getattr(self, k) for k in self._ctor_params}

    def set_params(self, **params):
        for k, v in params.items():
            if k in self._ctor_params:
                setattr(self, k, v)
            else:
                self.sarimax_kwargs[k] = v
        return self

    def __repr__(self):
        """Show only what differs from the defaults, as scikit-learn does."""
        parts = []
        for k in self._ctor_params:
            v = getattr(self, k)
            if k in self._ctor_defaults and v == self._ctor_defaults[k]:
                continue
            parts.append(f"{k}={v!r}")
        return f"ARIMA({', '.join(parts)})"

    def __str__(self):
        """The debug string `auto_arima` prints in its trace."""
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        int_str = "intercept"
        return " ARIMA({p},{d},{q})({P},{D},{Q})[{m}] {intercept}".format(
            p=p,
            d=d,
            q=q,
            P=P,
            D=D,
            Q=Q,
            m=m,
            # just for consistent spacing
            intercept=int_str if self.with_intercept else " " * len(int_str),
        )

    # ------------------------------------------------------------------ spec
    def _resolve_trend(self):
        trend = self.trend
        if trend is None and self.with_intercept:
            trend = "c"
        return trend

    def _build_spec(self, k_exog):
        kw = dict(self.sarimax_kwargs)
        if kw.get("hamilton_representation") and (
            self.order[1] > 0 or self.seasonal_order[1] > 0
        ):
            # statsmodels refuses this combination; the representation only
            # exists for models with no differencing states.
            raise ValueError(
                "The Hamilton representation is only available for models in "
                "which there is no differencing integrated into the state "
                "vector. Set `simple_differencing` to True or set "
                "`hamilton_representation` to False"
            )
        return Spec(
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self._resolve_trend(),
            k_exog=k_exog,
            enforce_stationarity=kw.get("enforce_stationarity", True),
            enforce_invertibility=kw.get("enforce_invertibility", True),
            concentrate_scale=kw.get("concentrate_scale", False),
            trend_offset=kw.get("trend_offset", 1),
        )

    # ------------------------------------------------------------------- fit
    def _fit(self, y, X=None, **fit_args):
        y_arr = np.asarray(y, dtype=float).ravel()
        k_exog = 0 if X is None else np.asarray(X).reshape(len(y_arr), -1).shape[1]
        spec = self._build_spec(k_exog)

        method = fit_args.pop("method", self.method)
        if method is None:
            raise ValueError("Expected non-None value for `method`")

        if method not in VALID_METHODS:
            raise ValueError(
                "method must be one of: %s"
                % ", ".join(repr(m) for m in VALID_METHODS)
            )

        maxiter = fit_args.pop("maxiter", self.maxiter)
        if maxiter is None:
            raise ValueError("Expected non-None value for `maxiter`")
        start_params = fit_args.pop("start_params", self.start_params)
        fit_args.pop("disp", None)

        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            if method != "lbfgs":
                # Say so rather than pretending. `lbfgs` is pmdarima's default
                # and the only solver its own `auto_arima` selects, so this is
                # a rarely-trodden path - but silently substituting a
                # different optimiser would make a differing fit look like a
                # bug. It sits inside the suppression block so that
                # `suppress_warnings` means what it does everywhere else.
                warnings.warn(
                    f"method={method!r} is not implemented; using 'lbfgs'. It "
                    "is pmdarima's default and the only method auto_arima "
                    "uses.",
                    UserWarning,
                    stacklevel=3,
                )
            res = _fitmod.fit(
                spec,
                y_arr,
                exog=X,
                start_params=start_params,
                maxiter=maxiter,
                **{k: v for k, v in fit_args.items() if k in ("parallel", "restarts")},
            )

        self.spec_ = spec
        self.res_ = res
        self._y = y_arr
        self._X = None if X is None else np.asarray(X, dtype=float).reshape(len(y_arr), -1)
        self.fit_with_exog_ = X is not None
        self.nobs_ = y_arr.shape[0]
        self.arima_res_ = _ResultsShim(self)
        self._fit_date = datetime.date.today().strftime("%a, %d %b %Y")
        self._fit_time = datetime.datetime.now().strftime("%H:%M:%S")
        from .. import __version__

        self.pkg_version_ = __version__
        return self

    def fit(self, y, X=None, **fit_args):
        y = check_endog(y, copy=False, preserve_series=True)
        n_samples = y.shape[0]
        index = y.index if isinstance(y, pd.Series) else None
        if not _supported_index(index, n_samples):
            raise ValueError("No supported index is available.")
        self.endog_index_ = index
        self._endog_name = getattr(y, "name", None) or "y"
        if X is not None:
            X = check_exog(X, force_all_finite=False, copy=False)
            if not np.all(np.isfinite(np.asarray(X, dtype=float))):
                # statsmodels rejects a design it cannot filter; doing it here
                # keeps the failure at the call the user made.
                from ..compat import MissingDataError

                raise MissingDataError("exog contains inf or nans")

        cv = max(int(self.out_of_sample_size), 0)
        scoring = _get_scoring(self.scoring)
        if cv >= n_samples:
            raise ValueError("out-of-sample size must be less than number of samples!")

        cv_samples = cv_exog = None
        if cv:
            cv_samples = np.asarray(y)[-cv:]
            y = np.asarray(y)[:-cv]
            if index is not None:
                self.endog_index_ = index[:-cv]
            if X is not None:
                Xa = np.asarray(X)
                cv_exog = Xa[len(Xa) - cv :]
                X = Xa[: len(Xa) - cv]

        self._fit(y, X, **fit_args)

        if cv_samples is not None:
            pred = self.predict(n_periods=cv, X=cv_exog)
            scoring_args = self.scoring_args or {}
            self.oob_ = scoring(cv_samples, np.asarray(pred), **scoring_args)
            self.oob_preds_ = pred
            # `pmdarima` folds the held-out tail back in so the returned model
            # is fit on everything, and `update` is how it does it.
            self.endog_index_ = index
            self.update(cv_samples, cv_exog, **fit_args)
        else:
            self.oob_ = np.nan
            self.oob_preds_ = None
        return self

    def fit_predict(self, y, X=None, n_periods=10, **fit_args):
        self.fit(y, X, **fit_args)
        return self.predict(n_periods=n_periods, X=X)

    def _check_exog(self, X):
        if self.fit_with_exog_:
            if X is None:
                raise ValueError(
                    "When an ARIMA is fit with an X array, it must also be "
                    "provided one for predicting or updating observations."
                )
            X = check_exog(X, force_all_finite=True)
            k_exog = self.spec_.k_exog
            if np.asarray(X).shape[1] != k_exog:
                raise ValueError(
                    "Provided exogenous values are not of the appropriate "
                    f"shape. Required (n, {k_exog}), got "
                    f"{np.asarray(X).shape}."
                )
            return X
        return None

    # -------------------------------------------------------------- predict
    def _rs_kwargs(self, exog_flat):
        spec = self.spec_
        return dict(
            order=spec.order,
            seasonal_order=(spec.bp, spec.bd, spec.bq, spec.s),
            trend_powers=list(spec.trend_powers),
            exog=exog_flat,
            k_exog=spec.k_exog,
            trend_offset=spec.trend_offset,
            enforce_stationarity=spec.enforce_stationarity,
            enforce_invertibility=spec.enforce_invertibility,
            concentrate_scale=spec.concentrate_scale,
        )

    def _forecast(self, n_periods, X=None, y=None, X_in=None):
        """Filter `y` (the training data by default), then forecast."""
        from .. import _pmdarima_rs as _rs

        spec = self.spec_
        y = self._y if y is None else np.ascontiguousarray(np.asarray(y, float))
        X_in = self._X if X_in is None else X_in
        exog = _fitmod.flatten_exog(X_in, len(y)) if X_in is not None else None
        exog_f = None
        if X is not None and n_periods:
            Xa = np.asarray(X, dtype=float).reshape(n_periods, -1)
            exog_f = np.asfortranarray(Xa).ravel(order="F")
        kw = self._rs_kwargs(exog)
        kw.pop("exog")
        return _rs.forecast(
            y,
            self.res_.params,
            kw.pop("order"),
            kw.pop("seasonal_order"),
            kw.pop("trend_powers"),
            n_periods,
            exog=exog,
            exog_future=exog_f,
            **kw,
        )

    def _index_slice(self, start, end):
        """The index labels for in-sample positions `start..end`, if any."""
        idx = getattr(self, "endog_index_", None)
        if idx is None:
            return None
        n = self.nobs_
        if end < n:
            return idx[start : end + 1]
        head = idx[start:n]
        tail = self._forecast_index(end - n + 1)
        if tail is None:
            return None
        return head.append(tail)

    def _forecast_index(self, n_periods):
        idx = getattr(self, "endog_index_", None)
        if idx is None or n_periods <= 0:
            return None
        try:
            if isinstance(idx, pd.PeriodIndex):
                return pd.period_range(idx[-1] + 1, periods=n_periods, freq=idx.freq)
            if isinstance(idx, pd.DatetimeIndex):
                freq = idx.freq or pd.infer_freq(idx)
                if freq is None:
                    return None
                return pd.date_range(
                    idx[-1] + pd.tseries.frequencies.to_offset(freq),
                    periods=n_periods,
                    freq=freq,
                )
            if isinstance(idx, pd.RangeIndex) or np.issubdtype(
                np.asarray(idx).dtype, np.integer
            ):
                last = int(np.asarray(idx)[-1])
                return pd.RangeIndex(last + 1, last + 1 + n_periods)
        except Exception:  # pragma: no cover - index shapes vary widely
            return None
        return None

    @staticmethod
    def _check_alpha(alpha):
        if not (0 < alpha < 2):
            raise ValueError(
                "alpha must be between 0 and 2 exclusive; a confidence "
                f"interval is not defined at alpha={alpha}"
            )

    def predict(self, n_periods=10, X=None, return_conf_int=False, alpha=0.05, **kwargs):
        check_is_fitted(self, "arima_res_")
        if not isinstance(n_periods, (int, np.integer)):
            raise TypeError("n_periods must be an int")
        X = self._check_exog(X)
        if X is not None and np.asarray(X).shape[0] != n_periods:
            raise ValueError(
                "X array dims (n_rows) != n_periods. Received "
                f"n_rows={np.asarray(X).shape[0]} and n_periods={n_periods}"
            )
        if return_conf_int:
            self._check_alpha(alpha)

        _, _, mean, var = self._forecast(n_periods, X)
        index = self._forecast_index(n_periods)
        out = pd.Series(mean, index=index) if index is not None else mean

        if not return_conf_int:
            return out
        q = _norm_ppf(1 - alpha / 2.0)
        hw = q * np.sqrt(np.maximum(var, 0.0))
        conf = np.column_stack([mean - hw, mean + hw])
        return out, conf

    def predict_in_sample(
        self, X=None, start=None, end=None, dynamic=False, return_conf_int=False,
        alpha=0.05, **kwargs
    ):
        check_is_fitted(self, "arima_res_")
        d = self.order[1]
        if isinstance(start, (int, np.integer)) and start < d:
            raise ValueError(
                f"In-sample predictions undefined for start={start} when d={d}"
            )
        X = self._check_exog(X)
        if dynamic and return_conf_int:
            warnings.warn(
                "Cannot produce in-sample confidence intervals for "
                "dynamic=True. Setting dynamic=False"
            )
            dynamic = False
        if return_conf_int:
            self._check_alpha(alpha)

        n = self.nobs_
        lo = 0 if start is None else int(start)
        hi = n - 1 if end is None else int(end)
        if hi < lo:
            raise ValueError("Prediction must have `end` after `start`.")
        # `X` describes the periods past the end of the sample, exactly as it
        # does for `statsmodels.predict(start, end, exog)`: the in-sample
        # design is the one the model was fit with.
        n_future = max(hi + 1 - n, 0)
        # In-sample only: statsmodels ignores `X` entirely there, so a
        # full-length design is accepted rather than rejected for its shape.
        if n_future and X is not None and np.asarray(X).shape[0] != n_future:
            raise ValueError(
                "Provided exogenous values are not of the appropriate shape. "
                f"Required ({n_future}, {self.spec_.k_exog}), got "
                f"{np.asarray(X).shape}."
            )

        preds, var = self._predict_range(lo, hi, X, dynamic)
        index = self._index_slice(lo, hi)
        out = pd.Series(preds, index=index) if index is not None else preds

        if not return_conf_int:
            return out
        q = _norm_ppf(1 - alpha / 2.0)
        hw = q * np.sqrt(np.maximum(var, 0.0))
        return out, np.column_stack([preds - hw, preds + hw])

    def _predict_range(self, lo, hi, X, dynamic):
        """In-sample predictions for `lo..hi`, honouring `dynamic`.

        `dynamic` is an offset relative to `start`: before it the one-step
        forecasts use the observed data, and from it onward they are fed their
        own predictions, which is what makes it a genuine multi-step forecast
        of the training period rather than a re-filtering of it.
        """
        n = self.nobs_
        if dynamic is False or dynamic is None:
            dyn = None
        elif dynamic is True:
            dyn = lo
        else:
            dyn = lo + int(dynamic)
        # Anything past the end of the sample is a forecast whatever `dynamic`
        # says, so the split point is the earlier of the two.
        split = n if dyn is None else min(max(dyn, 0), n)

        X_in = None if self._X is None else self._X[:split]
        horizon = max(hi + 1 - split, 0)
        X_fut = None
        if horizon and self.spec_.k_exog:
            # Rows the model was fit on cover a dynamic start inside the
            # sample; anything beyond the end has to come from the caller.
            in_sample = min(horizon, max(n - split, 0))
            parts = []
            if in_sample and self._X is not None:
                parts.append(self._X[split : split + in_sample])
            if horizon > in_sample and X is not None:
                parts.append(np.asarray(X, dtype=float)[: horizon - in_sample])
            if parts:
                X_fut = np.vstack(parts)

        fitted, _, mean, fvar = self._forecast(
            horizon, X_fut, y=self._y[:split], X_in=X_in
        )
        _, _, in_var, _ = _in_sample_variance(self, y=self._y[:split], X=X_in)

        head_hi = min(hi, split - 1)
        preds = np.asarray(fitted[lo : head_hi + 1], dtype=float)
        var = np.asarray(in_var[lo : head_hi + 1], dtype=float)
        if horizon:
            preds = np.concatenate([preds, np.asarray(mean, dtype=float)])
            var = np.concatenate([var, np.asarray(fvar, dtype=float)])
        return preds, var

    # --------------------------------------------------------------- update
    def update(self, y, X=None, maxiter=None, **kwargs):
        """Append new observations and refit from the current parameters.

        `pmdarima` re-runs the optimiser starting at the existing solution, so
        an update is cheap when the new data is consistent with the old.
        """
        check_is_fitted(self, "arima_res_")
        if not is_iterable(y):
            y = [y]
        y = check_endog(y, copy=False, preserve_series=True)
        n_samples = y.shape[0]
        X = self._check_exog(X)
        if X is not None:
            n_exog, exog_dim = np.asarray(X).shape
            if n_exog != n_samples:
                raise ValueError(
                    f"Dim mismatch in n_samples (y={n_samples}, X={n_exog})"
                )

        new_index = y.index if isinstance(y, pd.Series) else None
        y_all = np.concatenate([self._y, np.asarray(y, dtype=float).ravel()])
        X_all = None
        if X is not None and self._X is not None:
            X_all = np.vstack([self._X, np.asarray(X, dtype=float).reshape(n_samples, -1)])

        # The index has to grow with the data, or every later prediction is
        # labelled from the old end of the series.
        if getattr(self, "endog_index_", None) is not None:
            tail = new_index if new_index is not None else self._forecast_index(n_samples)
            if tail is not None and len(tail) == n_samples:
                self.endog_index_ = self.endog_index_.append(pd.Index(tail))
            else:
                self.endog_index_ = None

        if maxiter is None:
            maxiter = max(5, n_samples // 10)
        kwargs.pop("start_params", None)
        self._fit(
            y_all,
            X_all,
            start_params=self.res_.params,
            maxiter=maxiter,
            **kwargs,
        )
        return self

    # ------------------------------------------------------ model summaries
    @if_has_delegate("arima_res_")
    def params(self):
        return self.res_.params

    @if_has_delegate("arima_res_")
    def aic(self):
        return self.res_.aic

    @if_has_delegate("arima_res_")
    def bic(self):
        return self.res_.bic

    @if_has_delegate("arima_res_")
    def aicc(self):
        return self.res_.aicc

    @if_has_delegate("arima_res_")
    def hqic(self):
        return self.res_.hqic

    @if_has_delegate("arima_res_")
    def oob(self):
        return self.oob_

    @if_has_delegate("arima_res_")
    def df_model(self):
        return self.res_.df_model

    @if_has_delegate("arima_res_")
    def df_resid(self):
        """Residual degrees of freedom, which `statsmodels` reports as `inf`.

        Its `LikelihoodModelResults` leaves this undefined for a
        maximum-likelihood state-space model, and `pmdarima` passes that
        straight through. The finite count callers usually want is
        ``arima_res_.nobs_effective - df_model()``.
        """
        return np.inf

    def _wrap(self, values):
        """Wrap a full-length in-sample vector in the endog index, if any."""
        idx = getattr(self, "endog_index_", None)
        if idx is not None and len(idx) == len(values):
            return pd.Series(values, index=idx)
        return values

    @if_has_delegate("arima_res_")
    def resid(self):
        _, resid, _, _ = self._forecast(0)
        return self._wrap(resid)

    @if_has_delegate("arima_res_")
    def fittedvalues(self):
        fitted, _, _, _ = self._forecast(0)
        return self._wrap(fitted)

    @if_has_delegate("arima_res_")
    def arparams(self):
        s = self.spec_
        off = s.k_trend + s.k_exog
        return self.res_.params[off : off + s.p]

    @if_has_delegate("arima_res_")
    def maparams(self):
        s = self.spec_
        off = s.k_trend + s.k_exog + s.p
        return self.res_.params[off : off + s.q]

    def _reduced_polys(self):
        """The multiplied-out AR and MA operators, as statsmodels stores them.

        `auto_arima`'s near-non-invertibility guard tests the roots of these,
        not of the seasonal and non-seasonal factors separately, so getting
        this right is what keeps order selection aligned.
        """
        s = self.spec_
        p = np.asarray(self.res_.params, dtype=float)
        off = s.k_trend + s.k_exog
        ar = p[off : off + s.p]
        ma = p[off + s.p : off + s.p + s.q]
        sar = p[off + s.p + s.q : off + s.p + s.q + s.bp]
        sma = p[off + s.p + s.q + s.bp : off + s.p + s.q + s.bp + s.bq]

        poly_ar = np.r_[1.0, -ar]
        poly_ma = np.r_[1.0, ma]
        if s.bp:
            ps = np.zeros(s.bp * s.s + 1)
            ps[0] = 1.0
            for i, v in enumerate(sar):
                ps[(i + 1) * s.s] = -v
            poly_ar = np.convolve(poly_ar, ps)
        if s.bq:
            qs = np.zeros(s.bq * s.s + 1)
            qs[0] = 1.0
            for i, v in enumerate(sma):
                qs[(i + 1) * s.s] = v
            poly_ma = np.convolve(poly_ma, qs)
        return poly_ar, poly_ma

    @if_has_delegate("arima_res_")
    def arroots(self):
        poly = self._reduced_polys()[0]
        return np.roots(poly[::-1]) if poly.size > 1 else np.array([], dtype=complex)

    @if_has_delegate("arima_res_")
    def maroots(self):
        poly = self._reduced_polys()[1]
        return np.roots(poly[::-1]) if poly.size > 1 else np.array([], dtype=complex)

    @if_has_delegate("arima_res_")
    def bse(self):
        return _standard_errors(self)

    @if_has_delegate("arima_res_")
    def pvalues(self):
        from scipy.special import erfc

        z = np.abs(self.params() / self.bse())
        return erfc(z / np.sqrt(2.0))

    @if_has_delegate("arima_res_")
    def conf_int(self, alpha=0.05, **kwargs):
        self._check_alpha(alpha)
        q = _norm_ppf(1 - alpha / 2.0)
        se = self.bse()
        p = self.params()
        return np.column_stack([p - q * se, p + q * se])

    @if_has_delegate("arima_res_")
    def to_dict(self):
        return {
            "pvalues": self.pvalues(),
            "resid": np.asarray(self.resid()),
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "oob": self.oob_,
            "aic": self.aic(),
            "aicc": self.aicc(),
            "bic": self.bic(),
            "bse": self.bse(),
            "params": self.params(),
        }

    # ---------------------------------------------------------- diagnostics
    def _standardized_resid(self):
        """`v_t / sqrt(F_t)`, dropping the diffuse burn-in.

        The raw residuals of a differenced model start with values of order
        `sqrt(1e6 * sigma2)`, so any diagnostic computed on them is dominated
        by the initialisation rather than the fit.
        """
        _, resid, fvar, _ = _in_sample_variance(self)
        resid = np.asarray(resid, dtype=float)
        fvar = np.asarray(fvar, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            std = resid / np.sqrt(fvar)
        burn = self.spec_.loglikelihood_burn
        return std[burn:]

    def _ljung_box(self):
        return ljung_box(self._standardized_resid(), lags=1)

    def _jarque_bera(self):
        return jarque_bera(self._standardized_resid())

    def _heteroskedasticity(self):
        return break_variance(self._standardized_resid())

    @if_has_delegate("arima_res_")
    def summary(self):
        return Summary(self)

    @if_has_delegate("arima_res_")
    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None):
        """Standard residual diagnostics, on the standardized residuals."""
        import matplotlib.pyplot as plt

        from ..utils.wrapped import acf

        std = self._standardized_resid()
        std = std[np.isfinite(std)]

        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(221)
        ax.plot(std)
        ax.hlines(0, 0, len(std), alpha=0.5)
        ax.set_title("Standardized residual")

        ax = fig.add_subplot(222)
        ax.hist(std, bins=min(30, max(5, len(std) // 5)), density=True,
                label="Hist")
        grid = np.linspace(std.min(), std.max(), 100)
        from scipy import stats

        ax.plot(grid, stats.gaussian_kde(std)(grid), label="KDE")
        ax.plot(grid, stats.norm.pdf(grid), label="N(0,1)")
        ax.legend(loc="upper right")
        ax.set_title("Histogram plus estimated density")

        ax = fig.add_subplot(223)
        stats.probplot(std, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")

        ax = fig.add_subplot(224)
        vals = acf(std, nlags=lags)
        ax.vlines(np.arange(len(vals)), [0], vals)
        ax.axhline(y=0, color="k")
        ax.set_title("Correlogram")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------ pickling
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        self._warn_for_older_version()
        return self

    def _warn_for_older_version(self):
        from .. import __version__

        modl_version = getattr(self, "pkg_version_", None)
        if modl_version is None:
            if not hasattr(self, "arima_res_"):
                return
            modl_version = "<0.1.0"
        if modl_version != __version__:
            warnings.warn(
                "You've deserialized an ARIMA from a version (%s) that does "
                "not match your installed version of pmdarima-rs (%s). This "
                "could cause unforeseen behavior." % (modl_version, __version__),
                UserWarning,
            )


def _in_sample_variance(model, y=None, X=None):
    from .. import _pmdarima_rs as _rs

    y = model._y if y is None else y
    X = model._X if X is None else X
    exog = _fitmod.flatten_exog(X, len(y)) if X is not None else None
    kw = model._rs_kwargs(exog)
    return _rs.filter_paths(
        np.ascontiguousarray(np.asarray(y, dtype=float)),
        model.res_.params,
        kw.pop("order"),
        kw.pop("seasonal_order"),
        kw.pop("trend_powers"),
        **kw,
    )


def _roots(coefs, negate):
    coefs = np.asarray(coefs, dtype=float)
    if coefs.size == 0:
        return np.array([], dtype=complex)
    poly = np.r_[1.0, -coefs] if negate else np.r_[1.0, coefs]
    return np.roots(poly[::-1])


def _standard_errors(model):
    """Standard errors by the outer product of gradients.

    This is the estimator `statsmodels` uses for a fitted SARIMAX - its
    results object reports `cov_type='opg'` - so it is the one that reproduces
    `pmdarima`'s numbers. The covariance is `inv(G'G)` where row `t` of `G`
    holds the score of observation `t`.

    statsmodels differentiates `loglikeobs` by complex step, which is exact.
    We cannot take a complex step through the filter, so the score is taken by
    a five-point central difference instead: its truncation error is `O(h^4)`
    rather than `O(h^2)`, which matters because `G'G` is badly conditioned for
    the near-boundary fits ARIMA likelihoods like to land on, and a second
    order score there moves the standard errors by percent, not digits.
    """
    from .. import _pmdarima_rs as _rs

    spec = model.spec_
    p = np.asarray(model.res_.params, dtype=float)
    k = p.size
    if k == 0:
        return np.zeros(0)

    exog = (
        _fitmod.flatten_exog(model._X, model.nobs_) if model._X is not None else None
    )
    kw = model._rs_kwargs(exog)

    def llobs(theta):
        return _rs.loglikeobs(model._y, np.ascontiguousarray(theta), **kw)

    # `eps ** (1/5)` balances truncation against round-off for a five-point
    # rule, the way `eps ** (1/3)` does for a three-point one.
    step = np.finfo(float).eps ** (1 / 5) * np.maximum(np.abs(p), 0.1)
    G = np.empty((model.nobs_, k))
    for j in range(k):
        h = step[j]
        pts = []
        for mult in (-2.0, -1.0, 1.0, 2.0):
            theta = p.copy()
            theta[j] += mult * h
            pts.append(llobs(theta))
        G[:, j] = (pts[0] - 8.0 * pts[1] + 8.0 * pts[2] - pts[3]) / (12.0 * h)

    try:
        cov = np.linalg.inv(G.T.dot(G))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(G.T.dot(G))
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


class _ResultsShim:
    """The subset of `SARIMAXResults` that `pmdarima` users actually touch.

    `pmdarima` exposes its internal statsmodels results object as
    `arima_res_`, and code in the wild reads `.llf`, `.params`, `.aic` and
    friends off it. Providing those keeps such code working without dragging
    statsmodels back in as a dependency.
    """

    def __init__(self, model):
        self._model = model

    # --- basic quantities ---------------------------------------------
    @property
    def params(self):
        return self._model.res_.params

    @property
    def llf(self):
        return self._model.res_.loglike

    @property
    def nobs(self):
        return self._model.res_.nobs

    @property
    def nobs_effective(self):
        return self._model.res_.nobs_effective

    @property
    def nobs_diffuse(self):
        return 0

    @property
    def df_model(self):
        return self._model.res_.df_model

    @property
    def df_resid(self):
        # statsmodels leaves this at infinity for MLE state-space results.
        return np.inf

    @property
    def aic(self):
        return self._model.res_.aic

    @property
    def aicc(self):
        return self._model.res_.aicc

    @property
    def bic(self):
        return self._model.res_.bic

    @property
    def hqic(self):
        return self._model.res_.hqic

    @property
    def scale(self):
        return self._model.res_.scale

    @property
    def loglikelihood_burn(self):
        return self._model.spec_.loglikelihood_burn

    @property
    def param_names(self):
        return self._model.spec_.param_names

    @property
    def specification(self):
        s = self._model.spec_
        return {
            "seasonal_periods": s.s,
            "measurement_error": False,
            "time_varying_regression": False,
            "simple_differencing": False,
            "enforce_stationarity": s.enforce_stationarity,
            "enforce_invertibility": s.enforce_invertibility,
            "hamilton_representation": False,
            "concentrate_scale": s.concentrate_scale,
            "trend_offset": s.trend_offset,
            "order": s.order,
            "seasonal_order": (s.bp, s.bd, s.bq, s.s),
            "k_diff": s.d,
            "k_seasonal_diff": s.bd,
            "k_ar": s.p,
            "k_ma": s.q,
            "k_seasonal_ar": s.bp,
            "k_seasonal_ma": s.bq,
            "k_exog": s.k_exog,
            "k_trend": s.k_trend,
        }

    @property
    def mle_retvals(self):
        res = self._model.res_
        return {
            "fopt": -res.loglike / max(res.nobs, 1),
            "converged": res.converged,
            "iterations": res.n_iter,
            "fcalls": res.n_fev,
            "warnflag": 0 if res.converged else 1,
            "task": res.message,
        }

    @property
    def mle_settings(self):
        return {"optimizer": "lbfgs", "maxiter": self._model.maxiter}

    # --- inference ------------------------------------------------------
    @property
    def bse(self):
        return self._model.bse()

    @property
    def zvalues(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._model.params() / self._model.bse()

    tvalues = zvalues

    @property
    def pvalues(self):
        return self._model.pvalues()

    def cov_params(self):
        se = self._model.bse()
        return np.diag(se**2)

    def conf_int(self, alpha=0.05):
        return self._model.conf_int(alpha=alpha)

    # --- fitted quantities ---------------------------------------------
    @property
    def resid(self):
        return self._model.resid()

    @property
    def fittedvalues(self):
        return self._model.fittedvalues()

    @property
    def standardized_forecasts_error(self):
        return self._model._standardized_resid()

    @property
    def arparams(self):
        return self._model.arparams()

    @property
    def maparams(self):
        return self._model.maparams()

    @property
    def arroots(self):
        return self._model.arroots()

    @property
    def maroots(self):
        return self._model.maroots()

    @property
    def arfreq(self):
        z = self._model.arroots()
        return np.angle(z) / (2 * np.pi) if z.size else z

    @property
    def mafreq(self):
        z = self._model.maroots()
        return np.angle(z) / (2 * np.pi) if z.size else z

    @property
    def polynomial_reduced_ar(self):
        return self._model._reduced_polys()[0]

    @property
    def polynomial_reduced_ma(self):
        return self._model._reduced_polys()[1]

    # --- forecasting ----------------------------------------------------
    def forecast(self, steps=1, exog=None):
        return self._model.predict(n_periods=steps, X=exog)

    def predict(self, start=None, end=None, exog=None, dynamic=False):
        return self._model.predict_in_sample(
            X=exog, start=start, end=end, dynamic=dynamic
        )

    def test_serial_correlation(self, method="ljungbox", lags=1):
        return np.array([self._model._ljung_box()])

    def test_normality(self, method="jarquebera"):
        return np.array([self._model._jarque_bera()])

    def test_heteroskedasticity(self, method="breakvar"):
        return np.array([self._model._heteroskedasticity()])

    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None):
        return self._model.plot_diagnostics(
            variable=variable, lags=lags, fig=fig, figsize=figsize
        )

    def summary(self):
        return self._model.summary()
