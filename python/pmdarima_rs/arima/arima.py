"""The `ARIMA` estimator.

A drop-in for `pmdarima.arima.ARIMA`: same constructor arguments, same
methods, same return types, including the pandas index behaviour that decides
whether you get an array or a `Series` back.

The only compiled part is what runs thousands of times - the likelihood and
its gradient. Everything visible here runs once per fit.
"""

import warnings

import numpy as np
import pandas as pd

from .. import _fit as _fitmod
from .._ssm import Spec
from ..utils.array import check_endog, check_exog

__all__ = ["ARIMA"]


def _mse(y_true, y_pred, **kw):
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def _mae(y_true, y_pred, **kw):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


VALID_SCORING = {"mse": _mse, "mean_squared_error": _mse, "mae": _mae,
                 "mean_absolute_error": _mae}


def _get_scoring(scoring):
    if callable(scoring):
        return scoring
    try:
        return VALID_SCORING[scoring]
    except KeyError:
        raise ValueError(
            f"scoring must be callable or one of {set(VALID_SCORING)}"
        ) from None


def _norm_ppf(q):
    """Inverse standard normal CDF, so scipy.stats is not imported per call."""
    from scipy.special import ndtri

    return float(ndtri(q))


class ARIMA:
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
        self.sarimax_kwargs = sarimax_kwargs

    # ---------------------------------------------------------------- sklearn
    _ctor_params = (
        "order",
        "seasonal_order",
        "start_params",
        "method",
        "maxiter",
        "suppress_warnings",
        "out_of_sample_size",
        "scoring",
        "scoring_args",
        "trend",
        "with_intercept",
    )

    def get_params(self, deep=True):
        out = {k: getattr(self, k) for k in self._ctor_params}
        out.update(self.sarimax_kwargs)
        return out

    def set_params(self, **params):
        for k, v in params.items():
            if k in self._ctor_params:
                setattr(self, k, v)
            else:
                self.sarimax_kwargs[k] = v
        return self

    def __repr__(self):
        args = ", ".join(
            f"{k}={getattr(self, k)!r}"
            for k in ("order", "seasonal_order")
        )
        return f"ARIMA({args})"

    # ------------------------------------------------------------------ spec
    def _resolve_trend(self):
        trend = self.trend
        if trend is None and self.with_intercept:
            trend = "c"
        return trend

    def _build_spec(self, k_exog):
        kw = dict(self.sarimax_kwargs)
        return Spec(
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self._resolve_trend(),
            k_exog=k_exog,
            enforce_stationarity=kw.get("enforce_stationarity", True),
            enforce_invertibility=kw.get("enforce_invertibility", True),
            concentrate_scale=kw.get("concentrate_scale", False),
        )

    # ------------------------------------------------------------------- fit
    def _fit(self, y, X=None, **fit_args):
        y_arr = np.asarray(y, dtype=float).ravel()
        k_exog = 0 if X is None else np.asarray(X).reshape(len(y_arr), -1).shape[1]
        spec = self._build_spec(k_exog)

        maxiter = fit_args.pop("maxiter", self.maxiter)
        if maxiter is None:
            raise ValueError("Expected non-None value for `maxiter`")
        start_params = fit_args.pop("start_params", self.start_params)
        fit_args.pop("disp", None)

        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
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
        return self

    def fit(self, y, X=None, **fit_args):
        y = check_endog(y, copy=False, preserve_series=True)
        n_samples = y.shape[0]
        self.endog_index_ = y.index if isinstance(y, pd.Series) else None
        if X is not None:
            X = check_exog(X, force_all_finite=False, copy=False)

        cv = max(int(self.out_of_sample_size), 0)
        scoring = _get_scoring(self.scoring)
        if cv >= n_samples:
            raise ValueError("out-of-sample size must be less than number of samples!")

        cv_samples = cv_exog = None
        if cv:
            cv_samples = np.asarray(y)[-cv:]
            y = np.asarray(y)[:-cv]
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
            self.update(cv_samples, cv_exog, **fit_args)
        else:
            self.oob_ = np.nan
            self.oob_preds_ = None
        return self

    def fit_predict(self, y, X=None, n_periods=10, **fit_args):
        self.fit(y, X, **fit_args)
        return self.predict(n_periods=n_periods, X=X)

    def _check_is_fitted(self):
        if not hasattr(self, "res_"):
            raise ValueError(
                "This ARIMA instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )

    def _check_exog(self, X):
        if self.fit_with_exog_:
            if X is None:
                raise ValueError(
                    "When an ARIMA is fit with an X array, it must also be "
                    "provided one for predicting or updating observations."
                )
            return check_exog(X, force_all_finite=True)
        return None

    # -------------------------------------------------------------- predict
    def _forecast(self, n_periods, X=None):
        from .. import _pmdarima_rs as _rs

        spec = self.spec_
        exog = _fitmod.flatten_exog(self._X, self.nobs_) if self._X is not None else None
        exog_f = None
        if X is not None:
            Xa = np.asarray(X, dtype=float).reshape(n_periods, -1)
            exog_f = np.asfortranarray(Xa).ravel(order="F")
        return _rs.forecast(
            self._y,
            self.res_.params,
            spec.order,
            (spec.bp, spec.bd, spec.bq, spec.s),
            list(spec.trend_powers),
            n_periods,
            exog=exog,
            exog_future=exog_f,
            k_exog=spec.k_exog,
            enforce_stationarity=spec.enforce_stationarity,
            enforce_invertibility=spec.enforce_invertibility,
            concentrate_scale=spec.concentrate_scale,
        )

    def _forecast_index(self, n_periods):
        idx = getattr(self, "endog_index_", None)
        if idx is None:
            return None
        try:
            if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None:
                return pd.date_range(
                    idx[-1] + idx.freq, periods=n_periods, freq=idx.freq
                )
            if isinstance(idx, pd.RangeIndex) or np.issubdtype(
                np.asarray(idx).dtype, np.integer
            ):
                last = int(np.asarray(idx)[-1])
                return pd.RangeIndex(last + 1, last + 1 + n_periods)
        except Exception:  # pragma: no cover - index shapes vary widely
            return None
        return None

    def predict(self, n_periods=10, X=None, return_conf_int=False, alpha=0.05, **kwargs):
        self._check_is_fitted()
        if not isinstance(n_periods, (int, np.integer)):
            raise TypeError("n_periods must be an int")
        X = self._check_exog(X)
        if X is not None and np.asarray(X).shape[0] != n_periods:
            raise ValueError(
                "X array dims (n_rows) != n_periods. Received "
                f"n_rows={np.asarray(X).shape[0]} and n_periods={n_periods}"
            )

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
        self._check_is_fitted()
        d = self.order[1]
        if isinstance(start, (int, np.integer)) and start < d:
            raise ValueError(
                f"In-sample predictions undefined for start={start} when d={d}"
            )
        self._check_exog(X)
        if dynamic and return_conf_int:
            warnings.warn(
                "Cannot produce in-sample confidence intervals for "
                "dynamic=True. Setting dynamic=False"
            )
            dynamic = False

        fitted, resid, _, _ = self._forecast(1, None)
        n = self.nobs_
        # `pmdarima` defaults to the whole sample, index 0 included, even
        # though the first d predictions are not meaningful for d > 0. It only
        # rejects an *explicit* start below d.
        lo = 0 if start is None else int(start)
        hi = n - 1 if end is None else int(end)
        preds = fitted[lo : hi + 1]

        index = None
        if getattr(self, "endog_index_", None) is not None:
            index = self.endog_index_[lo : hi + 1]
        out = pd.Series(preds, index=index) if index is not None else preds

        if not return_conf_int:
            return out
        # Variance of a one-step-ahead in-sample forecast is F_t.
        _, _, fvar, _ = _in_sample_variance(self)
        q = _norm_ppf(1 - alpha / 2.0)
        hw = q * np.sqrt(np.maximum(fvar[lo : hi + 1], 0.0))
        return out, np.column_stack([preds - hw, preds + hw])

    # --------------------------------------------------------------- update
    def update(self, y, X=None, maxiter=None, **kwargs):
        """Append new observations and refit from the current parameters.

        `pmdarima` re-runs the optimiser starting at the existing solution, so
        an update is cheap when the new data is consistent with the old.
        """
        self._check_is_fitted()
        y = check_endog(y, copy=False, preserve_series=False)
        X = self._check_exog(X)

        self._y = np.concatenate([self._y, np.asarray(y, dtype=float).ravel()])
        if X is not None and self._X is not None:
            self._X = np.vstack([self._X, np.asarray(X, dtype=float).reshape(len(y), -1)])
        self.nobs_ = self._y.shape[0]

        if maxiter is None:
            maxiter = self.maxiter
        if maxiter == 0:
            # A pure "absorb the data" update: refilter, do not re-optimise.
            self.res_.nobs = self.nobs_
            return self

        kwargs.pop("start_params", None)
        with warnings.catch_warnings():
            if self.suppress_warnings:
                warnings.simplefilter("ignore")
            self.res_ = _fitmod.fit(
                self.spec_,
                self._y,
                exog=self._X,
                start_params=self.res_.params,
                maxiter=maxiter,
            )
        self.arima_res_ = _ResultsShim(self)
        return self

    # ------------------------------------------------------ model summaries
    def params(self):
        self._check_is_fitted()
        return self.res_.params

    def aic(self):
        self._check_is_fitted()
        return self.res_.aic

    def bic(self):
        self._check_is_fitted()
        return self.res_.bic

    def aicc(self):
        self._check_is_fitted()
        return self.res_.aicc

    def hqic(self):
        self._check_is_fitted()
        return self.res_.hqic

    def oob(self):
        self._check_is_fitted()
        return self.oob_

    def df_model(self):
        self._check_is_fitted()
        return self.res_.df_model

    def df_resid(self):
        self._check_is_fitted()
        return self.res_.nobs_effective - self.res_.df_model

    def resid(self):
        self._check_is_fitted()
        _, resid, _, _ = self._forecast(1, None)
        return resid

    def fittedvalues(self):
        self._check_is_fitted()
        fitted, _, _, _ = self._forecast(1, None)
        return fitted

    def arparams(self):
        self._check_is_fitted()
        s = self.spec_
        off = s.k_trend + s.k_exog
        return self.res_.params[off : off + s.p]

    def maparams(self):
        self._check_is_fitted()
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

    def arroots(self):
        self._check_is_fitted()
        poly = self._reduced_polys()[0]
        return np.roots(poly[::-1]) if poly.size > 1 else np.array([], dtype=complex)

    def maroots(self):
        self._check_is_fitted()
        poly = self._reduced_polys()[1]
        return np.roots(poly[::-1]) if poly.size > 1 else np.array([], dtype=complex)

    def bse(self):
        self._check_is_fitted()
        return _standard_errors(self)

    def pvalues(self):
        self._check_is_fitted()
        from scipy.special import erfc

        z = np.abs(self.params() / self.bse())
        return erfc(z / np.sqrt(2.0))

    def conf_int(self, alpha=0.05, **kwargs):
        self._check_is_fitted()
        q = _norm_ppf(1 - alpha / 2.0)
        se = self.bse()
        p = self.params()
        return np.column_stack([p - q * se, p + q * se])

    def to_dict(self):
        self._check_is_fitted()
        return {
            "pvalues": self.pvalues(),
            "resid": self.resid(),
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "oob": self.oob_,
            "aic": self.aic(),
            "aicc": self.aicc(),
            "bic": self.bic(),
            "bse": self.bse(),
            "params": self.params(),
        }

    def summary(self):
        self._check_is_fitted()
        return _Summary(self)

    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None):
        """Standard residual diagnostics.

        Kept because `pmdarima` exposes it; the plotting itself is matplotlib's
        and only imported when actually asked for.
        """
        import matplotlib.pyplot as plt

        resid = np.asarray(self.resid(), dtype=float)
        burn = self.spec_.loglikelihood_burn
        resid = resid[burn:]
        std = resid / np.std(resid)

        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(221)
        ax.plot(std)
        ax.set_title("Standardized residual")
        ax = fig.add_subplot(222)
        ax.hist(std, bins=min(30, max(5, len(std) // 5)), density=True)
        ax.set_title("Histogram plus estimated density")
        ax = fig.add_subplot(223)
        from scipy import stats

        stats.probplot(std, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q")
        ax = fig.add_subplot(224)
        ax.acorr(std - std.mean(), maxlags=lags)
        ax.set_title("Correlogram")
        fig.tight_layout()
        return fig


def _in_sample_variance(model):
    from .. import _pmdarima_rs as _rs

    spec = model.spec_
    exog = (
        _fitmod.flatten_exog(model._X, model.nobs_) if model._X is not None else None
    )
    return _rs.filter_paths(
        model._y,
        model.res_.params,
        spec.order,
        (spec.bp, spec.bd, spec.bq, spec.s),
        list(spec.trend_powers),
        exog=exog,
        k_exog=spec.k_exog,
        enforce_stationarity=spec.enforce_stationarity,
        enforce_invertibility=spec.enforce_invertibility,
        concentrate_scale=spec.concentrate_scale,
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

    An observed-information estimate (inverting the numerical Hessian) is a
    defensible alternative but a *different* one, and it disagrees visibly
    near the invertibility boundary, exactly where fitted ARIMA models like to
    sit. Matching the reference matters more than picking a favourite.
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
    kw = dict(
        order=spec.order,
        seasonal_order=(spec.bp, spec.bd, spec.bq, spec.s),
        trend_powers=list(spec.trend_powers),
        exog=exog,
        k_exog=spec.k_exog,
        enforce_stationarity=spec.enforce_stationarity,
        enforce_invertibility=spec.enforce_invertibility,
        concentrate_scale=spec.concentrate_scale,
    )

    def llobs(theta):
        return _rs.loglikeobs(model._y, np.ascontiguousarray(theta), **kw)

    # statsmodels differentiates by complex step, which is exact; a central
    # difference with a cube-root-epsilon step is the closest real-arithmetic
    # equivalent.
    step = np.finfo(float).eps ** (1 / 3) * np.maximum(np.abs(p), 0.1)
    G = np.empty((model.nobs_, k))
    for j in range(k):
        up, dn = p.copy(), p.copy()
        up[j] += step[j]
        dn[j] -= step[j]
        G[:, j] = (llobs(up) - llobs(dn)) / (2 * step[j])

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
    def df_model(self):
        return self._model.res_.df_model

    @property
    def aic(self):
        return self._model.res_.aic

    @property
    def bic(self):
        return self._model.res_.bic

    @property
    def hqic(self):
        return self._model.res_.hqic

    @property
    def loglikelihood_burn(self):
        return self._model.spec_.loglikelihood_burn

    @property
    def param_names(self):
        return self._model.spec_.param_names

    @property
    def bse(self):
        return self._model.bse()

    @property
    def pvalues(self):
        return self._model.pvalues()

    @property
    def resid(self):
        return self._model.resid()

    @property
    def fittedvalues(self):
        return self._model.fittedvalues()

    def conf_int(self, alpha=0.05):
        return self._model.conf_int(alpha=alpha)

    def summary(self):
        return self._model.summary()


class _Summary:
    """A readable text summary, in the spirit of statsmodels' table."""

    def __init__(self, model):
        self.model = model

    def __str__(self):
        m = self.model
        s = m.spec_
        names = s.param_names
        p = np.asarray(m.params(), dtype=float)
        se = np.asarray(m.bse(), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = p / se
        pv = np.asarray(m.pvalues(), dtype=float)
        ci = m.conf_int()

        order = f"{s.order}"
        sorder = f"x{(s.bp, s.bd, s.bq, s.s)}" if s.s else ""
        lines = [
            "                               SARIMAX Results                                ",
            "=" * 78,
            f"Dep. Variable:                      y   No. Observations:  {m.nobs_:>15d}",
            f"Model:               SARIMAX{order}{sorder}   Log Likelihood {m.res_.loglike:>17.3f}",
            f"                                        AIC            {m.aic():>17.3f}",
            f"                                        BIC            {m.bic():>17.3f}",
            f"                                        HQIC           {m.hqic():>17.3f}",
            "=" * 78,
            f"{'':>14}{'coef':>10}{'std err':>10}{'z':>9}{'P>|z|':>9}{'[0.025':>10}{'0.975]':>10}",
            "-" * 78,
        ]
        for i, nm in enumerate(names):
            lines.append(
                f"{nm:>14}{p[i]:>10.4f}{se[i]:>10.3f}{z[i]:>9.3f}"
                f"{pv[i]:>9.3f}{ci[i, 0]:>10.3f}{ci[i, 1]:>10.3f}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)

    __repr__ = __str__
