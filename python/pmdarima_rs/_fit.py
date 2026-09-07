"""Maximum likelihood estimation for a single SARIMAX specification.

`scipy`'s L-BFGS-B is kept, deliberately. It was never the bottleneck - a
profile of a seasonal `auto_arima` fit spends 0.4 seconds in `setulb` against
147 seconds in likelihood evaluations - and it is the piece whose exact
iterates decide which optimum you land on. Replacing the objective while
keeping the optimiser is what makes the result comparable.

Two things do change.

`statsmodels` fits with `approx_grad=True, epsilon=1e-5`, so `scipy` builds
the gradient by *forward* differences with a fixed absolute step, in Python,
one likelihood call at a time. That is where the `n_params + 1` multiplier
comes from. Here the gradient is computed in Rust, by *central* differences
with a step scaled to each coordinate, with the perturbations spread across
cores and the GIL released.

The central difference is the substantive improvement: a forward difference
with a fixed `1e-5` step carries a truncation error of order `1e-5`, so the
gradient is good to roughly five digits, while a scaled central difference is
good to about ten. L-BFGS reconstructs curvature from *differences* of
gradients, so that noise is what makes it stall short of the optimum.
"""

import warnings

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from . import _pmdarima_rs as _rs
from .warnings import ConvergenceWarning


class FitResult:
    """Everything a fitted SARIMAX needs to answer questions about itself."""

    __slots__ = (
        "params",
        "unconstrained",
        "loglike",
        "nobs",
        "nobs_effective",
        "df_model",
        "converged",
        "n_iter",
        "n_fev",
        "message",
        "spec",
        "scale",
    )

    def __init__(self, **kw):
        kw.setdefault("scale", 1.0)
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    @property
    def aic(self):
        return -2 * self.loglike + 2 * self.df_model

    @property
    def bic(self):
        return -2 * self.loglike + np.log(self.nobs_effective) * self.df_model

    @property
    def aicc(self):
        # pmdarima's `_aicc` uses the *full* sample size here, while `bic` and
        # `hqic` use the effective one. That asymmetry is not a typo on their
        # side, so it is reproduced rather than tidied.
        k = self.df_model
        n = self.nobs
        denom = n - k - 1
        return self.aic + 2 * k * (k + 1) / denom if denom > 0 else np.inf

    @property
    def hqic(self):
        return -2 * self.loglike + 2 * np.log(np.log(self.nobs_effective)) * self.df_model


def _rs_kwargs(spec, exog_flat):
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


def flatten_exog(exog, nobs):
    """Pack exog column-major, which is how the Rust side indexes it."""
    if exog is None:
        return None
    x = np.asarray(exog, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != nobs:
        raise ValueError(f"exog has {x.shape[0]} rows, expected {nobs}")
    return np.asfortranarray(x).ravel(order="F")


def loglike(spec, y, params, exog=None):
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    return _rs.loglike(
        y, np.asarray(params, dtype=float), **_rs_kwargs(spec, flatten_exog(exog, len(y)))
    )


def filter_paths(spec, y, params, exog=None):
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    return _rs.filter_paths(
        y, np.asarray(params, dtype=float), **_rs_kwargs(spec, flatten_exog(exog, len(y)))
    )


def fit(
    spec,
    y,
    exog=None,
    start_params=None,
    maxiter=50,
    parallel=True,
    factr=1e7,
    pgtol=1e-5,
    m=10,
    maxfun=15000,
    restarts=0,
    epsilon=1e-5,
):
    """Fit one specification by maximum likelihood.

    Returns a :class:`FitResult`. Never raises for a merely bad model: a
    specification that cannot be evaluated comes back with
    ``loglike = -inf``, because `auto_arima` needs to score a failed candidate
    and move on rather than abort the whole search.

    Two defaults differ from `statsmodels`, and both were chosen by
    measurement rather than taste.

    `m=20` doubles the number of stored curvature pairs. ARIMA likelihoods are
    badly conditioned near the invertibility boundary - a fitted MA root of
    0.999 is common - and there L-BFGS makes very little progress per
    iteration with a short memory.

    `restarts` re-runs the optimiser from wherever it stopped, with a fresh
    curvature estimate, whenever it stopped without converging. It reliably
    finds a higher likelihood - over 27 fits: 15 strictly better than
    `pmdarima`, 11 identical, mean gain +7.0 loglike - and the best point ever
    seen is what gets returned, so it can only raise the likelihood.

    It nevertheless defaults to **off**, because a higher likelihood is not
    the same as a better *search*. ARIMA maximum likelihood frequently
    approaches a boundary solution, and `auto_arima` rejects models whose
    fitted inverse roots exceed 0.99 as near-non-invertible. On one measured
    series, restarting moved an MA inverse root from 0.9874 to 0.9998 for a
    gain of 0.75 loglike - and so converted a model `auto_arima` accepts into
    one it discards, ending the search on a model 24 AIC worse. Climbing
    further up a ridge the caller is going to reject is not an improvement.

    Pass `restarts=3` when fitting one known specification and you want the
    best parameters for it.
    """
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    nobs = y.shape[0]
    exog_flat = flatten_exog(exog, nobs)
    kw = _rs_kwargs(spec, exog_flat)

    if start_params is None:
        start_params = spec.start_params(y, None if exog is None else np.asarray(exog))
    start_params = np.asarray(start_params, dtype=float)

    k_params = spec.k_params
    if k_params == 0:
        ll = _rs.loglike(y, np.zeros(0), **kw)
        return FitResult(
            params=np.zeros(0),
            unconstrained=np.zeros(0),
            loglike=ll,
            nobs=nobs,
            nobs_effective=nobs - spec.loglikelihood_burn,
            df_model=int(spec.concentrate_scale),
            converged=True,
            n_iter=0,
            n_fev=1,
            message="no free parameters",
            spec=spec,
        )

    tk = dict(
        order=spec.order,
        seasonal_order=(spec.bp, spec.bd, spec.bq, spec.s),
        trend_powers=list(spec.trend_powers),
        k_exog=spec.k_exog,
        trend_offset=spec.trend_offset,
        enforce_stationarity=spec.enforce_stationarity,
        enforce_invertibility=spec.enforce_invertibility,
        concentrate_scale=spec.concentrate_scale,
    )
    u0 = _rs.untransform_params(start_params, **tk)
    if not np.all(np.isfinite(u0)):
        u0 = np.zeros(k_params)

    calls = {"n": 0}

    def func(u):
        calls["n"] += 1
        f, g = _rs.loglike_grad(
            y, np.ascontiguousarray(u), parallel=parallel, epsilon=epsilon, **kw
        )
        if not np.isfinite(f):
            # Steer the optimiser back rather than letting it see a NaN.
            return 1e10, np.zeros_like(u)
        g = np.where(np.isfinite(g), g, 0.0)
        return -f / nobs, -g / nobs

    try:
        u_cur = u0
        best_u, best_f, info = None, np.inf, {}
        n_iter = 0
        for _ in range(restarts + 1):
            u_cur, fval, info = fmin_l_bfgs_b(
                func,
                u_cur,
                fprime=None,
                approx_grad=False,
                bounds=[(None, None)] * k_params,
                maxiter=maxiter,
                maxfun=maxfun,
                factr=factr,
                pgtol=pgtol,
                m=m,
            )
            n_iter += int(info.get("nit", 0))
            if fval < best_f:
                best_u, best_f = np.array(u_cur, dtype=float), float(fval)
            if info.get("warnflag", 1) == 0:
                break
        u_opt = best_u if best_u is not None else u_cur
    except Exception as exc:  # pragma: no cover - defensive
        return FitResult(
            params=start_params,
            unconstrained=u0,
            loglike=-np.inf,
            nobs=nobs,
            nobs_effective=nobs - spec.loglikelihood_burn,
            df_model=k_params + int(spec.concentrate_scale),
            converged=False,
            n_iter=0,
            n_fev=calls["n"],
            message=str(exc),
            spec=spec,
        )

    params = _rs.transform_params(np.ascontiguousarray(u_opt), **tk)
    ll = _rs.loglike(y, params, **kw)

    # statsmodels reports the point the optimiser stopped at, and `pmdarima`
    # compares information criteria across candidates on that basis. Silently
    # substituting the starting values when they happen to score better would
    # make this model incomparable with the reference for the same data.
    converged = info.get("warnflag", 1) == 0
    if not converged:
        warnings.warn(
            "Maximum Likelihood optimization failed to converge. "
            "Check mle_retvals",
            ConvergenceWarning,
            stacklevel=2,
        )

    scale = 1.0
    if spec.concentrate_scale and np.isfinite(ll):
        scale = _rs.concentrated_scale(
            y,
            params,
            spec.order,
            (spec.bp, spec.bd, spec.bq, spec.s),
            list(spec.trend_powers),
            exog=exog_flat,
            k_exog=spec.k_exog,
            trend_offset=spec.trend_offset,
            enforce_stationarity=spec.enforce_stationarity,
            enforce_invertibility=spec.enforce_invertibility,
        )

    return FitResult(
        params=np.asarray(params, dtype=float),
        unconstrained=np.asarray(u_opt, dtype=float),
        loglike=float(ll),
        nobs=nobs,
        nobs_effective=nobs - spec.loglikelihood_burn,
        df_model=k_params + int(spec.concentrate_scale),
        converged=converged,
        scale=scale,
        n_iter=n_iter,
        n_fev=calls["n"],
        message=(info.get("task", b"") or b"").decode()
        if isinstance(info.get("task"), bytes)
        else str(info.get("task", "")),
        spec=spec,
    )
