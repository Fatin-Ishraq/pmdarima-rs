"""Model bookkeeping around the compiled Kalman filter.

Everything here runs *once per candidate model*: parsing the order, building
starting values, driving the optimiser. The likelihood, which runs thousands
of times per model, is the only part that lives in Rust. Splitting it that way
keeps the interesting arithmetic compiled and the fiddly compatibility logic
readable.

The starting values reproduce `statsmodels.SARIMAX.start_params` exactly,
including its two-stage conditional-sum-of-squares regression and its
fallbacks. That matters more than it looks: maximum likelihood for ARIMA is
not convex, so a different starting point can land on a different optimum and
therefore select a different order.
"""

import warnings

import numpy as np

TREND_POWERS = {
    None: [],
    "n": [],
    "c": [0],
    "t": [1],
    "ct": [0, 1],
}


def parse_trend(trend, k_diff=0):
    """Map a statsmodels trend string (or polynomial list) to its powers."""
    if trend is None:
        return []
    if isinstance(trend, str):
        try:
            return list(TREND_POWERS[trend])
        except KeyError:
            raise ValueError(f"unrecognised trend '{trend}'") from None
    # A list like [1, 0, 1] selects powers by position, as statsmodels allows.
    return [i for i, v in enumerate(trend) if v]


def diff(series, k_diff=1, k_seasonal_diff=None, seasonal_periods=1):
    """Seasonal then simple differencing, matching statsmodels' order."""
    out = np.asarray(series)
    if k_seasonal_diff:
        for _ in range(k_seasonal_diff):
            out = out[seasonal_periods:] - out[:-seasonal_periods]
    if k_diff:
        out = np.diff(out, k_diff, axis=0)
    return out


def lagmat(x, maxlag, trim="forward"):
    """A faithful port of `statsmodels.tsa.tsatools.lagmat(original='ex')`."""
    x = np.asarray(x)
    orig_1d = x.ndim == 1
    if orig_1d:
        x = x[:, None]
    nobs, nvar = x.shape
    if maxlag == 0:
        return np.empty((nobs if trim in ("forward", "none") else nobs, 0))
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(maxlag + 1):
        lm[maxlag - k : nobs + maxlag - k, nvar * (maxlag - k) : nvar * (maxlag - k + 1)] = x
    startobs = 0 if trim in ("none", "forward") else maxlag
    stopobs = len(lm) if trim in ("none", "backward") else nobs
    return lm[startobs:stopobs, nvar:]


def is_invertible(polynomial, threshold=1.0):
    """True when every root of `polynomial` lies outside the unit circle.

    `polynomial` is given lowest-order-first with a leading 1, matching
    statsmodels. Companion eigenvalues are the reciprocals of the roots, so
    the test is on their moduli.
    """
    poly = np.asarray(polynomial, dtype=float)
    n = len(poly) - 1
    if n <= 0:
        return True
    comp = np.zeros((n, n))
    comp[:, 0] = -poly[1:] / poly[0]
    comp[:-1, 1:] = np.eye(n - 1)
    return bool(np.all(np.abs(np.linalg.eigvals(comp)) < threshold))


def _simple_poly(coefs):
    return np.r_[1.0, np.asarray(coefs, dtype=float)]


def _seasonal_poly(n, periods):
    p = np.zeros(n * periods + 1)
    p[0] = 1.0
    for i in range(n):
        p[(i + 1) * periods] = 1.0
    return p


def _conditional_sum_squares(
    endog, k_ar, polynomial_ar, k_ma, polynomial_ma, k_trend=0, trend_data=None
):
    """Port of `SARIMAX._conditional_sum_squares`.

    Two stages: fit a long AR to recover residuals, then regress the series on
    its own lags and those residuals. It is a Hannan-Rissanen starting value,
    not an estimate, so it only has to be reproduced faithfully -- not
    justified.
    """
    k = 2 * k_ma
    r = max(k + k_ma, k_ar)

    k_params_ar = 0 if k_ar == 0 else len(np.nonzero(polynomial_ar)[0]) - 1
    k_params_ma = 0 if k_ma == 0 else len(np.nonzero(polynomial_ma)[0]) - 1

    residuals = None
    params = None
    if k_ar + k_ma + k_trend > 0:
        try:
            if k_ma > 0:
                Y = endog[k:]
                X = lagmat(endog, k, trim="both")
                params_ar = np.linalg.pinv(X).dot(Y)
                residuals = Y - np.dot(X, params_ar)

            Y = endog[r:]
            X = np.empty((Y.shape[0], 0))
            if k_trend > 0:
                if trend_data is None:
                    raise ValueError("trend data required when k_trend > 0")
                X = np.c_[X, trend_data[: (-r if r > 0 else None), :]]
            if k_ar > 0:
                cols = np.nonzero(polynomial_ar)[0][1:] - 1
                X = np.c_[X, lagmat(endog, k_ar)[r:, cols]]
            if k_ma > 0:
                cols = np.nonzero(polynomial_ma)[0][1:] - 1
                X = np.c_[X, lagmat(residuals, k_ma)[r - k :, cols]]

            params = np.linalg.pinv(X).dot(Y)
            residuals = Y - np.dot(X, params)
        except (ValueError, np.linalg.LinAlgError):
            params = np.zeros(k_trend + k_ar + k_ma)
            if len(endog) == 0:
                residuals = np.ones(k_params_ma * 2 + 1)
            else:
                residuals = np.r_[np.zeros(k_params_ma * 2), endog - np.mean(endog)]

    params_trend, params_ar, params_ma, params_variance = [], [], [], []
    offset = 0
    if k_trend > 0:
        params_trend = params[offset : k_trend + offset]
        offset += k_trend
    if k_ar > 0:
        params_ar = params[offset : k_params_ar + offset]
        offset += k_params_ar
    if k_ma > 0:
        params_ma = params[offset : k_params_ma + offset]
        offset += k_params_ma
    if residuals is not None:
        if len(residuals) > max(1, k_params_ma):
            params_variance = (residuals[k_params_ma:] ** 2).mean()
        else:
            params_variance = np.var(endog)
    return params_trend, params_ar, params_ma, params_variance


class Spec:
    """A parsed SARIMAX specification.

    Mirrors the parts of `statsmodels.SARIMAX` that this package needs, using
    the same names so the two can be compared directly in tests.
    """

    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        k_exog=0,
        enforce_stationarity=True,
        enforce_invertibility=True,
        concentrate_scale=False,
    ):
        self.order = tuple(int(v) for v in order)
        so = tuple(seasonal_order)
        if len(so) != 4:
            raise ValueError("seasonal_order must have four elements")
        self.seasonal_order = tuple(int(v) for v in so)
        self.p, self.d, self.q = self.order
        self.bp, self.bd, self.bq, self.s = self.seasonal_order
        if self.s == 1:
            # statsmodels treats m=1 as non-seasonal
            self.bp = self.bd = self.bq = 0
            self.s = 0
        self.trend = trend
        self.trend_powers = parse_trend(trend)
        self.k_trend = len(self.trend_powers)
        self.k_exog = int(k_exog)
        self.enforce_stationarity = bool(enforce_stationarity)
        self.enforce_invertibility = bool(enforce_invertibility)
        self.concentrate_scale = bool(concentrate_scale)

        self.k_ar = self.p
        self.k_ma = self.q
        self.k_seasonal_ar = self.bp * self.s
        self.k_seasonal_ma = self.bq * self.s
        self._k_order = max(
            self.k_ar + self.k_seasonal_ar, self.k_ma + self.k_seasonal_ma + 1
        )
        self._k_states_diff = self.d + self.bd * self.s
        self.k_states = self._k_order + self._k_states_diff
        self.loglikelihood_burn = (
            self._k_states_diff if self.enforce_stationarity else self.k_states
        )

    @property
    def k_params(self):
        return (
            self.k_trend
            + self.k_exog
            + self.p
            + self.q
            + self.bp
            + self.bq
            + (0 if self.concentrate_scale else 1)
        )

    @property
    def param_names(self):
        names = []
        for pw in self.trend_powers:
            names.append({0: "intercept", 1: "drift"}.get(pw, f"trend.{pw}"))
        names += [f"x{i + 1}" for i in range(self.k_exog)]
        names += [f"ar.L{i + 1}" for i in range(self.p)]
        names += [f"ma.L{i + 1}" for i in range(self.q)]
        names += [f"ar.S.L{(i + 1) * self.s}" for i in range(self.bp)]
        names += [f"ma.S.L{(i + 1) * self.s}" for i in range(self.bq)]
        if not self.concentrate_scale:
            names.append("sigma2")
        return names

    def trend_data(self, nobs, offset=1):
        if self.k_trend == 0:
            return None
        t = np.arange(offset, nobs + offset, dtype=float)
        return np.column_stack([t**pw for pw in self.trend_powers])

    # ------------------------------------------------------------------
    def start_params(self, endog, exog=None):
        """Reproduce `SARIMAX.start_params`."""
        endog = np.asarray(endog, dtype=float)
        nobs = endog.shape[0]
        trend_data = self.trend_data(nobs)

        if self.d > 0 or self.bd > 0:
            e = diff(endog, self.d, self.bd, self.s)
            x = diff(exog, self.d, self.bd, self.s) if exog is not None else None
            if trend_data is not None:
                trend_data = trend_data[: e.shape[0], :]
        else:
            e = endog.copy()
            x = None if exog is None else np.array(exog, dtype=float)
        e = np.squeeze(e)

        if np.any(np.isnan(e)):
            mask = ~np.isnan(e)
            e = e[mask]
            if x is not None:
                x = x[mask]
            if trend_data is not None:
                trend_data = trend_data[mask]

        params_exog = []
        if self.k_exog > 0 and x is not None:
            params_exog = np.linalg.pinv(x).dot(e)
            e = e - np.dot(x, params_exog)

        poly_ar = _simple_poly(np.zeros(self.p)) if self.p else np.r_[1.0]
        poly_ar = np.r_[1.0, np.ones(self.p)] if self.p else np.r_[1.0]
        poly_ma = np.r_[1.0, np.ones(self.q)] if self.q else np.r_[1.0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params_trend, params_ar, params_ma, params_variance = (
                _conditional_sum_squares(
                    e, self.k_ar, poly_ar, self.k_ma, poly_ma, self.k_trend, trend_data
                )
            )

        if (
            self.k_ar > 0
            and self.enforce_stationarity
            and not is_invertible(np.r_[1, -np.asarray(params_ar)])
        ):
            params_ar = np.asarray(params_ar) * 0
        if (
            self.k_ma > 0
            and self.enforce_invertibility
            and not is_invertible(np.r_[1, np.asarray(params_ma)])
        ):
            params_ma = np.asarray(params_ma) * 0

        spoly_ar = _seasonal_poly(self.bp, self.s) if self.bp else np.r_[1.0]
        spoly_ma = _seasonal_poly(self.bq, self.s) if self.bq else np.r_[1.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, params_sar, params_sma, params_svar = _conditional_sum_squares(
                e, self.k_seasonal_ar, spoly_ar, self.k_seasonal_ma, spoly_ma
            )

        if (
            self.k_seasonal_ar > 0
            and self.enforce_stationarity
            and not is_invertible(np.r_[1, -np.asarray(params_sar)])
        ):
            params_sar = np.asarray(params_sar) * 0
        if (
            self.k_seasonal_ma > 0
            and self.enforce_invertibility
            and not is_invertible(np.r_[1, np.asarray(params_sma)])
        ):
            params_sma = np.asarray(params_sma) * 0

        if isinstance(params_variance, list) and len(params_variance) == 0:
            if not (isinstance(params_svar, list) and len(params_svar) == 0):
                params_variance = params_svar
            elif self.k_exog > 0:
                params_variance = np.inner(e, e)
            else:
                params_variance = np.inner(e, e) / nobs

        params_variance = np.atleast_1d(np.array(params_variance))
        if params_variance.size:
            params_variance = np.atleast_1d(max(params_variance[0], 1e-10))
        if self.concentrate_scale:
            params_variance = []

        return np.r_[
            params_trend,
            params_exog,
            params_ar,
            params_ma,
            params_sar,
            params_sma,
            params_variance,
        ].astype(float)
