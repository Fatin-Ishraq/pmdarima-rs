"""Unit-root tests: ADF, KPSS and Phillips-Perron.

These decide `d`, so they have to agree with `pmdarima` exactly - a different
differencing order is a different model, not a slightly different number.

They are also not the bottleneck. Each runs once per series, in `O(n)` time,
against an `auto_arima` search that fits dozens of models; a profile of a
seasonal fit puts essentially all of the time in the Kalman filter. So these
are faithful ports of the `pmdarima` routines (MIT, Taylor G. Smith et al.),
which this package is also licensed under, rather than reimplementations that
would risk changing an answer to save nothing.
"""

import warnings

import numpy as np

from ..base import BaseEstimator
from ..utils.array import c, check_endog, diff  # noqa: F401  (re-exported use)

__all__ = ["ADFTest", "KPSSTest", "PPTest"]


def approx(x, y, xout, rule=2):
    """R's `approx` with `ties='mean'`, which is how `pmdarima` calls it.

    `numpy.interp` already clamps outside the range, which is what `rule=2`
    means; what it does not do is tolerate unsorted or duplicated `x`, so that
    part is done first.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x)
    x, y = x[order], y[order]
    ux = np.unique(x)
    if ux.shape[0] < x.shape[0]:
        y = np.array([y[x == u].mean() for u in ux])
        x = ux
    xout = np.atleast_1d(np.asarray(xout, dtype=float))
    return xout, np.interp(xout, x, y)


def pp_sum(u, n, l_, total):
    """Bartlett-kernel long-run variance correction.

    A direct translation of `pmdarima`'s `C_tseries_pp_sum`, itself from R's
    `tseries`. `l_` is small (4-12 for realistic `n`), so this stays `O(n*l)`.
    """
    u = np.asarray(u, dtype=float).ravel()
    for i in range(1, int(l_) + 1):
        if i >= n:
            # R's tseries stops at the end of the series; without this guard a
            # short series with `lshort=False` slices past the end and numpy
            # raises on the shape mismatch.
            break
        tmp = float(np.dot(u[i:n], u[0 : n - i]))
        tmp *= 1.0 - i / (l_ + 1.0)
        total += 2.0 * tmp / n
    return total


class _BaseStationarityTest(BaseEstimator):
    @staticmethod
    def _base_case(x):
        return x is not None and np.asarray(x).shape[0] != 0

    @staticmethod
    def _embed(x, k):
        """R's `embed`: rows are successively lagged windows."""
        x = np.asarray(x)
        n = x.shape[0]
        if k > n:
            raise ValueError("k cannot exceed y dim")
        return np.asarray([x[j : n - i] for i, j in enumerate(range(k - 1, -1, -1))])

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._param_names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        from ..base import repr_with_defaults

        return repr_with_defaults(self, self._param_names)


class _DifferencingStationarityTest(_BaseStationarityTest):
    def __init__(self, alpha):
        self.alpha = alpha

    def should_diff(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def is_stationary(self, x):
        """Deprecated alias kept for compatibility with older `pmdarima`."""
        warnings.warn(
            "is_stationary is deprecated and will be removed in a future "
            "release. Use should_diff instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.should_diff(x)


class KPSSTest(_DifferencingStationarityTest):
    """Kwiatkowski-Phillips-Schmidt-Shin test. Null: the series is stationary."""

    _param_names = ("alpha", "null", "lshort")
    _valid = {"trend", "level"}
    tablep = np.array([0.01, 0.025, 0.05, 0.10])

    def __init__(self, alpha=0.05, null="level", lshort=True):
        super().__init__(alpha=alpha)
        self.null = null
        self.lshort = lshort

    def should_diff(self, x):
        if not self._base_case(x):
            return np.nan, False
        x = check_endog(x, preserve_series=False, force_all_finite=True, input_name="y")
        n = x.shape[0]

        if self.null == "trend":
            t = np.arange(n).reshape(n, 1)
            table = np.array([0.216, 0.176, 0.146, 0.119])
        elif self.null == "level":
            t = np.ones(n).reshape(n, 1)
            table = np.array([0.739, 0.574, 0.463, 0.347])
        else:
            raise ValueError("null must be one of %r" % self._valid)

        # Ordinary least squares with an intercept, matching
        # sklearn.LinearRegression's default.
        tc = t - t.mean(axis=0)
        xc = x - x.mean()
        beta, *_ = np.linalg.lstsq(tc, xc, rcond=None)
        e = xc - tc.dot(beta)

        s = np.cumsum(e)
        eta = (s * s).sum() / (n**2)
        s2 = (e * e).sum() / n

        scalar = 4 if self.lshort else 12
        l_ = int(np.trunc(scalar * (n / 100.0) ** 0.25))
        s2 = pp_sum(e, n, l_, s2)

        stat = eta / s2
        _, pval = approx(table, self.tablep, xout=stat, rule=2)
        return pval[0], pval[0] < self.alpha


class ADFTest(_DifferencingStationarityTest):
    """Augmented Dickey-Fuller test. Null: the series has a unit root."""

    _param_names = ("alpha", "k")
    table = np.array(
        [
            (-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15),
            (-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24),
            (-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28),
            (-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31),
            (-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32),
            (-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33),
        ]
    )
    tablen = table.shape[1]
    tableT = np.array([25.0, 50.0, 100.0, 250.0, 500.0, 100000.0])
    tablep = np.array([0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99])

    def __init__(self, alpha=0.05, k=None):
        super().__init__(alpha=alpha)
        self.k = k
        if k is not None and k < 0:
            raise ValueError("k must be a positive integer (>= 0)")

    @staticmethod
    def _ols_t_stat(x, y, z, k):
        """OLS of the differenced series on level, trend and its own lags.

        `pmdarima` reaches for `statsmodels.OLS(...).fit(method='qr')` and then
        for `res.bse[1]`. The same t statistic is available from a QR solve
        plus the usual `sigma^2 (X'X)^-1` standard error, without pulling in
        statsmodels for one number.
        """
        n = y.shape[0]
        yt = z[:, 0]
        tt = np.arange(k - 1, n)
        xt1 = x[tt]
        tt = tt + 1
        _n = xt1.shape[0]
        X = np.hstack(
            [
                np.ones((_n, 1)),
                xt1.reshape((_n, 1)),
                tt.reshape((_n, 1)).astype(float),
            ]
        )
        if k > 1:
            X = np.hstack([X, z[:, 1:k]])

        q, r = np.linalg.qr(X)
        beta = np.linalg.solve(r, q.T.dot(yt))
        resid = yt - X.dot(beta)
        dof = X.shape[0] - X.shape[1]
        sigma2 = resid.dot(resid) / dof
        # statsmodels' `OLS.fit(method="qr")` forms `inv(R'R)` and reads its
        # diagonal. Doing the same arithmetic in the same order keeps the t
        # statistic equal on badly scaled series, where the two algebraically
        # identical routes drift apart.
        cov_diag = np.diag(np.linalg.inv(r.T.dot(r)))
        se = np.sqrt(sigma2 * cov_diag)
        return beta[1] / se[1]

    def should_diff(self, x):
        if not self._base_case(x):
            return np.nan, False
        x = check_endog(x, preserve_series=False, force_all_finite=True)

        k = self.k
        if k is None:
            k = np.trunc(np.power(x.shape[0] - 1, 1 / 3.0))
        k = int(k) + 1

        y = diff(x)
        n = y.shape[0]
        z = self._embed(y, k).T
        stat = self._ols_t_stat(x, y, z, k)

        tableipl = np.array(
            [
                approx(self.tableT, self.table[:, i], xout=n, rule=2)[1][0]
                for i in range(self.tablen)
            ]
        )
        _, interpol = approx(tableipl, self.tablep, xout=stat, rule=2)
        pval = interpol[0]
        # Note the direction: this p-value is for stationarity, so we do NOT
        # difference when it exceeds alpha.
        return pval, pval > self.alpha


class PPTest(_DifferencingStationarityTest):
    """Phillips-Perron test. Null: the series has a unit root."""

    _param_names = ("alpha", "lshort")
    table = -np.array(
        [
            (22.5, 25.7, 27.4, 28.4, 28.9, 29.5),
            (19.9, 22.4, 23.6, 24.4, 24.8, 25.1),
            (17.9, 19.8, 20.7, 21.3, 21.5, 21.8),
            (15.6, 16.8, 17.5, 18.0, 18.1, 18.3),
            (3.66, 3.71, 3.74, 3.75, 3.76, 3.77),
            (2.51, 2.60, 2.62, 2.64, 2.65, 2.66),
            (1.53, 1.66, 1.73, 1.78, 1.78, 1.79),
            (0.43, 0.65, 0.75, 0.82, 0.84, 0.87),
        ]
    ).T
    tablen = table.shape[1]
    tableT = np.array([25.0, 50.0, 100.0, 250.0, 500.0, 100000.0])
    tablep = np.array([0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99])

    def __init__(self, alpha=0.05, lshort=True):
        super().__init__(alpha=alpha)
        self.lshort = lshort

    def should_diff(self, x):
        if not self._base_case(x):
            return np.nan, False
        x = check_endog(x, preserve_series=False, force_all_finite=True, input_name="X")

        z = self._embed(x, 2)
        yt = z[0, :]
        yt1 = z[1, :]
        n = yt.shape[0]

        tt = (np.arange(n) + 1) - (n / 2.0)
        X = np.column_stack([tt, yt1])
        xc = X - X.mean(axis=0)
        yc = yt - yt.mean()
        beta, *_ = np.linalg.lstsq(xc, yc, rcond=None)
        u = yc - xc.dot(beta)

        ssqru = (u * u).sum() / float(n)
        scalar = 4 if self.lshort else 12
        l_ = int(np.trunc(scalar * np.power(n / 100.0, 0.25)))
        ssqrtl = pp_sum(u, n, l_, ssqru)

        n2 = n * n
        syt11n = (yt1 * (np.arange(n) + 1)).sum()
        trm1 = n2 * (n2 - 1) * (yt1**2).sum() / 12.0
        trm2 = n * (syt11n**2)
        trm3 = n * (n + 1) * syt11n * yt1.sum()
        trm4 = (n * (n + 1) * (2 * n + 1) * (yt1.sum() ** 2)) / 6.0
        dx = trm1 - trm2 + trm3 - trm4

        alpha = beta[1]
        stat = n * (alpha - 1) - (n**6) / (24.0 * dx) * (ssqrtl - ssqru)

        tableipl = np.array(
            [
                approx(self.tableT, self.table[:, i], xout=n, rule=2)[1][0]
                for i in range(self.tablen)
            ]
        )
        _, interpol = approx(tableipl, self.tablep, xout=stat, rule=2)
        pval = interpol[0]
        return pval, pval > self.alpha
