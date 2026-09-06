"""A minimal OLS, matching the pieces of `statsmodels.OLS` that are used here.

`OCSBTest` needs `params`, `tvalues`, `aic` and `bic` from a QR least-squares
fit, and `CHTest` needs residuals. Reproducing those four numbers is a few
lines; depending on `statsmodels` for them would pull the whole library back
into a package whose point is not to need it.

The definitions follow `statsmodels` exactly, including its convention that
the loglikelihood uses the *uncorrected* variance `ssr / nobs` while the
standard errors use the degrees-of-freedom-corrected one.
"""

import numpy as np


class OLSResult:
    __slots__ = ("params", "resid", "ssr", "nobs", "df_model", "k_constant", "_bse")

    def __init__(self, params, resid, nobs, rank, k_constant, bse):
        self.params = params
        self.resid = resid
        self.ssr = float(resid.dot(resid))
        self.nobs = float(nobs)
        # statsmodels' df_model excludes the constant
        self.df_model = float(rank - k_constant)
        self.k_constant = k_constant
        self._bse = bse

    @property
    def bse(self):
        return self._bse

    @property
    def llf(self):
        n = self.nobs
        return -0.5 * n * (np.log(2 * np.pi) + np.log(self.ssr / n) + 1)

    @property
    def df_modelwc(self):
        """Degrees of freedom *with* the constant, as statsmodels counts it."""
        return self.df_model + self.k_constant

    @property
    def aic(self):
        return -2 * self.llf + 2 * self.df_modelwc

    @property
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * self.df_modelwc

    @property
    def tvalues(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self._bse

    def predict(self, X):
        return np.asarray(X, dtype=float).dot(self.params)


def add_constant(X, prepend=True):
    """`statsmodels.tools.add_constant`, including its no-op behaviour.

    statsmodels declines to add a column that is already constant; OCSB relies
    on that when a lag matrix happens to contain one.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] > 0:
        is_const = np.ptp(X, axis=0) == 0
        if np.any(is_const):
            return X
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X]) if prepend else np.hstack([X, ones])


def ols(y, X):
    """Least squares by QR, raising on a singular design.

    This mirrors `statsmodels.OLS(...).fit(method='qr')`, which forms `QR` and
    then calls `numpy.linalg.solve(R, Q.T y)` - and so raises `LinAlgError` on
    a rank-deficient design rather than quietly returning a minimum-norm
    solution.

    That behaviour is load-bearing, not incidental. `OCSBTest` can select a
    lag of zero, which produces an all-zero regressor; `pmdarima` relies on
    the resulting `LinAlgError` to fall back to the best regression it already
    has. Using `lstsq` here instead would sail past that point and fail later,
    with a different exception, on a different line.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape

    q, r = np.linalg.qr(X)
    params = np.linalg.solve(r, q.T.dot(y))  # raises LinAlgError if singular
    rank = k
    resid = y - X.dot(params)
    dof = n - rank
    if dof <= 0:
        raise np.linalg.LinAlgError("no residual degrees of freedom")
    sigma2 = resid.dot(resid) / dof
    xtx = X.T.dot(X)
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    bse = np.sqrt(np.maximum(sigma2 * np.diag(xtx_inv), 0.0))

    k_constant = int(np.any(np.ptp(X, axis=0) == 0)) if k else 0
    return OLSResult(params, resid, n, rank, k_constant, bse)
