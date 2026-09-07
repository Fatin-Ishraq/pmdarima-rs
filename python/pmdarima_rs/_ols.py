"""A minimal OLS, matching the pieces of `statsmodels.OLS` that are used here.

`OCSBTest` needs `params`, `tvalues`, `aic` and `bic` from a QR least-squares
fit, and `CHTest` needs residuals. Reproducing those numbers is a few dozen
lines; depending on `statsmodels` for them would pull the whole library back
into a package whose point is not to need it.

The definitions follow `statsmodels.OLS(...).fit(method='qr')` exactly,
because `OCSBTest` selects its lag order by comparing these information
criteria and then reads a t statistic off the winning fit - so a different
degrees-of-freedom convention here is a different seasonal differencing term,
not a rounding difference.

In particular this reproduces statsmodels' behaviour on *degenerate* designs
rather than rejecting them. A design with no residual degrees of freedom is
not an error there: `scale = ssr / 0` is infinite, so the standard errors are
infinite and the t statistics are zero, while `llf` uses `ssr / nobs` and
comes out very large. `pmdarima` relies on that: for short series `OCSBTest`
picks such a fit on information criteria and reads its t statistic.
"""

import numpy as np

__all__ = ["OLSResult", "add_constant", "ols"]


class OLSResult:
    """The subset of `statsmodels`' `OLSResults` that this package reads."""

    __slots__ = (
        "params",
        "resid",
        "ssr",
        "nobs",
        "rank",
        "df_model",
        "df_resid",
        "k_constant",
        "scale",
        "_cov_diag",
    )

    def __init__(self, params, resid, nobs, rank, k_constant, cov_diag):
        self.params = params
        self.resid = resid
        self.ssr = float(resid.dot(resid))
        self.nobs = float(nobs)
        self.rank = int(rank)
        self.k_constant = int(k_constant)
        # statsmodels: df_model excludes the constant, df_resid = nobs - rank
        self.df_model = float(rank - k_constant)
        self.df_resid = float(nobs - rank)
        with np.errstate(divide="ignore", invalid="ignore"):
            # numpy division, not Python's: a design with no residual degrees
            # of freedom has to give an infinite scale (and so zero t
            # statistics), which is what statsmodels produces and what
            # `OCSBTest` then selects on for very short series.
            self.scale = float(np.float64(self.ssr) / np.float64(self.df_resid))
        self._cov_diag = cov_diag

    @property
    def bse(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(self._cov_diag * self.scale)

    @property
    def llf(self):
        n = self.nobs
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(
                -0.5 * n * (np.log(2 * np.pi) + np.log(np.float64(self.ssr) / n) + 1)
            )

    @property
    def _k_params(self):
        """`df_model + k_constant`, which is just the rank of the design."""
        return self.df_model + self.k_constant

    @property
    def aic(self):
        return -2 * self.llf + 2 * self._k_params

    @property
    def bic(self):
        return -2 * self.llf + np.log(self.nobs) * self._k_params

    @property
    def tvalues(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.params / self.bse

    def predict(self, X):
        return np.asarray(X, dtype=float).dot(self.params)


def add_constant(X, prepend=True):
    """`statsmodels.tools.add_constant`, including its no-op behaviour.

    statsmodels declines to add a column when one is already constant *and
    non-zero*; an all-zero column does not count, which matters because
    `OCSBTest` builds exactly that when its lag order is zero.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] > 0 and X.shape[0] > 0:
        is_nonzero_const = np.ptp(X, axis=0) == 0
        is_nonzero_const &= np.all(X != 0.0, axis=0)
        if is_nonzero_const.any():
            return X
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X]) if prepend else np.hstack([X, ones])


def _k_constant(X):
    """Reproduce `statsmodels.base.data._handle_constant`."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        return 0
    exog_max = np.max(X, axis=0)
    if not np.isfinite(exog_max).all():
        raise ValueError("exog contains inf or nans")
    exog_min = np.min(X, axis=0)
    const_idx = np.where(exog_max == exog_min)[0]
    k_constant = const_idx.size
    check_implicit = False

    if k_constant == 1:
        if X[:, const_idx[0]].mean() == 0:
            check_implicit = True
    elif k_constant > 1:
        values = [X[:, idx].mean() for idx in const_idx]
        if 1 in values:
            k_constant = 1
        else:
            pos = np.array(values) != 0
            if pos.any():
                k_constant = 1
            else:
                check_implicit = True
    else:
        check_implicit = True

    if check_implicit:
        augmented = np.column_stack((np.ones(X.shape[0]), X))
        k_constant = int(np.linalg.matrix_rank(X) == np.linalg.matrix_rank(augmented))
    return k_constant


def ols(y, X):
    """Least squares by QR, mirroring `statsmodels.OLS(...).fit(method='qr')`.

    `numpy.linalg.solve` raises `LinAlgError` on a singular `R`, and that is
    load-bearing: `OCSBTest` can select a lag of zero, which produces an
    all-zero regressor, and `pmdarima` relies on the resulting `LinAlgError`
    to fall back to the best regression it already has.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]

    q, r = np.linalg.qr(X)
    params = np.linalg.solve(r, q.T.dot(y))  # raises LinAlgError if singular
    rank = int(np.linalg.matrix_rank(r))
    # statsmodels forms `inv(R'R)` and reads its diagonal; doing the same
    # arithmetic in the same order is what keeps the standard errors - and so
    # the OCSB t statistic - equal on badly conditioned designs.
    cov_diag = np.diag(np.linalg.inv(r.T.dot(r)))
    resid = y - X.dot(params)
    return OLSResult(params, resid, n, rank, _k_constant(X), cov_diag)
