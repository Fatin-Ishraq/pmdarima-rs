"""Seasonal unit-root tests: Canova-Hansen and OCSB, plus `decompose`.

These decide `D`, so like the stationarity tests they have to agree with
`pmdarima` exactly, and like them they run once per series rather than once
per candidate model. Faithful ports of the `pmdarima` routines (MIT, Taylor
G. Smith et al.), which this package is also licensed under.

The one piece with no Python original is `_canova_hansen_sd_test`, which
`pmdarima` ships as a Cython extension. It is the Newey-West long-run
covariance of the score contributions, written out here.
"""

import math
from collections import namedtuple

import numpy as np
from scipy.linalg import svd

from .._ols import add_constant, ols
from ..utils.array import _assert_all_finite, c, check_endog, diff
from .stationarity import _BaseStationarityTest

__all__ = ["CHTest", "OCSBTest", "decompose"]


def decompose(x, type_, m, filter_=None):
    """Classical additive or multiplicative decomposition."""
    multiplicative, additive = "multiplicative", "additive"
    x = np.asarray(x, dtype=float).ravel()
    is_m_odd = m % 2 == 1

    def helper(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return a / b if type_ == multiplicative else a - b

    if not (isinstance(m, (int, np.integer)) and m > 1):
        raise ValueError("'f' should be a positive integer")
    if filter_ is None:
        filter_ = np.ones((m,)) / m
    if type_ not in (multiplicative, additive):
        raise ValueError(f"'type_' can only take values '{multiplicative}' or '{additive}'")
    if (x.shape[0] / m) < 2:
        raise ValueError("time series has no or less than 2 periods")

    half_m = m // 2
    trend = np.convolve(x, filter_, mode="valid")
    if not is_m_odd:
        trend = trend[:-1]

    sma_xs = range(half_m, len(trend) + half_m)
    detrend = helper(x[sma_xs], trend)

    num_seasons = math.ceil((1.0 * trend.shape[0]) / m)
    pad_length = (num_seasons * m) - trend.shape[0]
    if pad_length > 0:
        detrend = np.array(detrend.tolist() + pad_length * [np.nan])

    m_arr = np.reshape(detrend, (num_seasons, m))
    seasonal = np.nanmean(m_arr, axis=0).tolist()
    seasonal = np.array(seasonal[half_m:] + seasonal[:half_m])
    temp = seasonal
    for _ in range(m_arr.shape[0]):
        seasonal = np.concatenate((seasonal, temp))
    if pad_length > 0:
        seasonal = seasonal[:-pad_length]
    if is_m_odd:
        seasonal = seasonal[:-1]

    buffer = [np.nan] * half_m
    # `pmdarima` hands the trend back as a list; callers index and
    # concatenate it, and a numpy array behaves differently for both.
    trend = list(buffer + trend.tolist() + buffer)
    random = helper(helper(x, trend), seasonal)

    decomposed = namedtuple("decomposed", "x trend seasonal random")
    return decomposed(x, trend, seasonal, random)


class _SeasonalStationarityTest(_BaseStationarityTest):
    def __init__(self, m):
        self.m = m
        if m < 2:
            raise ValueError("m must be > 1")

    def estimate_seasonal_differencing_term(self, x):  # pragma: no cover
        raise NotImplementedError


def _canova_hansen_sd_test(ltrunc, ne, fhataux, frec, s):
    """Newey-West long-run covariance and the frequency selection matrix.

    This is `pmdarima`'s `C_canova_hansen_sd_test`, which in turn follows
    R's `forecast::SD.test`. `Omfhat` is the Bartlett-weighted autocovariance
    sum of the score contributions; `A` selects the columns belonging to the
    frequencies being tested.
    """
    fhataux = np.asarray(fhataux, dtype=float)
    ncol = fhataux.shape[1]

    omfhat = np.zeros((ncol, ncol))
    for i in range(1, int(ltrunc) + 1):
        if i >= ne:
            break
        aux = fhataux[i:ne].T.dot(fhataux[0 : ne - i])
        omfhat += (1.0 - i / (ltrunc + 1.0)) * (aux + aux.T)
    omfhat = (omfhat + fhataux.T.dot(fhataux)) / ne

    # Which columns of the seasonal dummy basis are under test.
    sq = np.arange(0, s - 1, 2)
    frecob = np.zeros(s - 1, dtype=int)
    for i in range(int(s / 2)):
        if frec[i] == 1 and i == s / 2 - 1:
            frecob[sq[i]] = 1
        if frec[i] == 1 and i < s / 2 - 1:
            frecob[sq[i]] = 1
            frecob[sq[i] + 1] = 1

    a = int(frecob.sum())
    A = np.zeros((s - 1, a))
    j = 0
    for i in range(s - 1):
        if frecob[i] == 1:
            A[i, j] = 1
            j += 1

    return A, A.T.dot(omfhat).dot(A)


class CHTest(_SeasonalStationarityTest):
    """Canova-Hansen test for seasonal differencing."""

    _param_names = ("m",)
    crit_vals = c(
        0.4617146,
        0.7479655,
        1.0007818,
        1.2375350,
        1.4625240,
        1.6920200,
        1.9043096,
        2.1169602,
        2.3268562,
        2.5406922,
        2.7391007,
    )

    def __init__(self, m):
        super().__init__(m=m)

    @staticmethod
    def _seas_dummy(x, m):
        n = x.shape[0]
        assert m > 1
        tt = np.arange(n) + 1
        fmat = np.ones((n, 2 * m)) * np.nan
        for i in range(1, m + 1):
            fmat[:, (2 * i) - 1] = np.sin(2 * np.pi * i * tt / m)
            fmat[:, 2 * (i - 1)] = np.cos(2 * np.pi * i * tt / m)
        return fmat[:, : m - 1]

    @staticmethod
    def _sd_test(wts, s):
        _assert_all_finite(wts)
        n = wts.shape[0]
        frec = np.ones(int((s + 1) / 2), dtype=np.intp)
        ltrunc = int(np.round(s * ((n / 100.0) ** 0.25)))

        R1 = CHTest._seas_dummy(wts, s)
        # pmdarima scales without centring, then fits with an intercept. The
        # scaling cancels out of the residuals, so plain OLS with a constant
        # gives the same `residuals`.
        design = np.hstack([np.ones((n, 1)), R1])
        beta, *_ = np.linalg.lstsq(design, wts, rcond=None)
        residuals = wts - design.dot(beta)

        fhataux = (R1.T * residuals).T.astype(np.float64)
        ne = fhataux.shape[0]
        A, at_omfhat_a = _canova_hansen_sd_test(ltrunc, ne, fhataux, frec, s)

        sv = svd(at_omfhat_a, compute_uv=False)
        if sv.min() < np.finfo(sv.dtype).eps:
            return 0

        fhat = fhataux.cumsum(axis=0)
        solved = np.linalg.solve(at_omfhat_a, np.identity(at_omfhat_a.shape[0]))
        return (
            (1.0 / n**2)
            * solved.dot(A.T).dot(fhat.T).dot(fhat).dot(A).diagonal().sum()
        )

    @staticmethod
    def _calc_ch_crit_val(m):
        if m <= 12:
            return CHTest.crit_vals[m - 2]
        if m == 24:
            return 5.098624
        if m == 52:
            return 10.341416
        if m == 365:
            return 65.44445
        return 0.269 * (m**0.928)

    def estimate_seasonal_differencing_term(self, x):
        if not self._base_case(x):
            return 0
        x = check_endog(x, preserve_series=False)
        n = x.shape[0]
        m = int(self.m)
        if n < 2 * m + 5:
            return 0
        chstat = self._sd_test(x, m)
        return int(chstat > self._calc_ch_crit_val(m))


def _aicc(res, nobs, add_const):
    """`pmdarima._aicc`: AIC with the small-sample correction."""
    df_model = res.df_model
    if add_const:
        df_model += 1
    return res.aic + 2.0 * df_model * (nobs / (nobs - df_model - 1.0) - 1.0)


class OCSBTest(_SeasonalStationarityTest):
    """Osborn-Chui-Smith-Birchenhall test for seasonal differencing."""

    _param_names = ("m", "lag_method", "max_lag")

    _ic_method_map = {
        "aic": lambda fit: fit.aic,
        "bic": lambda fit: fit.bic,
        "aicc": lambda fit: _aicc(fit, fit.nobs, False),
    }

    def __init__(self, m, lag_method="aic", max_lag=3):
        super().__init__(m=m)
        self.lag_method = lag_method
        self.max_lag = max_lag

    @staticmethod
    def _calc_ocsb_crit_val(m):
        log_m = np.log(m)
        return (
            -0.2937411
            * np.exp(
                -0.2850853 * (log_m - 0.7656451)
                + (-0.05983644) * ((log_m - 0.7656451) ** 2)
            )
            - 1.652202
        )

    @staticmethod
    def _do_lag(y, lag, omit_na=True):
        n = y.shape[0]
        if lag == 1:
            return y.reshape(n, 1)
        out = np.ones((n + (lag - 1), lag)) * np.nan
        for i in range(lag):
            out[i : i + n, i] = y
        if omit_na:
            out = out[~np.isnan(out).any(axis=1)]
        return out

    @staticmethod
    def _gen_lags(y, max_lag, omit_na=True):
        if max_lag <= 0:
            return np.zeros(y.shape[0])
        return OCSBTest._do_lag(y, max_lag, omit_na)

    @staticmethod
    def _fit_ocsb(x, m, lag, max_lag):
        y_first_order_diff = diff(x, m)
        if y_first_order_diff.shape[0] == 0:
            raise ValueError(
                "There are no more samples after a first-order "
                "seasonal differencing. See http://alkaline-ml.com/pmdarima/"
                "seasonal-differencing-issues.html for a more in-depth "
                "explanation and potential work-arounds."
            )
        y = diff(y_first_order_diff)
        ylag = OCSBTest._gen_lags(y, lag)

        if max_lag > -1:
            y = y[max_lag:]

        mf = ylag[: y.shape[0]]
        ar_fit = ols(y, add_constant(mf))

        z4_y = y_first_order_diff[lag:]
        z4_lag = OCSBTest._gen_lags(y_first_order_diff, lag)[: z4_y.shape[0], :]
        z4 = z4_y - ar_fit.predict(add_constant(z4_lag))

        z5_y = diff(x)
        z5_lag = OCSBTest._gen_lags(z5_y, lag)
        z5_y = z5_y[lag:]
        z5_lag = z5_lag[: z5_y.shape[0], :]
        z5 = z5_y - ar_fit.predict(add_constant(z5_lag))

        data = np.hstack(
            (
                mf,
                z4[: mf.shape[0]].reshape(-1, 1),
                z5[: mf.shape[0]].reshape(-1, 1),
            )
        )
        return ols(y, data)

    def _compute_test_statistic(self, x):
        _assert_all_finite(x)
        m = self.m
        maxlag = self.max_lag
        method = self.lag_method
        crit_regression = None

        if maxlag > 0 and method != "fixed":
            try:
                icfunc = self._ic_method_map[method]
            except KeyError as err:
                raise ValueError(
                    f"'{method}' is an invalid method. Must be one of "
                    "('aic', 'aicc', 'bic', 'fixed')"
                ) from err

            fits, icvals = [], []
            for lag_term in range(1, maxlag + 1):
                try:
                    fit = self._fit_ocsb(x, m, lag_term, maxlag)
                    fits.append(fit)
                    icvals.append(icfunc(fit))
                except np.linalg.LinAlgError:
                    icvals.append(np.nan)
                    fits.append(None)

            if np.isnan(icvals).all():
                raise ValueError(
                    "All lag values up to 'maxlag' produced "
                    "singular matrices. Consider using a longer "
                    "series, a different lag term or a different "
                    "test."
                )
            best_index = int(np.nanargmin(icvals))
            maxlag = best_index - 1
            crit_regression = fits[best_index]

        try:
            regression = self._fit_ocsb(x, m, maxlag, maxlag)
        except np.linalg.LinAlgError as err:
            if crit_regression is not None:
                regression = crit_regression
            else:
                raise ValueError(
                    "Could not find a solution. Try a longer "
                    "series, different lag term, or a different "
                    "test."
                ) from err

        # R keeps only the z5 coefficient's t statistic.
        return regression.tvalues[-2:][-1]

    def estimate_seasonal_differencing_term(self, x):
        if not self._base_case(x):
            return 0
        x = check_endog(x, preserve_series=False)
        stat = self._compute_test_statistic(x)
        return int(stat > self._calc_ocsb_crit_val(self.m))
