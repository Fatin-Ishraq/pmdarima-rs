"""Autocorrelation and partial autocorrelation.

`pmdarima` re-exports these from `statsmodels`. Implementing them here removes
the last reason for this package to import statsmodels at all; the definitions,
the method names and the validation follow `statsmodels.tsa.stattools` exactly,
and the test suite compares against it.
"""

import warnings

import numpy as np

__all__ = ["acf", "pacf"]


def _norm_ppf(q):
    from scipy.special import ndtri

    return float(ndtri(q))


def acovf(x, adjusted=False, demean=True, fft=True, missing="none", nlag=None):
    """Autocovariance function, matching `statsmodels.tsa.stattools.acovf`.

    The `missing` handling is the fiddly part and is reproduced rather than
    reinterpreted: which denominator each mode divides by is what decides the
    numbers, and "drop" does not simply mean "delete and carry on".
    """
    from ..compat import MissingDataError

    x = np.asarray(x, dtype=float).ravel()
    if missing not in ("none", "raise", "conservative", "drop"):
        raise ValueError(
            'missing must be one of ("none", "raise", "conservative", "drop")'
        )
    notmask_bool = ~np.isnan(x)
    deal_with_masked = missing != "none" and not notmask_bool.all()
    notmask_int = None
    if deal_with_masked:
        if missing == "raise":
            raise MissingDataError("NaNs were encountered in the data")
        if missing == "drop" and notmask_bool.sum() == 0:
            raise ValueError("All observations are missing after dropping.")
        if missing == "conservative":
            x = x.copy()
            x[~notmask_bool] = 0
        else:  # "drop"
            x = x[notmask_bool]
        notmask_int = notmask_bool.astype(int)

    if demean and deal_with_masked:
        xo = x - x.sum() / notmask_int.sum()
        if missing == "conservative":
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    lag_len = n - 1 if nlag is None else nlag
    if nlag is not None and nlag > n - 1:
        raise ValueError("nlag must be smaller than nobs - 1")

    if not fft and nlag is not None:
        acov = np.empty(lag_len + 1)
        acov[0] = xo.dot(xo)
        for i in range(lag_len):
            acov[i + 1] = xo[i + 1 :].dot(xo[: -(i + 1)])
        if not deal_with_masked or missing == "drop":
            if adjusted:
                acov /= n - np.arange(lag_len + 1)
            else:
                acov /= n
        elif adjusted:
            divisor = np.empty(lag_len + 1, dtype=np.int64)
            divisor[0] = notmask_int.sum()
            for i in range(lag_len):
                divisor[i + 1] = notmask_int[i + 1 :].dot(notmask_int[: -(i + 1)])
            divisor[divisor == 0] = 1
            acov /= divisor
        else:
            acov /= notmask_int.sum()
        return acov

    if adjusted and deal_with_masked and missing == "conservative":
        d = np.correlate(notmask_int, notmask_int, "full")
        d[d == 0] = 1
    elif adjusted:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked:
        d = notmask_int.sum() * np.ones(2 * n - 1)
    else:
        d = n * np.ones(2 * n - 1)

    if fft:
        from scipy.fft import next_fast_len

        nobs = len(xo)
        nfft = next_fast_len(2 * nobs + 1)
        frf = np.fft.fft(xo, n=nfft)
        acov = np.fft.ifft(frf * np.conjugate(frf))[:nobs] / d[nobs - 1 :]
        acov = acov.real
    else:
        acov = np.correlate(xo, xo, "full")[n - 1 :] / d[n - 1 :]

    if deal_with_masked and notmask_bool.sum() == 0:
        acov[:] = np.nan
    if nlag is not None:
        return acov[: lag_len + 1].copy()
    return acov


def q_stat(x, nobs):
    """Ljung-Box Q statistic and its p-values."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    ret = (
        nobs
        * (nobs + 2)
        * np.cumsum((1.0 / (nobs - np.arange(1, x.shape[0] + 1))) * x**2)
    )
    chi2 = stats.chi2.sf(ret, np.arange(1, x.shape[0] + 1))
    return ret, chi2


def acf(
    x,
    nlags=None,
    qstat=False,
    fft=None,
    alpha=None,
    missing="none",
    adjusted=False,
):
    """Autocorrelation function, matching `statsmodels.tsa.stattools.acf`."""
    x = np.asarray(x, dtype=float).ravel()
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)
    if fft is None:
        fft = True
    if missing in ("drop", "conservative"):
        nobs = int(np.sum(~np.isnan(x)))
        if nobs == 0:
            raise ValueError("All observations are missing after dropping.")

    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft, missing=missing)
    ret = avf[: nlags + 1] / avf[0]

    out = [ret]
    if alpha is not None:
        varacf = np.ones_like(ret) / nobs
        varacf[0] = 0
        if len(varacf) > 1:
            varacf[1] = 1.0 / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(ret[1:-1] ** 2)
        interval = _norm_ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        out.append(np.array(list(zip(ret - interval, ret + interval))))
    if qstat:
        qs, pv = q_stat(ret[1:], nobs)
        out.extend([qs, pv])
    return out[0] if len(out) == 1 else tuple(out)


def _levinson_durbin_pacf(r, nlags):
    """Partial autocorrelations from an autocorrelation sequence."""
    r = np.asarray(r, dtype=float)
    pacf_ = np.zeros(nlags + 1)
    pacf_[0] = 1.0
    prev = np.zeros(0)
    for k in range(1, nlags + 1):
        if k == 1:
            phi_kk = r[1]
            cur = np.array([phi_kk])
        else:
            num = r[k] - np.dot(prev, r[1:k][::-1])
            den = 1.0 - np.dot(prev, r[1:k])
            phi_kk = 0.0 if den == 0 else num / den
            cur = np.empty(k)
            cur[: k - 1] = prev - phi_kk * prev[::-1]
            cur[k - 1] = phi_kk
        pacf_[k] = phi_kk
        prev = cur
    return pacf_


def _pacf_levinson(x, nlags, adjusted):
    avf = acovf(x, adjusted=adjusted, demean=True, fft=False)
    return _levinson_durbin_pacf(avf[: nlags + 1] / avf[0], nlags)


def _lagmat_sep(x, maxlag, trim):
    """`lagmat(x, maxlag, original='sep')`: the lags and the leads."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    nobs = x.shape[0]
    lm = np.zeros((nobs + maxlag, maxlag + 1))
    for k in range(maxlag + 1):
        lm[maxlag - k : nobs + maxlag - k, maxlag - k] = x[:, 0]
    if trim == "both":
        start, stop = maxlag, nobs
    else:  # "forward"
        start, stop = 0, nobs
    lags = lm[start:stop, 1:]
    leads = lm[start:stop, :1]
    return lags, leads


def pacf_ols(x, nlags, efficient=True, adjusted=False):
    """`statsmodels.tsa.stattools.pacf_ols`."""
    x = np.asarray(x, dtype=float).ravel()
    nobs = x.shape[0]
    pacf_ = np.empty(nlags + 1)
    pacf_[0] = 1.0
    if efficient:
        # The efficient variant keeps the full sample for each lag order and
        # carries an explicit constant; the inefficient one demeans instead
        # and trims every regression to the shortest common sample.
        xlags, x0 = _lagmat_sep(x, nlags, trim="forward")
        xlags = np.hstack([np.ones((xlags.shape[0], 1)), xlags])
        for k in range(1, nlags + 1):
            params = np.linalg.lstsq(xlags[k:, : k + 1], x0[k:], rcond=None)[0]
            pacf_[k] = np.squeeze(params[-1])
    else:
        xc = x - np.mean(x)
        xlags, x0 = _lagmat_sep(xc, nlags, trim="both")
        for k in range(1, nlags + 1):
            params = np.linalg.lstsq(xlags[:, :k], x0, rcond=None)[0]
            pacf_[k] = np.squeeze(params[-1])
    if adjusted:
        pacf_ = pacf_ * (nobs / (nobs - np.arange(nlags + 1)))
    return pacf_


def pacf_burg(x, nlags=None, demean=True):
    """`statsmodels.tsa.stattools.pacf_burg`, returning only the pacf."""
    x = np.asarray(x, dtype=float).ravel()
    if demean:
        x = x - x.mean()
    nobs = x.shape[0]
    p = nlags if nlags is not None else min(int(10 * np.log10(nobs)), nobs - 1)
    p = max(p, 1)
    if p > nobs - 1:
        raise ValueError("nlags must be smaller than nobs - 1")
    d = np.zeros(p + 1)
    d[0] = 2 * x.dot(x)
    pacf_ = np.zeros(p + 1)
    u = x[::-1].copy()
    v = x[::-1].copy()
    d[1] = u[:-1].dot(u[:-1]) + v[1:].dot(v[1:])
    pacf_[1] = 2 / d[1] * v[1:].dot(u[:-1])
    last_u = np.empty_like(u)
    last_v = np.empty_like(v)
    for i in range(1, p):
        last_u[:] = u
        last_v[:] = v
        u[1:] = last_u[:-1] - pacf_[i] * last_v[1:]
        v[1:] = last_v[1:] - pacf_[i] * last_u[:-1]
        d[i + 1] = (1 - pacf_[i] ** 2) * d[i] - v[i] ** 2 - u[-1] ** 2
        pacf_[i + 1] = 2 / d[i + 1] * v[i + 1 :].dot(u[i:-1])
    pacf_[0] = 1
    return pacf_


_YW_ADJUSTED = ("yw", "ywa", "ywadjusted", "yw_adjusted")
_YW_MLE = ("ywm", "ywmle", "yw_mle")
_LD_ADJUSTED = ("ld", "lda", "ldadjusted", "ld_adjusted")
_LD_BIASED = ("ldb", "ldbiased", "ld_biased")
_OLS = ("ols", "ols-inefficient", "ols-adjusted")
VALID_PACF_METHODS = _OLS + _YW_ADJUSTED + _YW_MLE + _LD_ADJUSTED + _LD_BIASED + ("burg",)


def pacf(x, nlags=None, method="ywadjusted", alpha=None):
    """Partial autocorrelation, matching `statsmodels.tsa.stattools.pacf`.

    `pmdarima` keeps two deprecation shims in front of this - the `*unbiased`
    spellings and the `ydu`/`ywu`/`ldu` abbreviations - and code written
    against it still passes them, so they are honoured here with the same
    warning.
    """
    if isinstance(method, str) and "unbiased" in method:
        warnings.warn(
            "The `*unbiased` methods have been deprecated in "
            "statsmodels >= 0.13.0. Please use `*adjusted` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = method.replace("unbiased", "adjusted")
    elif method in ("ydu", "ywu", "ldu"):
        warnings.warn(
            "The `ydu`, `ywu`, and `ldu` methods have been deprecated in "
            "statsmodels >= 0.13.0. Please use `yda`, `ywa`, and `lda` "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = method.replace("u", "a")

    if method not in VALID_PACF_METHODS:
        raise ValueError(
            f"Unknown value for method. Only {VALID_PACF_METHODS} are supported"
        )

    x = np.asarray(x, dtype=float).ravel()
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    nlags = max(nlags, 1)
    if nlags > nobs // 2:
        raise ValueError(
            "Can only compute partial correlations for lags up to 50% of the "
            f"sample size. The requested nlags {nlags} must be < "
            f"{nobs // 2}."
        )

    if method in _OLS:
        ret = pacf_ols(
            x,
            nlags=nlags,
            efficient="inefficient" not in method,
            adjusted="adjusted" in method,
        )
    elif method in _YW_ADJUSTED or method in _LD_ADJUSTED:
        ret = _pacf_levinson(x, nlags, adjusted=True)
    elif method in _YW_MLE or method in _LD_BIASED:
        ret = _pacf_levinson(x, nlags, adjusted=False)
    else:  # burg
        ret = pacf_burg(x, nlags=nlags, demean=True)

    if alpha is not None:
        interval = _norm_ppf(1 - alpha / 2.0) * np.sqrt(1.0 / nobs)
        confint = np.array(list(zip(ret - interval, ret + interval)))
        confint[0] = ret[0]
        return ret, confint
    return ret
