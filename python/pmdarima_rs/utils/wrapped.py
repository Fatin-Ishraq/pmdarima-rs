"""Autocorrelation and partial autocorrelation.

`pmdarima` re-exports these from `statsmodels`. Implementing them here is a
few dozen lines and removes the last reason for this package to import
statsmodels at all; the definitions and defaults follow statsmodels exactly,
and the test suite compares against it.
"""

import numpy as np

__all__ = ["acf", "pacf"]


def _default_nlags(nobs):
    """statsmodels' default: `min(10 * log10(nobs), nobs - 1)`."""
    return int(min(10 * np.log10(nobs), nobs - 1))


def _demeaned(x):
    x = np.asarray(x, dtype=float).ravel()
    return x - x.mean(), x.shape[0]


def acovf(x, adjusted=False, demean=True, fft=True, nlag=None):
    """Autocovariance function."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    xo = x - x.mean() if demean else x

    if fft:
        # Zero-pad to the next power of two so the circular convolution the
        # FFT computes equals the linear one we want.
        nobs2 = 2 ** int(np.ceil(np.log2(2 * n + 1)))
        f = np.fft.rfft(xo, n=nobs2)
        acov = np.fft.irfft(f * np.conjugate(f))[:n] / n
        acov = acov.real
    else:
        acov = np.array([np.dot(xo[k:], xo[: n - k]) / n for k in range(n)])

    if adjusted:
        d = n - np.arange(n)
        acov = acov * n / d

    if nlag is not None:
        return acov[: nlag + 1]
    return acov


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
    if missing == "drop":
        x = x[~np.isnan(x)]
    nobs = x.shape[0]
    if nlags is None:
        nlags = _default_nlags(nobs)
    if fft is None:
        fft = True

    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft)
    ret = avf[: nlags + 1] / avf[0]

    out = [ret]
    if alpha is not None:
        varacf = np.ones_like(ret) / nobs
        varacf[0] = 0
        if len(varacf) > 1:
            varacf[1] = 1.0 / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(ret[1:-1] ** 2)
        interval = _norm_ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(list(zip(ret - interval, ret + interval)))
        out.append(confint)
    if qstat:
        qs, pv = q_stat(ret[1:], nobs)
        out.extend([qs, pv])
    return out[0] if len(out) == 1 else tuple(out)


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


def _norm_ppf(q):
    from scipy.special import ndtri

    return float(ndtri(q))


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


def pacf(x, nlags=None, method="ywadjusted", alpha=None):
    """Partial autocorrelation, matching `statsmodels.tsa.stattools.pacf`."""
    x = np.asarray(x, dtype=float).ravel()
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)

    if method in ("ywadjusted", "yw", "ywunbiased", "ywm", "ywmle", "ywadj"):
        adjusted = method in ("ywadjusted", "ywunbiased", "ywadj")
        avf = acovf(x, adjusted=adjusted, demean=True, fft=False)
        r = avf[: nlags + 1] / avf[0]
        ret = _levinson_durbin_pacf(r, nlags)
    elif method in ("ols", "ols-adjusted", "ols-inefficient"):
        ret = _pacf_ols(x, nlags)
    else:
        raise ValueError(f"unrecognised pacf method {method!r}")

    if alpha is not None:
        varacf = 1.0 / nobs
        interval = _norm_ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(list(zip(ret - interval, ret + interval)))
        confint[0] = ret[0]
        return ret, confint
    return ret


def _pacf_ols(x, nlags):
    xo = x - x.mean()
    n = xo.shape[0]
    ret = np.empty(nlags + 1)
    ret[0] = 1.0
    for k in range(1, nlags + 1):
        y = xo[k:]
        design = np.column_stack([xo[k - j : n - j] for j in range(1, k + 1)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        ret[k] = beta[-1]
    return ret
