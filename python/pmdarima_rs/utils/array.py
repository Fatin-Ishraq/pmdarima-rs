"""Array helpers, matching `pmdarima.utils.array`.

Small, dull, and load-bearing: `diff` and `diff_inv` define what differencing
*means* for this package, and `check_endog` decides what an acceptable series
is. Both are ports of the `pmdarima` originals (MIT, Taylor G. Smith et al.).
"""

import numpy as np
import pandas as pd

__all__ = [
    "as_series",
    "c",
    "check_endog",
    "check_exog",
    "diff",
    "diff_inv",
    "is_iterable",
]


def is_iterable(x):
    if isinstance(x, str):
        return False
    return hasattr(x, "__iter__")


def c(*args):
    """R's `c`: concatenate scalars and iterables into one 1-D array.

    Returns `None` for no arguments, matching `pmdarima`, which relies on it
    to distinguish "not supplied" from "empty".
    """
    if not args:
        return None
    if len(args) == 1 and is_iterable(args[0]):
        return np.asarray(args[0])
    return np.concatenate([a if is_iterable(a) else [a] for a in args])


def as_series(x):
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("cannot convert a multi-column DataFrame to a Series")
        return x[x.columns[0]]
    return pd.Series(np.asarray(x).ravel())


def check_endog(y, dtype=np.float64, copy=True, force_all_finite=False, preserve_series=True):
    """Validate a series and return it as a 1-D array (or Series).

    `force_all_finite=False` is the default on purpose: the Kalman filter
    handles missing values by skipping the update, so NaNs are legitimate
    input, and rejecting them here would remove a capability `pmdarima` has.
    """
    is_pandas = isinstance(y, (pd.Series, pd.DataFrame))
    if preserve_series and is_pandas:
        out = y.copy() if copy else y
        if isinstance(out, pd.DataFrame):
            if out.shape[1] != 1:
                raise ValueError("y must be one-dimensional")
            out = out[out.columns[0]]
        arr = np.asarray(out, dtype=dtype)
        if arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if force_all_finite and not np.all(np.isfinite(arr)):
            raise ValueError("y contains non-finite values")
        if arr.shape[0] < 1:
            raise ValueError("y is empty")
        return pd.Series(arr, index=out.index, name=getattr(out, "name", None))

    arr = np.asarray(y, dtype=dtype)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if force_all_finite and not np.all(np.isfinite(arr)):
        raise ValueError("y contains non-finite values")
    if arr.shape[0] < 1:
        raise ValueError("y is empty")
    return arr.copy() if copy else arr


def check_exog(X, dtype=np.float64, copy=True, force_all_finite=True):
    """Validate exogenous regressors into a 2-D array or DataFrame."""
    if isinstance(X, pd.DataFrame):
        out = X.copy() if copy else X
        arr = np.asarray(out, dtype=dtype)
        if force_all_finite and not np.all(np.isfinite(arr)):
            raise ValueError("X contains non-finite values")
        return out
    arr = np.asarray(X, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if force_all_finite and not np.all(np.isfinite(arr)):
        raise ValueError("X contains non-finite values")
    return arr.copy() if copy else arr


def _diff_vector(x, lag, differences):
    for _ in range(differences):
        if x.shape[0] <= lag:
            return x[:0]
        x = x[lag:] - x[:-lag]
    return x


def diff(x, lag=1, differences=1):
    """Lagged differences, matching `pmdarima.utils.diff`."""
    if any(v < 1 for v in (lag, differences)):
        raise ValueError("lag and differences must be positive (> 0) integers")
    is_series = isinstance(x, pd.Series)
    name, index = (x.name, x.index) if is_series else (None, None)
    arr = np.asarray(x, dtype=np.float64)
    two_d = arr.ndim == 2
    out = _diff_vector(arr, lag, differences)
    if is_series:
        return pd.Series(out, index=index[len(index) - out.shape[0] :], name=name)
    if two_d:
        return out
    return out


def diff_inv(x, lag=1, differences=1, xi=None):
    """Invert :func:`diff`.

    `xi` supplies the initial values that differencing discarded; when it is
    omitted, zeros are used, exactly as `pmdarima` does.
    """
    if any(v < 1 for v in (lag, differences)):
        raise ValueError("lag and differences must be positive (> 0) integers")
    arr = np.asarray(x, dtype=np.float64)
    one_d = arr.ndim == 1
    if one_d:
        arr = arr.reshape(-1, 1)
    n, k = arr.shape
    n_xi = lag * differences
    if xi is None:
        xi_arr = np.zeros((n_xi, k))
    else:
        xi_arr = np.asarray(xi, dtype=np.float64)
        if xi_arr.ndim == 1:
            xi_arr = xi_arr.reshape(-1, 1)
        if xi_arr.shape[0] != n_xi:
            raise IndexError(f"xi must have {n_xi} rows, got {xi_arr.shape[0]}")

    out = arr
    for d in range(differences):
        head = xi_arr[n_xi - lag * (d + 1) : n_xi - lag * d]
        cur = np.zeros((out.shape[0] + lag, k))
        cur[:lag] = head
        for i in range(lag, cur.shape[0]):
            cur[i] = out[i - lag] + cur[i - lag]
        out = cur
    return out.ravel() if one_d else out
