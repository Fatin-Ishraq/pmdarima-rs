"""Order-of-differencing selection: `ndiffs` and `nsdiffs`.

Ports of the `pmdarima` routines (MIT, Taylor G. Smith et al.). They wrap the
unit-root tests in the same iterate-until-stationary loop, including the same
early exits for constant and too-short series.
"""

import warnings

import numpy as np

from ..utils.array import check_endog, diff
from . import seasonality as seatest_lib
from . import stationarity as statest_lib

__all__ = ["is_constant", "ndiffs", "nsdiffs"]

VALID_TESTS = {
    "kpss": statest_lib.KPSSTest,
    "adf": statest_lib.ADFTest,
    "pp": statest_lib.PPTest,
}

VALID_STESTS = {
    "ocsb": seatest_lib.OCSBTest,
    "ch": seatest_lib.CHTest,
}


def _get_callable(key, mapping):
    if callable(key):
        return key
    try:
        return mapping[key]
    except KeyError:
        raise ValueError(f"key must be one of {set(mapping)}, but got {key!r}") from None


def is_constant(x):
    x = np.asarray(x).ravel()
    return bool((x == x[0]).all())


def ndiffs(x, alpha=0.05, test="kpss", max_d=2, **kwargs):
    """Number of non-seasonal differences needed to make `x` stationary."""
    if max_d <= 0:
        raise ValueError("max_d must be a positive integer")
    testfunc = _get_callable(test, VALID_TESTS)(alpha, **kwargs).should_diff
    x = check_endog(x, copy=False, preserve_series=False)

    d = 0
    if is_constant(x):
        return d

    try:
        pval, dodiff = testfunc(x)
        if np.isnan(pval):
            return 0
        while dodiff and d < max_d:
            d += 1
            x = diff(x)
            if is_constant(x):
                return d
            pval, dodiff = testfunc(x)
            if np.isnan(pval):
                return d - 1
    except np.linalg.LinAlgError as err:
        raise ValueError(
            f"Encountered exception in stationarity test ({test!r}). This can "
            "occur in seasonal settings when a large enough `m` coupled with a "
            "large enough `D` difference the training array into too few "
            f"samples for OLS (input contains {len(x)} samples). Try fitting "
            "on a larger training size"
        ) from err
    return d


def nsdiffs(x, m, max_D=2, test="ocsb", **kwargs):
    """Number of seasonal differences needed to make `x` stationary."""
    if max_D <= 0:
        raise ValueError("max_D must be a positive integer")
    testfunc = _get_callable(test, VALID_STESTS)(
        m, **kwargs
    ).estimate_seasonal_differencing_term
    x = check_endog(x, copy=False, preserve_series=False)

    if is_constant(x):
        return 0

    D = 0
    dodiff = testfunc(x)
    while dodiff == 1 and D < max_D:
        D += 1
        x = diff(x, lag=m)
        if is_constant(x):
            return D
        if len(x) < m:
            warnings.warn(
                "Appropriate D value may not have been reached; length of "
                f"seasonally-differenced array ({len(x)}) is shorter than m "
                f"({m}). Using D={D}"
            )
            return D
        dodiff = testfunc(x)
    return D
