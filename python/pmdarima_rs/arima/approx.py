"""`approx` and `ARMAtoMA`, matching `pmdarima.arima`.

`approx` is R's linear interpolation; `ARMAtoMA` is R's psi-weight recursion,
which expands an ARMA model into the coefficients of its infinite moving
average representation.
"""

import numpy as np

__all__ = ["ARMAtoMA", "approx"]

VALID_APPROX = {"constant": 2, "linear": 1}


def approx(
    x,
    y,
    xout,
    method="linear",
    rule=1,
    f=0,
    yleft=None,
    yright=None,
    ties="mean",
):
    """R's `approx`: interpolate `y` at `xout`.

    `rule=1` returns NaN outside the range of `x`; `rule=2` clamps to the
    endpoint values. Duplicate `x` are collapsed by `ties` first, matching R.
    """
    if method not in VALID_APPROX:
        raise ValueError(f"method must be one of {set(VALID_APPROX)}")

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"array dim mismatch: {x.shape[0]} != {y.shape[0]}")

    order = np.argsort(x)
    x, y = x[order], y[order]
    ux = np.unique(x)
    if ux.shape[0] < x.shape[0] and ties != "ordered":
        agg = np.mean if ties == "mean" else (ties if callable(ties) else np.mean)
        y = np.array([agg(y[x == u]) for u in ux])
        x = ux

    if x.shape[0] == 1 and method == "linear":
        raise ValueError("need at least two points to linearly interpolate")

    xout = np.atleast_1d(np.asarray(xout, dtype=float))

    if yleft is None:
        yleft = y[0] if rule != 1 else np.nan
    if yright is None:
        yright = y[-1] if rule != 1 else np.nan

    if method == "linear":
        yout = np.interp(xout, x, y, left=yleft, right=yright)
    else:
        # Constant interpolation: `f` blends the left and right values.
        idx = np.searchsorted(x, xout, side="right") - 1
        yout = np.empty(xout.shape[0], dtype=float)
        for i, (xo, j) in enumerate(zip(xout, idx)):
            if xo < x[0]:
                yout[i] = yleft
            elif xo >= x[-1]:
                yout[i] = y[-1] if xo == x[-1] else yright
            else:
                yout[i] = (1 - f) * y[j] + f * y[j + 1]
    return xout, np.asarray(yout)


def ARMAtoMA(ar, ma, max_deg):
    """Psi-weights of the MA(inf) representation of an ARMA model.

    `psi_j = ma_j + sum_i ar_i * psi_{j-i}`, with `psi_0 = 1` implied and not
    returned - the same convention as R's `ARMAtoMA`.
    """
    ar = np.asarray(ar, dtype=float).ravel()
    ma = np.asarray(ma, dtype=float).ravel()
    p, q = ar.shape[0], ma.shape[0]
    max_deg = int(max_deg)
    psi = np.zeros(max_deg, dtype=float)
    for i in range(max_deg):
        tmp = ma[i] if i < q else 0.0
        for j in range(min(i + 1, p)):
            tmp += ar[j] * (psi[i - j - 1] if i - j - 1 >= 0 else 1.0)
        psi[i] = tmp
    return psi
