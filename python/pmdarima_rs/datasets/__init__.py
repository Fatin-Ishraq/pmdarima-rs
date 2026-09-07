"""The example datasets `pmdarima` ships, vendored so this package stands alone.

Same names, same values, same `as_series` / `dtype` arguments. The data lives
in a single compressed archive next to this module and is loaded lazily, so
importing the package does not read it.
"""

import os

import numpy as np
import pandas as pd

__all__ = [
    "DTYPE",
    "load_airpassengers",
    "load_ausbeer",
    "load_austres",
    "load_gasoline",
    "load_heartrate",
    "load_lynx",
    "load_msft",
    "load_sunspots",
    "load_taylor",
    "load_wineind",
    "load_woolyrnq",
]

DTYPE = np.float64

_ARCHIVE = os.path.join(os.path.dirname(__file__), "data", "datasets.npz")
_CACHE = {}


def _raw(name):
    if not _CACHE:
        with np.load(_ARCHIVE, allow_pickle=True) as z:
            for k in z.files:
                _CACHE[k] = z[k]
    return _CACHE[name]


def _load(name, as_series, dtype):
    arr = np.asarray(_raw(name), dtype=dtype)
    if as_series:
        return pd.Series(arr)
    return arr


def load_airpassengers(as_series=False, dtype=DTYPE):
    """Monthly totals of international airline passengers, 1949-1960."""
    return _load("airpassengers", as_series, dtype)


def load_ausbeer(as_series=False, dtype=DTYPE):
    """Quarterly Australian beer production. Contains a trailing NaN."""
    return _load("ausbeer", as_series, dtype)


def load_austres(as_series=False, dtype=DTYPE):
    """Quarterly Australian resident population."""
    return _load("austres", as_series, dtype)


def load_gasoline(as_series=False, dtype=DTYPE):
    """Weekly US finished motor gasoline product supplied."""
    return _load("gasoline", as_series, dtype)


def load_heartrate(as_series=False, dtype=DTYPE):
    """Instantaneous heart rate, sampled every half second."""
    return _load("heartrate", as_series, dtype)


def load_lynx(as_series=False, dtype=DTYPE):
    """Annual Canadian lynx trappings, 1821-1934."""
    return _load("lynx", as_series, dtype)


def load_sunspots(as_series=False, dtype=DTYPE):
    """Monthly mean relative sunspot numbers, 1749-1983."""
    return _load("sunspots", as_series, dtype)


def load_taylor(as_series=False, dtype=DTYPE):
    """Half-hourly electricity demand in England and Wales."""
    return _load("taylor", as_series, dtype)


def load_wineind(as_series=False, dtype=DTYPE):
    """Monthly Australian wine sales, bottles under 1 litre."""
    return _load("wineind", as_series, dtype)


def load_woolyrnq(as_series=False, dtype=DTYPE):
    """Quarterly production of woollen yarn in Australia."""
    return _load("woolyrnq", as_series, dtype)


def load_msft():
    """Daily Microsoft stock prices, as a `DataFrame`."""
    cols = [str(c) for c in _raw("msft__cols")]
    dates = [str(v) for v in _raw("msft__date")]
    vals = np.asarray(_raw("msft__vals"), dtype=float)
    out = pd.DataFrame({"Date": dates})
    for i, c in enumerate([c for c in cols if c != "Date"]):
        col = vals[:, i]
        # Volume and OpenInt are counts in the original, not prices.
        out[c] = col.astype(np.int64) if c in ("Volume", "OpenInt") else col
    return out[cols]


# `pmdarima` gives every dataset a module as well as a loader, and both are
# reachable as attributes of this package. These sit at the bottom because
# each one imports the loader defined above.
from . import (  # noqa: E402,F401
    airpassengers,
    ausbeer,
    austres,
    gasoline,
    heartrate,
    lynx,
    stocks,
    sunspots,
    taylor,
    wineind,
    woolyrnq,
)
