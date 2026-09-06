"""Array, plotting and introspection helpers, matching `pmdarima.utils`."""

from .array import (
    as_series,
    c,
    check_endog,
    check_exog,
    diff,
    diff_inv,
    is_iterable,
)
from .metaestimators import if_has_delegate
from .visualization import (
    autocorr_plot,
    decomposed_plot,
    plot_acf,
    plot_pacf,
    tsdisplay,
)
from .wrapped import acf, pacf

__all__ = [
    "acf",
    "as_series",
    "autocorr_plot",
    "c",
    "check_endog",
    "check_exog",
    "decomposed_plot",
    "diff",
    "diff_inv",
    "get_callable",
    "if_has_delegate",
    "is_iterable",
    "pacf",
    "plot_acf",
    "plot_pacf",
    "tsdisplay",
]


def get_callable(key, dct):
    """Resolve `key` against `dct`, or pass a callable straight through."""
    if callable(key):
        return key
    if key in dct:
        return dct[key]
    raise ValueError(f"key must be one of {set(dct)}, but got {key!r}")
