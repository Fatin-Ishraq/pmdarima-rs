"""Matplotlib import shims, matching `pmdarima.compat.matplotlib`.

`matplotlib` stays optional here: it is imported inside the plotting helpers
rather than at module scope, so importing this package never requires it.
"""

import os
import sys

__all__ = ["get_compatible_pyplot", "mpl_hist_arg"]


def get_compatible_pyplot(backend=None, debug=True):
    """Import `matplotlib.pyplot`, optionally forcing a backend."""
    import matplotlib

    if backend is not None:
        matplotlib.use(backend)
    elif sys.platform == "darwin" or sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and sys.platform != "darwin":
            matplotlib.use("Agg")

    from matplotlib import pyplot as plt

    if debug:
        print(f"Using matplotlib backend {matplotlib.get_backend()}")
    return plt


def mpl_hist_arg(value=True):
    """The keyword that asks matplotlib's `hist` for a density."""
    return {"density": value}
