"""Pandas-facing shims, matching `pmdarima.compat.pandas`."""

import pandas as pd

__all__ = ["plotting"]

#: `pandas.plotting`, which moved several times across pandas versions.
plotting = pd.plotting
