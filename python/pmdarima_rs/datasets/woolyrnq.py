"""Quarterly production of woollen yarn in Australia.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_woolyrnq

__all__ = ["DTYPE", "load_woolyrnq"]
