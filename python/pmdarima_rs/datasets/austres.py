"""Quarterly Australian resident population.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_austres

__all__ = ["DTYPE", "load_austres"]
