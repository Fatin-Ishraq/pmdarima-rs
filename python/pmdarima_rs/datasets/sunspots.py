"""Monthly mean relative sunspot numbers, 1749-1983.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_sunspots

__all__ = ["DTYPE", "load_sunspots"]
