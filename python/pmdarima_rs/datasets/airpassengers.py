"""Monthly totals of international airline passengers, 1949-1960.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_airpassengers

__all__ = ["DTYPE", "load_airpassengers"]
