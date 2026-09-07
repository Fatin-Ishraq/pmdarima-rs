"""Half-hourly electricity demand in England and Wales.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_taylor

__all__ = ["DTYPE", "load_taylor"]
