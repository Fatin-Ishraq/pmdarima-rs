"""Annual Canadian lynx trappings, 1821-1934.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_lynx

__all__ = ["DTYPE", "load_lynx"]
