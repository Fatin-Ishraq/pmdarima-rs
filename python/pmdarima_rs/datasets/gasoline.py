"""Weekly US finished motor gasoline product supplied.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_gasoline

__all__ = ["DTYPE", "load_gasoline"]
