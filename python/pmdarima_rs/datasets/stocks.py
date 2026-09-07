"""Daily Microsoft stock prices.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_msft

__all__ = ["DTYPE", "load_msft"]
