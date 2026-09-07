"""Monthly Australian wine sales, bottles under 1 litre.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_wineind

__all__ = ["DTYPE", "load_wineind"]
