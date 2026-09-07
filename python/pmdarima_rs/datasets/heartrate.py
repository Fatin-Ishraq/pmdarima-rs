"""Instantaneous heart rate, sampled every half second.

`pmdarima` gives every dataset its own module as well as exporting the
loader from the package, and code imports it both ways.
"""

from . import DTYPE, load_heartrate

__all__ = ["DTYPE", "load_heartrate"]
