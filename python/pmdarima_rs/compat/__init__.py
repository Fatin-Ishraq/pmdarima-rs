"""Compatibility shims.

Each submodule is deliberately free of imports from the rest of the package,
so it can be used from anywhere without a circular import.
"""

from .matplotlib import *  # noqa: F401,F403
from .numpy import *  # noqa: F401,F403
from .pandas import *  # noqa: F401,F403
from .sklearn import *  # noqa: F401,F403
from .statsmodels import *  # noqa: F401,F403


class MissingDataError(ValueError):
    """Raised when NaNs reach a routine that cannot represent them.

    `statsmodels` defines this and `pmdarima` propagates it, so callers catch
    it by name; it derives from `ValueError` there too.
    """


__all__ = [s for s in dir() if not s.startswith("_")]
