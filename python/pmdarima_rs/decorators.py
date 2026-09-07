"""Decorators, matching `pmdarima.decorators`."""

import functools
import warnings

__all__ = ["deprecated"]


def deprecated(use_instead, notes=None):
    """Mark a function as deprecated, naming its replacement."""
    notes = "" if notes is None else " " + notes

    def wrapped_func(func):
        @functools.wraps(func)
        def _inner(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)  # un-filter
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in a "
                f"future release of pmdarima. Use {use_instead} instead."
                f"{notes}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", DeprecationWarning)  # re-filter
            return func(*args, **kwargs)

        return _inner

    return wrapped_func
