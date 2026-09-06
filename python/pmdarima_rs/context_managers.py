"""Small context managers, matching `pmdarima.context_managers`."""

import contextlib

__all__ = ["except_and_reraise"]


@contextlib.contextmanager
def except_and_reraise(*exceptions, raise_err, raise_msg):
    """Re-raise one exception type as another, with a clearer message."""
    try:
        yield
    except exceptions as err:
        raise raise_err(raise_msg) from err
