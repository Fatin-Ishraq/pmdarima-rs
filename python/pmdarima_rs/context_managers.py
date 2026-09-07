"""Small context managers, matching `pmdarima.context_managers`."""

import contextlib

__all__ = ["except_and_reraise"]


@contextlib.contextmanager
def except_and_reraise(*except_errs, raise_err=None, raise_msg=None):
    """Re-raise one exception type as another, with a clearer message.

    `raise_err` and `raise_msg` are keyword-only in practice: `pmdarima`
    rejects positional use explicitly rather than letting them be swallowed
    into `*except_errs`.
    """
    if raise_err is None:
        raise TypeError("raise_err must be used as a key-word arg")
    if raise_msg is None:
        raise TypeError("raise_msg must be used as a key-word arg")
    try:
        yield
    except except_errs as e:
        message = "%s (raised from %s: %s)" % (
            raise_msg,
            e.__class__.__name__,
            str(e),
        )
        raise raise_err(message)
