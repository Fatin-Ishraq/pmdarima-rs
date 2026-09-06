"""Delegation helpers, matching `pmdarima.utils.metaestimators`."""

from functools import update_wrapper

__all__ = ["if_has_delegate"]


class _IffHasDelegate:
    """Expose a method only when the delegated-to attribute exists."""

    def __init__(self, fn, delegate_names):
        self.fn = fn
        self.delegate_names = delegate_names
        update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        if obj is not None:
            for name in self.delegate_names:
                try:
                    getattr(obj, name)
                except AttributeError:
                    continue
                else:
                    break
            else:
                raise AttributeError(
                    f"{type(obj).__name__} has none of {self.delegate_names}"
                )

        def out(*args, **kwargs):
            return self.fn(obj, *args, **kwargs)

        update_wrapper(out, self.fn)
        return out


def if_has_delegate(delegate):
    """Decorator: only expose the method if `delegate` is present on `self`."""
    if isinstance(delegate, str):
        delegate = (delegate,)
    return lambda fn: _IffHasDelegate(fn, delegate)
