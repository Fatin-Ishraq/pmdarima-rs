"""Context managers that tune the stepwise search, matching `pmdarima`.

A port of `pmdarima.arima._context` (MIT, Taylor G. Smith et al.). The store
is a stack per context type, so nested `with` blocks behave the way callers
expect and the innermost one wins.
"""

from abc import ABC, abstractmethod
from enum import Enum

__all__ = ["AbstractContext", "ContextStore", "ContextType", "StepwiseContext"]


class _CtxSingleton:
    store = {}


_ctx = _CtxSingleton()


class ContextType(Enum):
    EMPTY = 0
    STEPWISE = 1


class AbstractContext(ABC):
    def __init__(self, **kwargs):
        self.props = {k: v for k, v in kwargs.items() if v is not None} if kwargs else {}

    def __enter__(self):
        ContextStore._add_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ContextStore._remove_context(self)

    def __getattr__(self, item):
        # Only reached for attributes not found normally, so `props` itself is
        # never routed here.
        return self.props.get(item)

    def __contains__(self, item):
        return item in self.props

    def __getitem__(self, item):
        return self.props.get(item)

    def __iter__(self):
        return iter(self.props)

    def keys(self):
        return self.props.keys()

    def values(self):
        return self.props.values()

    def items(self):
        return self.props.items()

    def update(self, other):
        parent_props = dict(other)
        parent_props.update(self.props)
        self.props = parent_props

    def __repr__(self):
        return repr(self.props)

    @abstractmethod
    def get_type(self):
        """The context type this instance registers under."""


class _emptyContext(AbstractContext):
    def get_type(self):
        return ContextType.EMPTY


class StepwiseContext(AbstractContext):
    """Bound the stepwise search by number of steps or wall time."""

    def __init__(self, max_steps=None, max_dur=None):
        if max_steps is not None and not 0 < max_steps <= 1000:
            raise ValueError("max_steps must be between 1 and 1000")
        if max_dur is not None and max_dur <= 0:
            raise ValueError("max_dur must be positive")
        super().__init__(max_steps=max_steps, max_dur=max_dur)

    def get_type(self):
        return ContextType.STEPWISE


class ContextStore:
    @staticmethod
    def get_context(context_type):
        if not isinstance(context_type, ContextType):
            raise ValueError("context_type must be an instance of ContextType")
        stack = _ctx.store.get(context_type)
        return stack[-1] if stack else None

    @staticmethod
    def get_or_default(context_type, default):
        ctx = ContextStore.get_context(context_type)
        return ctx if ctx else default

    @staticmethod
    def get_or_empty(context_type):
        return ContextStore.get_or_default(context_type, _emptyContext())

    @staticmethod
    def _add_context(ctx):
        if not isinstance(ctx, AbstractContext):
            raise ValueError("ctx must be an instance of AbstractContext")
        ct = ctx.get_type()
        stack = _ctx.store.setdefault(ct, [])
        if stack:
            ctx.update(stack[-1])
        stack.append(ctx)

    @staticmethod
    def _remove_context(ctx):
        stack = _ctx.store.get(ctx.get_type())
        if stack:
            stack.pop()
