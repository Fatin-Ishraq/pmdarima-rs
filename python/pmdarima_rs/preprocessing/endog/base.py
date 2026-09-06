"""Base class for endogenous transformers."""

import abc

from ..base import BaseTransformer

__all__ = ["BaseEndogTransformer"]


class BaseEndogTransformer(BaseTransformer, metaclass=abc.ABCMeta):
    def _check_y_X(self, y, X):
        y, X = super()._check_y_X(y, X)
        if y is None:
            raise ValueError("y must be non-None for endogenous transformers")
        return y, X

    @abc.abstractmethod
    def inverse_transform(self, y, X=None):
        """Undo :meth:`transform`."""
