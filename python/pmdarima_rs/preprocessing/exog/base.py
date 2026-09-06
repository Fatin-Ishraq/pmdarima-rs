"""Base classes for exogenous featurisers."""

import abc

from ..base import BaseTransformer, safe_hstack

__all__ = ["BaseExogTransformer", "BaseExogFeaturizer"]


class BaseExogTransformer(BaseTransformer, metaclass=abc.ABCMeta):
    def _check_y_X(self, y, X, null_allowed=False):
        y, X = super()._check_y_X(y, X)
        if X is None and not null_allowed:
            raise ValueError("X must be non-None for exog transformers")
        return y, X


class BaseExogFeaturizer(BaseExogTransformer, metaclass=abc.ABCMeta):
    def __init__(self, prefix=None):
        self.prefix = prefix

    @abc.abstractmethod
    def _get_prefix(self):
        """The column-name prefix this featuriser stamps on its output."""

    def _get_feature_names(self, X):
        pfx = self._get_prefix()
        return [f"{pfx}_{i}" for i in range(X.shape[1])]

    def _safe_hstack(self, X, features):
        import pandas as pd

        if X is None or isinstance(X, pd.DataFrame):
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame.from_records(features)
            names = self._get_feature_names(features)
            return safe_hstack(X, features, names)
        return safe_hstack(X, features, None)

    @abc.abstractmethod
    def transform(self, y, X=None, n_periods=0, **kwargs):
        """Append the generated features to `X`."""
