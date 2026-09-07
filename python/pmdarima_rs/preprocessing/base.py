"""Transformer base classes, matching `pmdarima.preprocessing.base`."""

import abc

import numpy as np
import pandas as pd

from ..base import BaseEstimator, TransformerMixin
from ..utils.array import check_endog, check_exog

__all__ = ["BaseTransformer", "UpdatableMixin"]


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    """Transforms `(y, X)` jointly and returns both.

    The two-argument signature is the whole point: an endogenous transform
    such as Box-Cox has to hand `X` back untouched so the pair can be piped
    onward, and an exogenous featuriser has to hand `y` back for the same
    reason.
    """

    @staticmethod
    def _check_y_X(y, X):
        if y is not None:
            y = check_endog(
                y, copy=True, force_all_finite=False, preserve_series=False
            )
        if X is not None:
            X = check_exog(X, dtype=None, copy=True, force_all_finite=False)
        return y, X

    def fit_transform(self, y, X=None, **kwargs):
        self.fit(y, X)
        return self.transform(y, X, **kwargs)

    @abc.abstractmethod
    def fit(self, y, X):
        """Learn whatever the transform needs from the data."""

    @abc.abstractmethod
    def transform(self, y, X, **kwargs):
        """Apply the transform, returning `(y, X)`."""

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._param_names}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        from ..base import repr_with_defaults

        return repr_with_defaults(self, self._param_names)


class UpdatableMixin:
    def _check_endog(self, y):
        if y is None:
            raise ValueError("endog array cannot be None when updating")

    def update_and_transform(self, y, X=None, **kwargs):  # pragma: no cover
        raise NotImplementedError


def check_is_fitted(estimator, attr):
    if not hasattr(estimator, attr):
        raise ValueError(
            f"This {type(estimator).__name__} instance is not fitted yet. Call "
            "'fit' with appropriate arguments before using this estimator."
        )


def safe_hstack(X, features, names):
    """Concatenate features onto `X`, preserving a DataFrame when given one."""
    if X is None or isinstance(X, pd.DataFrame):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame.from_records(features)
        features.columns = names
        if X is not None:
            X = X.copy()
            X.index = features.index = np.arange(X.shape[0])
            return pd.concat([X, features], axis=1)
        return features
    return np.hstack([X, features])
