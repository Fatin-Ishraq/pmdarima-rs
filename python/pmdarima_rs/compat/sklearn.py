"""Estimator-protocol helpers, matching `pmdarima.compat.sklearn`.

`pmdarima` gets these from scikit-learn. This package does not depend on
scikit-learn, so they are written out; `NotFittedError` is taken from sklearn
when it happens to be installed, so that code catching *its* exception class
keeps working, and otherwise a local class with the same bases is used.
"""

__all__ = [
    "NotFittedError",
    "check_is_fitted",
    "if_delegate_has_method",
    "safe_indexing",
]

try:  # pragma: no cover - depends on the environment
    from sklearn.exceptions import NotFittedError
except ImportError:  # pragma: no cover

    class NotFittedError(ValueError, AttributeError):
        """Raised when an estimator is used before it has been fit.

        The bases match scikit-learn's, so `except ValueError` and
        `except AttributeError` both catch it wherever it is raised.
        """


def check_is_fitted(estimator, attributes):
    """Raise `NotFittedError` unless one of `attributes` is present."""
    if isinstance(attributes, str):
        attributes = [attributes]
    if not hasattr(attributes, "__iter__"):
        raise TypeError("attributes must be a string or iterable")
    for attr in attributes:
        if hasattr(estimator, attr):
            return
    raise NotFittedError("Model has not been fit!")


def safe_indexing(X, indices):
    """Slice an array or dataframe along its first axis."""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    if hasattr(X, "ndim") and X.ndim == 2:
        return X[indices, :]
    return X[indices]


def if_delegate_has_method(attr):
    """Expose a method only when the delegated-to attribute exists."""
    from ..utils.metaestimators import if_has_delegate

    return if_has_delegate(attr)
