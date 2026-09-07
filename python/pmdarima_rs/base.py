"""The estimator interface every forecaster in this package implements.

`pmdarima` defines this on top of scikit-learn's `BaseEstimator`. This package
does not depend on scikit-learn, so the pieces `clone`, `get_params` and
`GridSearchCV` actually need - a keyword-only constructor whose arguments are
readable back off the instance - are provided directly.
"""

import abc
import copy

from abc import ABCMeta  # noqa: F401  (re-exported, as pmdarima does)

__all__ = ["BaseARIMA", "BaseEstimator", "TransformerMixin"]


def repr_with_defaults(obj, names):
    """`Name(a=1)`, showing only what differs from the constructor defaults.

    scikit-learn's repr does this, and `pmdarima`'s estimators inherit it, so
    reprs printed in traces and notebooks match rather than listing every
    argument.
    """
    import inspect

    try:
        defaults = {
            name: param.default
            for name, param in inspect.signature(type(obj).__init__).parameters.items()
        }
    except (TypeError, ValueError):  # pragma: no cover
        defaults = {}
    parts = []
    for name in names:
        value = getattr(obj, name)
        default = defaults.get(name, inspect.Parameter.empty)
        if default is not inspect.Parameter.empty:
            try:
                if value is default or value == default:
                    continue
            except ValueError:  # pragma: no cover - array-valued params
                pass
        parts.append(f"{name}={value!r}")
    return f"{type(obj).__name__}({', '.join(parts)})"


class _LocalBaseEstimator:
    """A minimal stand-in for `sklearn.base.BaseEstimator`.

    `get_params`/`set_params` follow scikit-learn's contract, including the
    `step__param` nesting, so `sklearn.base.clone` and the model-selection
    tools work on these estimators without scikit-learn being a dependency.
    """

    @classmethod
    def _param_names(cls):
        import inspect

        init = cls.__init__
        if init is object.__init__:
            return []
        sig = inspect.signature(init)
        return sorted(
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind is not p.VAR_KEYWORD
        )

    def get_params(self, deep=True):
        out = {}
        for key in self._param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update(
                    (f"{key}__{k}", v) for k, v in value.get_params().items()
                )
            out[key] = value
        return out

    def set_params(self, **params):
        if not params:
            return self
        valid = self.get_params(deep=True)
        nested = {}
        for key, value in params.items():
            head, _, sub = key.partition("__")
            if head not in valid:
                raise ValueError(
                    f"Invalid parameter {head!r} for estimator "
                    f"{type(self).__name__}."
                )
            if sub:
                nested.setdefault(head, {})[sub] = value
            else:
                setattr(self, head, value)
        for head, sub_params in nested.items():
            getattr(self, head).set_params(**sub_params)
        return self

    def __repr__(self):
        return repr_with_defaults(self, sorted(self.get_params(deep=False)))

    def __sklearn_clone__(self):  # pragma: no cover - used only with sklearn
        params = self.get_params(deep=False)
        return type(self)(**{k: copy.deepcopy(v) for k, v in params.items()})


try:  # pragma: no cover - depends on the environment
    # `pmdarima` builds on scikit-learn, so its estimators carry scikit-learn's
    # plumbing - `set_output`, metadata routing, the `_repr_html_` used in
    # notebooks. Inheriting the real bases when scikit-learn happens to be
    # installed hands all of that back, and the local fallback keeps
    # scikit-learn from becoming a dependency of this package.
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # pragma: no cover
    BaseEstimator = _LocalBaseEstimator

    class TransformerMixin:
        """A no-op stand-in for `sklearn.base.TransformerMixin`."""


class BaseARIMA(BaseEstimator, metaclass=abc.ABCMeta):
    """The forecasting interface shared by `ARIMA`, `AutoARIMA` and `Pipeline`."""

    @abc.abstractmethod
    def fit(self, y, X, **fit_args):
        """Fit the estimator to a series."""

    def fit_predict(self, y, X=None, n_periods=10, **fit_args):
        """Fit, then forecast `n_periods` beyond the training data."""
        self.fit(y, X, **fit_args)
        return self.predict(n_periods=n_periods, X=X)

    @abc.abstractmethod
    def predict(self, n_periods, X, return_conf_int=False, alpha=0.05, **kwargs):
        """Forecast beyond the end of the training data."""

    @abc.abstractmethod
    def predict_in_sample(self, X, start, end, dynamic, **kwargs):
        """Predict the training period."""

    @abc.abstractmethod
    def update(self, y, X=None, maxiter=None, **kwargs):
        """Add new observations and refit."""
