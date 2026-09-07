"""A transformer pipeline ending in an ARIMA.

A port of `pmdarima.pipeline` (MIT, Taylor G. Smith et al.). It differs from
sklearn's `Pipeline` in the way that matters for forecasting: transformers
carry `(y, X)` together, endogenous transforms are *inverted* on the way back
out of `predict`, and exogenous featurisers are told how many future periods
they must generate.
"""

import copy
import warnings
from itertools import islice

import pandas as pd

from .arima.arima import ARIMA
from .base import BaseARIMA
from .preprocessing.base import BaseTransformer, check_is_fitted
from .preprocessing.endog.base import BaseEndogTransformer
from .preprocessing.exog.base import BaseExogFeaturizer, BaseExogTransformer
from .utils.array import check_endog

__all__ = ["Pipeline"]


def _warn_for_deprecated(**kwargs):
    for k in ("typ",):
        if kwargs.pop(k, None):
            warnings.warn(
                f"'{k}' is deprecated and will be removed in a future release",
                DeprecationWarning,
            )
    return kwargs


def _clone(estimator):
    """A local `sklearn.base.clone`: rebuild from constructor parameters."""
    if hasattr(estimator, "get_params"):
        params = estimator.get_params(deep=False)
        return type(estimator)(**{k: copy.deepcopy(v) for k, v in params.items()})
    return copy.deepcopy(estimator)


class Pipeline(BaseARIMA):
    """A chain of transformers ending in an ARIMA."""

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError(f"Names provided are not unique: {list(names)!r}")
        # A step named after a constructor argument would make `step__param`
        # ambiguous, so scikit-learn rejects it and so does pmdarima.
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError(
                "Estimator names conflict with constructor arguments: "
                f"{sorted(invalid_names)!r}"
            )
        invalid = [name for name in names if "__" in name]
        if invalid:
            raise ValueError(f"Estimator names must not contain __: got {invalid!r}")

    def _validate_steps(self):
        names, estimators = zip(*self.steps)
        self._validate_names(names)
        for t in estimators[:-1]:
            if not isinstance(t, BaseTransformer):
                raise TypeError(
                    "All intermediate steps should be instances of "
                    f"BaseTransformer, but '{t}' (type {type(t)}) is not"
                )
        estimator = estimators[-1]
        if not isinstance(estimator, ARIMA) and not hasattr(estimator, "predict"):
            raise TypeError(
                "Last step of Pipeline should be an ARIMA. "
                f"'{estimator}' (type {type(estimator)}) isn't"
            )
        return list(self.steps)

    def _iter(self, with_final=True):
        stop = len(self.steps_)
        if not with_final:
            stop -= 1
        for idx, (name, trans) in enumerate(islice(self.steps_, 0, stop)):
            yield idx, name, trans

    def _get_kwargs(self, **params):
        params_steps = {name: {} for name, step in self.steps if step is not None}
        for pname, pval in params.items():
            step, param = pname.split("__", 1)
            params_steps[step][param] = pval
        return params_steps

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.named_steps[item]
        return self.steps[item][1]

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    def _check_n_periods(self, n_periods, exog):
        if exog is not None and len(exog) != n_periods:
            n_periods = len(exog)
        return n_periods

    def set_params(self, **params):
        if "steps" in params:
            self.steps = params.pop("steps")
        for pname, pval in params.items():
            step, param = pname.split("__", 1)
            self.named_steps[step].set_params(**{param: pval})
        return self

    def fit(self, y, X=None, **fit_kwargs):
        steps = self.steps_ = self._validate_steps()
        yt = check_endog(y, copy=False, preserve_series=True)
        Xt = X
        named_kwargs = self._get_kwargs(**fit_kwargs)
        self.n_samples_ = yt.shape[0]

        for step_idx, name, transformer in self._iter(with_final=False):
            cloned = _clone(transformer)
            yt, Xt = cloned.fit_transform(yt, Xt, **named_kwargs[name])
            steps[step_idx] = (name, cloned)

        self.x_feats_ = Xt.columns.tolist() if isinstance(Xt, pd.DataFrame) else None
        self._final_estimator.fit(yt, X=Xt, **named_kwargs[steps[-1][0]])
        return self

    def _pre_predict(self, n_periods, X, **kwargs):
        check_is_fitted(self, "steps_")
        Xt = X
        named_kwargs = self._get_kwargs(**kwargs)
        for _, name, transformer in self._iter(with_final=False):
            if isinstance(transformer, BaseExogTransformer):
                kw = named_kwargs[name]
                if isinstance(transformer, BaseExogFeaturizer):
                    num_p = kw.get("n_periods", None)
                    if num_p is not None and num_p != n_periods:
                        raise ValueError(
                            f"Manually set 'n_periods' kwarg for step '{name}' "
                            f"differs from forecasting n_periods ({num_p!r} != "
                            f"{n_periods!r})"
                        )
                    kw["n_periods"] = n_periods
                _, Xt = transformer.transform(None, Xt, **kw)
        if self.x_feats_ is not None:
            Xt = Xt[self.x_feats_]
        nm, est = self.steps_[-1]
        return Xt, est, named_kwargs[nm]

    def transform(self, n_periods=10, X=None, **kwargs):
        n_periods = self._check_n_periods(n_periods, X)
        kwargs = _warn_for_deprecated(**kwargs)
        Xt, _, _ = self._pre_predict(n_periods, X, **kwargs)
        return Xt

    def predict_in_sample(
        self,
        X=None,
        start=None,
        end=None,
        dynamic=False,
        return_conf_int=False,
        alpha=0.05,
        inverse_transform=True,
        **kwargs,
    ):
        kwargs = _warn_for_deprecated(**kwargs)
        Xt, est, predict_kwargs = self._pre_predict(0, X, **kwargs)
        return_vals = est.predict_in_sample(
            X=Xt,
            start=start,
            end=end,
            return_conf_int=return_conf_int,
            alpha=alpha,
            dynamic=dynamic,
            **predict_kwargs,
        )
        return self._post_predict(Xt, return_vals, return_conf_int, inverse_transform)

    def predict(
        self,
        n_periods=10,
        X=None,
        return_conf_int=False,
        alpha=0.05,
        inverse_transform=True,
        **kwargs,
    ):
        n_periods = self._check_n_periods(n_periods, X)
        kwargs = _warn_for_deprecated(**kwargs)
        Xt, est, predict_kwargs = self._pre_predict(n_periods, X, **kwargs)
        return_vals = est.predict(
            n_periods=n_periods,
            X=Xt,
            return_conf_int=return_conf_int,
            alpha=alpha,
            **predict_kwargs,
        )
        return self._post_predict(Xt, return_vals, return_conf_int, inverse_transform)

    def _post_predict(self, Xt, return_vals, return_conf_int, inverse_transform):
        if not inverse_transform:
            return return_vals
        y_pred = return_vals
        conf_ints = None
        if return_conf_int:
            y_pred, conf_ints = y_pred
        for _, transformer in self.steps_[::-1]:
            if isinstance(transformer, BaseEndogTransformer):
                y_pred, Xt = transformer.inverse_transform(y_pred, Xt)
                if return_conf_int:
                    conf_ints[:, 0], _ = transformer.inverse_transform(
                        conf_ints[:, 0], Xt
                    )
                    conf_ints[:, 1], _ = transformer.inverse_transform(
                        conf_ints[:, 1], Xt
                    )
        if return_conf_int:
            return y_pred, conf_ints
        return y_pred

    def summary(self):
        return self._final_estimator.summary()

    def update(self, y, X=None, maxiter=None, **kwargs):
        check_is_fitted(self, "steps_")
        yt, Xt = y, X
        named_kwargs = self._get_kwargs(**kwargs)
        for _, name, transformer in self._iter(with_final=False):
            kw = named_kwargs[name]
            if hasattr(transformer, "update_and_transform"):
                yt, Xt = transformer.update_and_transform(y=yt, X=Xt, **kw)
            else:
                yt, Xt = transformer.transform(yt, Xt, **kw)
        if self.x_feats_ is not None:
            Xt = Xt[self.x_feats_]
        nm, est = self.steps_[-1]
        return est.update(yt, X=Xt, maxiter=maxiter, **named_kwargs[nm])

