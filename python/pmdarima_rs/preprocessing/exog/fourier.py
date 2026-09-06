"""Fourier terms as exogenous regressors.

The standard way to give a SARIMAX a long seasonal period without paying for
a state vector of that length: `m = 365` as a seasonal order is unusable,
while `m = 365` as a handful of Fourier terms is cheap.
"""

import numpy as np

from ..base import UpdatableMixin, check_is_fitted
from .base import BaseExogFeaturizer

__all__ = ["FourierFeaturizer"]


def _fourier_terms(p, times):
    """Interleaved sine and cosine terms, one pair per frequency.

    `pmdarima` builds these in Cython; the column order it produces is
    `[sin(p_0), cos(p_0), sin(p_1), cos(p_1), ...]`, which its own feature
    names confirm by alternating "S" and "C".
    """
    p = np.asarray(p, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.empty((times.shape[0], 2 * p.shape[0]))
    for i, pi in enumerate(p):
        arg = 2.0 * np.pi * pi * times
        out[:, 2 * i] = np.sin(arg)
        out[:, 2 * i + 1] = np.cos(arg)
    return out


class FourierFeaturizer(BaseExogFeaturizer, UpdatableMixin):
    """Seasonal Fourier terms of period `m`, `k` pairs of them."""

    _param_names = ("m", "k", "prefix")

    def __init__(self, m, k=None, prefix=None):
        self.m = m
        self.k = k
        super().__init__(prefix)

    def _get_prefix(self):
        return "FOURIER" if self.prefix is None else self.prefix

    def _get_feature_names(self, X):
        pfx = self._get_prefix()
        return [
            f"{pfx}_{'S' if i % 2 == 0 else 'C'}{self.m}-{i // 2}"
            for i in range(X.shape[1])
        ]

    def fit(self, y, X=None):
        _, _ = self._check_y_X(y, X, null_allowed=True)
        m, k = self.m, self.k
        if k is None:
            k = m // 2
        if 2 * k > m or k < 1:
            raise ValueError("k must be a positive integer not greater than m//2")
        self.p_ = ((np.arange(k) + 1) / m).astype(np.float64)
        self.k_ = k
        self.n_ = y.shape[0]
        return self

    def transform(self, y, X=None, n_periods=0, **kwargs):
        check_is_fitted(self, "p_")
        _, X = self._check_y_X(y, X, null_allowed=True)
        if n_periods and X is not None and n_periods != X.shape[0]:
            raise ValueError(
                "If n_periods and X are specified, n_periods must match dims "
                f"of X ({n_periods} != {X.shape[0]})"
            )
        times = np.arange(self.n_ + n_periods, dtype=np.float64) + 1
        X_fourier = _fourier_terms(self.p_, times)
        if n_periods:
            X_fourier = X_fourier[-n_periods:, :]
        return y, self._safe_hstack(X, X_fourier)

    def update_and_transform(self, y, X=None, **kwargs):
        check_is_fitted(self, "p_")
        self._check_endog(y)
        _, Xt = self.transform(y, X, n_periods=len(y), **kwargs)
        self.n_ += len(y)
        return y, Xt
