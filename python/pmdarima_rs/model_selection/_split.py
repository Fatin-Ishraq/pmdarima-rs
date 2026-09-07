"""Time-series cross-validation splitters.

Ports of `pmdarima.model_selection._split` (MIT, Taylor G. Smith et al.).
Unlike k-fold, these never let a fold train on data that comes after its test
window - the ordering is the point.
"""

import abc

import numpy as np

from ..base import BaseEstimator

__all__ = [
    "check_cv",
    "train_test_split",
    "RollingForecastCV",
    "SlidingWindowForecastCV",
]


def _take(a, idx):
    try:
        return a[idx]
    except (KeyError, TypeError, IndexError):
        return a.iloc[idx]


def train_test_split(*arrays, test_size=None, train_size=None):
    """Split without shuffling, preserving time order."""
    if not arrays:
        raise ValueError("At least one array required as input")
    n = len(arrays[0])
    for a in arrays:
        if len(a) != n:
            raise ValueError("All arrays must have the same length")

    if test_size is None and train_size is None:
        test_size = 0.25

    def _count(v, default):
        if v is None:
            return default
        return int(np.ceil(v * n)) if isinstance(v, float) else int(v)

    if test_size is not None:
        n_test = _count(test_size, None)
        n_train = n - n_test if train_size is None else _count(train_size, None)
    else:
        n_train = _count(train_size, None)
        n_test = n - n_train

    if n_train <= 0 or n_test < 0 or n_train + n_test > n:
        raise ValueError(
            "train_size / test_size are incompatible with the input length"
        )

    idx = np.arange(n)
    out = []
    for a in arrays:
        out.append(_take(a, idx[:n_train]))
        out.append(_take(a, idx[n_train : n_train + n_test]))
    return out


class BaseTSCrossValidator(BaseEstimator, metaclass=abc.ABCMeta):
    def __init__(self, h, step):
        if h < 1:
            raise ValueError("h must be a positive value")
        if step < 1:
            raise ValueError("step must be a positive value")
        self.h = h
        self.step = step

    @property
    def horizon(self):
        return self.h

    def split(self, y, X=None):
        y = np.asarray(y)
        indices = np.arange(y.shape[0])
        for train_index, test_index in self._iter_train_test_indices(y, X):
            yield indices[train_index], indices[test_index]

    @abc.abstractmethod
    def _iter_train_test_indices(self, y, X):
        """Yield (train_indices, test_indices) for each fold."""


class RollingForecastCV(BaseTSCrossValidator):
    """An expanding training window with a fixed forecast horizon."""


    def __init__(self, h=1, step=1, initial=None):
        super().__init__(h, step)
        self.initial = initial

    def _iter_train_test_indices(self, y, X):
        n_samples = y.shape[0]
        initial, step, h = self.initial, self.step, self.h
        if initial is not None:
            if initial < 1:
                raise ValueError("Initial training size must be a positive integer")
            if initial + h > n_samples:
                raise ValueError(
                    "The initial training size + forecasting horizon would "
                    "exceed the length of the given timeseries!"
                )
        else:
            initial = max(1, n_samples // 3)

        all_indices = np.arange(n_samples)
        window_start, window_end = 0, initial
        while True:
            if window_end + h > n_samples:
                break
            yield (
                all_indices[window_start:window_end],
                all_indices[window_end : window_end + h],
            )
            window_end += step


class SlidingWindowForecastCV(BaseTSCrossValidator):
    """A fixed-width training window that slides forward."""


    def __init__(self, h=1, step=1, window_size=None):
        super().__init__(h, step)
        self.window_size = window_size

    def _iter_train_test_indices(self, y, X):
        n_samples = y.shape[0]
        window_size, step, h = self.window_size, self.step, self.h
        if window_size is not None:
            if window_size + h > n_samples:
                raise ValueError(
                    "The window_size + forecasting horizon would exceed the "
                    "length of the given timeseries!"
                )
        else:
            window_size = max(3, n_samples // 5)
        if window_size < 3:
            raise ValueError("window_size must be > 2")

        indices = np.arange(n_samples)
        window_start = 0
        while True:
            window_end = window_start + window_size
            if window_end + h > n_samples:
                break
            yield (
                indices[window_start:window_end],
                indices[window_end : window_end + h],
            )
            window_start += step


def check_cv(cv=None):
    cv = RollingForecastCV() if cv is None else cv
    if not isinstance(cv, BaseTSCrossValidator):
        raise TypeError(
            "cv should be an instance of BaseTSCrossValidator or None, but got "
            f"{cv!r} (type={type(cv)})"
        )
    return cv
