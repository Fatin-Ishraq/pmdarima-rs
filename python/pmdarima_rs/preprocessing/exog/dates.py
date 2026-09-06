"""Calendar features from a datetime column."""

import warnings

import numpy as np
import pandas as pd

from .base import BaseExogFeaturizer

__all__ = ["DateFeaturizer"]


def _safe_hstack_numpy(left, right):
    return right if left is None else np.hstack([left, right])


class DateFeaturizer(BaseExogFeaturizer):
    """One-hot day-of-week and day-of-month from a datetime column."""

    _param_names = ("column_name", "with_day_of_week", "with_day_of_month", "prefix")

    def __init__(
        self, column_name, with_day_of_week=True, with_day_of_month=True, prefix=None
    ):
        super().__init__(prefix=prefix)
        self.column_name = column_name
        self.with_day_of_week = with_day_of_week
        self.with_day_of_month = with_day_of_month

    def _check_X(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a DataFrame to use the DateFeaturizer, but got "
                f"type={type(X)}"
            )
        name = self.column_name
        if not (name in X.columns and "datetime64" in X[name].dtype.name):
            raise ValueError(
                f"column '{name}' must exist in exog as a pd.Timestamp type"
            )

    def _get_prefix(self):
        return "DATE" if self.prefix is None else self.prefix

    def _get_feature_names(self, X):
        pfx = self._get_prefix()
        out = []
        if self.with_day_of_week:
            out += [f"{pfx}-WEEKDAY-{i}" for i in range(7)]
        if self.with_day_of_month:
            out += [f"{pfx}-DAY-OF-MONTH"]
        return out

    def fit(self, y, X=None, **kwargs):
        y, X = self._check_y_X(y, X, null_allowed=False)
        self._check_X(X)
        if not (self.with_day_of_month or self.with_day_of_week):
            warnings.warn(
                "DateTransformer will have no effect given disabled parameters"
            )
        return self

    def transform(self, y, X=None, **kwargs):
        y, X = self._check_y_X(y, X, null_allowed=True)
        self._check_X(X)
        date_series = X[self.column_name]
        m = X.shape[0]
        right_side = None
        if self.with_day_of_week:
            zeros = np.zeros((m, 7), dtype=int)
            zeros[np.arange(m), date_series.dt.weekday.values] = 1
            right_side = zeros
        if self.with_day_of_month:
            right_side = _safe_hstack_numpy(
                right_side, date_series.dt.day.values.reshape(-1, 1)
            )
        if right_side is not None:
            X = self._safe_hstack(X.drop(self.column_name, axis=1), right_side)
        return y, X
