"""Forecast accuracy metrics."""

import numpy as np

from .utils.array import check_endog

__all__ = ["smape"]


def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error, on a 0-200 scale."""
    y_true = check_endog(y_true, copy=False, preserve_series=False)
    y_pred = check_endog(y_pred, copy=False, preserve_series=False)
    abs_diff = np.abs(y_pred - y_true)
    return np.mean(abs_diff * 200 / (np.abs(y_pred) + np.abs(y_true)))
