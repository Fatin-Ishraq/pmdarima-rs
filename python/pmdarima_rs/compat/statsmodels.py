"""Shims for statsmodels conventions, matching `pmdarima.compat.statsmodels`.

Nothing here imports statsmodels. `check_seasonal_order` is the one that
matters: statsmodels rejects a seasonal periodicity of 1, but `auto_arima`
carries `(0, 0, 0, 1)` around internally to mean "not seasonal", so it has to
be normalised before it reaches a model.
"""

from collections.abc import Iterable

__all__ = ["bind_df_model", "check_seasonal_order"]


def check_seasonal_order(order):
    """Normalise a null seasonal order with `m == 1` to `(0, 0, 0, 0)`."""
    if isinstance(order[0], Iterable) and not isinstance(order[0], str):
        return order
    if sum(order[:3]) == 0 and order[-1] == 1:
        return (0, 0, 0, 0)
    return order


def bind_df_model(model_fit, arima_results):
    """Ensure `df_model` is present on a results object.

    Old statsmodels releases did not define it on SARIMAX results, so
    `pmdarima` computes and attaches it. This package always provides it, so
    this only has to stay compatible with callers that still invoke it.
    """
    if not hasattr(arima_results, "df_model"):
        df_model = model_fit.k_exog + model_fit.k_trend + model_fit.k_ar + model_fit.k_ma
        arima_results.df_model = df_model
    return arima_results
