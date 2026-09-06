"""Warning categories, matching `pmdarima.warnings`."""

__all__ = ["ModelFitWarning"]


class ModelFitWarning(UserWarning):
    """Raised when a candidate model fails to fit during a search."""
