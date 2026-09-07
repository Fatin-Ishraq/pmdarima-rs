"""Warning categories, matching `pmdarima.warnings`."""

__all__ = ["ModelFitWarning", "ConvergenceWarning", "EstimationWarning"]


class ModelFitWarning(UserWarning):
    """Raised when a candidate model fails to fit during a search."""


try:  # pragma: no cover - depends on the environment
    # Prefer statsmodels' own categories when it happens to be installed, so
    # that code written against pmdarima and filtering on them keeps working.
    # This is not a dependency: the import is guarded and never required.
    from statsmodels.tools.sm_exceptions import (  # noqa: F401
        ConvergenceWarning,
        EstimationWarning,
    )
except ImportError:  # pragma: no cover

    class EstimationWarning(UserWarning):
        """Raised when starting values cannot be estimated from the data."""

    class ConvergenceWarning(UserWarning):
        """Raised when the optimiser stops before it has converged."""
