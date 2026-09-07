"""Test helpers, matching `pmdarima.compat.pytest`."""

__all__ = ["pytest_error_str", "pytest_warning_messages", "raises"]


def pytest_error_str(error):
    """The message from a pytest `ExceptionInfo`, or from a bare exception."""
    return str(getattr(error, "value", error))


def pytest_warning_messages(warnings):
    """The messages carried by a list of recorded warnings."""
    return [str(w.message) for w in warnings]


def raises(exception):
    """`pytest.raises`, imported lazily so pytest stays a test-only need."""
    import pytest

    return pytest.raises(exception)
