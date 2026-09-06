"""pmdarima-rs: fast, drop-in ARIMA and auto-ARIMA for Python.

A reimplementation of the `pmdarima` API with the Kalman filter in Rust.
Replace

    import pmdarima as pm

with

    import pmdarima_rs as pm

and nothing else changes - same classes, same arguments, same selected orders.

The compiled part is deliberately small. Only the likelihood and its gradient
run thousands of times per fitted model, so only those are in Rust; the order
search, the unit-root tests and the estimator API run once per model and stay
in Python where fidelity to `pmdarima` is easy to see and to test.
"""

from . import arima, datasets, metrics, model_selection, pipeline, preprocessing, utils
from ._pmdarima_rs import __version__
from .arima import (
    ADFTest,
    ARIMA,
    AutoARIMA,
    CHTest,
    KPSSTest,
    OCSBTest,
    PPTest,
    StepwiseContext,
    auto_arima,
    decompose,
    ndiffs,
    nsdiffs,
)
from .context_managers import except_and_reraise
from .metrics import smape
from .pipeline import Pipeline
from .preprocessing import (
    BoxCoxEndogTransformer,
    DateFeaturizer,
    FourierFeaturizer,
    LogEndogTransformer,
)
from .utils import (
    acf,
    autocorr_plot,
    c,
    pacf,
    plot_acf,
    plot_pacf,
    tsdisplay,
)


def show_versions():
    """Print the versions of this package and its dependencies."""
    import platform
    import sys

    import numpy
    import pandas
    import scipy

    print("\nSystem:")
    print(f"    python: {sys.version.splitlines()[0]}")
    print(f"  platform: {platform.platform()}")
    print("\nPython dependencies:")
    print(f"pmdarima-rs: {__version__}")
    print(f"      numpy: {numpy.__version__}")
    print(f"     pandas: {pandas.__version__}")
    print(f"      scipy: {scipy.__version__}")
    try:
        import sklearn

        print(f"    sklearn: {sklearn.__version__}")
    except ImportError:
        print("    sklearn: not installed")


def install():
    """Alias this package into :data:`sys.modules` as ``pmdarima``.

    For code you cannot edit. After ``pmdarima_rs.install()``, an
    ``import pmdarima`` anywhere in the process resolves here. Call it before
    the first ``import pmdarima``; if the real package is already imported
    this raises rather than leaving a half-patched module graph.
    """
    import sys

    if "pmdarima" in sys.modules and sys.modules["pmdarima"] is not sys.modules[__name__]:
        raise RuntimeError(
            "`pmdarima` is already imported; call install() before importing it"
        )
    for name in (
        "",
        ".arima",
        ".arima.arima",
        ".arima.auto",
        ".arima.seasonality",
        ".arima.stationarity",
        ".arima.utils",
        ".context_managers",
        ".datasets",
        ".metrics",
        ".model_selection",
        ".pipeline",
        ".preprocessing",
        ".preprocessing.endog",
        ".preprocessing.exog",
        ".utils",
        ".utils.array",
    ):
        mod = sys.modules.get(__name__ + name)
        if mod is not None:
            sys.modules["pmdarima" + name] = mod
    return sys.modules[__name__]


__all__ = [
    # estimators
    "ARIMA",
    "AutoARIMA",
    "auto_arima",
    "Pipeline",
    # diagnostics
    "ADFTest",
    "KPSSTest",
    "PPTest",
    "CHTest",
    "OCSBTest",
    "ndiffs",
    "nsdiffs",
    "decompose",
    "StepwiseContext",
    # preprocessing
    "BoxCoxEndogTransformer",
    "LogEndogTransformer",
    "FourierFeaturizer",
    "DateFeaturizer",
    # utilities
    "acf",
    "pacf",
    "c",
    "smape",
    "plot_acf",
    "plot_pacf",
    "autocorr_plot",
    "tsdisplay",
    "except_and_reraise",
    "show_versions",
    "install",
    # submodules
    "arima",
    "datasets",
    "metrics",
    "model_selection",
    "pipeline",
    "preprocessing",
    "utils",
    "__version__",
]
