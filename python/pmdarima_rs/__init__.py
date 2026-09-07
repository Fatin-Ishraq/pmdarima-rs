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


#: The `pmdarima` API level this package implements. `install()` reports this
#: as ``pmdarima.__version__`` so that version gates in code you cannot edit
#: keep working; this package's own version stays on ``pmdarima_rs``.
PMDARIMA_API_VERSION = "2.1.1"


import types as _types


class _AliasModule(_types.ModuleType):
    """A stand-in module that forwards everything to the real one.

    Used only for the top-level ``pmdarima`` alias, so that
    ``pmdarima.__version__`` can report the API level while every other
    attribute - classes included, so ``isinstance`` still works - is the very
    same object this package exposes.
    """

    def __init__(self, name, target, **overrides):
        super().__init__(name, target.__doc__)
        object.__setattr__(self, "_alias_target", target)
        object.__setattr__(self, "_alias_overrides", overrides)

    def __getattr__(self, item):
        # Only reached when the name is not in this module's own dict.
        overrides = object.__getattribute__(self, "_alias_overrides")
        if item in overrides:
            return overrides[item]
        return getattr(object.__getattribute__(self, "_alias_target"), item)

    def __dir__(self):
        target = object.__getattribute__(self, "_alias_target")
        overrides = object.__getattribute__(self, "_alias_overrides")
        return sorted(set(dir(target)) | set(overrides))


def _submodule_names():
    """Every importable submodule of this package, as dotted suffixes."""
    import pkgutil

    names = []
    # Walk with the real prefix: pkgutil imports subpackages to descend into
    # them, and a relative prefix makes that import fail silently, leaving
    # most of the tree unaliased.
    for info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
        names.append(info.name[len(__name__) :])
    return names


def install(api_version=PMDARIMA_API_VERSION):
    """Alias this package into :data:`sys.modules` as ``pmdarima``.

    For code you cannot edit. After ``pmdarima_rs.install()``, an
    ``import pmdarima`` anywhere in the process resolves here - including
    submodules that have not been imported yet, which are resolved on demand
    by a finder installed on ``sys.meta_path``. Call it before the first
    ``import pmdarima``; if the real package is already imported this raises
    rather than leaving a half-patched module graph.
    """
    import importlib
    import importlib.abc
    import importlib.util
    import sys

    existing = sys.modules.get("pmdarima")
    if existing is not None and getattr(existing, "__pmdarima_rs__", False) is not True:
        raise RuntimeError(
            "`pmdarima` is already imported; call install() before importing it"
        )

    self = sys.modules[__name__]
    alias = _AliasModule(
        "pmdarima",
        self,
        __version__=api_version,
        __pmdarima_rs__=True,
        __pmdarima_rs_version__=__version__,
    )
    sys.modules["pmdarima"] = alias

    # Alias everything that is already imported, so the common names resolve
    # without going through the finder at all.
    for suffix in _submodule_names():
        if not suffix:
            continue
        mod = sys.modules.get(__name__ + suffix)
        if mod is not None:
            sys.modules["pmdarima" + suffix] = mod

    class _AliasLoader(importlib.abc.Loader):
        def __init__(self, module):
            self._module = module

        def create_module(self, spec):
            return self._module

        def exec_module(self, module):
            pass

    class _AliasFinder(importlib.abc.MetaPathFinder):
        """Resolve any `pmdarima.*` import to the matching module here."""

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "pmdarima" and not fullname.startswith("pmdarima."):
                return None
            if fullname in sys.modules:
                return None
            mapped = __name__ + fullname[len("pmdarima") :]
            try:
                mod = importlib.import_module(mapped)
            except ImportError:
                return None
            sys.modules[fullname] = mod
            return importlib.util.spec_from_loader(fullname, _AliasLoader(mod))

    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
    return alias


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
    "PMDARIMA_API_VERSION",
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
