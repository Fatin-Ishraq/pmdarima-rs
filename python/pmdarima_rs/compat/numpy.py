"""Numpy-facing constants, matching `pmdarima.compat.numpy`."""

import numpy as np

__all__ = ["DTYPE"]

#: The dtype every array in this package is coerced to.
DTYPE = np.float64
