"""Box-Cox and log transforms of the endogenous series."""

import warnings

import numpy as np
from scipy import stats

from ..base import check_is_fitted
from .base import BaseEndogTransformer

__all__ = ["BoxCoxEndogTransformer", "LogEndogTransformer"]


class BoxCoxEndogTransformer(BaseEndogTransformer):
    """Box-Cox transform, estimating lambda by maximum likelihood if unset."""

    _param_names = ("lmbda", "lmbda2", "neg_action", "floor")

    def __init__(self, lmbda=None, lmbda2=0, neg_action="raise", floor=1e-16):
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.neg_action = neg_action
        self.floor = floor

    def fit(self, y, X=None):
        lam1 = self.lmbda
        lam2 = self.lmbda2
        if lam2 < 0:
            raise ValueError("lmbda2 must be a non-negative scalar value")
        if lam1 is None:
            y, _ = self._check_y_X(y, X)
            _, lam1 = stats.boxcox(y + lam2, lmbda=None, alpha=None)
        self.lam1_ = lam1
        self.lam2_ = lam2
        return self

    def transform(self, y, X=None, **kwargs):
        check_is_fitted(self, "lam1_")
        lam1, lam2 = self.lam1_, self.lam2_
        y, exog = self._check_y_X(y, X)
        y = y + lam2
        neg_mask = y <= 0.0
        if neg_mask.any():
            msg = "Negative or zero values present in y"
            if self.neg_action == "raise":
                raise ValueError(msg)
            if self.neg_action == "warn":
                warnings.warn(msg, UserWarning)
            y[neg_mask] = self.floor
        if lam1 == 0:
            return np.log(y), exog
        return (y**lam1 - 1) / lam1, exog

    def inverse_transform(self, y, X=None):
        check_is_fitted(self, "lam1_")
        lam1, lam2 = self.lam1_, self.lam2_
        y, exog = self._check_y_X(y, X)
        if lam1 == 0:
            return np.exp(y) - lam2, exog
        numer = y * lam1 + 1.0
        return numer ** (1.0 / lam1) - lam2, exog


class LogEndogTransformer(BoxCoxEndogTransformer):
    """A Box-Cox transform with lambda pinned to zero, i.e. a log."""

    _param_names = ("lmbda", "neg_action", "floor")

    def __init__(self, lmbda=0, neg_action="raise", floor=1e-16):
        super().__init__(neg_action=neg_action, floor=floor)
        self.lmbda = 0
        self.lmbda2 = lmbda

    def get_params(self, deep=True):
        params = {
            "neg_action": self.neg_action,
            "floor": self.floor,
            "lmbda": self.lmbda2,
        }
        return params
