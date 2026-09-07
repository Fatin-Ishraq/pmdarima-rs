"""Argument validation for `auto_arima`.

Kept in its own module, as `pmdarima` does, so the checks can be tested
without fitting dozens of models.
"""

import warnings

import numpy as np

from ..warnings import ModelFitWarning

__all__ = [
    "auto_intercept",
    "check_information_criterion",
    "check_kwargs",
    "check_m",
    "check_n_jobs",
    "check_start_max_values",
    "check_trace",
    "get_scoring_metric",
    "warn_for_D",
]

VALID_CRITERIA = {"aic", "aicc", "bic", "hqic", "oob"}


def auto_intercept(with_intercept, default):
    if with_intercept == "auto":
        return default
    return with_intercept


def check_information_criterion(information_criterion, out_of_sample_size):
    if information_criterion not in VALID_CRITERIA:
        raise ValueError(
            "auto_arima not defined for information_criteria=%s. "
            "Valid information criteria include: %r"
            % (information_criterion, VALID_CRITERIA)
        )
    if information_criterion == "oob" and out_of_sample_size == 0:
        information_criterion = "aic"
        warnings.warn(
            "information_criterion cannot be 'oob' with "
            "out_of_sample_size = 0. "
            "Falling back to information criterion = aic."
        )
    return information_criterion


def check_kwargs(kwargs):
    """Turn a `None` default into an empty dict, avoiding a mutable default."""
    if kwargs:
        return kwargs
    return {}


def check_m(m, seasonal):
    if (m < 1 and seasonal) or m < 0:
        raise ValueError("m must be a positive integer (> 0)")
    if not seasonal:
        if m > 1:
            warnings.warn("m (%i) set for non-seasonal fit. Setting to 0" % m)
        m = 0
    return m


def check_n_jobs(stepwise, n_jobs):
    """The stepwise walk is sequential by construction, so it cannot fan out."""
    if stepwise and n_jobs != 1:
        n_jobs = 1
        warnings.warn(
            "stepwise model cannot be fit in parallel (n_jobs=%i). "
            "Falling back to stepwise parameter search." % n_jobs
        )
    return n_jobs


def check_start_max_values(st, mx, argname):
    if mx is None:
        mx = np.inf
    if st is None:
        raise ValueError("start_%s cannot be None" % argname)
    if st < 0:
        raise ValueError("start_%s must be positive" % argname)
    if mx < st:
        raise ValueError("max_%s must be >= start_%s" % (argname, argname))
    return st, mx


def check_trace(trace):
    if trace is None:
        return 0
    if isinstance(trace, (int, bool, np.integer)):
        return int(trace)
    if trace:
        return 1
    return 0


def get_scoring_metric(metric):
    """Resolve a scoring name or callable, as `pmdarima` does."""
    from .arima import _get_scoring

    return _get_scoring(metric)


def warn_for_D(d, D):
    """Warn when the total amount of differencing is implausible."""
    if D >= 2:
        warnings.warn(
            "Having more than one seasonal differences is "
            "not recommended. Please consider using only one "
            "seasonal difference.",
            ModelFitWarning,
        )
    elif D + d > 2 or d > 2:
        warnings.warn(
            "Having 3 or more differencing operations is not "
            "recommended. Please consider reducing the total "
            "number of differences.",
            ModelFitWarning,
        )
