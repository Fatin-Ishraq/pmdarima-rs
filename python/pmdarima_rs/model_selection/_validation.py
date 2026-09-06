"""Cross-validation drivers for forecasting models.

Ports of `pmdarima.model_selection._validation` (MIT, Taylor G. Smith et al.).

Folds run on threads rather than processes. `pmdarima` reaches for `joblib`
because a Python-bound fit cannot share a core; here the likelihood releases
the GIL, so threads overlap properly without pickling the series to a worker
or paying interpreter startup per fold.
"""

import copy
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..metrics import smape
from ._split import check_cv

__all__ = ["cross_validate", "cross_val_predict", "cross_val_score"]


def _mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


_METRICS = {
    "mean_squared_error": _mse,
    "mse": _mse,
    "mean_absolute_error": _mae,
    "mae": _mae,
    "smape": smape,
}


def _get_metric(scoring):
    if callable(scoring):
        return scoring
    try:
        return _METRICS[scoring]
    except KeyError:
        raise ValueError(
            f"scoring must be callable or one of {sorted(_METRICS)}"
        ) from None


def _safe_index(a, idx):
    if a is None:
        return None
    try:
        return a[idx]
    except (KeyError, TypeError, IndexError):
        return a.iloc[idx]


def _clone(est):
    if hasattr(est, "get_params"):
        params = est.get_params(deep=False)
        return type(est)(**{k: copy.deepcopy(v) for k, v in params.items()})
    return copy.deepcopy(est)


def _fit_and_score(estimator, y, X, train, test, scorer, error_score):
    est = _clone(estimator)
    start = time.time()
    y_train, y_test = _safe_index(y, train), _safe_index(y, test)
    X_train = _safe_index(X, train) if X is not None else None
    X_test = _safe_index(X, test) if X is not None else None
    try:
        est.fit(y_train, X=X_train)
        pred = est.predict(n_periods=len(test), X=X_test)
        score = scorer(np.asarray(y_test), np.asarray(pred))
    except Exception as exc:
        if error_score == "raise":
            raise
        warnings.warn(f"Fold failed: {exc}")
        score, pred = error_score, np.full(len(test), np.nan)
    return score, time.time() - start, np.asarray(pred, dtype=float)


def _run_folds(estimator, y, X, folds, scorer, error_score, n_jobs):
    def run(fold):
        return _fit_and_score(estimator, y, X, fold[0], fold[1], scorer, error_score)

    if n_jobs in (None, 0, 1) or len(folds) < 2:
        return [run(f) for f in folds]
    workers = None if n_jobs < 0 else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run, folds))


def cross_validate(
    estimator,
    y,
    X=None,
    scoring=None,
    cv=None,
    verbose=0,
    n_jobs=None,
    error_score=np.nan,
):
    """Score `estimator` on each fold, returning fit times and test scores."""
    cv = check_cv(cv)
    scorer = _get_metric(scoring or "mean_squared_error")
    folds = list(cv.split(np.asarray(y), X))
    results = _run_folds(estimator, y, X, folds, scorer, error_score, n_jobs)
    return {
        "test_score": np.array([r[0] for r in results], dtype=float),
        "fit_time": np.array([r[1] for r in results], dtype=float),
    }


def cross_val_score(
    estimator,
    y,
    X=None,
    scoring=None,
    cv=None,
    verbose=0,
    n_jobs=None,
    error_score=np.nan,
):
    return cross_validate(
        estimator,
        y,
        X=X,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        error_score=error_score,
    )["test_score"]


def cross_val_predict(
    estimator,
    y,
    X=None,
    cv=None,
    verbose=0,
    n_jobs=None,
    averaging="mean",
    return_raw_predictions=False,
    error_score=np.nan,
):
    """Out-of-sample predictions, combined across the folds that cover them.

    Later folds re-forecast periods that earlier folds already covered, so a
    predicted index can carry several values; `averaging` decides how they
    collapse.
    """
    y_arr = np.asarray(y, dtype=float)
    n = y_arr.shape[0]
    cv = check_cv(cv)
    if cv.step > cv.horizon:
        raise ValueError(
            "CV step cannot be > CV horizon, or there will be a gap in "
            "predictions between folds"
        )
    folds = list(cv.split(y_arr, X))
    if not folds:
        raise ValueError("cv produced no folds for this series")

    scorer = _get_metric("mean_squared_error")
    results = _run_folds(estimator, y, X, folds, scorer, error_score, n_jobs)

    raw = np.full((n, len(folds)), np.nan)
    for j, ((_, test), (_, _, pred)) in enumerate(zip(folds, results)):
        raw[test, j] = pred

    if return_raw_predictions:
        return raw

    if averaging == "mean":
        combine = np.nanmean
    elif averaging == "median":
        combine = np.nanmedian
    elif callable(averaging):
        combine = averaging
    else:
        raise ValueError("averaging must be 'mean', 'median' or a callable")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = combine(raw, axis=1)

    # Only the span the folds actually covered. Anything after the last test
    # window was never predicted, and returning it as NaN would silently
    # lengthen the result relative to `pmdarima`.
    first_test = folds[0][1][0]
    last_test = folds[-1][1][-1]
    return out[first_test : last_test + 1]
