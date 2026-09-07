"""Cross-validation drivers for forecasting models.

Ports of `pmdarima.model_selection._validation` (MIT, Taylor G. Smith et al.).
Folds run on threads rather than processes. `pmdarima` reaches for `joblib`
because a Python-bound fit cannot share a core; here the likelihood releases
the GIL, so threads overlap properly without pickling the series to a worker
or paying interpreter startup per fold.
"""

import copy
import numbers
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from traceback import format_exception_only

import numpy as np

from ..metrics import smape
from ..utils.array import check_endog
from ..warnings import ModelFitWarning
from ._split import check_cv

__all__ = ["cross_validate", "cross_val_predict", "cross_val_score"]


def _mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


_VALID_SCORING = {
    "mean_absolute_error": _mae,
    "mean_squared_error": _mse,
    "smape": smape,
}

_VALID_AVERAGING = {
    "mean": np.nanmean,
    "median": np.nanmedian,
}


def _check_callables(x, dct, varname):
    if callable(x):
        return x
    if isinstance(x, str):
        try:
            return dct[x]
        except KeyError:
            raise ValueError(
                "%s can be a callable or a string in %s" % (varname, list(dct))
            ) from None
    raise TypeError(
        "expected a callable or a string, but got %r (type=%s)" % (x, type(x))
    )


def _check_averaging(method):
    return _check_callables(method, _VALID_AVERAGING, "averaging")


def _check_scoring(metric):
    return _check_callables(metric, _VALID_SCORING, "metric")


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


def _split_arrays(y, X, train, test):
    y_train, y_test = _safe_index(y, train), _safe_index(y, test)
    X_train = _safe_index(X, train) if X is not None else None
    X_test = _safe_index(X, test) if X is not None else None
    return y_train, y_test, X_train, X_test


def _fit_and_score(fold, estimator, y, X, scorer, train, test, verbose, error_score):
    """Fit on `train`, score the forecast of `test`."""
    msg = "fold=%i" % fold
    if verbose > 1:
        print("[CV] %s %s" % (msg, (64 - len(msg)) * "."))

    start_time = time.time()
    y_train, y_test, X_train, X_test = _split_arrays(y, X, train, test)
    try:
        estimator.fit(y_train, X=X_train)
    except Exception as e:
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        test_scores = error_score
        warnings.warn(
            "Estimator fit failed. The score on this train-test partition "
            "will be set to %f. Details: \n%s"
            % (error_score, format_exception_only(type(e), e)[0]),
            ModelFitWarning,
        )
    else:
        fit_time = time.time() - start_time
        preds = estimator.predict(n_periods=len(test), X=X_test)
        test_scores = scorer(np.asarray(y_test), np.asarray(preds))
        score_time = time.time() - start_time - fit_time

    if verbose > 2:
        total_time = score_time + fit_time
        print(msg + ", score=%.3f [time=%.3f sec]" % (test_scores, total_time))
    return test_scores, fit_time, score_time


def _fit_and_predict(fold, estimator, y, X, train, test, verbose):
    """Fit on `train` and forecast `test`; failures are the caller's problem."""
    msg = "fold=%i" % fold
    if verbose > 1:
        print("[CV] %s %s" % (msg, (64 - len(msg)) * "."))

    start_time = time.time()
    y_train, _, X_train, X_test = _split_arrays(y, X, train, test)
    estimator.fit(y_train, X=X_train)
    fit_time = time.time() - start_time

    start_time = time.time()
    preds = estimator.predict(n_periods=len(test), X=X_test)
    pred_time = time.time() - start_time
    if verbose > 2:
        print(msg + " [time=%.3f sec]" % (pred_time + fit_time))
    return np.asarray(preds, dtype=float), test


def _run(tasks, n_jobs):
    """Run per-fold work, on threads when asked for."""
    if n_jobs in (None, 0, 1) or len(tasks) < 2:
        return [t() for t in tasks]
    workers = None if n_jobs < 0 else n_jobs
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda t: t(), tasks))


def _check_error_score(error_score):
    if not (error_score == "raise" or isinstance(error_score, numbers.Number)):
        raise ValueError(
            'error_score should be the string "raise" or a numeric value'
        )


def cross_validate(
    estimator,
    y,
    X=None,
    scoring=None,
    cv=None,
    verbose=0,
    error_score=np.nan,
    n_jobs=None,
):
    """Score `estimator` on each fold, with the fit and score times."""
    y = check_endog(y, copy=False, preserve_series=True)
    cv = check_cv(cv)
    scoring = _check_scoring(scoring)
    _check_error_score(error_score)

    folds = list(cv.split(np.asarray(y), X))
    tasks = [
        (
            lambda fold=fold, train=train, test=test: _fit_and_score(
                fold,
                _clone(estimator),
                y,
                X,
                scorer=scoring,
                train=train,
                test=test,
                verbose=verbose,
                error_score=error_score,
            )
        )
        for fold, (train, test) in enumerate(folds)
    ]
    results = _run(tasks, n_jobs)
    scores, fit_times, score_times = list(zip(*results)) if results else ((), (), ())
    return {
        "test_score": np.array(scores, dtype=float),
        "fit_time": np.array(fit_times, dtype=float),
        "score_time": np.array(score_times, dtype=float),
    }


def cross_val_score(
    estimator,
    y,
    X=None,
    scoring=None,
    cv=None,
    verbose=0,
    error_score=np.nan,
    n_jobs=None,
):
    """The `test_score` column of :func:`cross_validate`."""
    return cross_validate(
        estimator,
        y,
        X=X,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        error_score=error_score,
        n_jobs=n_jobs,
    )["test_score"]


def cross_val_predict(
    estimator,
    y,
    X=None,
    cv=None,
    verbose=0,
    averaging="mean",
    return_raw_predictions=False,
    n_jobs=None,
):
    """Out-of-sample predictions, combined across the folds that cover them.

    Later folds re-forecast periods that earlier folds already covered, so a
    predicted index can carry several values; `averaging` decides how they
    collapse. A fold that fails to fit raises rather than being scored away -
    there is no score here to put a sentinel into.
    """
    y = check_endog(y, copy=False, preserve_series=True)
    cv = check_cv(cv)
    avgfunc = _check_averaging(averaging)
    if cv.step > cv.horizon:
        raise ValueError(
            "CV step cannot be > CV horizon, or there will be a gap in "
            "predictions between folds"
        )

    folds = list(cv.split(np.asarray(y), X))
    if not folds:
        raise ValueError("cv produced no folds for this series")
    tasks = [
        (
            lambda fold=fold, train=train, test=test: _fit_and_predict(
                fold, _clone(estimator), y, X, train=train, test=test, verbose=verbose
            )
        )
        for fold, (train, test) in enumerate(folds)
    ]
    prediction_blocks = _run(tasks, n_jobs)

    n = np.asarray(y).shape[0]
    pred_matrix = np.ones((n, len(prediction_blocks))) * np.nan
    for i, (pred_block, test_indices) in enumerate(prediction_blocks):
        pred_matrix[test_indices, i] = pred_block

    if return_raw_predictions:
        predictions = np.ones((n, cv.horizon)) * np.nan
        for pred_block, test_indices in prediction_blocks:
            predictions[test_indices[0]] = pred_block
        return predictions

    test_mask = ~(np.isnan(pred_matrix).all(axis=1))
    predictions = pred_matrix[test_mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return avgfunc(predictions, axis=1)
