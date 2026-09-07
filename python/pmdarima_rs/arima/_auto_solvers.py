"""Order search: the stepwise walk and the grid/random alternative.

A port of `pmdarima`'s solvers (MIT, Taylor G. Smith et al.), because the walk
*is* the algorithm - which neighbours are tried, in which order, and when the
walk stops - and any deviation selects a different model. The speed comes from
underneath: each candidate fit is the Rust likelihood rather than the
statsmodels one.

One thing does change. `pmdarima` runs the non-stepwise search under `joblib`,
which means processes, because a Python-bound fit cannot share a core. Here
the likelihood releases the GIL, so threads are enough - no pickling of the
series per worker, no interpreter startup, and the results come back in the
same order.
"""

import functools
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from ..warnings import ModelFitWarning
from ..compat.statsmodels import check_seasonal_order
from ._context import ContextStore, ContextType
from .arima import ARIMA

__all__ = ["_StepwiseFitWrapper", "_RandomFitWrapper", "_fit_candidate_model"]


def _root_test(model, ic, trace):
    """Reject models whose fitted roots sit on the unit circle.

    A model with an inverse root at 0.999 has technically maximised the
    likelihood, but it is a boundary solution rather than a description of the
    data, and its information criterion is not comparable with the others.
    `pmdarima` scores those as infinite; so do we.
    """
    max_invroot = 0
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order
    if p + P > 0:
        roots = model.arroots()
        if roots.size:
            max_invroot = max(0, *np.abs(1 / roots))
    if q + Q > 0 and np.isfinite(ic):
        roots = model.maroots()
        if roots.size:
            max_invroot = max(0, *np.abs(1 / roots))
    if max_invroot > 1 - 1e-2:
        ic = np.inf
        if trace > 1:
            print(
                "Near non-invertible roots for order "
                "(%i, %i, %i)(%i, %i, %i, %i); setting score to inf (at "
                "least one inverse root too close to the border of the "
                "unit circle: %.3f)" % (p, d, q, P, D, Q, m, max_invroot)
            )
    return ic


def _fit_candidate_model(
    y,
    X,
    order,
    seasonal_order,
    start_params,
    trend,
    method,
    maxiter,
    fit_params,
    suppress_warnings,
    trace,
    error_action,
    out_of_sample_size,
    scoring,
    scoring_args,
    with_intercept,
    information_criterion,
    **kwargs,
):
    start = time.time()
    fit_time = np.nan
    ic = np.inf
    fit = ARIMA(
        order=order,
        seasonal_order=seasonal_order,
        start_params=start_params,
        trend=trend,
        method=method,
        maxiter=maxiter,
        suppress_warnings=suppress_warnings,
        out_of_sample_size=out_of_sample_size,
        scoring=scoring,
        scoring_args=scoring_args,
        with_intercept=with_intercept,
        **kwargs,
    )
    try:
        fit.fit(y, X=X, **fit_params)
    except (np.linalg.LinAlgError, ValueError) as v:
        if error_action == "raise":
            raise v
        if error_action in ("warn", "trace"):
            msg = (
                f"Error fitting {fit} (if you do not want to see these "
                'warnings, run with error_action="ignore").'
            )
            if error_action == "trace":
                msg += "\nTraceback:\n" + traceback.format_exc()
            warnings.warn(msg, ModelFitWarning)
    else:
        fit_time = time.time() - start
        ic = getattr(fit, information_criterion)()
        ic = _root_test(fit, ic, trace)

    if trace:
        print(
            f"{fit}   : {information_criterion.upper()}={ic:.3f}, "
            f"Time={fit_time:.2f} sec"
        )
    return fit, fit_time, ic


def _sort_and_filter_fits(models):
    if not isinstance(models, list):
        models = [models]
    filtered = [(mod, ic) for mod, _, ic in models if mod is not None and np.isfinite(ic)]
    if not filtered:
        raise ValueError(
            "Could not successfully fit a viable ARIMA model "
            "to input data.\nSee "
            "http://alkaline-ml.com/pmdarima/no-successful-model.html "
            "for more information on why this can happen."
        )
    sorted_res = sorted(filtered, key=lambda mod_ic: mod_ic[1])
    models, _ = zip(*sorted_res)
    return list(models)


class _RandomFitWrapper:
    """Grid or random search over the whole order space."""

    def __init__(
        self,
        y,
        X,
        fit_partial,
        d,
        D,
        m,
        max_order,
        max_p,
        max_q,
        max_P,
        max_Q,
        random,
        random_state,
        n_fits,
        n_jobs,
        seasonal,
        trace,
        with_intercept,
        sarimax_kwargs,
    ):
        if seasonal:
            gen = [
                ((p, d, q), (P, D, Q, m))
                for p in range(0, max_p + 1)
                for q in range(0, max_q + 1)
                for P in range(0, max_P + 1)
                for Q in range(0, max_Q + 1)
                if p + q + P + Q <= max_order
            ]
        else:
            gen = [
                ((p, d, q), (0, 0, 0, 0))
                for p in range(0, max_p + 1)
                for q in range(0, max_q + 1)
                if p + q <= max_order
            ]
        if random:
            rs = np.random.mtrand._rand if random_state is None else None
            if rs is None:
                rs = (
                    random_state
                    if isinstance(random_state, np.random.RandomState)
                    else np.random.RandomState(random_state)
                )
            gen = list(rs.permutation(np.array(gen, dtype="object"))[:n_fits])

        self.gen = gen
        self.n_jobs = n_jobs
        self.trace = trace
        self.fit_partial = functools.partial(
            fit_partial,
            y=y,
            X=X,
            with_intercept=with_intercept,
            **sarimax_kwargs,
        )

    def solve(self):
        fit_partial = self.fit_partial
        gen = list(self.gen)
        n_jobs = self.n_jobs

        def run(spec):
            order, seasonal_order = spec
            return fit_partial(order=order, seasonal_order=seasonal_order)

        if n_jobs in (0, 1) or len(gen) < 2:
            all_res = [run(s) for s in gen]
        else:
            workers = None if n_jobs < 0 else n_jobs
            # Threads, not processes: the likelihood releases the GIL, so the
            # candidates genuinely overlap without pickling the series to a
            # worker or paying interpreter startup.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                all_res = list(ex.map(run, gen))

        sorted_fits = _sort_and_filter_fits(all_res)
        if self.trace and sorted_fits:
            print(f"\nBest model: {sorted_fits[0]}")
        return sorted_fits


class _StepwiseFitWrapper:
    """Hyndman & Khandakar's stepwise walk over neighbouring orders."""

    def __init__(
        self,
        y,
        X,
        start_params,
        trend,
        method,
        maxiter,
        fit_params,
        suppress_warnings,
        trace,
        error_action,
        out_of_sample_size,
        scoring,
        scoring_args,
        p,
        d,
        q,
        P,
        D,
        Q,
        m,
        max_p,
        max_q,
        max_P,
        max_Q,
        seasonal,
        information_criterion,
        with_intercept,
        **kwargs,
    ):
        self.trace = trace
        self._fit_arima = functools.partial(
            _fit_candidate_model,
            y=y,
            X=X,
            start_params=start_params,
            trend=trend,
            method=method,
            maxiter=maxiter,
            fit_params=fit_params,
            suppress_warnings=suppress_warnings,
            trace=trace,
            error_action=error_action,
            out_of_sample_size=out_of_sample_size,
            scoring=scoring,
            scoring_args=scoring_args,
            information_criterion=information_criterion,
            **kwargs,
        )
        self.information_criterion = information_criterion
        self.with_intercept = with_intercept
        self.p, self.d, self.q = p, d, q
        self.P, self.D, self.Q, self.m = P, D, Q, m
        self.max_p, self.max_q = max_p, max_q
        self.max_P, self.max_Q = max_P, max_Q
        self.seasonal = seasonal
        self.exec_context = ContextStore.get_or_empty(ContextType.STEPWISE)
        self.k = self.start_k = 0
        self.max_k = 100 if self.exec_context.max_steps is None else self.exec_context.max_steps
        self.max_dur = self.exec_context.max_dur
        self.results_dict = {}
        self.ic_dict = {}
        self.fit_time_dict = {}
        self.bestfit = None
        self.bestfit_key = None

    def _do_fit(self, order, seasonal_order, constant=None):
        if not self.seasonal:
            seasonal_order = (0, 0, 0, 0)
        # A null seasonal order with m == 1 means "not seasonal"; statsmodels
        # rejects a periodicity of 1, and pmdarima reports (0, 0, 0, 0).
        seasonal_order = check_seasonal_order(tuple(seasonal_order))
        if constant is None:
            constant = self.with_intercept

        if (order, seasonal_order, constant) not in self.results_dict:
            self.k += 1
            fit, fit_time, new_ic = self._fit_arima(
                order=order, seasonal_order=seasonal_order, with_intercept=constant
            )
            self.results_dict[(order, seasonal_order, constant)] = fit
            self.ic_dict[(order, seasonal_order, constant)] = new_ic
            self.fit_time_dict[(order, seasonal_order, constant)] = fit_time

            if fit is None or np.isinf(new_ic):
                return False
            if self.bestfit is None:
                self.bestfit = fit
                self.bestfit_key = (order, seasonal_order, constant)
                if self.trace > 1:
                    print("First viable model found (%.3f)" % new_ic)
                return True

            current_ic = self.ic_dict[self.bestfit_key]
            if new_ic < current_ic:
                if self.trace > 1:
                    print("New best model found (%.3f < %.3f)" % (new_ic, current_ic))
                self.bestfit = fit
                self.bestfit_key = (order, seasonal_order, constant)
                return True
        return False

    def solve(self):
        start_time = datetime.now()
        p, d, q = self.p, self.d, self.q
        P, D, Q, m = self.P, self.D, self.Q, self.m
        max_p, max_q = self.max_p, self.max_q
        max_P, max_Q = self.max_P, self.max_Q

        if self.trace:
            print(f"Performing stepwise search to minimize {self.information_criterion}")

        self._do_fit((p, d, q), (P, D, Q, m))
        if self._do_fit((0, d, 0), (0, D, 0, m)):
            p = q = P = Q = 0

        if max_p > 0 or max_P > 0:
            _p = 1 if max_p > 0 else 0
            _P = 1 if (m > 1 and max_P > 0) else 0
            if self._do_fit((_p, d, 0), (_P, D, 0, m)):
                p, P = _p, _P
                q = Q = 0

        if max_q > 0 or max_Q > 0:
            _q = 1 if max_q > 0 else 0
            _Q = 1 if (m > 1 and max_Q > 0) else 0
            if self._do_fit((0, d, _q), (0, D, _Q, m)):
                p = P = 0
                Q, q = _Q, _q

        if self.with_intercept:
            if self._do_fit((0, d, 0), (0, D, 0, m), constant=False):
                p = q = P = Q = 0

        while self.start_k < self.k < self.max_k:
            self.start_k = self.k
            dur = (datetime.now() - start_time).total_seconds()
            if self.max_dur and dur > self.max_dur:
                warnings.warn(
                    "early termination of stepwise search due to max_dur "
                    "threshold (%.3f > %.3f)" % (dur, self.max_dur)
                )
                break

            # Seasonal neighbours first, then non-seasonal, then the
            # intercept. The order is part of the algorithm.
            moves = (
                (P > 0, (p, d, q), (P - 1, D, Q, m), (-1, 0)),
                (Q > 0, (p, d, q), (P, D, Q - 1, m), (0, -1)),
                (P < max_P, (p, d, q), (P + 1, D, Q, m), (1, 0)),
                (Q < max_Q, (p, d, q), (P, D, Q + 1, m), (0, 1)),
                (Q > 0 and P > 0, (p, d, q), (P - 1, D, Q - 1, m), (-1, -1)),
                (Q < max_Q and P > 0, (p, d, q), (P - 1, D, Q + 1, m), (-1, 1)),
                (Q > 0 and P < max_P, (p, d, q), (P + 1, D, Q - 1, m), (1, -1)),
                (Q < max_Q and P < max_P, (p, d, q), (P + 1, D, Q + 1, m), (1, 1)),
            )
            moved = False
            for cond, o, so, (dP, dQ) in moves:
                if cond and self.k < self.max_k and self._do_fit(o, so):
                    P += dP
                    Q += dQ
                    moved = True
                    break
            if moved:
                continue

            pq_moves = (
                (p > 0, (p - 1, d, q), (-1, 0)),
                (q > 0, (p, d, q - 1), (0, -1)),
                (p < max_p, (p + 1, d, q), (1, 0)),
                (q < max_q, (p, d, q + 1), (0, 1)),
                (q > 0 and p > 0, (p - 1, d, q - 1), (-1, -1)),
                (q < max_q and p > 0, (p - 1, d, q + 1), (-1, 1)),
                (q > 0 and p < max_p, (p + 1, d, q - 1), (1, -1)),
                (q < max_q and p < max_p, (p + 1, d, q + 1), (1, 1)),
            )
            for cond, o, (dp, dq) in pq_moves:
                if cond and self.k < self.max_k and self._do_fit(o, (P, D, Q, m)):
                    p += dp
                    q += dq
                    moved = True
                    break
            if moved:
                continue

            if self.k < self.max_k and self._do_fit(
                (p, d, q), (P, D, Q, m), constant=not self.with_intercept
            ):
                self.with_intercept = not self.with_intercept
                continue

        if self.exec_context.max_steps is not None and self.k >= self.exec_context.max_steps:
            warnings.warn(
                "stepwise search has reached the maximum number of tries to "
                "find the best fit model"
            )

        filtered_models_ics = sorted(
            [
                (v, self.fit_time_dict[k], self.ic_dict[k])
                for k, v in self.results_dict.items()
                if v is not None
            ],
            key=lambda fit_ic: fit_ic[1],
        )
        sorted_fits = _sort_and_filter_fits(filtered_models_ics)
        if self.trace and sorted_fits:
            print(f"\nBest model: {sorted_fits[0]}")
        return sorted_fits
