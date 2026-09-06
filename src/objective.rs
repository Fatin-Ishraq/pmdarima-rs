//! The objective the optimiser actually sees.
//!
//! `statsmodels` hands `scipy` a likelihood and lets it build the gradient by
//! finite differences in Python. That costs `n_params + 1` full Kalman filter
//! passes per gradient, and a profile of one seasonal `auto_arima` fit shows
//! the consequence plainly: 12,193 likelihood evaluations for 1,241
//! gradients, with `scipy._numdiff._dense_difference` sitting on 147 seconds
//! of cumulative time.
//!
//! Two things change here.
//!
//! * The whole gradient is computed in one call, so the per-evaluation Python
//!   round trip disappears.
//! * The perturbed evaluations are independent, so they run on all cores.
//!
//! The differencing is *central* rather than forward. That doubles the number
//! of passes but buys roughly eight extra digits of gradient accuracy, and an
//! accurate gradient is worth more than a cheap one here: L-BFGS builds its
//! curvature estimate from differences of gradients, so noise in the gradient
//! costs iterations. Since the passes are parallel, the extra work is close
//! to free in wall-clock terms.

use crate::filter::kalman_filter;
use crate::ssm::Spec;
use rayon::prelude::*;

/// Evaluate the loglikelihood at an *unconstrained* parameter vector.
pub fn loglike_unconstrained(
    spec: &Spec,
    y: &[f64],
    exog: Option<&[f64]>,
    u: &[f64],
    diffuse_variance: f64,
    tolerance: f64,
) -> f64 {
    let params = spec.transform(u);
    let parts = spec.split(&params);
    let sys = spec.build(&parts, y.len());
    let obs = exog_intercept(spec, exog, &parts, y.len());
    let out = kalman_filter(
        spec,
        &sys,
        y,
        obs.as_deref(),
        diffuse_variance,
        false,
        tolerance,
    );
    if out.loglike.is_finite() {
        out.loglike
    } else {
        f64::NEG_INFINITY
    }
}

/// Exogenous regressors enter as a time-varying observation intercept, which
/// is equivalent to subtracting `X beta` from the series.
pub fn exog_intercept(
    spec: &Spec,
    exog: Option<&[f64]>,
    parts: &crate::ssm::Parts,
    n: usize,
) -> Option<Vec<f64>> {
    let exog = exog?;
    if spec.k_exog == 0 {
        return None;
    }
    let mut out = vec![0.0; n];
    for (t, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for j in 0..spec.k_exog {
            acc += exog[j * n + t] * parts.exog[j];
        }
        *slot = acc;
    }
    Some(out)
}

/// The step used for each coordinate of the central difference.
///
/// `cbrt(eps)` is the step that balances truncation against round-off for a
/// central difference, just as `sqrt(eps)` is for a forward one. Scaling by
/// `max(1, |x|)` keeps it meaningful for both small and large coordinates.
fn step_for(x: f64) -> f64 {
    const CBRT_EPS: f64 = 6.055_454_452_393_343e-6; // f64::EPSILON.cbrt()
    CBRT_EPS * x.abs().max(1.0)
}

/// How the gradient is approximated.
#[derive(Clone, Copy, PartialEq)]
pub enum GradKind {
    /// Central differences with a per-coordinate step. More accurate.
    Central,
    /// Forward differences with a fixed absolute step - what `statsmodels`
    /// asks `scipy` for (`approx_grad=True, epsilon=1e-5`).
    ///
    /// This exists to *match*, not to be good. ARIMA likelihoods routinely
    /// have their maximum at the invertibility boundary, and `auto_arima`
    /// discards any model whose fitted inverse roots exceed 0.99. A more
    /// accurate gradient climbs closer to that boundary and so gets more
    /// models discarded - which changes which order is selected. Reproducing
    /// the reference optimiser's blunter gradient is what keeps the search
    /// on the same path.
    Forward(f64),
}

/// Loglikelihood and its gradient with respect to the unconstrained
/// parameters, in one call.
///
/// `parallel` is a choice, not a default: `auto_arima` already runs candidate
/// models across cores, and nesting a parallel gradient inside that would
/// oversubscribe. Fitting a single model has no such outer loop, so it takes
/// the cores.
pub fn loglike_and_grad(
    spec: &Spec,
    y: &[f64],
    exog: Option<&[f64]>,
    u: &[f64],
    diffuse_variance: f64,
    tolerance: f64,
    parallel: bool,
    kind: GradKind,
) -> (f64, Vec<f64>) {
    let f0 = loglike_unconstrained(spec, y, exog, u, diffuse_variance, tolerance);
    let n = u.len();

    let one = |i: usize| -> f64 {
        match kind {
            GradKind::Forward(eps) => {
                let mut up = u.to_vec();
                up[i] += eps;
                let fu = loglike_unconstrained(spec, y, exog, &up, diffuse_variance, tolerance);
                if fu.is_finite() && f0.is_finite() {
                    (fu - f0) / eps
                } else {
                    0.0
                }
            }
            GradKind::Central => {
                let h = step_for(u[i]);
                let mut up = u.to_vec();
                let mut dn = u.to_vec();
                up[i] += h;
                dn[i] -= h;
                let fu = loglike_unconstrained(spec, y, exog, &up, diffuse_variance, tolerance);
                let fd = loglike_unconstrained(spec, y, exog, &dn, diffuse_variance, tolerance);
                if fu.is_finite() && fd.is_finite() {
                    (fu - fd) / (2.0 * h)
                } else if fu.is_finite() {
                    (fu - f0) / h
                } else if fd.is_finite() {
                    (f0 - fd) / h
                } else {
                    0.0
                }
            }
        }
    };

    let grad: Vec<f64> = if parallel && n > 1 {
        (0..n).into_par_iter().map(one).collect()
    } else {
        (0..n).map(one).collect()
    };
    (f0, grad)
}
