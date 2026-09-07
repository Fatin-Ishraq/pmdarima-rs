#![allow(clippy::needless_range_loop)]
//! The Kalman filter, specialised to the SARIMAX structure.
//!
//! Two things make this cheaper than a general filter without changing a
//! single digit of the answer:
//!
//! 1. **The observation is univariate.** `Z P Z'` is a scalar, so the matrix
//!    inversion at the heart of a general filter becomes one divide, and the
//!    gain is a vector rather than a matrix.
//! 2. **The transition is sparse and structured.** Held as sparse rows,
//!    `T P T'` costs `O(k^2)` instead of `O(k^3)`.
//!
//! The initial covariance of the stationary block solves a discrete Lyapunov
//! equation. `statsmodels` reaches for a general solver, which builds and
//! factors a `k^2 x k^2` system - the reason `_initialize_state` is 12% of a
//! profile. Squaring converges quadratically and never leaves `k x k`.

use crate::ssm::{SparseRow, Spec, System};

pub const LN_2PI: f64 = 1.837_877_066_409_345_6;

/// Result of one filtering pass.
pub struct FilterOut {
    pub loglike: f64,
    /// One-step-ahead forecasts `Z a_t` (plus intercepts).
    pub fitted: Vec<f64>,
    /// Prediction errors `v_t`.
    pub resid: Vec<f64>,
    /// Prediction error variances `F_t`, in units of `sigma2`.
    pub fvar: Vec<f64>,
    /// `sum v_t^2 / F_t` over the non-burned observations.
    pub ssr: f64,
    /// `sum log F_t` over the non-burned observations.
    pub sum_log_f: f64,
    /// Number of observations contributing to the loglikelihood.
    pub n_eff: usize,
    /// The profiled-out scale, when `concentrate_scale` is set; else 1.
    pub scale: f64,
    /// Per-observation loglikelihood contributions, zero on burned periods.
    ///
    /// `statsmodels` reports standard errors from the outer product of
    /// gradients, which needs the score of each observation separately rather
    /// than only the total.
    pub llobs: Vec<f64>,
    /// Final state and covariance, for forecasting.
    pub a_final: Vec<f64>,
    pub p_final: Vec<f64>,
}

/// `M[i][j] = sum_c T[i][c] * P[c][j]`, exploiting sparsity of `T`.
fn t_times(t_rows: &[SparseRow], p: &[f64], k: usize, out: &mut [f64]) {
    out.fill(0.0);
    for (i, row) in t_rows.iter().enumerate() {
        let obase = i * k;
        for &(c, v) in row.iter() {
            let pbase = c * k;
            if v == 1.0 {
                for j in 0..k {
                    out[obase + j] += p[pbase + j];
                }
            } else {
                for j in 0..k {
                    out[obase + j] += v * p[pbase + j];
                }
            }
        }
    }
}

/// `out = T P T' + sigma2 * R R'`, given `m = T P`.
fn t_p_tt_plus_rqr(
    t_rows: &[SparseRow],
    m: &[f64],
    r_col: &[f64],
    sigma2: f64,
    k: usize,
    out: &mut [f64],
) {
    // (T P T')[i][j] = sum_c T[j][c] * (T P)[i][c]
    for i in 0..k {
        let mbase = i * k;
        for j in i..k {
            let mut s = 0.0;
            for &(c, v) in t_rows[j].iter() {
                s += v * m[mbase + c];
            }
            let val = s + sigma2 * r_col[i] * r_col[j];
            out[i * k + j] = val;
            out[j * k + i] = val;
        }
    }
}

/// Dense `C = A * B` for small square matrices.
fn matmul(a: &[f64], b: &[f64], n: usize, out: &mut [f64]) {
    out.fill(0.0);
    for i in 0..n {
        for l in 0..n {
            let av = a[i * n + l];
            if av == 0.0 {
                continue;
            }
            let bbase = l * n;
            let obase = i * n;
            for j in 0..n {
                out[obase + j] += av * b[bbase + j];
            }
        }
    }
}

/// Solve `P = A P A' + Q` for a stable `A`, by squaring.
///
/// Iterating `P <- P + A P A'`, `A <- A^2` accumulates
/// `sum_i A^i Q A'^i` with the number of terms doubling each round, so it
/// converges in `O(log)` iterations rather than the `O(k^6)` of forming and
/// factoring `(I - A (x) A)`.
pub fn solve_lyapunov(a_init: &[f64], q: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let mut p = q.to_vec();
    let mut a = a_init.to_vec();
    let mut tmp = vec![0.0; n * n];
    let mut tmp2 = vec![0.0; n * n];
    for _ in 0..64 {
        // P += A P A'
        matmul(&a, &p, n, &mut tmp); // tmp = A P
                                     // tmp2 = tmp * A'
        tmp2.fill(0.0);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..n {
                    s += tmp[i * n + l] * a[j * n + l];
                }
                tmp2[i * n + j] = s;
            }
        }
        let mut delta = 0.0;
        let mut scale = 0.0;
        for i in 0..n * n {
            p[i] += tmp2[i];
            delta += tmp2[i].abs();
            scale += p[i].abs();
        }
        // A <- A^2
        matmul(&a, &a, n, &mut tmp);
        a.copy_from_slice(&tmp);
        let anorm: f64 = a.iter().map(|v| v.abs()).sum();
        if delta <= 1e-15 * scale.max(1e-300) || anorm < 1e-300 || !anorm.is_finite() {
            break;
        }
    }
    // Symmetrise: the recursion is symmetric in exact arithmetic, and
    // enforcing it stops rounding error from accumulating asymmetry.
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (p[i * n + j] + p[j * n + i]);
            p[i * n + j] = v;
            p[j * n + i] = v;
        }
    }
    p
}

/// Initial state mean.
///
/// The differencing states start at zero (they are diffuse). The stationary
/// ARMA block starts at its unconditional mean, which `statsmodels` obtains
/// with a general `solve(I - T, c)`. For a companion `T` that system unrolls
/// analytically: with `mu = c / (1 - sum(phi))`,
/// `x_i = (1 - phi_1 - ... - phi_i) * mu - c`, so the whole thing is `O(r)`
/// instead of `O(r^3)`.
pub fn initial_state(spec: &Spec, sys: &System) -> Vec<f64> {
    let k = spec.k;
    let (kd, r) = (spec.kd, spec.r);
    let mut a0 = vec![0.0; k];
    if !spec.enforce_stationarity || r == 0 {
        return a0;
    }
    let c0 = sys.c_const[kd];
    if c0 == 0.0 {
        return a0;
    }
    let sum_phi: f64 = sys.red_ar.iter().take(r).sum();
    let denom = 1.0 - sum_phi;
    let mu = c0 / denom;
    a0[kd] = mu;
    let mut running = 0.0;
    for i in 1..r {
        running += sys.red_ar.get(i - 1).copied().unwrap_or(0.0);
        a0[kd + i] = (1.0 - running) * mu - c0;
    }
    a0
}

/// Build the initial state covariance: approximate-diffuse on the
/// differencing states, stationary on the ARMA block.
///
/// When stationarity is not enforced the model may be explosive, so
/// `statsmodels` makes *every* state approximately diffuse rather than
/// attempting a stationary solve. We follow that exactly.
pub fn initial_covariance(spec: &Spec, sys: &System, diffuse_variance: f64) -> Vec<f64> {
    let k = spec.k;
    let (kd, r) = (spec.kd, spec.r);
    let mut p0 = vec![0.0; k * k];
    if !spec.enforce_stationarity {
        for i in 0..k {
            p0[i * k + i] = diffuse_variance;
        }
        return p0;
    }
    for i in 0..kd {
        p0[i * k + i] = diffuse_variance;
    }
    if r > 0 {
        // The ARMA block of T, densified and re-based to the block's origin.
        let mut a = vec![0.0; r * r];
        for i in 0..r {
            for &(c, v) in sys.t_rows[kd + i].iter() {
                if c >= kd {
                    a[i * r + (c - kd)] = v;
                }
            }
        }
        // Q = sigma2 * R R' restricted to the ARMA block.
        let mut q = vec![0.0; r * r];
        for i in 0..r {
            for j in 0..r {
                q[i * r + j] = sys.sigma2 * sys.r_col[kd + i] * sys.r_col[kd + j];
            }
        }
        let block = solve_lyapunov(&a, &q, r);
        for i in 0..r {
            for j in 0..r {
                p0[(kd + i) * k + (kd + j)] = block[i * r + j];
            }
        }
    }
    p0
}

/// Run the filter over `y`, with `obs_intercept` already evaluated per period.
///
/// `y` may contain NaN for missing observations; those periods skip the
/// update and only propagate, which matches `statsmodels`' handling.
#[allow(clippy::too_many_arguments)]
pub fn kalman_filter(
    spec: &Spec,
    sys: &System,
    y: &[f64],
    obs_intercept: Option<&[f64]>,
    diffuse_variance: f64,
    want_paths: bool,
    tolerance: f64,
) -> FilterOut {
    let k = spec.k;
    let n = y.len();
    let mut a = initial_state(spec, sys);
    let mut p = initial_covariance(spec, sys, diffuse_variance);

    let mut a_filt = vec![0.0; k];
    let mut p_filt = vec![0.0; k * k];
    let mut m = vec![0.0; k];
    let mut tp = vec![0.0; k * k];

    // Steady-state short-circuit.
    //
    // For a time-invariant system the Riccati recursion converges, after
    // which `P`, `F` and the gain stop changing. statsmodels detects this
    // (`||P_t - P_{t+1}||_F^2 < tolerance`) and freezes them; matching that is
    // required for bit-level agreement, and it is also a large win in its own
    // right, because every step after convergence drops from `O(k^2)` to
    // `O(k)`. The system is only time-invariant when there is no trend and no
    // exogenous regressor - both enter as per-period intercepts.
    let time_invariant = spec.k_trend == 0 && spec.k_exog == 0;
    let mut converged = false;
    let mut conv_f = 0.0;
    let mut conv_m: Vec<f64> = Vec::new();
    let mut p_prev = vec![0.0; k * k];

    let mut loglike = 0.0;
    let mut ssr = 0.0;
    let mut sum_log_f = 0.0;
    let mut n_eff = 0usize;
    let mut fitted = if want_paths { vec![0.0; n] } else { Vec::new() };
    let mut resid = if want_paths { vec![0.0; n] } else { Vec::new() };
    let mut fvar = if want_paths { vec![0.0; n] } else { Vec::new() };
    let mut llobs = if want_paths { vec![0.0; n] } else { Vec::new() };

    for t in 0..n {
        // --- forecast: Z is a 0/1 vector, so Z a is a sum of entries ---
        let mut za = 0.0;
        for &i in sys.z_idx.iter() {
            za += a[i];
        }
        let d_t = obs_intercept.map_or(0.0, |o| o[t]);
        let forecast = za + d_t;
        let v = y[t] - forecast;

        // Only NaN marks a missing observation, exactly as `numpy.isnan`
        // decides it for statsmodels. An infinite `y` or an overflowed
        // forecast is *not* missing: it has to poison the likelihood rather
        // than be quietly skipped, or an explosive model scores as a good one.
        let missing = y[t].is_nan();

        // A missing observation invalidates the steady state: with no
        // observation to correct against, `P_{t|t} = P_t` rather than
        // `P_t - M M'/F`, so the Riccati recursion leaves its fixed point and
        // has to find it again. statsmodels drops the converged flag here and
        // re-converges a few periods later; matching that is what keeps the
        // likelihood equal when the series has holes.
        if missing {
            converged = false;
        }

        // --- M = P Z', F = Z P Z' ---
        // Once converged these are constant, so we skip the O(k) sweep and
        // reuse the frozen values, exactly as statsmodels does.
        let f;
        if converged {
            m.copy_from_slice(&conv_m);
            f = conv_f;
        } else {
            for i in 0..k {
                let base = i * k;
                let mut s = 0.0;
                for &j in sys.z_idx.iter() {
                    s += p[base + j];
                }
                m[i] = s;
            }
            let mut acc = 0.0;
            for &i in sys.z_idx.iter() {
                acc += m[i];
            }
            f = acc;
        }

        if want_paths {
            fitted[t] = forecast;
            resid[t] = v;
            fvar[t] = f;
        }

        if !missing && f > 0.0 {
            if t >= spec.burn {
                let ll_t = -0.5 * (LN_2PI + f.ln() + v * v / f);
                loglike += ll_t;
                if want_paths {
                    llobs[t] = ll_t;
                }
                ssr += v * v / f;
                sum_log_f += f.ln();
                n_eff += 1;
            }
            // --- contemporaneous update: rank-1, no inversion ---
            let vf = v / f;
            for i in 0..k {
                a_filt[i] = a[i] + m[i] * vf;
            }
            if !converged {
                for i in 0..k {
                    let mi = m[i] / f;
                    for j in i..k {
                        let val = p[i * k + j] - mi * m[j];
                        p_filt[i * k + j] = val;
                        p_filt[j * k + i] = val;
                    }
                }
            }
        } else {
            if !missing && t >= spec.burn {
                // A non-positive or NaN prediction variance is a degenerate
                // model, not a missing observation. Poison the likelihood so
                // the caller scores it as unusable instead of silently
                // dropping the period and reporting a better fit than it has.
                loglike = f64::NAN;
            }
            a_filt.copy_from_slice(&a);
            if !converged {
                p_filt.copy_from_slice(&p);
            }
        }

        // --- predict: a = T a_filt + c, P = T P_filt T' + sigma2 R R' ---
        for i in 0..k {
            // The intercept only ever occupies row `kd`; when the trend is
            // time-varying we read this period's value there.
            let mut s = match (&sys.c_time, i == spec.kd) {
                (Some(ct), true) => ct[t],
                _ => sys.c_const[i],
            };
            for &(c, val) in sys.t_rows[i].iter() {
                s += val * a_filt[c];
            }
            a[i] = s;
        }
        if !converged {
            p_prev.copy_from_slice(&p);
            t_times(&sys.t_rows, &p_filt, k, &mut tp);
            t_p_tt_plus_rqr(&sys.t_rows, &tp, &sys.r_col, sys.sigma2, k, &mut p);

            // statsmodels compares the squared Frobenius norm of
            // `P_t - P_{t+1}` against `tolerance`, skipping t = 0 and any
            // period adjacent to a missing observation.
            let adjacent_missing = missing || (t > 0 && y[t - 1].is_nan());
            if time_invariant && t >= 1 && !adjacent_missing {
                let mut sq = 0.0;
                for i in 0..k * k {
                    let d = p_prev[i] - p[i];
                    sq += d * d;
                }
                if sq < tolerance {
                    converged = true;
                    conv_f = f;
                    conv_m = m.clone();
                }
            }
        }
    }

    // With the scale concentrated out, the filter above ran at `sigma2 = 1`,
    // so `F_t` is in units of the scale and the maximising scale has a closed
    // form: `s = (1/n) sum v_t^2 / F_t`. statsmodels' FILTER_CONCENTRATED does
    // the same, and reports `params` without a `sigma2` entry.
    let mut scale = 1.0;
    if spec.concentrate_scale && loglike.is_finite() {
        if n_eff > 0 {
            scale = ssr / n_eff as f64;
            let n = n_eff as f64;
            loglike = -0.5 * (n * (LN_2PI + scale.ln() + 1.0) + sum_log_f);
            if want_paths {
                for t in 0..llobs.len() {
                    llobs[t] = if fvar[t] > 0.0 && !resid[t].is_nan() && t >= spec.burn {
                        -0.5 * (LN_2PI
                            + scale.ln()
                            + fvar[t].ln()
                            + resid[t] * resid[t] / (fvar[t] * scale))
                    } else {
                        0.0
                    };
                }
            }
        } else {
            loglike = 0.0;
        }
    }

    FilterOut {
        loglike,
        fitted,
        resid,
        fvar,
        llobs,
        ssr,
        sum_log_f,
        n_eff,
        scale,
        a_final: a,
        p_final: p,
    }
}

/// Forecast `h` periods beyond the end of `y`.
///
/// Once the filter has run, forecasting is the prediction step alone, with no
/// observation to correct against: the state mean and covariance simply
/// propagate. Returns the forecast mean and its variance for each horizon.
pub fn forecast(
    spec: &Spec,
    sys: &System,
    filtered: &FilterOut,
    horizon: usize,
    exog_future: Option<&[f64]>,
    nobs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let k = spec.k;
    let mut a = filtered.a_final.clone();
    let mut p = filtered.p_final.clone();
    let mut a_next = vec![0.0; k];
    let mut tp = vec![0.0; k * k];
    let mut p_next = vec![0.0; k * k];

    let mut mean = Vec::with_capacity(horizon);
    let mut var = Vec::with_capacity(horizon);

    for h in 0..horizon {
        // `a` and `p` already hold the one-step-ahead prediction for this
        // period, produced by the last iteration of the filter.
        let mut za = 0.0;
        for &i in sys.z_idx.iter() {
            za += a[i];
        }
        let d_t = exog_future.map_or(0.0, |e| e[h]);
        mean.push(za + d_t);

        let mut f = 0.0;
        for &i in sys.z_idx.iter() {
            for &j in sys.z_idx.iter() {
                f += p[i * k + j];
            }
        }
        var.push(f);

        if h + 1 == horizon {
            break;
        }
        for i in 0..k {
            // A time-varying trend continues past the sample, so the system
            // is built over `nobs + horizon` periods and indexed absolutely
            // here rather than repeating the last in-sample value.
            let mut s = match (&sys.c_time, i == spec.kd) {
                (Some(ct), true) => ct[(nobs + h).min(ct.len() - 1)],
                _ => sys.c_const[i],
            };
            for &(c, val) in sys.t_rows[i].iter() {
                s += val * a[c];
            }
            a_next[i] = s;
        }
        a.copy_from_slice(&a_next);
        t_times(&sys.t_rows, &p, k, &mut tp);
        t_p_tt_plus_rqr(&sys.t_rows, &tp, &sys.r_col, sys.sigma2, k, &mut p_next);
        p.copy_from_slice(&p_next);
    }
    (mean, var)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyapunov_solves_ar1() {
        // P = phi^2 P + sigma2  =>  P = sigma2 / (1 - phi^2)
        let phi = 0.6;
        let p = solve_lyapunov(&[phi], &[1.0], 1);
        assert!((p[0] - 1.0 / (1.0 - phi * phi)).abs() < 1e-12);
    }

    #[test]
    fn lyapunov_residual_is_zero() {
        let n = 3;
        let a = vec![0.5, 0.2, 0.0, 0.1, -0.3, 0.2, 0.0, 0.4, 0.1];
        let q = vec![1.0, 0.3, 0.0, 0.3, 2.0, 0.1, 0.0, 0.1, 1.5];
        let p = solve_lyapunov(&a, &q, n);
        let mut ap = vec![0.0; n * n];
        matmul(&a, &p, n, &mut ap);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..n {
                    s += ap[i * n + l] * a[j * n + l];
                }
                let resid = p[i * n + j] - (s + q[i * n + j]);
                assert!(resid.abs() < 1e-10, "residual {resid}");
            }
        }
    }
}
