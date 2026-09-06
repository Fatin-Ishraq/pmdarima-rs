//! The SARIMAX state-space model and its Kalman filter.
//!
//! This reproduces the state space that `statsmodels` builds for
//! `SARIMAX(..., simple_differencing=False, hamilton_representation=False)`,
//! which is the form `pmdarima` always uses. The *representation* is
//! identical; the *arithmetic* is not, and that is where the speed comes from.
//!
//! `statsmodels` stores the transition as a dense `k x k` matrix and runs a
//! general multivariate filter over it, so every step costs `O(k^3)`. But the
//! transition of a SARIMAX model is almost entirely structure:
//!
//! * the ARMA block is a companion matrix - a superdiagonal identity plus one
//!   column of AR coefficients,
//! * the differencing rows are a handful of ones,
//! * the design vector is 0/1 with a few nonzeros,
//! * the selection matrix is a single column.
//!
//! Holding the transition as sparse rows makes `T P T'` cost `O(k^2)`, and the
//! observation update collapses from a matrix inversion to a scalar divide.
//! For a seasonal model with `m = 12` the state dimension is around 27, so
//! that is roughly an order of magnitude of arithmetic removed before a single
//! line has been micro-optimised.

use crate::poly::{constrain_stationary, polymul, seasonal_poly, simple_poly};

/// `statsmodels` evaluates trend terms at `t + 1`, not `t`.
pub const TREND_OFFSET: usize = 1;

/// A sparse row of the transition matrix: `(column, value)` pairs.
pub type SparseRow = Vec<(usize, f64)>;

/// Everything about a SARIMAX model that does not depend on the parameters.
#[derive(Clone, Debug)]
pub struct Spec {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    pub bp: usize,
    pub bd: usize,
    pub bq: usize,
    pub s: usize,
    /// Number of trend parameters (1 for `trend='c'`, 2 for `'ct'`, ...).
    pub k_trend: usize,
    /// Exponents of the trend polynomial actually included: `[0]` for `'c'`,
    /// `[0, 1]` for `'ct'`, `[1]` for `'t'`.
    pub trend_powers: Vec<usize>,
    pub k_exog: usize,
    pub enforce_stationarity: bool,
    pub enforce_invertibility: bool,
    pub concentrate_scale: bool,
    // ---- derived ----
    /// Reduced (multiplied-out) AR degree: `p + s*P`.
    pub k_ar_red: usize,
    /// Reduced MA degree: `q + s*Q`.
    pub k_ma_red: usize,
    /// `_k_order` in statsmodels: the ARMA state block size.
    pub r: usize,
    /// `_k_states_diff`: number of differencing states, `d + s*D`.
    pub kd: usize,
    /// Total state dimension.
    pub k: usize,
    /// Observations skipped in the loglikelihood (the diffuse ones).
    pub burn: usize,
}

impl Spec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        bp: usize,
        bd: usize,
        bq: usize,
        s: usize,
        trend_powers: Vec<usize>,
        k_exog: usize,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
        concentrate_scale: bool,
    ) -> Self {
        let k_ar_red = p + bp * s;
        let k_ma_red = q + bq * s;
        let r = k_ar_red.max(k_ma_red + 1);
        let kd = d + bd * s;
        let k = r + kd;
        Spec {
            p,
            d,
            q,
            bp,
            bd,
            bq,
            s,
            k_trend: trend_powers.len(),
            trend_powers,
            k_exog,
            enforce_stationarity,
            enforce_invertibility,
            concentrate_scale,
            k_ar_red,
            k_ma_red,
            r,
            kd,
            k,
            // statsmodels: k_diffuse_states = k_states, minus _k_order only
            // when stationarity is enforced (otherwise nothing is stationary).
            burn: if enforce_stationarity { kd } else { k },
        }
    }

    /// Number of free parameters, in `statsmodels` order:
    /// trend, exog, ar, ma, seasonal ar, seasonal ma, sigma2.
    pub fn n_params(&self) -> usize {
        self.k_trend
            + self.k_exog
            + self.p
            + self.q
            + self.bp
            + self.bq
            + usize::from(!self.concentrate_scale)
    }

    pub fn param_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for &pw in &self.trend_powers {
            names.push(match pw {
                0 => "intercept".to_string(),
                1 => "drift".to_string(),
                n => format!("trend.{n}"),
            });
        }
        for i in 0..self.k_exog {
            names.push(format!("x{}", i + 1));
        }
        for i in 0..self.p {
            names.push(format!("ar.L{}", i + 1));
        }
        for i in 0..self.q {
            names.push(format!("ma.L{}", i + 1));
        }
        for i in 0..self.bp {
            names.push(format!("ar.S.L{}", (i + 1) * self.s));
        }
        for i in 0..self.bq {
            names.push(format!("ma.S.L{}", (i + 1) * self.s));
        }
        if !self.concentrate_scale {
            names.push("sigma2".to_string());
        }
        names
    }

    /// Offset of the ARMA block within the parameter vector.
    pub fn arma_offset(&self) -> usize {
        self.k_trend + self.k_exog
    }

    /// Apply the stationarity/invertibility reparametrisation, mirroring
    /// `statsmodels.SARIMAX.transform_params`.
    pub fn transform(&self, unconstrained: &[f64]) -> Vec<f64> {
        let mut out = unconstrained.to_vec();
        let mut i = self.arma_offset();
        if self.enforce_stationarity && self.p > 0 {
            out[i..i + self.p]
                .copy_from_slice(&constrain_stationary(&unconstrained[i..i + self.p]));
        }
        i += self.p;
        if self.enforce_invertibility && self.q > 0 {
            // statsmodels negates the MA block so the stored convention is
            // `1 + theta_1 L + ...` rather than the AR sign convention.
            let c = constrain_stationary(&unconstrained[i..i + self.q]);
            for (o, v) in out[i..i + self.q].iter_mut().zip(c.iter()) {
                *o = -*v;
            }
        }
        i += self.q;
        if self.enforce_stationarity && self.bp > 0 {
            out[i..i + self.bp]
                .copy_from_slice(&constrain_stationary(&unconstrained[i..i + self.bp]));
        }
        i += self.bp;
        if self.enforce_invertibility && self.bq > 0 {
            let c = constrain_stationary(&unconstrained[i..i + self.bq]);
            for (o, v) in out[i..i + self.bq].iter_mut().zip(c.iter()) {
                *o = -*v;
            }
        }
        i += self.bq;
        if !self.concentrate_scale {
            out[i] = unconstrained[i] * unconstrained[i];
        }
        out
    }

    /// Inverse of [`Spec::transform`].
    pub fn untransform(&self, constrained: &[f64]) -> Vec<f64> {
        use crate::poly::unconstrain_stationary;
        let mut out = constrained.to_vec();
        let mut i = self.arma_offset();
        if self.enforce_stationarity && self.p > 0 {
            out[i..i + self.p]
                .copy_from_slice(&unconstrain_stationary(&constrained[i..i + self.p]));
        }
        i += self.p;
        if self.enforce_invertibility && self.q > 0 {
            let neg: Vec<f64> = constrained[i..i + self.q].iter().map(|v| -v).collect();
            out[i..i + self.q].copy_from_slice(&unconstrain_stationary(&neg));
        }
        i += self.q;
        if self.enforce_stationarity && self.bp > 0 {
            out[i..i + self.bp]
                .copy_from_slice(&unconstrain_stationary(&constrained[i..i + self.bp]));
        }
        i += self.bp;
        if self.enforce_invertibility && self.bq > 0 {
            let neg: Vec<f64> = constrained[i..i + self.bq].iter().map(|v| -v).collect();
            out[i..i + self.bq].copy_from_slice(&unconstrain_stationary(&neg));
        }
        i += self.bq;
        if !self.concentrate_scale {
            out[i] = constrained[i].max(0.0).sqrt();
        }
        out
    }
}

/// A parameter vector split into its named blocks.
pub struct Parts<'a> {
    pub trend: &'a [f64],
    pub exog: &'a [f64],
    pub ar: &'a [f64],
    pub ma: &'a [f64],
    pub sar: &'a [f64],
    pub sma: &'a [f64],
    pub sigma2: f64,
}

impl Spec {
    pub fn split<'a>(&self, params: &'a [f64]) -> Parts<'a> {
        let mut i = 0;
        let trend = &params[i..i + self.k_trend];
        i += self.k_trend;
        let exog = &params[i..i + self.k_exog];
        i += self.k_exog;
        let ar = &params[i..i + self.p];
        i += self.p;
        let ma = &params[i..i + self.q];
        i += self.q;
        let sar = &params[i..i + self.bp];
        i += self.bp;
        let sma = &params[i..i + self.bq];
        i += self.bq;
        let sigma2 = if self.concentrate_scale {
            1.0
        } else {
            params[i]
        };
        Parts {
            trend,
            exog,
            ar,
            ma,
            sar,
            sma,
            sigma2,
        }
    }
}

/// Parameter-dependent state-space matrices, in structured form.
pub struct System {
    /// Sparse rows of the transition matrix.
    pub t_rows: Vec<SparseRow>,
    /// Indices where the design vector is 1 (all its nonzeros are 1).
    pub z_idx: Vec<usize>,
    /// Selection column `R` (length `k`).
    pub r_col: Vec<f64>,
    /// Time-invariant state intercept.
    pub c_const: Vec<f64>,
    /// Time-varying state intercept at row `kd`, when the trend has a term of
    /// degree 1 or higher. `statsmodels` always stores the intercept as a
    /// `k x nobs` array; we only materialise a vector when it actually varies,
    /// so the common `trend='c'` path stays a single scalar add.
    pub c_time: Option<Vec<f64>>,
    pub sigma2: f64,
    /// Reduced AR coefficients, i.e. `reduced_polynomial_ar[1..]`.
    pub red_ar: Vec<f64>,
    /// Reduced MA coefficients, i.e. `reduced_polynomial_ma[1..]`.
    pub red_ma: Vec<f64>,
}

impl Spec {
    /// Build the structured system matrices for one parameter vector.
    ///
    /// `nobs` is needed only to expand a time-varying trend; for `trend='c'`
    /// (the path `auto_arima` takes) nothing of size `nobs` is allocated.
    pub fn build(&self, parts: &Parts, nobs: usize) -> System {
        // Reduced-form lag polynomials: multiply the seasonal and
        // non-seasonal operators out, exactly as statsmodels does.
        let poly_ar = simple_poly(&parts.ar.iter().map(|v| -v).collect::<Vec<_>>());
        let poly_sar = seasonal_poly(&parts.sar.iter().map(|v| -v).collect::<Vec<_>>(), self.s);
        let poly_ma = simple_poly(parts.ma);
        let poly_sma = seasonal_poly(parts.sma, self.s);

        let red_ar_full = if self.bp > 0 {
            polymul(&poly_ar, &poly_sar)
        } else {
            poly_ar
        };
        let red_ma_full = if self.bq > 0 {
            polymul(&poly_ma, &poly_sma)
        } else {
            poly_ma
        };
        // statsmodels stores `-polynomial_ar` and then reads `[1:]`.
        let red_ar: Vec<f64> = red_ar_full[1..].iter().map(|v| -v).collect();
        let red_ma: Vec<f64> = red_ma_full[1..].to_vec();

        let (k, kd, r, d, bd, s) = (self.k, self.kd, self.r, self.d, self.bd, self.s);
        let mut t_rows: Vec<SparseRow> = vec![Vec::new(); k];

        // --- differencing rows: upper triangle of ones, plus accumulators ---
        for i in 0..d {
            for j in i..d {
                t_rows[i].push((j, 1.0));
            }
            // ([0]*(s-1) + [1]) repeated D times, laid over columns d..kd
            for dd in 0..bd {
                t_rows[i].push((d + (dd + 1) * s - 1, 1.0));
            }
            if r > 0 {
                t_rows[i].push((kd, 1.0));
            }
        }

        // --- seasonal differencing rows ---
        for dd in 0..bd {
            let start = d + dd * s;
            let end = d + (dd + 1) * s;
            // Seasonal companion: row 0 picks up the last column of its own
            // block, and a subdiagonal identity shifts the rest along.
            t_rows[start].push((end - 1, 1.0));
            for i in 1..s {
                t_rows[start + i].push((start + i - 1, 1.0));
            }
            if dd + 1 < bd {
                t_rows[start].push((end + s - 1, 1.0));
            }
            if r > 0 {
                t_rows[start].push((kd, 1.0));
            }
        }

        // --- ARMA companion block: first column plus superdiagonal ---
        for i in 0..r {
            if i < red_ar.len() && red_ar[i] != 0.0 {
                t_rows[kd + i].push((kd, red_ar[i]));
            }
            if i + 1 < r {
                t_rows[kd + i].push((kd + i + 1, 1.0));
            }
        }

        // --- design ---
        let mut z_idx = Vec::new();
        for i in 0..d {
            z_idx.push(i);
        }
        for dd in 0..bd {
            z_idx.push(d + (dd + 1) * s - 1);
        }
        if r > 0 {
            z_idx.push(kd);
        }
        if z_idx.is_empty() {
            z_idx.push(0);
        }

        // --- selection ---
        let mut r_col = vec![0.0; k];
        if r > 0 {
            r_col[kd] = 1.0;
            for (i, &v) in red_ma.iter().enumerate() {
                if kd + 1 + i < k {
                    r_col[kd + 1 + i] = v;
                }
            }
        }

        // --- state intercept ---
        // The trend enters at row `kd`, the head of the stationary block.
        // statsmodels evaluates `trend_data[t] = (t + offset)^power` with
        // `trend_offset = 1`, so `trend='c'` is invariant and anything with a
        // positive power is not.
        let mut c_const = vec![0.0; k];
        let mut c_time = None;
        if self.k_trend > 0 && r > 0 {
            let varying = self.trend_powers.iter().any(|&pw| pw > 0);
            if varying {
                let mut v = vec![0.0; nobs];
                for (t, slot) in v.iter_mut().enumerate() {
                    let base = (t + TREND_OFFSET) as f64;
                    let mut acc = 0.0;
                    for (j, &pw) in self.trend_powers.iter().enumerate() {
                        acc += parts.trend[j] * base.powi(pw as i32);
                    }
                    *slot = acc;
                }
                // The initial state mean is built from the t = 0 column.
                c_const[kd] = v.first().copied().unwrap_or(0.0);
                c_time = Some(v);
            } else if let Some(pos) = self.trend_powers.iter().position(|&pw| pw == 0) {
                c_const[kd] = parts.trend[pos];
            }
        }

        System {
            t_rows,
            z_idx,
            r_col,
            c_const,
            c_time,
            sigma2: parts.sigma2,
            red_ar,
            red_ma,
        }
    }
}
