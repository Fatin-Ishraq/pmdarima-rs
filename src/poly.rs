//! Lag polynomials and the stationarity/invertibility reparametrisation.
//!
//! These mirror `statsmodels.tsa.statespace.tools` exactly. They are small,
//! but they sit inside the optimiser's inner loop and they define the
//! parameter space, so any deviation here changes the answer rather than just
//! the speed.

/// Multiply two polynomials given lowest-order-first.
///
/// `numpy.polymul` documents highest-order-first, but multiplication is
/// symmetric under reversal, so `statsmodels` uses it lowest-first and so do
/// we.
pub fn polymul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Build the seasonal lag polynomial `1 + c_1 L^s + c_2 L^{2s} + ...`.
pub fn seasonal_poly(coefs: &[f64], periods: usize) -> Vec<f64> {
    let mut p = vec![0.0; coefs.len() * periods + 1];
    p[0] = 1.0;
    for (i, &c) in coefs.iter().enumerate() {
        p[(i + 1) * periods] = c;
    }
    p
}

/// Build the non-seasonal lag polynomial `1 + c_1 L + c_2 L^2 + ...`.
pub fn simple_poly(coefs: &[f64]) -> Vec<f64> {
    let mut p = Vec::with_capacity(coefs.len() + 1);
    p.push(1.0);
    p.extend_from_slice(coefs);
    p
}

/// Map an unconstrained vector to stationary AR coefficients.
///
/// This is Monahan's (1984) partial-autocorrelation reparametrisation, the
/// same one `statsmodels` applies in `constrain_stationary_univariate`. The
/// optimiser works in the unconstrained space; the filter always sees
/// coefficients whose characteristic roots lie outside the unit circle, so no
/// likelihood evaluation is ever wasted on an explosive model.
pub fn constrain_stationary(unconstrained: &[f64]) -> Vec<f64> {
    let n = unconstrained.len();
    if n == 0 {
        return Vec::new();
    }
    let r: Vec<f64> = unconstrained
        .iter()
        .map(|&u| u / (1.0 + u * u).sqrt())
        .collect();
    // `y` is built row by row; only the previous row is ever read.
    let mut prev = vec![0.0; n];
    let mut cur = vec![0.0; n];
    for k in 0..n {
        for i in 0..k {
            cur[i] = prev[i] + r[k] * prev[k - i - 1];
        }
        cur[k] = r[k];
        std::mem::swap(&mut prev, &mut cur);
    }
    prev.iter().take(n).map(|v| -v).collect()
}

/// Inverse of [`constrain_stationary`].
pub fn unconstrain_stationary(constrained: &[f64]) -> Vec<f64> {
    let n = constrained.len();
    if n == 0 {
        return Vec::new();
    }
    let mut y = vec![vec![0.0; n]; n];
    for i in 0..n {
        y[n - 1][i] = -constrained[i];
    }
    // Undo the Levinson-Durbin style recursion from the last row upward.
    for k in (1..n).rev() {
        let rk = y[k][k];
        let denom = 1.0 - rk * rk;
        for i in 0..k {
            y[k - 1][i] = (y[k][i] - rk * y[k][k - i - 1]) / denom;
        }
    }
    (0..n)
        .map(|k| {
            let rk = y[k][k];
            rk / (1.0 - rk * rk).sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polymul_matches_convolution() {
        assert_eq!(
            polymul(&[1.0, 0.4], &[1.0, 0.0, 0.0, 0.0, -0.5]),
            vec![1.0, 0.4, 0.0, 0.0, -0.5, -0.2]
        );
    }

    #[test]
    fn constrain_roundtrip() {
        for raw in [vec![0.5], vec![0.3, -0.2], vec![0.9, -0.4, 0.15]] {
            let c = constrain_stationary(&raw);
            let back = unconstrain_stationary(&c);
            for (a, b) in raw.iter().zip(back.iter()) {
                assert!((a - b).abs() < 1e-10, "{a} vs {b}");
            }
        }
    }
}
