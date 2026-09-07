//! pmdarima-rs: fast, drop-in ARIMA and auto-ARIMA for Python.
//!
//! The compiled surface is deliberately small. Only the Kalman filter and the
//! objective built on it run thousands of times per fitted model; everything
//! else - parsing orders, starting values, order search, the estimator API -
//! runs once per model and lives in Python, where it is easier to keep
//! faithful to `pmdarima`.
//!
//! Every entry point validates its arguments before touching the core. These
//! functions are reachable from Python, and a Rust panic crossing the FFI
//! boundary arrives as `pyo3_runtime.PanicException`, which inherits from
//! `BaseException` - so it slips past the `except (LinAlgError, ValueError)`
//! that `auto_arima` uses to skip a bad candidate and aborts the whole search.
//! A wrong shape has to be a `ValueError`.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod filter;
pub mod objective;
pub mod poly;
pub mod ssm;

use filter::kalman_filter;
use objective::{exog_intercept, loglike_and_grad, loglike_unconstrained, GradKind};
use ssm::Spec;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

type FilterPaths<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
);
type ForecastOut<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

/// The knobs every entry point shares, so the signatures stay readable.
#[derive(Clone, Copy)]
struct Opts {
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            enforce_stationarity: true,
            enforce_invertibility: true,
            concentrate_scale: false,
        }
    }
}

/// Build a [`Spec`], rejecting seasonal orders that have no period to attach
/// to. `statsmodels` raises for both of these, and so must we: with `s = 0`
/// the seasonal state block has a zero-length stride and the index arithmetic
/// that builds the transition matrix underflows.
///
/// The validation is kept separate from the Python exception it turns into,
/// so the unit tests can exercise it without an interpreter: a `cargo test`
/// binary is not linked against libpython under the `extension-module`
/// feature, and touching the runtime there fails to link.
fn checked_spec(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    trend_offset: f64,
    o: Opts,
) -> PyResult<Spec> {
    let (bp, bd, bq, s) = seasonal_order;
    let has_seasonal = bp > 0 || bd > 0 || bq > 0;
    if s == 0 && has_seasonal {
        return Err(PyValueError::new_err(
            "Must include nonzero seasonal periodicity if including seasonal \
             AR, MA, or differencing.",
        ));
    }
    if s == 1 && has_seasonal {
        return Err(PyValueError::new_err(
            "Seasonal periodicity must be greater than 1.",
        ));
    }
    Ok(Spec::new(
        order.0,
        order.1,
        order.2,
        bp,
        bd,
        bq,
        s,
        trend_powers,
        k_exog,
        trend_offset,
        o.enforce_stationarity,
        o.enforce_invertibility,
        o.concentrate_scale,
    ))
}

fn make_spec(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    trend_offset: f64,
    o: Opts,
) -> PyResult<Spec> {
    checked_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)
        .map_err(PyValueError::new_err)
}

fn check_params(spec: &Spec, params: &[f64], what: &str) -> PyResult<()> {
    if params.len() != spec.n_params() {
        return Err(PyValueError::new_err(format!(
            "{what} has {} elements, but this specification has {} parameters",
            params.len(),
            spec.n_params()
        )));
    }
    Ok(())
}

/// Borrow `exog` as a flat column-major slice, checking that it is contiguous
/// and has exactly `k_exog * n` entries.
fn exog_slice<'a>(
    exog: &'a Option<PyReadonlyArray1<'a, f64>>,
    k_exog: usize,
    n: usize,
    what: &str,
) -> PyResult<Option<&'a [f64]>> {
    match exog {
        None => Ok(None),
        Some(e) => {
            let s = e.as_slice().map_err(|_| {
                PyValueError::new_err(format!("{what} must be a contiguous float64 array"))
            })?;
            if s.len() != k_exog * n {
                return Err(PyValueError::new_err(format!(
                    "{what} has {} entries, expected k_exog * n = {}",
                    s.len(),
                    k_exog * n
                )));
            }
            Ok(Some(s))
        }
    }
}

/// Evaluate the loglikelihood at a *constrained* parameter vector.
///
/// This mirrors `statsmodels.SARIMAX.loglike`, and exists mainly so the test
/// suite can compare evaluation for evaluation rather than only comparing
/// final fits - a fit that lands in the right place can hide a filter that is
/// wrong everywhere else.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19
))]
#[allow(clippy::too_many_arguments)]
fn loglike(
    y: PyReadonlyArray1<f64>,
    params: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    exog: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, params, "params")?;
    let ex = exog_slice(&exog, k_exog, y.len(), "exog")?;
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let obs = exog_intercept(&spec, ex, &parts, y.len());
    let out = kalman_filter(
        &spec,
        &sys,
        y,
        obs.as_deref(),
        diffuse_variance,
        false,
        tolerance,
    );
    Ok(out.loglike)
}

/// Loglikelihood at an unconstrained parameter vector.
#[pyfunction]
#[pyo3(signature = (
    y, u, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19
))]
#[allow(clippy::too_many_arguments)]
fn loglike_u(
    y: PyReadonlyArray1<f64>,
    u: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    exog: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let y = y.as_slice()?;
    let u = u.as_slice()?;
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, u, "u")?;
    let ex = exog_slice(&exog, k_exog, y.len(), "exog")?;
    Ok(loglike_unconstrained(
        &spec,
        y,
        ex,
        u,
        diffuse_variance,
        tolerance,
    ))
}

/// Loglikelihood and gradient at an unconstrained parameter vector.
///
/// Returning both together is the point: it removes the `n_params + 1`
/// separate Python round trips that `scipy`'s finite differencing would
/// otherwise make per gradient.
#[pyfunction]
#[pyo3(signature = (
    y, u, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19,
    parallel=true, epsilon=None
))]
#[allow(clippy::too_many_arguments)]
fn loglike_grad<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    u: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    exog: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
    parallel: bool,
    epsilon: Option<f64>,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let yv = y.as_slice()?.to_vec();
    let uv = u.as_slice()?.to_vec();
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, &uv, "u")?;
    let ev = exog_slice(&exog, k_exog, yv.len(), "exog")?.map(|e| e.to_vec());
    // Release the GIL: the gradient fans out over cores and touches nothing
    // Python-owned while it runs.
    let kind = match epsilon {
        Some(e) if e > 0.0 => GradKind::Forward(e),
        _ => GradKind::Central,
    };
    let (f, g) = py.allow_threads(|| {
        loglike_and_grad(
            &spec,
            &yv,
            ev.as_deref(),
            &uv,
            diffuse_variance,
            tolerance,
            parallel,
            kind,
        )
    });
    Ok((f, g.to_pyarray(py)))
}

/// One-step-ahead forecasts, prediction errors and their variances.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19
))]
#[allow(clippy::too_many_arguments)]
fn filter_paths<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    params: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    exog: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<FilterPaths<'py>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, params, "params")?;
    let ex = exog_slice(&exog, k_exog, y.len(), "exog")?;
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let obs = exog_intercept(&spec, ex, &parts, y.len());
    let out = kalman_filter(
        &spec,
        &sys,
        y,
        obs.as_deref(),
        diffuse_variance,
        true,
        tolerance,
    );
    // With the scale concentrated out the filter runs at `sigma2 = 1`, so the
    // reported variances have to be put back on the data's scale.
    let mut fvar = out.fvar;
    if spec.concentrate_scale {
        for v in fvar.iter_mut() {
            *v *= out.scale;
        }
    }
    Ok((
        out.fitted.to_pyarray(py),
        out.resid.to_pyarray(py),
        fvar.to_pyarray(py),
        out.loglike,
    ))
}

/// Per-observation loglikelihood contributions.
///
/// Mirrors `statsmodels.SARIMAX.loglikeobs`, and exists for the same reason
/// statsmodels has it: standard errors come from the outer product of
/// gradients, which needs each observation's score, not just their sum.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19
))]
#[allow(clippy::too_many_arguments)]
fn loglikeobs<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    params: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    exog: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, params, "params")?;
    let ex = exog_slice(&exog, k_exog, y.len(), "exog")?;
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let obs = exog_intercept(&spec, ex, &parts, y.len());
    let out = kalman_filter(
        &spec,
        &sys,
        y,
        obs.as_deref(),
        diffuse_variance,
        true,
        tolerance,
    );
    Ok(out.llobs.to_pyarray(py))
}

/// Filter `y`, then forecast `horizon` periods beyond it.
///
/// Returns the in-sample one-step forecasts and residuals alongside the
/// out-of-sample mean and variance, because `pmdarima` callers routinely want
/// both and running the filter twice would be wasteful.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, horizon,
    exog=None, exog_future=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19
))]
#[allow(clippy::too_many_arguments)]
fn forecast<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    params: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    horizon: usize,
    exog: Option<PyReadonlyArray1<f64>>,
    exog_future: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<ForecastOut<'py>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let y = y.as_slice()?;
    let n = y.len();
    let params = params.as_slice()?;
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, params, "params")?;
    let ex = exog_slice(&exog, k_exog, n, "exog")?;
    let exf = exog_slice(&exog_future, k_exog, horizon, "exog_future")?;
    let parts = spec.split(params);
    // Build over n + horizon so a time-varying trend genuinely covers the
    // forecast period rather than repeating its last in-sample value.
    let sys = spec.build(&parts, n + horizon);
    let obs = exog_intercept(&spec, ex, &parts, n);
    let out = kalman_filter(
        &spec,
        &sys,
        y,
        obs.as_deref(),
        diffuse_variance,
        true,
        tolerance,
    );
    let obs_future = exf.map(|e| {
        let mut v = vec![0.0; horizon];
        for (h, slot) in v.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..spec.k_exog {
                acc += e[j * horizon + h] * parts.exog[j];
            }
            *slot = acc;
        }
        v
    });
    let (mean, mut var) = filter::forecast(&spec, &sys, &out, horizon, obs_future.as_deref(), n);
    let fitted = out.fitted;
    let resid = out.resid;
    // `fitted` is already on the data's scale; only the variances carry the
    // concentrated scale, so they are the only thing that needs rescaling.
    if spec.concentrate_scale {
        for v in var.iter_mut() {
            *v *= out.scale;
        }
    }
    Ok((
        fitted.to_pyarray(py),
        resid.to_pyarray(py),
        mean.to_pyarray(py),
        var.to_pyarray(py),
    ))
}

/// Apply the stationarity/invertibility reparametrisation.
#[pyfunction]
#[pyo3(signature = (
    u, order, seasonal_order, trend_powers, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true, concentrate_scale=false,
    trend_offset=1.0
))]
#[allow(clippy::too_many_arguments)]
fn transform_params<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    let u = u.as_slice()?;
    check_params(&spec, u, "u")?;
    Ok(spec.transform(u).to_pyarray(py))
}

/// Inverse of [`transform_params`].
#[pyfunction]
#[pyo3(signature = (
    p, order, seasonal_order, trend_powers, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true, concentrate_scale=false,
    trend_offset=1.0
))]
#[allow(clippy::too_many_arguments)]
fn untransform_params<'py>(
    py: Python<'py>,
    p: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    trend_offset: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    };
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    let p = p.as_slice()?;
    check_params(&spec, p, "params")?;
    Ok(spec.untransform(p).to_pyarray(py))
}

/// Derived state-space dimensions, so tests can assert the representation
/// matches statsmodels rather than only that the answer does.
#[pyfunction]
#[pyo3(signature = (order, seasonal_order, trend_powers, k_exog=0, trend_offset=1.0))]
fn spec_dims(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    trend_offset: f64,
) -> PyResult<(usize, usize, usize, usize, usize)> {
    let spec = make_spec(
        order,
        seasonal_order,
        trend_powers,
        k_exog,
        trend_offset,
        Opts::default(),
    )?;
    Ok((spec.k, spec.r, spec.kd, spec.burn, spec.n_params()))
}

/// The scale the filter profiles out when `concentrate_scale` is set.
///
/// `statsmodels` reports this as `SARIMAXResults.scale`, and it is what a
/// concentrated fit has instead of a `sigma2` parameter.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    trend_offset=1.0, diffuse_variance=1e6, tolerance=1e-19
))]
#[allow(clippy::too_many_arguments)]
fn concentrated_scale(
    y: PyReadonlyArray1<f64>,
    params: PyReadonlyArray1<f64>,
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    exog: Option<PyReadonlyArray1<f64>>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    trend_offset: f64,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale: true,
    };
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = make_spec(order, seasonal_order, trend_powers, k_exog, trend_offset, o)?;
    check_params(&spec, params, "params")?;
    let ex = exog_slice(&exog, k_exog, y.len(), "exog")?;
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let obs = exog_intercept(&spec, ex, &parts, y.len());
    let out = kalman_filter(
        &spec,
        &sys,
        y,
        obs.as_deref(),
        diffuse_variance,
        false,
        tolerance,
    );
    Ok(out.scale)
}

#[pymodule]
fn _pmdarima_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(loglike, m)?)?;
    m.add_function(wrap_pyfunction!(loglike_u, m)?)?;
    m.add_function(wrap_pyfunction!(loglike_grad, m)?)?;
    m.add_function(wrap_pyfunction!(filter_paths, m)?)?;
    m.add_function(wrap_pyfunction!(forecast, m)?)?;
    m.add_function(wrap_pyfunction!(loglikeobs, m)?)?;
    m.add_function(wrap_pyfunction!(transform_params, m)?)?;
    m.add_function(wrap_pyfunction!(untransform_params, m)?)?;
    m.add_function(wrap_pyfunction!(spec_dims, m)?)?;
    m.add_function(wrap_pyfunction!(concentrated_scale, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seasonal_order_without_a_period_is_rejected() {
        let o = Opts::default();
        assert!(checked_spec((1, 0, 0), (1, 0, 0, 0), vec![], 0, 1.0, o).is_err());
        assert!(checked_spec((1, 0, 0), (0, 1, 0, 0), vec![], 0, 1.0, o).is_err());
        assert!(checked_spec((1, 0, 0), (1, 0, 0, 1), vec![], 0, 1.0, o).is_err());
        assert!(checked_spec((1, 0, 0), (0, 0, 0, 0), vec![], 0, 1.0, o).is_ok());
        assert!(checked_spec((1, 0, 0), (0, 0, 0, 12), vec![], 0, 1.0, o).is_ok());
    }
}
