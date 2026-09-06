//! pmdarima-rs: fast, drop-in ARIMA and auto-ARIMA for Python.
//!
//! The compiled surface is deliberately small. Only the Kalman filter and the
//! objective built on it run thousands of times per fitted model; everything
//! else - parsing orders, starting values, order search, the estimator API -
//! runs once per model and lives in Python, where it is easier to keep
//! faithful to `pmdarima`.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

pub mod filter;
pub mod objective;
pub mod poly;
pub mod ssm;

use filter::kalman_filter;
use objective::{exog_intercept, loglike_and_grad, loglike_unconstrained};
use ssm::Spec;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The knobs every entry point shares, so the signatures stay readable.
#[derive(Clone, Copy)]
struct Opts {
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    diffuse_variance: f64,
    tolerance: f64,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            enforce_stationarity: true,
            enforce_invertibility: true,
            concentrate_scale: false,
            diffuse_variance: 1e6,
            tolerance: 1e-19,
        }
    }
}

fn build_spec(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    o: Opts,
) -> Spec {
    Spec::new(
        order.0,
        order.1,
        order.2,
        seasonal_order.0,
        seasonal_order.1,
        seasonal_order.2,
        seasonal_order.3,
        trend_powers,
        k_exog,
        o.enforce_stationarity,
        o.enforce_invertibility,
        o.concentrate_scale,
    )
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
    concentrate_scale=false, diffuse_variance=1e6, tolerance=1e-19
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
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        diffuse_variance,
        tolerance,
    };
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, o);
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let ex = exog.as_ref().map(|e| e.as_slice().unwrap());
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
    concentrate_scale=false, diffuse_variance=1e6, tolerance=1e-19
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
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let y = y.as_slice()?;
    let u = u.as_slice()?;
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        diffuse_variance,
        tolerance,
    };
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, o);
    let ex = exog.as_ref().map(|e| e.as_slice().unwrap());
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
    concentrate_scale=false, diffuse_variance=1e6, tolerance=1e-19,
    parallel=true
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
    diffuse_variance: f64,
    tolerance: f64,
    parallel: bool,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        diffuse_variance,
        tolerance,
    };
    let yv = y.as_slice()?.to_vec();
    let uv = u.as_slice()?.to_vec();
    let ev = exog.as_ref().map(|e| e.as_slice().unwrap().to_vec());
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, o);
    // Release the GIL: the gradient fans out over cores and touches nothing
    // Python-owned while it runs.
    let (f, g) = py.allow_threads(|| {
        loglike_and_grad(
            &spec,
            &yv,
            ev.as_deref(),
            &uv,
            diffuse_variance,
            tolerance,
            parallel,
        )
    });
    Ok((f, g.to_pyarray(py)))
}

/// One-step-ahead forecasts, prediction errors and their variances.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, exog=None, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true,
    concentrate_scale=false, diffuse_variance=1e6, tolerance=1e-19
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
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    f64,
)> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        diffuse_variance,
        tolerance,
    };
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, o);
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let ex = exog.as_ref().map(|e| e.as_slice().unwrap());
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
    Ok((
        out.fitted.to_pyarray(py),
        out.resid.to_pyarray(py),
        out.fvar.to_pyarray(py),
        out.loglike,
    ))
}

/// Apply the stationarity/invertibility reparametrisation.
#[pyfunction]
#[pyo3(signature = (
    u, order, seasonal_order, trend_powers, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true, concentrate_scale=false
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
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        ..Default::default()
    };
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, o);
    Ok(spec.transform(u.as_slice()?).to_pyarray(py))
}

/// Inverse of [`transform_params`].
#[pyfunction]
#[pyo3(signature = (
    p, order, seasonal_order, trend_powers, k_exog=0,
    enforce_stationarity=true, enforce_invertibility=true, concentrate_scale=false
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
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let o = Opts {
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
        ..Default::default()
    };
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, o);
    Ok(spec.untransform(p.as_slice()?).to_pyarray(py))
}

/// Derived state-space dimensions, so tests can assert the representation
/// matches statsmodels rather than only that the answer does.
#[pyfunction]
#[pyo3(signature = (order, seasonal_order, trend_powers, k_exog=0))]
fn spec_dims(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
) -> (usize, usize, usize, usize, usize) {
    let spec = build_spec(order, seasonal_order, trend_powers, k_exog, Opts::default());
    (spec.k, spec.r, spec.kd, spec.burn, spec.n_params())
}

#[pymodule]
fn _pmdarima_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(loglike, m)?)?;
    m.add_function(wrap_pyfunction!(loglike_u, m)?)?;
    m.add_function(wrap_pyfunction!(loglike_grad, m)?)?;
    m.add_function(wrap_pyfunction!(filter_paths, m)?)?;
    m.add_function(wrap_pyfunction!(transform_params, m)?)?;
    m.add_function(wrap_pyfunction!(untransform_params, m)?)?;
    m.add_function(wrap_pyfunction!(spec_dims, m)?)?;
    Ok(())
}
