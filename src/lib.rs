//! pmdarima-rs: fast, drop-in ARIMA and auto-ARIMA for Python.

use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

pub mod filter;
pub mod poly;
pub mod ssm;

use filter::kalman_filter;
use ssm::Spec;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[allow(clippy::too_many_arguments)]
fn make_spec(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
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
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    )
}

/// Evaluate the SARIMAX loglikelihood at `params` (already transformed).
///
/// Exposed mainly so the test suite can compare against `statsmodels`
/// evaluation for evaluation, rather than only comparing final fits.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers, exog=None,
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
    enforce_stationarity: bool,
    enforce_invertibility: bool,
    concentrate_scale: bool,
    diffuse_variance: f64,
    tolerance: f64,
) -> PyResult<f64> {
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let k_exog = usize::from(exog.is_some());
    let spec = make_spec(
        order,
        seasonal_order,
        trend_powers,
        k_exog,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    );
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let obs_int = exog.as_ref().map(|e| {
        let e = e.as_slice().unwrap();
        e.iter().map(|v| v * parts.exog[0]).collect::<Vec<f64>>()
    });
    let out = kalman_filter(
        &spec,
        &sys,
        y,
        obs_int.as_deref(),
        diffuse_variance,
        false,
        tolerance,
    );
    Ok(out.loglike)
}

/// One-step-ahead prediction errors and their variances.
#[pyfunction]
#[pyo3(signature = (
    y, params, order, seasonal_order, trend_powers,
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
    let y = y.as_slice()?;
    let params = params.as_slice()?;
    let spec = make_spec(
        order,
        seasonal_order,
        trend_powers,
        0,
        enforce_stationarity,
        enforce_invertibility,
        concentrate_scale,
    );
    let parts = spec.split(params);
    let sys = spec.build(&parts, y.len());
    let out = kalman_filter(&spec, &sys, y, None, diffuse_variance, true, tolerance);
    Ok((
        out.fitted.to_pyarray(py),
        out.resid.to_pyarray(py),
        out.fvar.to_pyarray(py),
        out.loglike,
    ))
}

/// Report the derived state-space dimensions, so tests can assert that the
/// representation matches statsmodels rather than only that the answer does.
#[pyfunction]
#[pyo3(signature = (order, seasonal_order, trend_powers, k_exog=0))]
fn spec_dims(
    order: (usize, usize, usize),
    seasonal_order: (usize, usize, usize, usize),
    trend_powers: Vec<usize>,
    k_exog: usize,
) -> (usize, usize, usize, usize, usize) {
    let spec = make_spec(
        order,
        seasonal_order,
        trend_powers,
        k_exog,
        true,
        true,
        false,
    );
    (spec.k, spec.r, spec.kd, spec.burn, spec.n_params())
}

#[pymodule]
fn _pmdarima_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(loglike, m)?)?;
    m.add_function(wrap_pyfunction!(filter_paths, m)?)?;
    m.add_function(wrap_pyfunction!(spec_dims, m)?)?;
    Ok(())
}
