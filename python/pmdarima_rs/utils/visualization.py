"""Plotting helpers, matching `pmdarima.utils.visualization`.

`matplotlib` is imported inside the functions, not at module scope, so it
stays an optional dependency: importing `pmdarima_rs` never requires it.
"""

import numpy as np

__all__ = [
    "autocorr_plot",
    "decomposed_plot",
    "plot_acf",
    "plot_pacf",
    "tsdisplay",
]


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting; install it with "
            "`pip install matplotlib`"
        ) from err
    return plt


def _show_or_return(fig, show):
    plt = _mpl()
    if show:
        plt.show()
        return None
    return fig


def _corr_plot(
    values, confint, ax, lags, alpha, title, zero, use_vlines, vlines_kwargs, show, kwargs
):
    """Draw a correlogram with optional Bartlett confidence bands."""
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    values = np.asarray(values, dtype=float)
    if not zero:
        values = values[1:]
        if confint is not None:
            confint = confint[1:]
        x = np.arange(1, len(values) + 1)
    else:
        x = np.arange(len(values))

    if use_vlines:
        ax.vlines(x, [0], values, **(vlines_kwargs or {}))
        ax.axhline(y=0, color="k")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 5)
    kwargs.setdefault("linestyle", "None")
    ax.margins(0.05)
    ax.plot(x, values, **kwargs)
    ax.set_title(title)

    if confint is not None:
        # The band is drawn around zero, and the lag-0 spike is dropped from
        # it, which is what statsmodels' plot does.
        lo = confint[:, 0] - values
        hi = confint[:, 1] - values
        first = 1 if zero else 0
        ax.fill_between(
            x[first:], lo[first:], hi[first:], alpha=0.25, color="tab:blue"
        )
    return _show_or_return(fig, show)


def plot_acf(
    series,
    ax=None,
    lags=None,
    alpha=None,
    use_vlines=True,
    unbiased=False,
    fft=True,
    title="Autocorrelation",
    zero=True,
    vlines_kwargs=None,
    show=True,
    **kwargs,
):
    """Plot the autocorrelation function of a series."""
    from .wrapped import acf

    out = acf(series, nlags=lags, fft=fft, adjusted=unbiased, alpha=alpha)
    vals, confint = out if alpha is not None else (out, None)
    return _corr_plot(
        vals, confint, ax, lags, alpha, title, zero, use_vlines, vlines_kwargs,
        show, kwargs,
    )


def plot_pacf(
    series,
    ax=None,
    lags=None,
    alpha=None,
    method="yw",
    use_vlines=True,
    title="Partial Autocorrelation",
    zero=True,
    vlines_kwargs=None,
    show=True,
    **kwargs,
):
    """Plot the partial autocorrelation function of a series."""
    from .wrapped import pacf

    out = pacf(series, nlags=lags, method=method, alpha=alpha)
    vals, confint = out if alpha is not None else (out, None)
    return _corr_plot(
        vals, confint, ax, lags, alpha, title, zero, use_vlines, vlines_kwargs,
        show, kwargs,
    )


def autocorr_plot(series, show=True):
    """A pandas-style autocorrelation plot."""
    plt = _mpl()
    from pandas.plotting import autocorrelation_plot

    fig = plt.figure()
    autocorrelation_plot(np.asarray(series, dtype=float))
    return _show_or_return(fig, show)


def decomposed_plot(decomposed_tuple, figure_kwargs=None, show=True):
    """Plot the four series returned by :func:`pmdarima_rs.arima.decompose`."""
    plt = _mpl()
    figure_kwargs = figure_kwargs or {}
    fig, axes = plt.subplots(4, 1, sharex=True, **figure_kwargs)
    for ax, name in zip(axes, ("x", "trend", "seasonal", "random")):
        ax.plot(np.asarray(getattr(decomposed_tuple, name), dtype=float))
        ax.set_ylabel(name)
    fig.tight_layout()
    return _show_or_return(fig, show)


def tsdisplay(
    y,
    lag_max=50,
    figsize=(8, 6),
    title=None,
    bins=25,
    series_kwargs=None,
    acf_kwargs=None,
    hist_kwargs=None,
    show=True,
):
    """The series, its histogram, and its autocorrelation, in one figure."""
    plt = _mpl()
    from .array import check_endog
    from .wrapped import acf

    y = check_endog(y, copy=False, preserve_series=True)
    if lag_max >= y.shape[0]:
        raise ValueError(
            f"lag_max ({lag_max}) must be < length of the "
            f"series ({y.shape[0]})"
        )
    y = np.asarray(y, dtype=float).ravel()
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    ax1.plot(y, **(series_kwargs or {}))
    ax1.set_title(title or "")

    ax2 = fig.add_subplot(223)
    ax2.hist(y, bins=bins, **(hist_kwargs or {}))

    ax3 = fig.add_subplot(224)
    vals = acf(y, nlags=min(lag_max, len(y) - 1))
    x = np.arange(len(vals))
    ax3.vlines(x, [0], vals, **(acf_kwargs or {}))
    ax3.axhline(y=0, color="k")
    fig.tight_layout()
    return _show_or_return(fig, show)
