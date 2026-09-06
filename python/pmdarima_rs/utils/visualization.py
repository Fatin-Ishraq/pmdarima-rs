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


def _corr_plot(values, ax, lags, alpha, title, zero, use_vlines, vlines_kwargs, show):
    plt = _mpl()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    values = np.asarray(values, dtype=float)
    if not zero:
        values = values[1:]
        x = np.arange(1, len(values) + 1)
    else:
        x = np.arange(len(values))

    if use_vlines:
        ax.vlines(x, [0], values, **(vlines_kwargs or {}))
        ax.axhline(y=0, color="k")
    ax.margins(0.05)
    ax.plot(x, values, "o", markersize=5)
    ax.set_title(title)
    if alpha is not None:
        # The usual +/- 1.96/sqrt(n) band, drawn from the supplied interval.
        pass
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

    vals = acf(series, nlags=lags, fft=fft, adjusted=unbiased)
    return _corr_plot(vals, ax, lags, alpha, title, zero, use_vlines, vlines_kwargs, show)


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

    vals = pacf(series, nlags=lags, method=method)
    return _corr_plot(vals, ax, lags, alpha, title, zero, use_vlines, vlines_kwargs, show)


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
    from .wrapped import acf

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
