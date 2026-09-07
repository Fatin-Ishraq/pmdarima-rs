"""A statsmodels-shaped summary for a fitted model.

`pmdarima` hands back `SARIMAXResults.summary()`, and people read it, diff it
and paste it into reports. Reproducing it means reproducing the residual
diagnostics too - Ljung-Box, Jarque-Bera and the break-variance
heteroskedasticity test - which are computed here from the standardized
one-step forecast errors, exactly as `statsmodels` computes them.
"""

import numpy as np

__all__ = ["Summary", "jarque_bera", "ljung_box", "break_variance"]


def _fmt(value, width, prec=4):
    """Fixed point where it fits, scientific where it does not."""
    if value is None:
        return " " * width
    if isinstance(value, str):
        return f"{value:>{width}}"
    if not np.isfinite(value):
        return f"{'nan':>{width}}"
    txt = f"{value:.{prec}f}"
    if len(txt) > width - 1:
        txt = f"{value:.4g}"
    if len(txt) > width - 1:
        txt = f"{value:.3e}"
    return f"{txt:>{width}}"


def ljung_box(resid, lags=1):
    """Ljung-Box Q at `lags`, and its p-value."""
    from scipy import stats

    x = np.asarray(resid, dtype=float)
    x = x[~np.isnan(x)]
    nobs = x.shape[0]
    if nobs <= lags + 1:
        return np.nan, np.nan
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        return np.nan, np.nan
    acf = np.array([np.dot(xc[k:], xc[:-k]) / denom for k in range(1, lags + 1)])
    q = nobs * (nobs + 2) * np.sum(acf**2 / (nobs - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def jarque_bera(resid):
    """Jarque-Bera statistic, p-value, skew and kurtosis."""
    from scipy import stats

    x = np.asarray(resid, dtype=float)
    x = x[~np.isnan(x)]
    n = x.shape[0]
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    xc = x - x.mean()
    s2 = np.mean(xc**2)
    if s2 == 0:
        return np.nan, np.nan, np.nan, np.nan
    skew = np.mean(xc**3) / s2**1.5
    kurtosis = np.mean(xc**4) / s2**2
    jb = n / 6.0 * (skew**2 + (kurtosis - 3) ** 2 / 4.0)
    return float(jb), float(stats.chi2.sf(jb, 2)), float(skew), float(kurtosis)


def break_variance(resid):
    """statsmodels' 'breakvar' heteroskedasticity test, two-sided.

    Compares the sum of squares of the last third of the residuals with that
    of the first third; under homoskedasticity the ratio is F distributed.
    """
    from scipy import stats

    x = np.asarray(resid, dtype=float)
    x = x[~np.isnan(x)]
    nobs = x.shape[0]
    h = int(np.round(nobs / 3))
    if h < 1:
        return np.nan, np.nan
    numer = np.sum(x[-h:] ** 2)
    denom = np.sum(x[:h] ** 2)
    if denom == 0:
        return np.nan, np.nan
    stat = numer / denom
    pval = 2 * min(
        stats.f.cdf(stat, h, h),
        stats.f.sf(stat, h, h),
    )
    return float(stat), float(min(pval, 1.0))


class Summary:
    """The pieces of a `statsmodels` summary that callers actually use."""

    def __init__(self, model):
        self.model = model
        self._text = None

    # ------------------------------------------------------------------
    @property
    def tables(self):
        """The three blocks, as lists of rows - the statsmodels layout."""
        if not hasattr(self, "_tables"):
            self._tables = self._build_tables()
        return self._tables

    def _build_tables(self):
        m = self.model
        s = m.spec_
        names = s.param_names
        p = np.asarray(m.params(), dtype=float)
        se = np.asarray(m.bse(), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = p / se
        pv = np.asarray(m.pvalues(), dtype=float)
        ci = m.conf_int()

        order = f"{tuple(m.order)}"
        sorder = f"x{tuple(m.seasonal_order)}" if s.s else ""

        top = [
            ["Dep. Variable:", m._endog_name, "No. Observations:", f"{m.nobs_:d}"],
            ["Model:", f"SARIMAX{order}{sorder}", "Log Likelihood", f"{m.res_.loglike:.3f}"],
            ["Date:", m._fit_date, "AIC", f"{m.aic():.3f}"],
            ["Time:", m._fit_time, "BIC", f"{m.bic():.3f}"],
            ["Sample:", "0", "HQIC", f"{m.hqic():.3f}"],
            ["", f"- {m.nobs_}", "", ""],
            ["Covariance Type:", "opg", "", ""],
        ]
        coefs = [["", "coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]]
        for i, nm in enumerate(names):
            coefs.append([nm, p[i], se[i], z[i], pv[i], ci[i, 0], ci[i, 1]])

        lb, lbp = m._ljung_box()
        jb, jbp, skew, kurt = m._jarque_bera()
        het, hetp = m._heteroskedasticity()
        diag = [
            ["Ljung-Box (L1) (Q):", lb, "Jarque-Bera (JB):", jb],
            ["Prob(Q):", lbp, "Prob(JB):", jbp],
            ["Heteroskedasticity (H):", het, "Skew:", skew],
            ["Prob(H) (two-sided):", hetp, "Kurtosis:", kurt],
        ]
        return [top, coefs, diag]

    # ------------------------------------------------------------------
    def as_text(self):
        if self._text is None:
            self._text = self._render()
        return self._text

    def _render(self):
        top, coefs, diag = self.tables
        names = [row[0] for row in coefs[1:]]
        name_w = max(14, max((len(n) for n in names), default=0) + 2)
        total = name_w + 13 + 12 + 10 + 9 + 13 + 13

        lines = ["SARIMAX Results".center(total), "=" * total]
        lw = max(len(r[0]) for r in top) + 2
        lv = max(len(str(r[1])) for r in top)
        rw = max(len(r[2]) for r in top) + 2
        rv = max(len(str(r[3])) for r in top)
        for lk, lval, rk, rval in top:
            lhs = f"{lk:<{lw}}{str(lval):<{lv}}"
            rhs = f"{rk:<{rw}}{str(rval):>{rv}}" if rk or rval else ""
            lines.append(f"{lhs}   {rhs}".rstrip())

        lines += [
            "=" * total,
            f"{'':>{name_w}}{'coef':>13}{'std err':>12}{'z':>10}"
            f"{'P>|z|':>9}{'[0.025':>13}{'0.975]':>13}",
            "-" * total,
        ]
        for row in coefs[1:]:
            lines.append(
                f"{row[0]:>{name_w}}{_fmt(row[1], 13)}{_fmt(row[2], 12)}"
                f"{_fmt(row[3], 10)}{_fmt(row[4], 9)}{_fmt(row[5], 13)}"
                f"{_fmt(row[6], 13)}"
            )
        lines.append("=" * total)
        for lk, lval, rk, rval in diag:
            lines.append(
                f"{lk:<26}{_fmt(lval, 8, 2)}   {rk:<22}{_fmt(rval, 8, 2)}"
            )
        lines.append("=" * total)
        lines.append("")
        lines.append(
            "[1] Covariance matrix calculated using the outer product of "
            "gradients (central difference)."
        )
        return "\n".join(lines)

    def as_csv(self):
        rows = []
        for table in self.tables:
            for row in table:
                rows.append(",".join("" if c is None else str(c) for c in row))
            rows.append("")
        return "\n".join(rows)

    def as_html(self):
        parts = ["<h3>SARIMAX Results</h3>"]
        for table in self.tables:
            parts.append("<table>")
            for row in table:
                cells = "".join(
                    f"<td>{'' if c is None else c}</td>" for c in row
                )
                parts.append(f"<tr>{cells}</tr>")
            parts.append("</table>")
        return "\n".join(parts)

    def as_latex(self):
        parts = []
        for table in self.tables:
            ncol = max(len(r) for r in table)
            parts.append("\\begin{tabular}{" + "l" * ncol + "}")
            for row in table:
                parts.append(
                    " & ".join("" if c is None else str(c) for c in row) + " \\\\"
                )
            parts.append("\\end{tabular}")
        return "\n".join(parts)

    def __str__(self):
        return self.as_text()

    __repr__ = __str__
