"""Regenerate the images the README uses.

    python assets/make_assets.py

Nothing here is mocked up. The terminal panels are rendered from output
captured by actually running the code shown in them, and the bar chart is
parsed out of `bench/results.md`, which is itself the stdout of
`bench/bench.py`. Re-run this after re-running the benchmark and the pictures
follow.
"""

import contextlib
import io
import pathlib
import re
import warnings

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# The palette the logo uses, so the whole README reads as one thing.
BG = "#0b1220"
PANEL = "#0e1729"
EDGE = "#ffffff"
FG = "#cbd5e1"
DIM = "#64748b"
ACCENT = "#fb923c"
COOL = "#5eead4"
BRIGHT = "#e2e8f0"

MONO = ("ui-monospace, SFMono-Regular, Menlo, Consolas, "
        "'DejaVu Sans Mono', 'Liberation Mono', monospace")
SANS = ("ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, "
        "Helvetica, Arial, sans-serif")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# terminal panels


def runs(line):
    """Split one captured line into coloured runs.

    Colouring is by pattern and never rewrites anything: the runs always
    concatenate back to the line exactly as the program printed it.
    """
    if line.startswith("$ "):
        return [("$ ", COOL), (line[2:], BRIGHT)]
    if line.startswith(">>> ") or line.startswith("... "):
        return [(line[:4], COOL), (line[4:], BRIGHT)]
    m = re.match(r"^(Best model:\s+)(\S+)(\s*)$", line)
    if m:
        return [(m.group(1), FG), (m.group(2), ACCENT), (m.group(3), FG)]
    if line.startswith("Total fit time"):
        return [(line, COOL)]
    stripped = line.strip()
    if stripped and set(stripped) <= {"=", "-"}:
        return [(line, DIM)]
    m = re.match(r"^(.*?)(AIC=[\d.a-z]+)(.*)$", line)
    if m:
        return [(m.group(1), DIM), (m.group(2), FG), (m.group(3), DIM)]
    return [(line, FG)]


def terminal(lines, out, title, fs=13, lh=20, pad=22):
    cols = max(len(line) for line in lines)
    adv = fs * 0.62
    width = int(pad * 2 + cols * adv) + 8
    top = 42
    height = int(top + pad + len(lines) * lh + pad * 0.4)

    body, y = [], top + pad + fs
    for line in lines:
        parts = "".join(
            '<tspan fill="%s">%s</tspan>' % (c, esc(t)) for t, c in runs(line) if t
        )
        body.append(
            '    <text x="%d" y="%d" xml:space="preserve">%s</text>' % (pad, y, parts)
        )
        y += lh

    dots = "".join(
        '<circle cx="%d" cy="21" r="6" fill="%s" fill-opacity="0.85"/>' % (22 + i * 20, c)
        for i, c in enumerate(("#f87171", "#fbbf24", "#4ade80"))
    )

    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{t}">
  <title>{t}</title>
  <rect width="{w}" height="{h}" rx="12" fill="{bg}"/>
  <rect x="0.5" y="0.5" width="{w1}" height="{h1}" rx="11.5" fill="none" stroke="{edge}" stroke-opacity="0.10"/>
  <path d="M0 {top}H{w}" stroke="{edge}" stroke-opacity="0.08"/>
  {dots}
  <text x="{mid}" y="26" font-family="{sans}" font-size="12" fill="{dim}" text-anchor="middle">{t}</text>
  <g font-family="{mono}" font-size="{fs}">
{body}
  </g>
</svg>
""".format(w=width, h=height, w1=width - 1, h1=height - 1, t=esc(title), bg=BG,
           edge=EDGE, top=top, dots=dots, mid=width // 2, sans=SANS, dim=DIM,
           mono=MONO, fs=fs, body="\n".join(body))

    (HERE / out).write_text(svg, encoding="utf-8")
    print("wrote %s (%dx%d)" % (out, width, height))


def capture_quickstart():
    """Run the quickstart for real and keep what it printed."""
    warnings.simplefilter("ignore")
    import pmdarima_rs as pm

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        y = pm.datasets.load_wineind()
        model = pm.auto_arima(y, seasonal=True, m=12, trace=True,
                              suppress_warnings=True)
    trace = [line.rstrip() for line in buf.getvalue().splitlines() if line.strip()]
    return model, trace, repr(model.predict(n_periods=3))


def build_terminals(model, trace, forecast):
    head = [
        "$ pip install pmdarima-rs",
        "$ python",
        ">>> import pmdarima_rs as pm",
        ">>> y = pm.datasets.load_wineind()",
        ">>> model = pm.auto_arima(y, seasonal=True, m=12, trace=True)",
    ]
    shown, hidden = trace[:5], len(trace) - 7
    body = shown + ["  ... %d more candidates ..." % hidden] + trace[-2:]
    tail = [">>> model.predict(n_periods=3)", forecast]
    terminal(head + body + tail, "quickstart.svg",
             "auto_arima on pmdarima's wineind dataset")

    summary = [line.rstrip() for line in str(model.summary()).splitlines()]
    terminal([">>> model.summary()"] + summary, "summary.svg",
             "model.summary()", fs=12, lh=17)


# ---------------------------------------------------------------------------
# speedup chart, parsed from the benchmark's own output


def parse_auto_table(md):
    rows, inside = [], False
    for line in md.splitlines():
        if line.startswith("## `auto_arima` on every dataset"):
            inside = True
            continue
        if inside and line.startswith("## "):
            break
        if inside and line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            skip = cells[0] in ("dataset", "") or cells[0].startswith(("-", "**"))
            if len(cells) == 8 and not skip:
                rows.append((cells[0],
                             float(cells[4].rstrip(" s")),
                             float(cells[5].rstrip(" s")),
                             float(cells[6].strip("*x")),
                             cells[7] == "yes"))
    return rows


def chart(rows, out="speedup.svg"):
    rows = sorted(rows, key=lambda r: -r[3])
    pad_l, top, lh = 118, 58, 30
    width = 880
    height = top + len(rows) * lh + 26
    span = width - pad_l - 214
    hi = max(r[3] for r in rows)

    bars = []
    for i, (name, ref, mine, speed, _) in enumerate(rows):
        y = top + i * lh
        bw = span * speed / hi
        bars.append(
            '<rect x="%d" y="%d" width="%d" height="18" rx="4" fill="%s" fill-opacity="0.04"/>'
            '<rect x="%d" y="%d" width="%.1f" height="18" rx="4" fill="url(#bar)"/>'
            '<text x="%d" y="%d" text-anchor="end" font-size="14" fill="%s" font-family="%s">%s</text>'
            '<text x="%.0f" y="%d" font-size="14" font-weight="600" fill="%s" font-family="%s">%.1f&#215;</text>'
            '<text x="%d" y="%d" text-anchor="end" font-size="12" fill="%s" font-family="%s">%.2fs &#8594; %.2fs</text>'
            % (pad_l, y, span, EDGE,
               pad_l, y, bw,
               pad_l - 12, y + 14, FG, MONO, esc(name),
               pad_l + bw + 10, y + 14, ACCENT, SANS, speed,
               width - 14, y + 14, DIM, MONO, ref, mine)
        )

    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="auto_arima speedup over pmdarima, by dataset">
  <title>auto_arima speedup over pmdarima, by dataset</title>
  <defs>
    <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{cool}"/>
      <stop offset="1" stop-color="{accent}"/>
    </linearGradient>
  </defs>
  <rect width="{w}" height="{h}" rx="14" fill="{panel}"/>
  <rect x="0.5" y="0.5" width="{w1}" height="{h1}" rx="13.5" fill="none" stroke="{edge}" stroke-opacity="0.10"/>
  <text x="24" y="34" font-family="{sans}" font-size="16" font-weight="600" fill="{bright}">auto_arima, end to end &#8212; same order selected on {same} of {n}</text>
  <g>{bars}</g>
</svg>
""".format(w=width, h=height, w1=width - 1, h1=height - 1, cool=COOL,
           accent=ACCENT, panel=PANEL, edge=EDGE, sans=SANS, bright=BRIGHT,
           n=len(rows), same=sum(r[4] for r in rows), bars="".join(bars))

    (HERE / out).write_text(svg, encoding="utf-8")
    print("wrote %s (%dx%d)" % (out, width, height))


# ---------------------------------------------------------------------------
# forecast plot


def forecast_png(out="forecast.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    import pmdarima_rs as pm

    warnings.simplefilter("ignore")
    y = np.asarray(pm.datasets.load_wineind(), dtype=float)
    model = pm.auto_arima(y, seasonal=True, m=12, suppress_warnings=True)
    n = 24
    forecast, ci = model.predict(n_periods=n, return_conf_int=True, alpha=0.05)

    # Only the tail, so the forecast is not a sliver at the edge.
    keep = 84
    hx = np.arange(len(y) - keep, len(y))
    fx = np.arange(len(y), len(y) + n)
    y = y[-keep:]

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.fill_between(fx, ci[:, 0], ci[:, 1], color=ACCENT, alpha=0.16, lw=0,
                    label="95% interval")
    ax.plot(hx, y, color=COOL, lw=1.4, label="observed")
    ax.plot(np.r_[hx[-1], fx], np.r_[y[-1], forecast], color=ACCENT, lw=2.0,
            label="forecast")
    ax.axvline(hx[-1] + 0.5, color="#ffffff", alpha=0.14, lw=1, ls=(0, (4, 4)))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=DIM, labelsize=8, length=0)
    ax.grid(color="#ffffff", alpha=0.05, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("wineind — auto_arima chose %s, forecasting %d periods ahead"
                 % (str(model).strip(), n),
                 color=BRIGHT, fontsize=11, loc="left", pad=10)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper left", ncols=3)
    for text in legend.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(HERE / out, facecolor=BG)
    print("wrote %s" % out)


def main():
    model, trace, forecast = capture_quickstart()
    build_terminals(model, trace, forecast)
    results = ROOT / "bench" / "results.md"
    if results.exists():
        chart(parse_auto_table(results.read_text(encoding="utf-8")))
    else:
        print("no bench/results.md; skipping the chart")
    forecast_png()


if __name__ == "__main__":
    main()
