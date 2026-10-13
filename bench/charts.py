"""The charts of README.md and docs/benchmarks.md: SVG files in docs/images, a light and a dark version of each
(`<name>.svg`, `<name>-dark.svg`; the docs choose with <picture> and prefers-color-scheme).

    python bench/charts.py

The numbers are those of docs/benchmarks.md's tables (from results/, as bench/report.py prints them), and every chart
has its table next to it there. One style throughout: flash-jev in the accent blue, every other engine in gray, the
value at each bar's tip, hairline grid; the colors are the light and dark tokens below (the accent and the line gray
checked with the dataviz palette validator against both surfaces).
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  accent="#2a78d6", other="#b4b2aa", line2="#898781", border="#e9e8e2"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 accent="#3987e5", other="#62615c", line2="#898781", border="#2c2c2a"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
W = 840              # chart width, px (the docs show every chart at this width)
PAD = 22             # outer padding
ROW = 26             # one bar's row
BAR = 14             # bar thickness (<= 24)
GROUP_BAR = 13       # bars in a group: 2 px apart
GAP = 30             # between side-by-side panels


def text_w(s, size):
    """Rendered width of s in the system sans, roughly (enough to lay out labels without clipping)."""
    w = 0.0
    for c in s:
        if c in "il.,:;|!'":
            w += 0.27
        elif c in "jtfr()[]{}-/ ":
            w += 0.35
        elif c in "mwMW":
            w += 0.86
        elif c.isupper():
            w += 0.68
        elif c.isdigit():
            w += 0.58
        elif ord(c) > 0x2000:
            w += 0.62
        else:
            w += 0.54
    return w * size


def wrap(s, size, width):
    lines, cur = [], ""
    for word in s.split():
        nxt = word if not cur else cur + " " + word
        if text_w(nxt, size) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = nxt
    return lines + ([cur] if cur else [])


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_max(v):
    """The axis end above v: 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 x 10^k."""
    k = 1.0
    while k * 10 < v:
        k *= 10
    while k > v:
        k /= 10
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * k >= v * 1.0001:
            return m * k
    return 10 * k


def ticks(vmax, n=4):
    """Clean tick values from 0 to vmax: the finest 1 / 2 / 2.5 / 5 x 10^k step with at most n + 1 intervals."""
    k = 1.0
    while k * 10 <= vmax:
        k *= 10
    k /= 1000
    step = next(m * k * 10 ** e for e in range(6) for m in (1, 2, 2.5, 5) if vmax / (m * k * 10 ** e) <= n + 1.0001)
    out, t = [], 0.0
    while t <= vmax * 1.0001:
        out.append(round(t, 10))
        t += step
    return out


def fmt_tick(t, unit=""):
    if t >= 1000:
        s = f"{t:,.0f}"
    elif t == int(t):
        s = f"{int(t)}"
    else:
        s = f"{t:g}"
    return s + unit


class Svg:
    def __init__(self, theme, width=W):
        self.c = THEMES[theme]
        self.w = width
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, color="ink", anchor="start", weight=400, num=False, halo=False):
        style = ' style="font-variant-numeric: tabular-nums"' if num else ""
        if halo:                                        # a surface-colored outline: gridlines stop short of the text
            style += f' stroke="{self.c["surface"]}" stroke-width="3" stroke-linejoin="round" paint-order="stroke"'
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.c[color]}" text-anchor="{anchor}"'
                 f' font-weight="{weight}"{style}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, color, width=1):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{self.c[color]}"'
                 f' stroke-width="{width}" shape-rendering="crispEdges"/>')

    def hbar(self, x0, y, w, h, color, tip, r=4.0):
        """A horizontal bar from the baseline x0: square there, 4 px rounded at its data end."""
        w = max(w, 1.0)
        r = min(r, w / 2, h / 2)
        d = (f"M{x0:.1f},{y:.1f} H{x0 + w - r:.1f} A{r},{r} 0 0 1 {x0 + w:.1f},{y + r:.1f} V{y + h - r:.1f}"
             f" A{r},{r} 0 0 1 {x0 + w - r:.1f},{y + h:.1f} H{x0:.1f} Z")
        self.add(f'<path d="{d}" fill="{self.c[color]}"><title>{esc(tip)}</title></path>')

    def doc(self, height, title, desc):
        c = self.c
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{height:.0f}" '
                f'viewBox="0 0 {self.w} {height:.0f}" role="img" aria-labelledby="title desc" font-family="{FONT}">\n'
                f'<title id="title">{esc(title)}</title>\n<desc id="desc">{esc(desc)}</desc>\n'
                f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{height - 1:.0f}" rx="8" fill="{c["surface"]}"'
                f' stroke="{c["border"]}"/>\n' + "\n".join(self.parts) + "\n</svg>\n")


def header(g, title, subtitle, legend=None, line_legend=False):
    """Title, subtitle and legend; returns the y below them."""
    y = PAD + 16
    g.text(PAD, y, title, size=16, weight=600)
    y += 6
    for ln in wrap(subtitle, 12.5, g.w - 2 * PAD):
        y += 18
        g.text(PAD, y, ln, size=12.5, color="ink2")
    if legend:
        y += 24
        x = PAD
        for color, label in legend:
            if line_legend:
                g.add(f'<line x1="{x:.1f}" y1="{y - 4:.1f}" x2="{x + 18:.1f}" y2="{y - 4:.1f}" stroke="{g.c[color]}"'
                      f' stroke-width="2" stroke-linecap="round"/>')
                g.add(f'<circle cx="{x + 9:.1f}" cy="{y - 4:.1f}" r="4" fill="{g.c[color]}" stroke="{g.c["surface"]}"'
                      f' stroke-width="2"/>')
                x += 24
            else:
                g.add(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="11" height="11" rx="2" fill="{g.c[color]}"/>')
                x += 17
            g.text(x, y, label, size=12.5, color="ink2")
            x += text_w(label, 12.5) + 22
    return y


def footer(g, y, note):
    if note:
        y += 8
        for ln in wrap(note, 11.5, g.w - 2 * PAD):
            y += 16
            g.text(PAD, y, ln, size=11.5, color="muted")
    return y + PAD


def bar_chart(theme, title, subtitle, rows, panels, legend=None, note=None, desc="", width=W):
    """Horizontal bars, side-by-side panels sharing the row labels.

    rows: [(label, highlighted)]; a panel: dict(head, vals (one per row: a number, or a string written in place of
    the bar), shown (the value texts), max (axis end; larger values are cut and marked), unit, tick_unit)."""
    g = Svg(theme, width)
    y = header(g, title, subtitle, legend)
    lw = max(text_w(lbl, 12.5) * 1.06 + 14 for lbl, _ in rows)
    pw = (g.w - 2 * PAD - lw - GAP * (len(panels) - 1)) / len(panels)
    heads = [wrap(p["head"], 12.5, pw) for p in panels]
    y += 16 + 17 * max(len(h) for h in heads)
    y0 = y + 6                                                  # top of the first row
    bottom = y0 + ROW * len(rows)
    for i, p in enumerate(panels):
        px0 = PAD + lw + i * (pw + GAP)
        for j, ln in enumerate(heads[i]):
            g.text(px0, y0 - 12 - 17 * (len(heads[i]) - 1 - j), ln, size=12.5, weight=600)
        nums = [v for v in p["vals"] if not isinstance(v, str)]
        vmax = p.get("max") or nice_max(max(nums))
        vl = max(text_w(s, 12) * 1.08 for v, s in zip(p["vals"], p["shown"]) if not isinstance(v, str)) + 8
        plot = pw - vl
        for t in ticks(vmax, max(2, min(5, int(plot / 58)))):
            x = px0 + t / vmax * plot
            if t > 0:
                g.line(x, y0 - 4, x, bottom, "grid")
            g.text(x, bottom + 15, fmt_tick(t, p.get("tick_unit", "")), size=11, color="muted", anchor="middle",
                   num=True)
        g.line(px0, y0 - 4, px0, bottom, "axis")
        for r, ((label, hi), v, shown) in enumerate(zip(rows, p["vals"], p["shown"])):
            cy = y0 + r * ROW + ROW / 2
            if isinstance(v, str):
                g.text(px0 + 6, cy + 4, v, size=11.5, color="muted", halo=True)
                continue
            cut = v > vmax
            w = plot if cut else v / vmax * plot
            g.hbar(px0, cy - BAR / 2, w, BAR, "accent" if hi else "other", f"{label}: {shown} {p.get('unit', '')}")
            if cut:                                             # the bar is cut: a gap across its end
                for dx in (14, 9):
                    g.add(f'<line x1="{px0 + w - dx - 2:.1f}" y1="{cy + BAR / 2 + 2:.1f}" x2="{px0 + w - dx + 2:.1f}"'
                          f' y2="{cy - BAR / 2 - 2:.1f}" stroke="{g.c["surface"]}" stroke-width="2.5"/>')
            g.text(px0 + w + 6, cy + 4.3, shown, size=12, color="ink" if hi else "ink2", weight=600 if hi else 400,
                   num=True, halo=True)
    for r, (label, hi) in enumerate(rows):
        g.text(PAD + lw - 12, y0 + r * ROW + ROW / 2 + 4.3, label, size=12.5, color="ink" if hi else "ink2",
               anchor="end", weight=600 if hi else 400)
    y = footer(g, bottom + 22, note)
    return g.doc(y, title, desc or subtitle)


def group_chart(theme, title, subtitle, groups, panels, legend, note=None, desc=""):
    """Grouped horizontal bars (flash-jev, then another engine), panels stacked; a panel: dict(head, vals: one
    (ours, theirs) pair per group, names: the other engine per group (or None), max, unit, and optionally groups (this
    panel's own group labels), fmt (the values' format, default ",.1f") and extra (a text after our value per group,
    e.g. a speed-up))."""
    g = Svg(theme)
    y = header(g, title, subtitle, legend)
    lw = max(text_w(lbl, 12.5) * 1.06 + 14 for p in panels for lbl in p.get("groups", groups))
    px0 = PAD + lw
    pw = g.w - PAD - px0
    gh = 2 * GROUP_BAR + 2                                      # a group's bars, 2 px apart
    for p in panels:
        groups_p = p.get("groups", groups)
        fmt = p.get("fmt", ",.1f")
        y += 30
        for ln in wrap(p["head"], 12.5, g.w - 2 * PAD):
            g.text(PAD, y, ln, size=12.5, weight=600)
            y += 17
        y0 = y + 2
        step = gh + 14
        bottom = y0 + step * len(groups_p) - 8
        shown = [[format(v, fmt) for v in pair] for pair in p["vals"]]
        extra = p.get("extra") or [None] * len(groups_p)
        tails = [[s[0] + (f"  {e}" if e else ""), s[1] + (f"  {n}" if n else "")]
                 for s, n, e in zip(shown, p.get("names") or [None] * len(groups_p), extra)]
        vl = max(text_w(t, 12) * 1.06 for pair in tails for t in pair) + 10
        plot = pw - vl
        vmax = p.get("max") or nice_max(max(max(pair) for pair in p["vals"]))
        for t in ticks(vmax):
            x = px0 + t / vmax * plot
            if t > 0:
                g.line(x, y0 - 4, x, bottom + 4, "grid")
            g.text(x, bottom + 19, fmt_tick(t), size=11, color="muted", anchor="middle", num=True)
        g.line(px0, y0 - 4, px0, bottom + 4, "axis")
        for k, (label, pair, tail) in enumerate(zip(groups_p, p["vals"], tails)):
            top = y0 + k * step
            g.text(px0 - 12, top + gh / 2 + 4.3, label, size=12.5, anchor="end", color="ink")
            for s, v in enumerate(pair):
                by = top + s * (GROUP_BAR + 2)
                w = v / vmax * plot
                name = legend[s][1] if s == 0 else (p["names"][k] if p.get("names") else legend[1][1])
                g.hbar(px0, by, w, GROUP_BAR, "accent" if s == 0 else "other",
                       f"{label}, {name}: {format(v, fmt)} {p['unit']}")
                if s == 0:
                    g.text(px0 + w + 6, by + 10.5, shown[k][0], size=12, weight=600, num=True, halo=True)
                    if extra[k]:
                        g.text(px0 + w + 6 + text_w(shown[k][0], 12) * 1.08 + 8, by + 10.5, extra[k], size=12,
                               color="ink2", halo=True)
                else:
                    g.text(px0 + w + 6, by + 10.5, shown[k][1], size=12, color="ink2", num=True, halo=True)
                    if p.get("names") and p["names"][k]:
                        g.text(px0 + w + 6 + text_w(shown[k][1], 12) * 1.06 + 8, by + 10.5, p["names"][k], size=12,
                               color="muted", halo=True)
        y = bottom + 19
    y = footer(g, y + 6, note)
    return g.doc(y, title, desc or subtitle)


def line_chart(theme, title, subtitle, panels, legend, note=None, desc=""):
    """Side-by-side line panels: x = prompt tokens (thousands), y = seconds; a panel: dict(head, xs, series (one list
    of y per legend entry; None = no answer), xmax, ymax)."""
    g = Svg(theme)
    y = header(g, title, subtitle, legend, line_legend=True)
    yl = 34                                                    # y tick labels
    pw = (g.w - 2 * PAD - GAP * (len(panels) - 1)) / len(panels)
    heads = [wrap(p["head"], 12.5, pw) for p in panels]
    y += 18 + 17 * max(len(h) for h in heads)
    ph = 200
    y0 = y + 8
    colors = [c for c, _ in legend]
    for i, p in enumerate(panels):
        x0 = PAD + i * (pw + GAP) + yl
        plot = pw - yl - 46                                    # end labels right of the plot
        for j, ln in enumerate(heads[i]):
            g.text(PAD + i * (pw + GAP), y0 - 14 - 17 * (len(heads[i]) - 1 - j), ln, size=12.5, weight=600)
        X = lambda v: x0 + v / p["xmax"] * plot                # noqa: E731
        Y = lambda v: y0 + ph - v / p["ymax"] * ph             # noqa: E731
        for t in ticks(p["ymax"]):
            if t > 0:
                g.line(x0, Y(t), x0 + plot, Y(t), "grid")
            g.text(x0 - 7, Y(t) + 4, fmt_tick(t, " s"), size=11, color="muted", anchor="end", num=True)
        g.line(x0, Y(0), x0 + plot, Y(0), "axis")
        for t in p["xs"]:
            g.text(X(t), y0 + ph + 16, f"{t}k", size=11, color="muted", anchor="middle", num=True)
        ends = []
        for s, ys in enumerate(p["series"]):
            pts = [(X(xv), Y(yv), xv, yv) for xv, yv in zip(p["xs"], ys) if yv is not None]
            d = " ".join(f"{'M' if k == 0 else 'L'}{px:.1f},{py:.1f}" for k, (px, py, _, _) in enumerate(pts))
            col = g.c[colors[s]]
            g.add(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"'
                  f' stroke-linecap="round"/>')
            for px, py, xv, yv in pts:
                g.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{col}" stroke="{g.c["surface"]}"'
                      f' stroke-width="2"><title>{esc(legend[s][1])}, {xv}k tokens: {yv:,.3f} s</title></circle>')
            ends.append((pts[-1], s))
        for (px, py, xv, yv), s in ends:                       # end labels: right of the last x, else above
            last = xv == max(p["xs"])
            g.text(px + 8 if last else px, py + 4 if last else py - 10, f"{yv:,.1f} s", size=12,
                   color="ink" if s == 0 else "ink2", weight=600 if s == 0 else 400,
                   anchor="start" if last else "middle", num=True, halo=True)
    y = footer(g, y0 + ph + 26, note)
    return g.doc(y, title, desc or subtitle)


# ---- the data (docs/benchmarks.md) -------------------------------------------------------------------------------

STAND_IN = "decider and JevK5 run StartLux's weights as same-shape stand-ins (they ship no StartLux checkpoint)."
MODELS = ("4B · RTX 5090", "9B · RTX 5090", "27B · H200")
# JevBench's 231 public items per engine: mean ms per item one request at a time, and requests/s from 32 clients, on
# (MODELS); rows in the order of their summed ranks
JEVBENCH = [
    ("flash-jev int8", (15.0, 21.6, 41.0), (85.2, 53.3, 30.6)),
    ("decider, compile + FP8", (33.4, 50.3, 66.8), (35.0, 22.6, 15.4)),
    ("flash-jev bf16", (29.6, 52.1, 68.2), (37.0, 19.9, 16.7)),
    ("SGLang native", (42.0, 59.8, 90.1), (39.5, 21.7, 18.8)),
    ("SGLang via adapter", (40.1, 62.1, 92.0), (37.4, 20.5, 17.7)),
    ("StartLux-Decision server", (47.0, 71.2, 96.0), (22.9, 14.2, 10.4)),
    ("decider, default", (48.5, 76.1, 99.9), (20.7, 12.6, 9.6)),
    ("JevK5", (49.8, 74.4, 101.1), (21.8, 13.6, 9.9)),
    ("vLLM via adapter", (55.3, 84.9, 119.1), (33.6, 18.9, 15.4)),
    ("llama.cpp (Q8_0)", (134.7, 162.7, 589.2), (6.5, 5.7, 1.4)),
]
LAT_MAX = (None, None, 150)          # axis ends: the 27B's llama.cpp bar (589.2 ms) is cut


def all_engines_chart(theme, title, subtitle, which, unit, note):
    """Every engine on the three models: one panel per model, the rows in JEVBENCH's order."""
    k = 1 if which == "lat" else 2
    return bar_chart(
        theme, title, subtitle, [(r[0], r[0] == "flash-jev int8") for r in JEVBENCH],
        [dict(head=MODELS[m], vals=[r[k][m] for r in JEVBENCH], shown=[f"{r[k][m]:,.1f}" for r in JEVBENCH],
              unit=unit, max=LAT_MAX[m] if which == "lat" else None) for m in range(3)],
        note=note)


def engines_chart(theme, title, m, note):
    """JevBench one request at a time (mean ms per item) and from 32 clients (requests/s) on MODELS[m], one row per
    engine, fastest first."""
    rows = sorted(JEVBENCH, key=lambda r: r[1][m])
    lat = [r[1][m] for r in rows]
    return bar_chart(
        theme, title,
        "The 231 public JevBench items; every engine alone on the GPU, through the same HTTP client.",
        [(r[0], r[0] == "flash-jev int8") for r in rows],
        [dict(head="One request at a time: mean ms per item (lower is better)", vals=lat,
              shown=[f"{v:,.1f}" for v in lat], unit="ms", max=LAT_MAX[m]),
         dict(head="From 32 clients: requests per second (higher is better)", vals=[r[2][m] for r in rows],
              shown=[f"{r[2][m]:,.1f}" for r in rows], unit="requests/s")],
        note=note)


CHARTS = {}


def chart(fn):
    CHARTS[fn.__name__] = fn
    return fn


@chart
def jevbench_latency(theme):
    return all_engines_chart(
        theme, "StartLux-Decision v1 on JevBench, one request at a time: every engine",
        "Mean ms per item over the 231 public items (lower is better). Every engine alone on the GPU with the same "
        "Q8_0 weights, through the same HTTP client.", "lat", "ms",
        STAND_IN + " The 27B's llama.cpp bar is cut (589.2 ms). Scores, p50 and p95: docs/benchmarks.md.")


@chart
def jevbench_throughput(theme):
    return all_engines_chart(
        theme, "StartLux-Decision v1 on JevBench from 32 clients: every engine",
        "Requests per second over three passes of the 231 public items (higher is better); no engine failed a request. "
        "The same Q8_0 weights for every engine.", "tput", "requests/s",
        STAND_IN + " 8 and 16 clients give every engine within a few percent of its 32-client rate.")


@chart
def rtx5090_4b(theme):
    return engines_chart(theme, "StartLux-Decision-v1-4B (Q8_0) on an RTX 5090", 0,
                         "Q8_0 weights for every engine (bf16 conversions of them for the PyTorch engines); "
                         "2026-10-10. " + STAND_IN)


@chart
def rtx5090_9b(theme):
    return engines_chart(theme, "StartLux-Decision-v1-9B (Q8_0) on an RTX 5090", 1,
                         "Q8_0 weights for every engine (bf16 conversions of them for the PyTorch engines); "
                         "2026-10-10. " + STAND_IN)


@chart
def h200_27b(theme):
    return engines_chart(theme, "StartLux-Decision-v1-27B (Q8_0) on an H200", 2,
                         "Q8_0 weights for every engine (bf16 conversions of them for the PyTorch engines); "
                         "2026-10-12. llama.cpp's latency bar is cut (589.2 ms). " + STAND_IN)


def long_chart(theme, title, rows, cut8, cut50, note):
    return bar_chart(
        theme, title,
        "One yes/no question about an application log of 8,000 or 50,000 tokens, one request at a time; every engine "
        "counted every token. " + STAND_IN,
        [(r[0], r[0] == "flash-jev int8") for r in rows],
        [dict(head="8,000 tokens: p50 ms (lower is better)", vals=[r[1] for r in rows],
              shown=[f"{r[1]:,}" if isinstance(r[1], int) else f"{r[1]:,.1f}" for r in rows], unit="ms", max=cut8),
         dict(head="50,000 tokens: p50 ms (lower is better)", vals=[r[2] for r in rows],
              shown=[r[2] if isinstance(r[2], str) else f"{r[2]:,}" for r in rows], unit="ms", max=cut50)],
        note=note)


@chart
def rtx5090_4b_long(theme):
    rows = [("flash-jev int8", 162.5, 1960), ("decider, compile + FP8", 328.7, 3708), ("SGLang via adapter", 342.2, 2911),
            ("JevK5", 417.7, 3263), ("StartLux-Decision server", 424.2, "fails above 32,768 tokens"),
            ("vLLM via adapter", 440.5, 3359), ("decider, default", 521.5, 3272), ("llama.cpp (Q8_0)", 730.3, 5326)]
    return long_chart(theme, "Long evidence: StartLux-Decision-v1-4B (Q8_0) on an RTX 5090", rows, None, None,
                      "2026-10-10. The StartLux-Decision server's chunked path fails on prompts above 32,768 tokens "
                      "with the transformers version installed.")


@chart
def h200_27b_long(theme):
    rows = [("flash-jev int8", 432, 3693), ("decider, compile + FP8", 698, 7301), ("SGLang via adapter", 704, 4936),
            ("vLLM via adapter", 833, 5743), ("JevK5", 847, 5783), ("StartLux-Decision server", 870, "fails above 32,768 tokens"),
            ("decider, default", 1062, 6779), ("llama.cpp (Q8_0)", 3514, 28623)]
    return long_chart(theme, "Long evidence: StartLux-Decision-v1-27B (Q8_0) on an H200", rows, 1200, 8000,
                      "2026-10-12. llama.cpp's bars are cut (3,514 and 28,623 ms). SGLang runs FlashAttention 3 here; "
                      "at 50,000 tokens attention is most of the work.")


@chart
def max_context(theme):
    return line_chart(
        theme, "The longest request in one pass (no chunked prefill)",
        "One request at a time, the whole prompt in one forward pass; seconds per request. vLLM does not start "
        "without chunked prefill on this model.",
        [dict(head="StartLux-Decision-v1-4B (Q8_0), RTX 5090", xs=[16, 32, 48, 64, 96],
              series=[[0.377, 0.978, 1.814, 2.861, 5.567], [0.692, 1.562, 2.559, 3.741, None]], xmax=100, ymax=6),
         dict(head="StartLux-Decision-v1-27B (Q8_0), H200", xs=[16, 32, 48, 64, 96, 128],
              series=[[0.910, 2.047, 3.460, 5.120, 9.234, 14.205], [1.424, 2.984, 4.697, 6.605, 10.821, 15.705]],
              xmax=132, ymax=16)],
        legend=[("accent", "flash-jev int8"), ("line2", "SGLang 0.5.21, --chunked-prefill-size -1")],
        note="Prompt length in thousands of tokens. On the RTX 5090 flash-jev runs with --prefix-cache 0 (its largest "
             "batch then 98,304 tokens) and SGLang with --mem-fraction-static 0.6 (it does not start otherwise); "
             "SGLang ran out of memory at 96k there, and between 128k and 192k on the H200. flash-jev refuses a "
             "request beyond its largest batch (HTTP 422); it never cuts one.")


@chart
def prefix_cache(theme):
    rows = [("flash-jev int8, prefix cache", 54.1, 5.4), ("vLLM, automatic prefix caching", 162.0, 42.6),
            ("flash-jev int8, cache off", 50.7, 48.0), ("SGLang, radix cache", 129.9, 113.3),
            ("decider, compile + FP8", 104.6, 157.8), ("StartLux-Decision server", 173.9, 173.6)]
    return bar_chart(
        theme, "Follow-up questions about evidence already sent",
        "StartLux-Decision-v1-4B (Q8_0) on an RTX 5090: the 20 public JevBench items with the longest evidence (2,000-3,700 "
        "tokens), each asked, then two more questions about the same evidence. p50 ms.",
        [(r[0], r[0] == "flash-jev int8, prefix cache") for r in rows],
        [dict(head="First request (new evidence)", vals=[r[1] for r in rows], shown=[f"{r[1]:,.1f}" for r in rows],
              unit="ms", max=200),
         dict(head="Follow-up requests (same evidence)", vals=[r[2] for r in rows], shown=[f"{r[2]:,.1f}" for r in rows],
              unit="ms", max=200)],
        note="2026-10-10. flash-jev answers a follow-up from the evidence's cached state, bit-identical to "
             "recomputing it.")


@chart
def q4km(theme):
    return group_chart(
        theme, "StartLux-Decision-v1 Q4_K_M files: flash-jev W4A8 against llama.cpp",
        "Each engine on the same Q4_K_M GGUF file: flash-jev on the file's own 4-bit weights (int8 activations, "
        "int8 tensor cores), llama.cpp b10639. JevBench, one request at a time.",
        ["4B · RTX 5090", "9B · RTX 5090", "27B · RTX 5090", "27B · H200"],
        [dict(head="Mean ms per JevBench item (lower is better)",
              vals=[(19.7, 135.8), (32.1, 160.3), (112.7, 439.3), (155.2, 626.7)], unit="ms")],
        legend=[("accent", "flash-jev W4A8"), ("other", "llama.cpp")],
        note="Scores on the same files: 201, 200, 207, 207 of 231 for flash-jev; 201, 200, 206, 207 for llama.cpp. "
             "On the H200 the W4A8 GEMMs still run on mma.sync, not on Hopper's warpgroup MMA.")


V11_ROWS = [("flash-jev int8", True), ("official engine, FP8", False), ("official engine, BF16", False)]
V11_NOTE = ("Official engine: Startlux-decide-inference (StartLux-Decision v1.1's own package, commit b70404d) with "
            "its release flags on vLLM 0.31.0, from the BF16 release (FP8: vLLM's --quantization fp8); the release has "
            "no MTP weights, so neither side drafts tokens. flash-jev int8 from the Q8_0 GGUF release. One H200 per "
            "server, 2026-10-13/14.")


@chart
def v11_official_latency(theme):
    return bar_chart(
        theme, "StartLux-Decision v1.1 27B AutoThink: flash-jev against the official engine",
        "One request at a time, new evidence in every request, the official engine's own client (startlux-decide "
        "bench). Mean ms per request (lower is better).",
        V11_ROWS,
        [dict(head="One yes/no question, ~256 tokens", vals=[19.0, 53.1, 54.2], shown=["19.0", "53.1", "54.2"],
              unit="ms"),
         dict(head="One yes/no question, ~8,192 tokens", vals=[439.8, 531.2, 728.8], shown=["439.8", "531.2", "728.8"],
              unit="ms"),
         dict(head="The README's three questions", vals=[17.4, 92.9, 93.1], shown=["17.4", "92.9", "93.1"],
              unit="ms")],
        note=V11_NOTE)


@chart
def v11_official_throughput(theme):
    return bar_chart(
        theme, "StartLux-Decision v1.1 27B AutoThink under load",
        "JevBench's 231 public items, requests per second (higher is better): read-outs only (think_mode never) from "
        "one and from 64 clients, and with thoughts (think_mode auto, the release's default) from 64 clients.",
        V11_ROWS,
        [dict(head="Read-outs, one client", vals=[24.2, 12.1, 10.6], shown=["24.2", "12.1", "10.6"], unit="req/s"),
         dict(head="Read-outs, 64 clients", vals=[30.7, 26.3, 18.3], shown=["30.7", "26.3", "18.3"], unit="req/s"),
         dict(head="With thoughts, 64 clients", vals=[7.61, 6.52, 5.89], shown=["7.61", "6.52", "5.89"],
              unit="req/s")],
        note=V11_NOTE + " Three passes over the items per measurement, every request with its own evidence.")


@chart
def v11_hero(theme):
    return group_chart(
        theme, "StartLux-Decision v1.1 27B AutoThink on one H200",
        "flash-jev int8 against the model's official engine (Startlux-decide-inference with its release flags on vLLM "
        "0.31) in FP8, its faster setting; the same requests to both.",
        [],
        [dict(head="One request at a time: mean ms per request (lower is better)",
              groups=["One question, ~256 tokens", "One question, ~8,192 tokens", "The README's three questions",
                      "The same, with thoughts"],
              vals=[(19.0, 53.1), (439.8, 531.2), (17.4, 92.9), (785.1, 1296.0)], unit="ms",
              extra=["2.8x faster", "1.2x faster", "5.3x faster", "1.7x faster"]),
         dict(head="JevBench's 231 items from many clients: requests per second (higher is better)",
              groups=["Read-outs, 1 client", "Read-outs, 64 clients", "With thoughts, 64 clients"],
              vals=[(24.2, 12.1), (30.7, 26.3), (7.61, 6.52)], unit="requests/s",
              extra=["2.0x", "1.17x", "1.17x"])],
        legend=[("accent", "flash-jev int8"), ("other", "official engine, FP8")],
        note="Same accuracy on StartLux-Decision v1's evaluation suites with thoughts: 92.94 (flash-jev) and 92.83 "
             "(FP8; 92.97 for the official BF16 server). 2026-10-13/14.")


@chart
def v11_precision(theme):
    return group_chart(
        theme, "flash-jev int8 against bf16: StartLux-Decision v1.1 27B on one H200",
        "The same Q8_0 weights: int8 (W8A8) or dequantized to bf16 (cuBLAS); the engine alone, no HTTP.",
        [],
        [dict(head="Prefill: prompt tokens per second (higher is better)",
              groups=["One prompt of 1,024 tokens", "One prompt of 8,192 tokens", "One prompt of 32,768 tokens",
                      "Passes of 8,192 tokens of short prompts"],
              vals=[(18865, 11571), (19219, 11424), (15810, 10155), (20855, 11958)], unit="tokens/s", fmt=",.0f",
              extra=["1.63x", "1.68x", "1.56x", "1.74x"]),
         dict(head="Decode: ms per step (lower is better)",
              groups=["1 sequence", "8 sequences", "64 sequences"],
              vals=[(10.47, 14.89), (11.06, 15.90), (16.70, 21.74)], unit="ms", fmt=",.2f",
              extra=["1.42x", "1.44x", "1.30x"])],
        legend=[("accent", "flash-jev int8"), ("other", "flash-jev bf16")],
        note="Decode at a 512-token context. Thoughts written by int8 are slightly less accurate on hard items "
             "(docs/benchmarks.md).")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in CHARTS.items():
        for theme in THEMES:
            path = os.path.join(OUT, name.replace("_", "-") + ("-dark" if theme == "dark" else "") + ".svg")
            with open(path, "w") as f:
                f.write(fn(theme))
    print(f"{2 * len(CHARTS)} files in {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
