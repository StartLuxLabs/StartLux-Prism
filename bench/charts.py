"""The charts of README.md and docs/benchmarks.md: SVG files in docs/images, a light and a dark version of each
(`<name>.svg`, `<name>-dark.svg`; the docs choose with <picture> and prefers-color-scheme).

    python bench/charts.py

The numbers are those of docs/benchmarks.md's tables (from results/, as bench/report.py prints them), and every chart
has its table next to it there. One style throughout: StartLux-Prism in teal, every other engine in soft slate, the
value at each bar's tip; the README's top charts lead with the speed-ups as big numbers (`hero_chart`). The colors are
the light and dark tokens below (the teal marks at 3:1 or more against both surfaces).
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images")

THEMES = {
    "light": dict(surface="#ffffff", ink="#0f172a", ink2="#475569", muted="#94a3b8", grid="#edf1f5", axis="#cbd5e1",
                  accent="#0d9488", other="#d5dce6", line2="#94a3b8", border="#e6ece9", pill="#f0fdfa",
                  pill_ink="#0f766e", kpi="#0d9488", kpi_bg="#f0fdfa"),
    "dark": dict(surface="#0d1117", ink="#f1f5f9", ink2="#cbd5e1", muted="#7d8899", grid="#1b222c", axis="#2b3443",
                 accent="#2dd4bf", other="#2b3443", line2="#7d8899", border="#232a35", pill="#042f2e",
                 pill_ink="#99f6e4", kpi="#2dd4bf", kpi_bg="#0b2a28"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
W = 840              # chart width, px (the docs show every chart at this width)
PAD = 26             # outer padding
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
                f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{height - 1:.0f}" rx="14" fill="{c["surface"]}"'
                f' stroke="{c["border"]}"/>\n' + "\n".join(self.parts) + "\n</svg>\n")


def header(g, title, subtitle, legend=None, line_legend=False):
    """Title, subtitle and legend; returns the y below them."""
    y = PAD + 18
    g.text(PAD, y, title, size=19, weight=700)
    y += 6
    for ln in wrap(subtitle, 13, g.w - 2 * PAD):
        y += 19
        g.text(PAD, y, ln, size=13, color="ink2")
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
                g.add(f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="22" height="10" rx="5" fill="{g.c[color]}"/>')
                x += 30
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
    """Grouped horizontal bars (StartLux-Prism, then another engine), panels stacked; a panel: dict(head, vals: one
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


def ratio(ours, theirs, kind):
    """The pill's text: how many times faster ("lat": lower is better) or how many times the requests/s."""
    r = theirs / ours if kind == "lat" else ours / theirs
    return f"{r:.1f}× faster" if kind == "lat" else (f"{r:.2f}× more" if r < 1.95 else f"{r:.1f}× more")


def hero_chart(theme, title, subtitle, kpis, panels, legend, note=None, desc=""):
    """The README's top charts: the speed-ups as big numbers, then one row per measurement with StartLux-Prism's bar
    over the other engine's, on the row's own scale (the slower engine's bar spans it), the values at the bar ends and
    the ratio in a pill. kpis: [(big number, caption)]; a panel: dict(head, kind ("lat": lower is better, or "tput"),
    rows: [(label, ours, theirs)])."""
    g = Svg(theme)
    c = g.c
    y = header(g, title, subtitle, legend)
    if kpis:
        y += 20
        gap = 12
        tw = (g.w - 2 * PAD - gap * (len(kpis) - 1)) / len(kpis)
        for i, (big, cap) in enumerate(kpis):
            tx = PAD + i * (tw + gap)
            g.add(f'<rect x="{tx:.1f}" y="{y:.1f}" width="{tw:.1f}" height="74" rx="12" fill="{c["kpi_bg"]}"/>')
            g.text(tx + 16, y + 40, big, size=30, color="kpi", weight=700, num=True)
            g.text(tx + 16, y + 61, cap, size=12, color="ink2")
        y += 74
    lw = max(text_w(lbl, 13.5) for p in panels for lbl, _, _ in p["rows"]) + 28
    pill_w, vw, bh, gap2, step = 96, 64, 13, 3, 50
    x0 = PAD + lw
    plot = g.w - PAD - pill_w - 16 - vw - x0
    fmt = lambda v: f"{v:,.1f}" if v >= 10 else f"{v:.2f}"              # noqa: E731
    for p in panels:
        y += 36
        g.add(f'<text x="{PAD}" y="{y:.1f}" font-size="11.5" fill="{c["muted"]}" font-weight="600" '
              f'letter-spacing="0.06em">{esc(p["head"].upper())}</text>')
        y += 14
        for label, ours, theirs in p["rows"]:
            cy = y + bh + gap2 / 2
            g.text(PAD, cy + 4.5, label, size=13.5, weight=500)
            m = max(ours, theirs)
            for k, (v, color, ink, weight) in enumerate(((ours, "accent", "ink", 700), (theirs, "other", "ink2", 400))):
                by = y + k * (bh + gap2)
                w = max(v / m * plot, bh)
                name = legend[k][1]
                g.add(f'<rect x="{x0:.1f}" y="{by:.1f}" width="{w:.1f}" height="{bh}" rx="{bh / 2}" '
                      f'fill="{c[color]}"><title>{esc(label)}, {esc(name)}: {fmt(v)}</title></rect>')
                g.text(x0 + w + 8, by + 10.5, fmt(v), size=12.5, color=ink, weight=weight, num=True)
            px = g.w - PAD - pill_w
            g.add(f'<rect x="{px:.1f}" y="{cy - 11:.1f}" width="{pill_w}" height="22" rx="11" fill="{c["pill"]}"/>')
            g.text(px + pill_w / 2, cy + 4.3, ratio(ours, theirs, p["kind"]), size=12, color="pill_ink",
                   anchor="middle", weight=700)
            y += step
        y -= step - (2 * bh + gap2)
    y = footer(g, y + 14, note)
    return g.doc(y, title, desc or subtitle)


# ---- the data (docs/benchmarks.md) -------------------------------------------------------------------------------

STAND_IN = "decider and JevK5 run StartLux's weights as same-shape stand-ins (they ship no StartLux checkpoint)."
MODELS = ("4B · RTX 5090", "9B · RTX 5090", "27B · H200")
# JevBench's 231 public items per engine: mean ms per item one request at a time, and requests/s from 32 clients, on
# (MODELS); rows in the order of their summed ranks
JEVBENCH = [
    ("StartLux-Prism int8", (15.0, 21.6, 41.0), (85.2, 53.3, 30.6)),
    ("decider, compile + FP8", (33.4, 50.3, 66.8), (35.0, 22.6, 15.4)),
    ("StartLux-Prism bf16", (29.6, 52.1, 68.2), (37.0, 19.9, 16.7)),
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
        theme, title, subtitle, [(r[0], r[0] == "StartLux-Prism int8") for r in JEVBENCH],
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
        [(r[0], r[0] == "StartLux-Prism int8") for r in rows],
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
        [(r[0], r[0] == "StartLux-Prism int8") for r in rows],
        [dict(head="8,000 tokens: p50 ms (lower is better)", vals=[r[1] for r in rows],
              shown=[f"{r[1]:,}" if isinstance(r[1], int) else f"{r[1]:,.1f}" for r in rows], unit="ms", max=cut8),
         dict(head="50,000 tokens: p50 ms (lower is better)", vals=[r[2] for r in rows],
              shown=[r[2] if isinstance(r[2], str) else f"{r[2]:,}" for r in rows], unit="ms", max=cut50)],
        note=note)


@chart
def rtx5090_4b_long(theme):
    rows = [("StartLux-Prism int8", 162.5, 1960), ("decider, compile + FP8", 328.7, 3708), ("SGLang via adapter", 342.2, 2911),
            ("JevK5", 417.7, 3263), ("StartLux-Decision server", 424.2, "fails above 32,768 tokens"),
            ("vLLM via adapter", 440.5, 3359), ("decider, default", 521.5, 3272), ("llama.cpp (Q8_0)", 730.3, 5326)]
    return long_chart(theme, "Long evidence: StartLux-Decision-v1-4B (Q8_0) on an RTX 5090", rows, None, None,
                      "2026-10-10. The StartLux-Decision server's chunked path fails on prompts above 32,768 tokens "
                      "with the transformers version installed.")


@chart
def h200_27b_long(theme):
    rows = [("StartLux-Prism int8", 432, 3693), ("decider, compile + FP8", 698, 7301), ("SGLang via adapter", 704, 4936),
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
        legend=[("accent", "StartLux-Prism int8"), ("line2", "SGLang 0.5.21, --chunked-prefill-size -1")],
        note="Prompt length in thousands of tokens. On the RTX 5090 StartLux-Prism runs with --prefix-cache 0 (its largest "
             "batch then 98,304 tokens) and SGLang with --mem-fraction-static 0.6 (it does not start otherwise); "
             "SGLang ran out of memory at 96k there, and between 128k and 192k on the H200. StartLux-Prism refuses a "
             "request beyond its largest batch (HTTP 422); it never cuts one.")


@chart
def prefix_cache(theme):
    rows = [("StartLux-Prism int8, prefix cache", 54.1, 5.4), ("vLLM, automatic prefix caching", 162.0, 42.6),
            ("StartLux-Prism int8, cache off", 50.7, 48.0), ("SGLang, radix cache", 129.9, 113.3),
            ("decider, compile + FP8", 104.6, 157.8), ("StartLux-Decision server", 173.9, 173.6)]
    return bar_chart(
        theme, "Follow-up questions about evidence already sent",
        "StartLux-Decision-v1-4B (Q8_0) on an RTX 5090: the 20 public JevBench items with the longest evidence (2,000-3,700 "
        "tokens), each asked, then two more questions about the same evidence. p50 ms.",
        [(r[0], r[0] == "StartLux-Prism int8, prefix cache") for r in rows],
        [dict(head="First request (new evidence)", vals=[r[1] for r in rows], shown=[f"{r[1]:,.1f}" for r in rows],
              unit="ms", max=200),
         dict(head="Follow-up requests (same evidence)", vals=[r[2] for r in rows], shown=[f"{r[2]:,.1f}" for r in rows],
              unit="ms", max=200)],
        note="2026-10-10. StartLux-Prism answers a follow-up from the evidence's cached state, bit-identical to "
             "recomputing it.")


@chart
def q4km(theme):
    return group_chart(
        theme, "StartLux-Decision-v1 Q4_K_M files: StartLux-Prism W4A8 against llama.cpp",
        "Each engine on the same Q4_K_M GGUF file: StartLux-Prism on the file's own 4-bit weights (int8 activations, "
        "int8 tensor cores), llama.cpp b10639. JevBench, one request at a time.",
        ["4B · RTX 5090", "9B · RTX 5090", "27B · RTX 5090", "27B · H200"],
        [dict(head="Mean ms per JevBench item (lower is better)",
              vals=[(19.7, 135.8), (32.1, 160.3), (112.7, 439.3), (155.2, 626.7)], unit="ms")],
        legend=[("accent", "StartLux-Prism W4A8"), ("other", "llama.cpp")],
        note="Scores on the same files: 201, 200, 207, 207 of 231 for StartLux-Prism; 201, 200, 206, 207 for llama.cpp. "
             "On the H200 the W4A8 GEMMs still run on mma.sync, not on Hopper's warpgroup MMA.")


V11_ROWS = [("StartLux-Prism int8", True), ("official engine, FP8", False), ("official engine, BF16", False)]
V11_NOTE = ("Official engine: Startlux-decide-inference (StartLux-Decision v1.1's own package, commit b70404d) with "
            "its release flags on vLLM 0.31.0, from the BF16 release (FP8: vLLM's --quantization fp8); the release has "
            "no MTP weights, so neither side drafts tokens. StartLux-Prism int8 from the Q8_0 GGUF release. One H200 per "
            "server, 2026-10-13/14.")


@chart
def v11_official_latency(theme):
    return bar_chart(
        theme, "StartLux-Decision v1.1 27B AutoThink: StartLux-Prism against the official engine",
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
    return hero_chart(
        theme, "StartLux-Decision v1.1 27B AutoThink on one H200",
        "StartLux-Prism int8 against the official engine (vLLM 0.31, FP8), the same requests to both.",
        [("2.8×", "faster on a short question"), ("5.3×", "faster on three questions"), ("1.7×", "faster with thoughts"),
         ("1.17×", "the throughput, 64 clients")],
        [dict(head="One request at a time  ·  ms, lower is better", kind="lat",
              rows=[("Short question, ~256 tokens", 19.0, 53.1), ("Long question, ~8,192 tokens", 439.8, 531.2),
                    ("The README's three questions", 17.4, 92.9), ("Three questions, with thoughts", 785.1, 1296.0)]),
         dict(head="JevBench, 231 items  ·  requests/s, higher is better", kind="tput",
              rows=[("1 client", 24.2, 12.1), ("64 clients", 30.7, 26.3), ("64 clients, with thoughts", 7.61, 6.52)])],
        legend=[("accent", "StartLux-Prism int8"), ("other", "official engine, FP8")],
        note="Each row's bars share one scale. The official engine: Startlux-decide-inference with its release flags. "
             "Methods and every run: docs/benchmarks.md.")


# StartLux-Decision v1 27B against the official engine in FP8 (docs/benchmarks.md): mean ms one request at a time
# (~256 and ~8,192 tokens, the README's three questions), JevBench requests/s from 1 and 64 clients; (ours, theirs)
V1_OFFICIAL = dict(lat=[(18.9, 54.5), (438.5, 539.1), (17.2, 96.4)], tput=[(24.35, 12.04), (30.85, 25.37)])


@chart
def v1_hero(theme):
    (s, s2), (l8, l82), (t3, t32) = V1_OFFICIAL["lat"]
    (c1, c12), (c64, c642) = V1_OFFICIAL["tput"]
    return hero_chart(
        theme, "StartLux-Decision v1 27B on one H200",
        "StartLux-Prism int8 against the official engine (vLLM 0.31, FP8), the same requests to both. v1 answers in one "
        "pass.",
        [(ratio(s, s2, "lat").split()[0], "faster on a short question"),
         (ratio(t3, t32, "lat").split()[0], "faster on three questions"),
         (ratio(c1, c12, "tput").split()[0], "the throughput, 1 client"),
         (ratio(c64, c642, "tput").split()[0], "the throughput, 64 clients")],
        [dict(head="One request at a time  ·  ms, lower is better", kind="lat",
              rows=[("Short question, ~256 tokens", s, s2), ("Long question, ~8,192 tokens", l8, l82),
                    ("The README's three questions", t3, t32)]),
         dict(head="JevBench, 231 items  ·  requests/s, higher is better", kind="tput",
              rows=[("1 client", c1, c12), ("64 clients", c64, c642)])],
        legend=[("accent", "StartLux-Prism int8"), ("other", "official engine, FP8")],
        note="Each row's bars share one scale. The official engine: Startlux-decide-inference with its release flags, "
             "from the BF16 release. Methods and every run: docs/benchmarks.md.")


V11_W4_SUITES, V11_W4_JB = 93.07, 226   # the Q4_K_M file in W4A8 (docs/benchmarks.md)


@chart
def v11_accuracy(theme):
    rows = [("StartLux-Prism int8", True), ("StartLux-Prism W4A8", True), ("official engine, BF16", False),
            ("official engine, FP8", False)]
    suites = [92.94, V11_W4_SUITES, 92.97, 92.83]
    jb = [224, V11_W4_JB, 226, 225]
    return bar_chart(
        theme, "StartLux-Decision v1.1 27B AutoThink: the official engine's accuracy",
        "With thoughts (think_mode auto, gate 0.9), every item once in file order. StartLux-Prism int8 on the Q8_0 "
        "file and W4A8 on the Q4_K_M file; the official engine (Startlux-decide-inference on vLLM 0.31) on the BF16 "
        "release, in BF16 and FP8.",
        rows,
        [dict(head="StartLux-Decision v1's seven evaluation suites (12,351 decisions): average accuracy, %",
              vals=suites, shown=[f"{v:.2f}" for v in suites], unit="%", max=100),
         dict(head="JevBench public: items right, of 231", vals=jb, shown=[f"{v}" for v in jb], unit="of 231",
              max=231)],
        note="Every suite, and the GPU each server ran on: docs/benchmarks.md.")


@chart
def v11_precision(theme):
    return group_chart(
        theme, "StartLux-Prism int8 against bf16: StartLux-Decision v1.1 27B on one H200",
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
        legend=[("accent", "StartLux-Prism int8"), ("other", "StartLux-Prism bf16")],
        note="Decode at a 512-token context. Thoughts written by int8 are slightly less accurate on hard items "
             "(docs/benchmarks.md).")


@chart
def several_gpus(theme):
    return bar_chart(
        theme, "Several GPUs: StartLux-Decision v1.1 27B on H200s",
        "StartLux-Prism int8, JevBench. Data parallelism (DP: whole servers behind the router) for throughput, tensor "
        "parallelism (TP: one model split over the GPUs) for the time per request. Every configuration answers as one "
        "GPU does, bit for bit; one GPU in gray, for scale.",
        [("one GPU", False), ("2 GPUs: TP 2", True), ("4 GPUs: TP 4", True), ("4 GPUs: DP 2 x TP 2", True),
         ("4 GPUs: DP 4", True)],
        [dict(head="Read-outs from many clients: requests/s (higher is better)", vals=[30.9, 43.3, 54.0, 86.0, 121.1],
              shown=["30.9", "43.3 (1.40x)", "54.0 (1.75x)", "86.0 (2.78x)", "121.1 (3.92x)"],
              unit="requests/s"),
         dict(head="One request at a time, with thoughts: mean ms per item (lower is better)",
              vals=[1209, 902, 768, 902, "as one GPU (1,209), plus ~1 ms for the router"],
              shown=["1,209", "902", "768", "902", ""], unit="ms")],
        note="Read-outs: the most requests/s over 32 to 256 clients (one GPU and TP: from 64 clients). With thoughts: "
             "the 231 items one at a time, 67 of them thinking. Second H200 server (NVSwitch), 2026-10-14.")


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
