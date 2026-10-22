"""The charts of README.md and docs: SVG files in docs/images, a light and a dark version of each (`<name>.svg`,
`<name>-dark.svg`; the pages choose with <picture> and prefers-color-scheme).

    python bench/charts.py [OUT_DIR]

The numbers are those of docs/benchmarks.md's tables (from results/, as bench/report.py prints them), and every chart
has its table next to it there. Plain matplotlib in the style of the StartLux-Decision technical report
(arXiv:2610.17309): its violet for StartLux-Prism, greys for every other engine, Arial, titles on the left, the value on
every bar, and one bar thickness in every chart. Text is
drawn as paths, so the SVGs look the same in every browser, and the files carry no date, so a rerun changes nothing.
"""
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import transforms  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs",
                                                         "images")

# The StartLux-Decision report's palette: StartLux violet for its own work, greys (told apart by lightness) for others
PRISM, PRISM_LIGHT = "#8A5CD6", "#B99BE8"          # StartLux-Prism int8; its other precisions and configurations
OTHER, OTHER_DARK = "#B9B9B9", "#7A7A7A"           # every other engine (the official one in FP8); the official in BF16
BAR = 0.22                                         # inches: the thickness of every bar, vertical or horizontal
PITCH = 0.34                                       # inches: one row of horizontal bars

THEMES = {   # GitHub's text, muted text, border and page colors in each theme
    "light": dict(text="#1f2328", muted="#59636e", grid="#d1d9e0", axis="#818b98", page="#ffffff"),
    "dark": dict(text="#d1d7e0", muted="#9198a1", grid="#3d444d", axis="#9198a1", page="#0d1117"),
}
WIDTH = 10.5                                       # inches: the pages show every chart 840 px wide


def style(theme):
    t = THEMES[theme]
    plt.rcdefaults()
    plt.rcParams.update({
        "svg.fonttype": "path", "svg.hashsalt": "startlux-prism",
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10,
        "text.color": t["text"], "axes.labelcolor": t["text"], "axes.edgecolor": t["axis"],
        "xtick.color": t["axis"], "ytick.color": t["axis"], "xtick.labelcolor": t["text"],
        "ytick.labelcolor": t["text"],
        "axes.titlesize": 10, "axes.titlelocation": "left", "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.7,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    })
    return t


def save(fig, name, theme):
    path = os.path.join(OUT, name + ("" if theme == "light" else "-dark") + ".svg")
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.1, transparent=True, metadata={"Date": None})
    plt.close(fig)
    return path


def num(v, fmt=None):
    if fmt:
        return format(v, fmt)
    return f"{v:,.2f}" if v < 10 else f"{v:,.1f}"


def frame(fig, axes, t, title, subtitle=None, handles=None, note=None, head=26, foot=30):
    """Title, subtitle and legend above the panels, the note under them: placed in points from the panels' edges
    (head: the room their titles take above them; foot: what their axes take below them)."""
    top = max(a.get_position().y1 for a in axes)
    bottom = min(a.get_position().y0 for a in axes)

    left = min(a.get_position().x0 for a in axes)

    def at(y, pts):
        return dict(x=left, y=y, transform=transforms.offset_copy(fig.transFigure, fig=fig, y=pts, units="points"))

    y = head
    if handles:
        fig.legend(handles=handles, loc="lower left", ncol=len(handles), borderaxespad=0, handlelength=1.5,
                   columnspacing=1.8, bbox_to_anchor=(left, top),
                   bbox_transform=transforms.offset_copy(fig.transFigure, fig=fig, y=y, units="points"))
        y += 21
    for line in reversed(textwrap.wrap(subtitle or "", 150)):
        fig.text(s=line, ha="left", va="bottom", fontsize=9, color=t["muted"], **at(top, y))
        y += 12.5
    fig.text(s=title, ha="left", va="bottom", fontsize=12, fontweight="bold", **at(top, y + 3))
    y = -foot
    for line in textwrap.wrap(note or "", 160):
        p = at(bottom, y)
        p["x"] = left
        fig.text(s=line, ha="left", va="top", fontsize=8.5, color=t["muted"], **p)
        y -= 11.5


def hbars(ax, t, labels, vals, colors, shown=None, xmax=None, title=None, xlabel=None, bold=()):
    """Horizontal bars, the first row on top, the value at each bar's end. A value past xmax is cut: a gap across the
    bar's end and the value written inside it. A string is written in place of the bar."""
    n = len(labels)
    nums = [v for v in vals if not isinstance(v, str)]
    shown = shown or [v if isinstance(v, str) else f"{v:,.1f}" for v in vals]
    end = xmax or max(nums) * 1.17
    ax.set_xlim(0, end)
    ax.set_ylim(n - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_yticks(range(n), labels)
    for i, (v, s, c) in enumerate(zip(vals, shown, colors)):
        if isinstance(v, str):
            ax.text(end * 0.012, i, v, va="center", ha="left", fontsize=8.5, color=t["muted"], style="italic")
            continue
        weight = "bold" if labels[i] in bold else "normal"
        if xmax and v > xmax:                            # cut: a gap near the bar's end, the value before it
            ax.barh(i, end, BAR / PITCH, color=c)
            for x in (end * 0.93, end * 0.955):
                ax.plot([x - end * 0.006, x + end * 0.006], [i + 0.38, i - 0.38], color=t["page"], lw=2.2,
                        solid_capstyle="butt")
            ax.text(end * 0.905, i, s, va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.barh(i, v, BAR / PITCH, color=c)
            ax.text(v + end * 0.012, i, s, va="center", ha="left", fontsize=8.5, fontweight=weight)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in bold:
            lbl.set_fontweight("bold")
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)


def vwidth(ax, ngroups):
    """The data width of a vertical bar BAR inches wide in ax, whose x axis spans ngroups (call after the layout)."""
    return BAR * ngroups / (ax.get_position().width * ax.figure.get_figwidth())


def hfig(nrows, ncols=1, head=1.0, foot=0.9, left=0.2, wspace=0.12):
    """A figure whose panels hold nrows of horizontal bars, PITCH inches apart, so every bar is BAR inches thick."""
    h = head + nrows * PITCH + foot
    fig, axes = plt.subplots(1, ncols, figsize=(WIDTH, h), sharey=True)
    fig.subplots_adjust(left=left, right=0.98, top=1 - head / h, bottom=foot / h, wspace=wspace)
    return fig, axes


def vgroups(ax, t, groups, series, colors, fmt=None, title=None, ylabel=None, ymax=None, labels=None):
    """Grouped vertical bars, series[s][g] for series s in group g, every bar BAR inches wide (call after the layout),
    the value (or labels[s][g]) on every bar."""
    x = np.arange(len(groups))
    k = len(series)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    w = vwidth(ax, len(groups))
    ax.grid(axis="x", visible=False)
    top = max(max(s) for s in series)
    for s, (vals, c) in enumerate(zip(series, colors)):
        bars = ax.bar(x + (s - (k - 1) / 2) * w * 1.25, vals, w, color=c)     # a quarter bar of space between bars
        ax.bar_label(bars, labels[s] if labels else [num(v, fmt) for v in vals], padding=2, fontsize=8.5)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ymax or top * 1.18)
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def patches(pairs):
    return [Patch(color=c, label=label) for c, label in pairs]


CHARTS = {}


def chart(fn):
    CHARTS[fn.__name__.replace("_", "-")] = fn
    return fn


# ---- the README ---------------------------------------------------------------------------------------------------

def hero(theme, title, lat, tput):
    """lat: [(label, StartLux-Prism ms, official ms)] one request at a time; tput: [(label, ours, theirs)] req/s."""
    t = style(theme)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.7), width_ratios=[len(lat), len(tput)])
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.17, wspace=0.22)
    vgroups(a, t, [label for label, _, _ in lat], [[p / o for _, p, o in lat], [1.0] * len(lat)], [PRISM, OTHER],
            labels=[[num(p) for _, p, _ in lat], [num(o) for _, _, o in lat]], ymax=1.2,
            title="One request at a time (lower is better; labels in ms)", ylabel="Latency, official engine = 1")
    vgroups(b, t, [label for label, _, _ in tput], [[p for _, p, _ in tput], [o for _, _, o in tput]],
            [PRISM, OTHER], title="JevBench, 231 items (higher is better)", ylabel="Requests per second")
    frame(fig, (a, b), t, title, handles=patches([(PRISM, "StartLux-Prism, int8"),
                                                  (OTHER, "Official engine, FP8 (vLLM 0.31)")]))
    return fig


@chart
def v11_hero(theme):
    return hero(theme, "StartLux-Decision-v1.1-27B-AutoThink on one H200",
                [("Short question\n~256 tokens", 19.0, 53.1), ("Long question\n~8,192 tokens", 439.8, 531.2),
                 ("README's three\nquestions", 17.4, 92.9), ("Three questions\nwith thoughts", 785.1, 1296.0)],
                [("1 client", 24.2, 12.1), ("64 clients", 30.7, 26.3), ("64 clients\nwith thoughts", 7.61, 6.52)])


@chart
def v1_hero(theme):
    return hero(theme, "StartLux-Decision-v1-27B on one H200",
                [("Short question\n~256 tokens", 18.9, 54.5), ("Long question\n~8,192 tokens", 438.5, 539.1),
                 ("README's three\nquestions", 17.2, 96.4)],
                [("1 client", 24.4, 12.0), ("64 clients", 30.9, 25.4)])


@chart
def v11_accuracy(theme):
    t = style(theme)
    names = ["StartLux-Prism\nint8", "StartLux-Prism\nW4A8", "Official\nBF16", "Official\nFP8"]
    colors = [PRISM, PRISM_LIGHT, OTHER_DARK, OTHER]
    suites = [92.94, 93.07, 92.97, 92.83]     # StartLux-Decision v1's seven suites, average accuracy %
    jevbench = [224, 226, 226, 225]           # JevBench public, items right of 231
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.17, wspace=0.25)
    for ax, vals, fmt, top, label, head in ((a, suites, ".2f", 110, "Average accuracy (%)",
                                             "Seven evaluation suites (12,351 decisions)"),
                                            (b, jevbench, "d", 254, "Items right, of 231",
                                             "JevBench public (231 items)")):
        ax.set_xlim(-0.5, 3.5)
        bars = ax.bar(range(4), vals, vwidth(ax, 4), color=colors)
        ax.bar_label(bars, [format(v, fmt) for v in vals], padding=2, fontsize=8.5)
        ax.set_xticks(range(4), names)
        ax.set_ylim(0, top)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(label)
        ax.set_title(head)
    frame(fig, (a, b), t, "StartLux-Decision-v1.1-27B-AutoThink with thoughts: accuracy")
    return fig


# ---- docs/benchmarks.md: StartLux-Decision v1 against every engine ------------------------------------------------

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
OURS = "StartLux-Prism int8"


def engine_color(name):
    return PRISM if name == OURS else PRISM_LIGHT if name.startswith("StartLux-Prism") else OTHER


def all_engines(theme, title, subtitle, which, xlabel, note):
    """Every engine on the three models: one panel per model, the rows in JEVBENCH's order."""
    t = style(theme)
    k = 1 if which == "lat" else 2
    fig, axes = hfig(len(JEVBENCH), 3, head=1.05, foot=1.1)
    labels = [r[0] for r in JEVBENCH]
    for m, ax in enumerate(axes):
        hbars(ax, t, labels, [r[k][m] for r in JEVBENCH], [engine_color(r[0]) for r in JEVBENCH],
              xmax=LAT_MAX[m] if which == "lat" else None, title=MODELS[m], xlabel=xlabel, bold=(OURS,))
    frame(fig, axes, t, title, subtitle, note=note, foot=40)
    return fig


@chart
def jevbench_latency(theme):
    return all_engines(
        theme, "StartLux-Decision-v1 on JevBench, one request at a time: every engine",
        "Mean ms per item over the 231 public items (lower is better). Every engine alone on the GPU with the same "
        "Q8_0 weights, through the same HTTP client.", "lat", "ms per item",
        STAND_IN + " The 27B's llama.cpp bar is cut (589.2 ms). Scores, p50 and p95: docs/benchmarks.md.")


@chart
def jevbench_throughput(theme):
    return all_engines(
        theme, "StartLux-Decision-v1 on JevBench from 32 clients: every engine",
        "Requests per second over three passes of the 231 public items (higher is better); no engine failed a request. "
        "The same Q8_0 weights for every engine.", "tput", "requests per second",
        STAND_IN + " 8 and 16 clients give every engine within a few percent of its 32-client rate.")


def engines(theme, title, m, note):
    """JevBench one request at a time (mean ms per item) and from 32 clients (requests/s) on MODELS[m], one row per
    engine, fastest first."""
    t = style(theme)
    rows = sorted(JEVBENCH, key=lambda r: r[1][m])
    labels = [r[0] for r in rows]
    colors = [engine_color(r[0]) for r in rows]
    fig, (a, b) = hfig(len(rows), 2, head=1.0, foot=1.0)
    hbars(a, t, labels, [r[1][m] for r in rows], colors, xmax=LAT_MAX[m], bold=(OURS,),
          title="One request at a time (lower is better)", xlabel="mean ms per item")
    hbars(b, t, labels, [r[2][m] for r in rows], colors, bold=(OURS,),
          title="From 32 clients (higher is better)", xlabel="requests per second")
    frame(fig, (a, b), t, title, "The 231 public JevBench items; every engine alone on the GPU, through the same HTTP "
                                 "client.", note=note, foot=40)
    return fig


@chart
def rtx5090_4b(theme):
    return engines(theme, "StartLux-Decision-v1-4B (Q8_0) on an RTX 5090", 0,
                   "Q8_0 weights for every engine (bf16 conversions of them for the PyTorch engines); 2026-10-10. "
                   + STAND_IN)


@chart
def rtx5090_9b(theme):
    return engines(theme, "StartLux-Decision-v1-9B (Q8_0) on an RTX 5090", 1,
                   "Q8_0 weights for every engine (bf16 conversions of them for the PyTorch engines); 2026-10-10. "
                   + STAND_IN)


@chart
def h200_27b(theme):
    return engines(theme, "StartLux-Decision-v1-27B (Q8_0) on an H200", 2,
                   "Q8_0 weights for every engine (bf16 conversions of them for the PyTorch engines); 2026-10-12. "
                   "llama.cpp's latency bar is cut (589.2 ms). " + STAND_IN)


def long_evidence(theme, title, rows, cut8, cut50, note):
    t = style(theme)
    labels = [r[0] for r in rows]
    colors = [engine_color(r[0]) for r in rows]
    fig, (a, b) = hfig(len(rows), 2, head=1.15, foot=0.9)
    hbars(a, t, labels, [r[1] for r in rows], colors, xmax=cut8, bold=(OURS,),
          shown=[f"{v:,}" if isinstance(v, int) else f"{v:,.1f}" for v in (r[1] for r in rows)],
          title="8,000 tokens (lower is better)", xlabel="p50 ms")
    hbars(b, t, labels, [r[2] for r in rows], colors, xmax=cut50, bold=(OURS,),
          shown=[v if isinstance(v, str) else f"{v:,}" for v in (r[2] for r in rows)],
          title="50,000 tokens (lower is better)", xlabel="p50 ms")
    frame(fig, (a, b), t, title, "One yes/no question about an application log of 8,000 or 50,000 tokens, one request "
                                 "at a time; every engine counted every token. " + STAND_IN, note=note, foot=40)
    return fig


@chart
def rtx5090_4b_long(theme):
    rows = [("StartLux-Prism int8", 162.5, 1960), ("decider, compile + FP8", 328.7, 3708),
            ("SGLang via adapter", 342.2, 2911), ("JevK5", 417.7, 3263),
            ("StartLux-Decision server", 424.2, "fails above 32,768 tokens"), ("vLLM via adapter", 440.5, 3359),
            ("decider, default", 521.5, 3272), ("llama.cpp (Q8_0)", 730.3, 5326)]
    return long_evidence(theme, "Long evidence: StartLux-Decision-v1-4B (Q8_0) on an RTX 5090", rows, None, None,
                         "2026-10-10. The StartLux-Decision server's chunked path fails on prompts above 32,768 tokens "
                         "with the transformers version installed.")


@chart
def h200_27b_long(theme):
    rows = [("StartLux-Prism int8", 432, 3693), ("decider, compile + FP8", 698, 7301), ("SGLang via adapter", 704, 4936),
            ("vLLM via adapter", 833, 5743), ("JevK5", 847, 5783),
            ("StartLux-Decision server", 870, "fails above 32,768 tokens"), ("decider, default", 1062, 6779),
            ("llama.cpp (Q8_0)", 3514, 28623)]
    return long_evidence(theme, "Long evidence: StartLux-Decision-v1-27B (Q8_0) on an H200", rows, 1200, 8000,
                         "2026-10-12. llama.cpp's bars are cut (3,514 and 28,623 ms). SGLang runs FlashAttention 3 "
                         "here; at 50,000 tokens attention is most of the work.")


@chart
def max_context(theme):
    t = style(theme)
    panels = [("StartLux-Decision-v1-4B (Q8_0), RTX 5090", [16, 32, 48, 64, 96],
               [[0.377, 0.978, 1.814, 2.861, 5.567], [0.692, 1.562, 2.559, 3.741, None]], 6),
              ("StartLux-Decision-v1-27B (Q8_0), H200", [16, 32, 48, 64, 96, 128],
               [[0.910, 2.047, 3.460, 5.120, 9.234, 14.205], [1.424, 2.984, 4.697, 6.605, 10.821, 15.705]], 17)]
    names = ["StartLux-Prism int8", "SGLang 0.5.21, --chunked-prefill-size -1"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5))
    for ax, (head, xs, series, ymax) in zip(axes, panels):
        for ys, c in zip(series, (PRISM, OTHER)):
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, lw=2, ms=5)
            x, y = pts[-1]
            ax.annotate(f"{y:,.1f} s", (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, fontweight="bold" if c == PRISM else "normal")
        ax.set_xticks(xs, [f"{x}k" for x in xs])
        ax.set_xlim(0, max(xs) * 1.14)
        ax.set_ylim(0, ymax)
        ax.set_title(head)
        ax.set_xlabel("prompt tokens")
        ax.set_ylabel("seconds per request")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.15, wspace=0.22)
    frame(fig, axes, t, "The longest request in one pass (no chunked prefill)",
          "One request at a time, the whole prompt in one forward pass; seconds per request. vLLM does not start "
          "without chunked prefill on this model.",
          handles=[Line2D([], [], color=c, marker="o", lw=2, ms=5, label=n) for c, n in zip((PRISM, OTHER), names)],
          note="On the RTX 5090 StartLux-Prism runs with --prefix-cache 0 (its largest batch then 98,304 tokens) and "
               "SGLang with --mem-fraction-static 0.6 (it does not start otherwise); SGLang ran out of memory at 96k "
               "there, and between 128k and 192k on the H200. StartLux-Prism refuses a request beyond its largest batch "
               "(HTTP 422); it never cuts one.", foot=34)
    return fig


@chart
def prefix_cache(theme):
    t = style(theme)
    rows = [("StartLux-Prism int8, prefix cache", 54.1, 5.4), ("vLLM, automatic prefix caching", 162.0, 42.6),
            ("StartLux-Prism int8, cache off", 50.7, 48.0), ("SGLang, radix cache", 129.9, 113.3),
            ("decider, compile + FP8", 104.6, 157.8), ("StartLux-Decision server", 173.9, 173.6)]
    labels = [r[0] for r in rows]
    colors = [PRISM if i == 0 else PRISM_LIGHT if r[0].startswith("StartLux-Prism") else OTHER
              for i, r in enumerate(rows)]
    fig, (a, b) = hfig(len(rows), 2, head=1.15, foot=0.85, left=0.23)
    hbars(a, t, labels, [r[1] for r in rows], colors, xmax=200, bold=(labels[0],),
          title="First request (new evidence)", xlabel="p50 ms")
    hbars(b, t, labels, [r[2] for r in rows], colors, xmax=200, bold=(labels[0],),
          title="Follow-up requests (same evidence)", xlabel="p50 ms")
    frame(fig, (a, b), t, "Follow-up questions about evidence already sent",
          "StartLux-Decision-v1-4B (Q8_0) on an RTX 5090: the 20 public JevBench items with the longest evidence "
          "(2,000-3,700 tokens), each asked, then two more questions about the same evidence. p50 ms.",
          note="2026-10-10. StartLux-Prism answers a follow-up from the evidence's cached state, bit-identical to "
               "recomputing it.", foot=40)
    return fig


@chart
def q4km(theme):
    t = style(theme)
    groups = ["4B · RTX 5090", "9B · RTX 5090", "27B · RTX 5090", "27B · H200"]
    vals = [(19.7, 135.8), (32.1, 160.3), (112.7, 439.3), (155.2, 626.7)]
    fig, ax = plt.subplots(figsize=(WIDTH, 3.1))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.12)
    vgroups(ax, t, groups, [[v[0] for v in vals], [v[1] for v in vals]], [PRISM, OTHER],
            title="Mean ms per JevBench item (lower is better)", ylabel="ms per item")
    frame(fig, (ax,), t, "StartLux-Decision-v1 Q4_K_M files: StartLux-Prism W4A8 against llama.cpp",
          "Each engine on the same Q4_K_M GGUF file: StartLux-Prism on the file's own 4-bit weights (int8 activations, "
          "int8 tensor cores), llama.cpp b10639. JevBench, one request at a time.",
          handles=patches([(PRISM, "StartLux-Prism W4A8"), (OTHER, "llama.cpp")]),
          note="Scores on the same files: 201, 200, 207, 207 of 231 for StartLux-Prism; 201, 200, 206, 207 for "
               "llama.cpp. On the H200 the W4A8 GEMMs still run on mma.sync, not on Hopper's warpgroup MMA.", foot=26)
    return fig


# ---- docs/benchmarks.md: StartLux-Decision v1.1 against its official engine ---------------------------------------

V11_NOTE = ("Official engine: Startlux-decide-inference (StartLux-Decision v1.1's own package, commit b70404d) with its "
            "release flags on vLLM 0.31.0, from the BF16 release (FP8: vLLM's --quantization fp8); the release has no "
            "MTP weights, so neither side drafts tokens. StartLux-Prism int8 from the Q8_0 GGUF release. One H200 per "
            "server, 2026-10-13/14.")
V11_SYSTEMS = [(PRISM, "StartLux-Prism int8"), (OTHER, "official engine, FP8"), (OTHER_DARK, "official engine, BF16")]


def v11_panels(theme, title, subtitle, panels, ylabel, note):
    t = style(theme)
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 3.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.06, wspace=0.28)
    colors = [c for c, _ in V11_SYSTEMS]
    for ax, (head, vals) in zip(axes, panels):
        ax.set_xlim(-0.5, 2.5)
        bars = ax.bar(range(3), vals, vwidth(ax, 3), color=colors)
        ax.bar_label(bars, [num(v) for v in vals], padding=2, fontsize=8.5)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(head)
    axes[0].set_ylabel(ylabel)
    frame(fig, axes, t, title, subtitle, handles=patches(V11_SYSTEMS), note=note, foot=10)
    return fig


@chart
def v11_official_latency(theme):
    return v11_panels(
        theme, "StartLux-Decision-v1.1-27B-AutoThink: StartLux-Prism against the official engine",
        "One request at a time, new evidence in every request, the official engine's own client (startlux-decide "
        "bench). Mean ms per request (lower is better).",
        [("One yes/no question, ~256 tokens", [19.0, 53.1, 54.2]),
         ("One yes/no question, ~8,192 tokens", [439.8, 531.2, 728.8]),
         ("The README's three questions", [17.4, 92.9, 93.1])], "ms per request", V11_NOTE)


@chart
def v11_official_throughput(theme):
    return v11_panels(
        theme, "StartLux-Decision-v1.1-27B-AutoThink under load",
        "JevBench's 231 public items, requests per second (higher is better): read-outs only (think_mode never) from "
        "one and from 64 clients, and with thoughts (think_mode auto, the release's default) from 64 clients.",
        [("Read-outs, one client", [24.2, 12.1, 10.6]), ("Read-outs, 64 clients", [30.7, 26.3, 18.3]),
         ("With thoughts, 64 clients", [7.61, 6.52, 5.89])], "requests per second",
        V11_NOTE + " Three passes over the items per measurement, every request with its own evidence.")


@chart
def v11_precision(theme):
    t = style(theme)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.5), width_ratios=[4, 3])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.2, wspace=0.25)
    vgroups(a, t, ["One prompt of\n1,024 tokens\n(1.63×)", "One prompt of\n8,192 tokens\n(1.68×)",
                   "One prompt of\n32,768 tokens\n(1.56×)", "Passes of 8,192\ntokens, short\nprompts (1.74×)"],
            [[18865, 19219, 15810, 20855], [11571, 11424, 10155, 11958]], [PRISM, PRISM_LIGHT], fmt=",.0f",
            title="Prefill (higher is better)", ylabel="prompt tokens per second")
    vgroups(b, t, ["1 sequence\n(1.42×)", "8 sequences\n(1.44×)", "64 sequences\n(1.30×)"],
            [[10.47, 11.06, 16.70], [14.89, 15.90, 21.74]], [PRISM, PRISM_LIGHT], fmt=",.2f",
            title="Decode (lower is better)", ylabel="ms per step")
    frame(fig, (a, b), t, "StartLux-Prism int8 against bf16: StartLux-Decision-v1.1-27B on one H200",
          "The same Q8_0 weights: int8 (W8A8) or dequantized to bf16 (cuBLAS); the engine alone, no HTTP. In "
          "parentheses: how many times faster int8 is.",
          handles=patches([(PRISM, "StartLux-Prism int8"), (PRISM_LIGHT, "StartLux-Prism bf16")]),
          note="Decode at a 512-token context. Thoughts written by int8 are slightly less accurate on hard items "
               "(docs/benchmarks.md).", foot=50)
    return fig


@chart
def several_gpus(theme):
    t = style(theme)
    labels = ["one GPU", "2 GPUs: TP 2", "4 GPUs: TP 4", "4 GPUs: DP 2 × TP 2", "4 GPUs: DP 4"]
    colors = [PRISM_LIGHT] + [PRISM] * 4
    fig, (a, b) = hfig(len(labels), 2, head=1.3, foot=0.95, left=0.15)
    hbars(a, t, labels, [30.9, 43.3, 54.0, 86.0, 121.1], colors,
          shown=["30.9", "43.3 (1.40×)", "54.0 (1.75×)", "86.0 (2.78×)", "121.1 (3.92×)"], xmax=175,
          title="Read-outs from many clients (higher is better)", xlabel="requests per second")
    hbars(b, t, labels, [1209, 902, 768, 902, "as one GPU (1,209), plus ~1 ms for the router"], colors,
          shown=["1,209", "902", "768", "902", ""], xmax=1450,
          title="One request at a time, with thoughts (lower is better)", xlabel="mean ms per item")
    frame(fig, (a, b), t, "Several GPUs: StartLux-Decision-v1.1-27B on H200s",
          "StartLux-Prism int8, JevBench. Data parallelism (DP: whole servers behind the router) for throughput, tensor "
          "parallelism (TP: one model split over the GPUs) for the time per request. Every configuration answers as "
          "one GPU does, bit for bit; one GPU in light violet, for scale.",
          note="Read-outs: the most requests/s over 32 to 256 clients (one GPU and TP: from 64 clients). With "
               "thoughts: the 231 items one at a time, 67 of them thinking. Second H200 server (NVSwitch), "
               "2026-10-14.", foot=40)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in CHARTS.items():
        for theme in THEMES:
            save(fn(theme), name, theme)
    print(f"{2 * len(CHARTS)} files in {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
