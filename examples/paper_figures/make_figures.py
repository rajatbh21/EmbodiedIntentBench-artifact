"""Render evaluation Figs. 4-7 (PDF + PNG) from ``figure_data.json``.

Responsibility: drawing only. Every plotted number comes from
``figure_data.json``, which ``verify_sources.py`` verifies against
the source artifacts; this script derives nothing except layout and
percentages of stated denominators.

Typography: all text is 20 pt Times New Roman (metric-compatible Liberation
Serif when Times New Roman is not installed; the font used is printed and
recorded in ``render_receipt.json``). Canvases are 16 in wide, so at ACM
``\\textwidth`` (7.0 in) the scale is 0.4375 and the text prints at 8.75 pt.
Fonts are embedded as TrueType (Type 42), never Type 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
OUTPUT = HERE
DATA = json.loads((HERE / "figure_data.json").read_text())

FONT_FAMILY = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
SIZE = 20
WIDTH = 16.0  # in; printed at \textwidth = 7.0 in
PRINT_WIDTH = 7.0
TIMES, EN_DASH = "\u00d7", "\u2013"

# Colour roles (validated with the dataviz palette checker; see README.md).
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#d8dee4"
BAND = "#eef1f4"
BLUE = "#2a78d6"  # Quality+risk, urgency-aware; Fig. 4 observed recall
ORANGE = "#eb6834"  # Fixed remote Mask R-CNN; Fig. 4 missions completed
GREEN = "#1baf7a"  # Fixed local Mask R-CNN
PURPLE = "#6250d6"  # Quality, urgency-aware (never shown beside BLUE)
GRAY = "#8c959f"  # urgency-blind counterpart; no-skip protocol
POLICY_COLOR = {
    "risk_current": BLUE,
    "risk_initial": GRAY,
    "fixed_remote_mask": ORANGE,
    "fixed_local_mask": GREEN,
    "quality_current": PURPLE,
}


def resolve_font():
    path = font_manager.findfont(
        font_manager.FontProperties(family=FONT_FAMILY), fallback_to_default=False
    )
    return font_manager.FontProperties(fname=path).get_name(), path


FONT_NAME, FONT_PATH = resolve_font()
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": FONT_FAMILY,
        "font.size": SIZE,
        "axes.titlesize": SIZE,
        "axes.labelsize": SIZE,
        "xtick.labelsize": SIZE,
        "ytick.labelsize": SIZE,
        "legend.fontsize": SIZE,
        "legend.title_fontsize": SIZE,
        "figure.titlesize": SIZE,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_NAME,
        "mathtext.it": FONT_NAME + ":italic",
        "mathtext.bf": FONT_NAME + ":bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "lines.linewidth": 3.0,
        "hatch.linewidth": 1.6,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 1.4,
        "legend.borderaxespad": 0.0,
        "legend.borderpad": 0.0,
        "svg.hashsalt": "missionbench-paper-figures",
    }
)


# ---------------------------------------------------------------- helpers
def figure(height):
    return plt.figure(figsize=(WIDTH, height))


def axes(fig, left, bottom, width, height):
    """Add axes at an absolute position in inches (exact, reproducible layout)."""
    w, h = fig.get_size_inches()
    return fig.add_axes((left / w, bottom / h, width / w, height / h))


def text_in(fig, x, y, s, **kw):
    w, h = fig.get_size_inches()
    return fig.text(x / w, y / h, s, **kw)


def panel_title(fig, x, y, label, title):
    """'(a) Title' with a bold panel label, baseline-aligned at (x, y) inches."""
    t = text_in(fig, x, y, label, fontweight="bold", ha="left", va="baseline")
    fig.canvas.draw()
    bb = t.get_window_extent()
    x2 = x + bb.width / fig.dpi + 0.12
    text_in(fig, x2, y, title, ha="left", va="baseline")


def bare(ax, *, left=False, bottom=True):
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if not left:
        ax.tick_params(axis="y", left=False, labelleft=False)


def save(fig, name):
    pdf = OUTPUT / f"{name}.pdf"
    png = OUTPUT / f"{name}.png"
    fig.savefig(
        pdf,
        metadata={"CreationDate": None, "ModDate": None, "Creator": "make_figures.py"},
    )
    fig.savefig(png, dpi=300, metadata={"Software": None})
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------- Fig. 4
def fig4():
    d = DATA["fig4"]
    rows = sorted(d["rows"], key=lambda r: -r["observed_recall"])
    n = len(rows)
    hl = set(d["highlight"])
    fig = figure(6.15)
    bottom, row_h = 0.95, 0.5
    top = bottom + n * row_h
    ax_r = axes(fig, 3.45, bottom, 3.55, n * row_h)
    ax_s = axes(fig, 7.45, bottom, 3.0, n * row_h)
    ax_p = axes(fig, 11.75, bottom, 4.05, n * row_h)
    ys = list(range(n))
    for ax in (ax_r, ax_s, ax_p):
        ax.set_ylim(n - 0.5, -0.5)
        for i, r in enumerate(rows):
            if r["config"] in hl:
                ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0, lw=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, lw=1.2)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax_r.set_yticks(ys, [r["label"] for r in rows])
    for tick, r in zip(ax_r.get_yticklabels(), rows, strict=True):
        if r["config"] in hl:
            tick.set_fontweight("bold")
    for ax in (ax_s, ax_p):
        ax.tick_params(axis="y", labelleft=False)

    # (a) observed recall | missions completed, on separate scales.
    rec = [r["observed_recall"] for r in rows]
    ax_r.barh(ys, rec, height=0.62, color=BLUE, zorder=2)
    for y, v in zip(ys, rec, strict=True):
        ax_r.text(v + 0.02, y, f"{v:.3f}", va="center", ha="left")
    ax_r.set_xlim(0, 1.16)
    ax_r.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".50", ".75", "1"])
    ax_r.set_xlabel(f"Observed recall (n = {d['visible_observations']})")

    succ = [r["successes"] for r in rows]
    ax_s.barh(ys, succ, height=0.62, color=ORANGE, zorder=2)
    for y, v in zip(ys, succ, strict=True):
        ax_s.text(v + 0.3, y, f"{v}/{d['conditions']}", va="center", ha="left")
    ax_s.set_xlim(0, 19.2)
    ax_s.set_xticks([0, 4, 8, 12, 16])
    ax_s.set_xlabel(f"Missions completed (of {d['conditions']})")

    # (b) frozen predicted vs observed recall (dumbbell).
    for y, r in zip(ys, rows, strict=True):
        p, o = r["predicted_recall"], r["observed_recall"]
        ax_p.plot([p, o], [y, y], color=GRAY, lw=2.4, zorder=2, solid_capstyle="butt")
        ax_p.plot(p, y, "o", ms=13, mfc="white", mec=INK, mew=2.2, zorder=3)
        ax_p.plot(o, y, "o", ms=13, mfc=BLUE, mec="white", mew=1.6, zorder=4)
    # Direct labels where the row has free space to the right of the marker.
    index = {r["config"]: i for i, r in enumerate(rows)}
    for config, key, word in (
        ("CFG_CABINET_UAVID_1280", "predicted_recall", "predicted"),
        ("CFG_YOLO11N_640_ROT0", "observed_recall", "observed"),
    ):
        i = index[config]
        ax_p.annotate(
            word,
            (rows[i][key], i),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=MUTED,
        )
    ax_p.set_xlim(-0.03, 1.03)
    ax_p.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".50", ".75", "1"])
    ax_p.set_xlabel("Recall")

    y_title = top + 0.34
    panel_title(fig, 0.08, y_title, "(a)", "Depot: full-bank recall vs. mission success")
    panel_title(fig, 11.55, y_title, "(b)", "Predicted vs. observed recall")
    return save(fig, "fig4")


# ---------------------------------------------------------------- Fig. 5
def fig5():
    d = DATA["fig5"]
    fig = figure(4.3)
    bottom, h = 0.58, 2.3
    mini_title_y = bottom + h + 0.2
    panel_y = mini_title_y + 0.46

    # (a) urgency adaptation: three metrics, three scales.
    u = d["urgency"]
    pols = ("risk_current", "risk_initial")
    metrics = (
        ("successes", f"Successes (of {u['risk_current']['runs']})", 48, "{:d}"),
        ("deadline_misses", "Late requests", 100, "{:d}"),
        ("mean_nonflight_energy_j", "Mean task energy (J)", 225, "{:.0f}"),
    )
    x0, w, gap = 0.15, 2.05, 0.55
    for k, (key, title, ymax, fmt) in enumerate(metrics):
        ax = axes(fig, x0 + k * (w + gap), bottom, w, h)
        vals = [u[p][key] for p in pols]
        ax.bar([0, 1], vals, width=0.66, color=[POLICY_COLOR[p] for p in pols], zorder=2)
        for x, v in zip([0, 1], vals, strict=True):
            ax.text(
                x,
                v + ymax * 0.03,
                fmt.format(round(v) if key.startswith("mean") else v),
                ha="center",
                va="bottom",
            )
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, ymax)
        ax.set_xticks([])
        bare(ax)
        text_in(fig, x0 + k * (w + gap) + w / 2, mini_title_y, title, ha="center", va="baseline")
    a_right = x0 + 3 * w + 2 * gap

    # (b) connectivity, (c) battery: successes of 24 per level, shared scale.
    def grouped(left, width, key, levels, ticklabels, policies):
        ax = axes(fig, left, bottom, width, h)
        for j, p in enumerate(policies):
            vals = [d[key][p]["levels"][lv]["successes"] for lv in levels]
            xs = [i + (j - 0.5) * 0.36 for i in range(len(levels))]
            ax.bar(xs, vals, width=0.34, color=POLICY_COLOR[p], zorder=2)
            for x, v in zip(xs, vals, strict=True):
                ax.text(x, v + 0.7, f"{v}", ha="center", va="bottom")
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 24)
        ax.set_xticks([0, 1], ticklabels)
        ax.tick_params(axis="x", length=0)
        bare(ax)
        runs = {d[key][p]["levels"][lv]["runs"] for p in policies for lv in levels}
        (n,) = runs
        text_in(
            fig, left + width / 2, mini_title_y, f"Successes (of {n})", ha="center", va="baseline"
        )
        return ax

    b_left = a_right + 0.85
    grouped(
        b_left,
        3.3,
        "link",
        ("good", "degraded"),
        ["Good link", "Degraded link"],
        ("fixed_remote_mask", "risk_current"),
    )
    c_left = b_left + 3.3 + 0.75
    grouped(
        c_left,
        3.3,
        "battery",
        ("ample", "residual_250j"),
        ["Ample", "Constrained"],
        ("fixed_local_mask", "risk_current"),
    )

    panel_title(fig, 0.08, panel_y, "(a)", "Lake: urgency adaptation")
    panel_title(fig, b_left - 0.1, panel_y, "(b)", "Connectivity")
    panel_title(fig, c_left - 0.1, panel_y, "(c)", "Battery")
    handles = [
        Patch(color=POLICY_COLOR[p], label=d_label(p))
        for p in ("risk_current", "risk_initial", "fixed_remote_mask", "fixed_local_mask")
    ]
    w_, h_ = fig.get_size_inches()
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=4,
        bbox_to_anchor=(0.08 / w_, (h_ - 0.08) / h_),
        handlelength=1.0,
    )
    return save(fig, "fig5")


def d_label(policy):
    labels = {
        "risk_current": "Quality+risk, urgency-aware",
        "risk_initial": "Quality+risk, urgency-blind",
        "quality_current": "Quality, urgency-aware",
        "fixed_remote_mask": "Fixed remote Mask R-CNN",
        "fixed_local_mask": "Fixed local Mask R-CNN",
    }
    return labels[policy]


# ---------------------------------------------------------------- Fig. 6
def fig6():
    d = DATA["fig6"]
    fig = figure(6.28)
    # Row 2 geometry first: (c) timeline of one predeclared run.
    r2_bottom, r2_h = 0.85, 1.45
    legend_y = r2_bottom + r2_h + 0.14
    title2 = legend_y + 0.5
    # Row 1: (a) latency sweep, (b) ranking reversal.
    r1_bottom = title2 + 0.78
    r1_h = 1.95
    title1 = r1_bottom + r1_h + 0.3

    ax = axes(fig, 1.3, r1_bottom, 6.9, r1_h)
    mults = [p["multiplier"] for p in d["sweep"]["causal"]]
    xs = list(range(len(mults)))
    styles = {
        "no_skip": dict(color=GRAY, ls=(0, (4, 2.2)), mfc="white", mec=GRAY),
        "causal": dict(color=INK, ls="-", mfc=INK, mec=INK),
    }
    for proto, st in styles.items():
        vals = [p["successes"] for p in d["sweep"][proto]]
        ax.plot(
            xs,
            vals,
            color=st["color"],
            ls=st["ls"],
            marker="o",
            ms=12,
            mfc=st["mfc"],
            mec=st["mec"],
            mew=2.4,
            zorder=3,
        )
        for x, v in zip(xs, vals, strict=True):
            if proto == "no_skip":
                off, ha, va = (0, 13), "center", "bottom"
            elif x == len(xs) - 1:
                off, ha, va = (14, -2), "left", "center"
            else:
                off, ha, va = (0, -13), "center", "top"
            ax.annotate(f"{v}", (x, v), xytext=off, textcoords="offset points", ha=ha, va=va)
    ax.text(1.5, 262, "No-skip", ha="center", va="bottom", color=MUTED)
    ax.text(1.5, 168, "Causal", ha="center", va="top", color=INK)
    ax.set_xticks(xs, [f"{m:g}{TIMES}" for m in mults])
    ax.set_xlim(-0.3, len(xs) - 0.55)
    ax.set_ylim(0, 340)
    ax.set_yticks([0, 100, 200, 300])
    n_runs = d["sweep"]["causal"][0]["runs"]
    ax.set_ylabel(f"Successes (of {n_runs})")
    ax.grid(axis="y", color=GRID, lw=1.2)
    ax.set_axisbelow(True)

    bx = axes(fig, 9.75, r1_bottom, 2.2, r1_h)
    rank = d["ranking"]
    for p in ("fixed_remote_mask", "quality_current"):
        vals = [rank[p]["causal"], rank[p]["no_skip"]]
        bx.plot([0, 1], vals, color=POLICY_COLOR[p], marker="o", ms=12, mew=0, zorder=3)
        bx.annotate(
            f"{vals[0]}",
            (0, vals[0]),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
        )
        bx.annotate(
            f"{vals[1]}",
            (1, vals[1]),
            xytext=(12, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )
        bx.annotate(
            d_label(p),
            (1, vals[1]),
            xytext=(50, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            annotation_clip=False,
        )
    bx.set_xticks([0, 1], ["Causal", "No-skip"])
    bx.set_xlim(-0.35, 1.12)
    bx.set_ylim(35, 50)
    bx.set_yticks([36, 40, 44, 48])
    bx.set_ylabel("Successes (of 72)")
    bx.grid(axis="y", color=GRID, lw=1.2)
    bx.set_axisbelow(True)
    bx.tick_params(axis="x", length=0)

    panel_title(fig, 0.08, title1, "(a)", "Lake: success vs. latency multiplier")
    panel_title(fig, 8.65, title1, "(b)", f"Ranking reversal at 1{TIMES} latency")

    t = d["timeline"]
    cx = axes(fig, 1.3, r2_bottom, 14.55, r2_h)
    rows = (("causal", 1.0, "Causal"), ("no_skip", 0.0, "No-skip"))
    for proto, y, _ in rows:
        caps = t[proto]["captures"]
        if proto == "causal":
            for c in caps:
                if c["processed"]:
                    cx.add_patch(
                        Rectangle(
                            (c["t"], y - 0.19),
                            c["done"] - c["t"],
                            0.38,
                            fc=BAND,
                            ec=GRAY,
                            lw=1.2,
                            zorder=1,
                        )
                    )
        for c in caps:
            if not c["processed"]:
                cx.plot(c["t"], y, "o", ms=12, mfc="white", mec=INK, mew=2.0, zorder=3)
            elif 3 in c["matched"]:
                cx.plot(c["t"], y, "o", ms=13, mfc=GREEN, mec=INK, mew=1.6, zorder=4)
            else:
                cx.plot(c["t"], y, "o", ms=12, mfc=INK, mec=INK, zorder=3)
        (rep,) = t[proto]["reports"]
        cx.plot(rep["done"], y + 0.42, "v", ms=15, color=INK, zorder=5, clip_on=False)
        cx.annotate(
            f"report {rep['done']:.1f} s",
            (rep["done"], y + 0.42),
            xytext=(-12, 0),
            textcoords="offset points",
            ha="right",
            va="center",
        )
        n_proc = sum(c["processed"] for c in caps)
        cx.text(37.0, y, f"{n_proc}/{len(caps)} processed", ha="left", va="center")
    cx.axvline(t["pass_end_s"], color=MUTED, lw=1.4, ls=":", zorder=1)
    cx.set_yticks([y for _, y, _ in rows], [lab for _, _, lab in rows])
    cx.tick_params(axis="y", length=0)
    cx.spines["left"].set_visible(False)
    cx.set_ylim(-0.45, 1.62)
    cx.set_xlim(-0.8, 42.2)
    cx.set_xticks(list(range(0, 37, 4)))
    cx.set_xlabel("Mission time (s); dotted line: end of pass")
    panel_title(fig, 0.08, title2, "(c)", f"{d_label('fixed_local_mask')}, one predeclared run")
    handles = [
        Line2D([], [], ls="", marker="o", ms=12, mfc=INK, mec=INK, label="capture processed"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=12,
            mfc="white",
            mec=INK,
            mew=2.0,
            label="skipped while busy",
        ),
        Patch(fc=BAND, ec=GRAY, lw=1.2, label="inference busy"),
        Line2D(
            [], [], ls="", marker="o", ms=13, mfc=GREEN, mec=INK, mew=1.6, label="person 3 matched"
        ),
        Line2D([], [], ls="", marker="v", ms=15, color=INK, label="report"),
    ]
    w_, h_ = fig.get_size_inches()
    fig.legend(
        handles=handles,
        loc="lower left",
        ncol=5,
        bbox_to_anchor=(1.3 / w_, legend_y / h_),
        handlelength=1.0,
        columnspacing=1.6,
    )
    return save(fig, "fig6")


# ---------------------------------------------------------------- Fig. 7
def fig7():
    d = DATA["fig7"]
    fig = figure(3.55)
    bottom, h = 0.12, 2.75
    title_y = bottom + h + 0.33

    # (a) tall segmented bar: 72 conditions, four mutually exclusive classes.
    ax = axes(fig, 0.15, bottom, 0.85, h)
    total = sum(b["conditions"] for b in d["feasibility"])
    fills = {
        "C_released_policy": dict(fc="#4a5560", hatch=None),
        "C_hindsight_witness": dict(fc="#8c959f", hatch=None),
        "B_unresolved": dict(fc="#d0d7de", hatch=None),
        "B_model_pool_infeasible": dict(fc="white", hatch="///"),
    }
    y = total
    centers = []
    for b in d["feasibility"]:
        y -= b["conditions"]
        f = fills[b["cls"]]
        ax.bar(
            0,
            b["conditions"],
            bottom=y,
            width=1.0,
            color=f["fc"],
            hatch=f["hatch"],
            edgecolor=INK if f["hatch"] else "white",
            lw=1.6 if f["hatch"] else 2.0,
        )
        centers.append((y + b["conditions"] / 2, b))
    for c, b in centers:
        ax.annotate(
            f"{b['conditions']}  {b['label']}",
            (0.5, c),
            xytext=(14, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            annotation_clip=False,
        )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, total)
    ax.axis("off")
    panel_title(fig, 0.08, title_y, "(a)", f"Lake: feasibility of {total} conditions")

    # (b) evidence funnel for Quality+risk, urgency-aware.
    f = d["funnel"]
    n = f["instances"]
    bx = axes(fig, 9.55, bottom + 0.05, 4.6, h - 0.1)
    st = f["stages"]
    ys = list(range(len(st)))
    vals = [s["count"] for s in st]
    bx.barh(ys, vals, height=0.62, color=BLUE)
    for y_, v in zip(ys, vals, strict=True):
        pct = "" if v == n else f" ({100 * v / n:.0f}%)"
        bx.text(v + n * 0.02, y_, f"{v}{pct}", ha="left", va="center")
    bx.set_yticks(ys, [s["label"] for s in st])
    bx.set_ylim(len(st) - 0.5, -0.5)
    bx.set_xlim(0, n)
    bx.tick_params(axis="y", length=0)
    bare(bx, bottom=False)
    bx.tick_params(axis="y", labelleft=True)
    bx.tick_params(axis="x", bottom=False, labelbottom=False)
    panel_title(
        fig,
        6.0,
        title_y,
        "(b)",
        f"{d_label('risk_current')}: {n} person{EN_DASH}condition instances",
    )
    return save(fig, "fig7")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    global OUTPUT, FONT_NAME, FONT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE, help="Directory for PDFs, PNGs and the render receipt")
    parser.add_argument("--font-dir", type=Path, help="Optional directory of locally licensed TTF/OTF fonts; fonts are not copied into the release")
    args = parser.parse_args()
    if args.font_dir:
        if not args.font_dir.is_dir():
            parser.error("--font-dir must name an existing directory")
        for path in sorted(args.font_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
                font_manager.fontManager.addfont(str(path))
        FONT_NAME, FONT_PATH = resolve_font()
        plt.rcParams.update({"mathtext.rm": FONT_NAME, "mathtext.it": FONT_NAME + ":italic",
                             "mathtext.bf": FONT_NAME + ":bold"})
    OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for build in (fig4, fig5, fig6, fig7):
        for path in build():
            outputs[path.name] = sha256(path)
    receipt = dict(
        font=dict(
            requested="Times New Roman",
            used=FONT_NAME,
            file=Path(FONT_PATH).name,
            sha256=sha256(FONT_PATH),
            size_pt=SIZE,
            print_size_pt=round(SIZE * PRINT_WIDTH / WIDTH, 2),
        ),
        canvas_width_in=WIDTH,
        print_width_in=PRINT_WIDTH,
        matplotlib=matplotlib.__version__,
        renderer_sha256=sha256(__file__),
        figure_data_sha256=sha256(HERE / "figure_data.json"),
        outputs=outputs,
    )
    (OUTPUT / "render_receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    if FONT_NAME != "Times New Roman":
        print(f"note: Times New Roman not installed; used {FONT_NAME}")
    print("wrote", ", ".join(sorted(outputs)))


if __name__ == "__main__":
    main()
