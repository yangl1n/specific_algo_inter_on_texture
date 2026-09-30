"""Wavelet-likeness figures of the paper.

    python plots.py --out <results dir> [--ds-root <deep_scatter dir>]

Reads ``<out>/metrics.parquet`` (per-filter metrics, from ``wavelet_likeness.run_metrics``)
and the ``summary.csv`` (from ``analyze.py``) of every results directory those runs
live in, for each run's fine-tuning distance ``filter_delta_relative`` =
``||theta - theta_0|| / ||theta_0||`` of the extractor, as logged by ``train.py``.
Writes to ``<out>/figures``:

* ``rho_vs_drift_checkpoint``: Morlet residual (median and worst filter of each
  checkpoint) against the checkpoint's fine-tuning distance.
* ``admissibility_vs_train_fraction``: relative DC component of the band-pass filters.
* ``param_shifted``: Morlet-parameter shifts from the matched Theta_0 filter.

All figures use the stored band-pass filters (layer_kind == 'stored',
filter_index >= 1).
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns


# ---------------------------------------------------------------------------
# styling (same dataset names, colours and markers as paper_plots.py)
# ---------------------------------------------------------------------------

DATASET_LABEL = {"dtd": "DTD", "eurosat": "EuroSat", "minc2500": "MinC-2500",
                 "surotex": "SuroTex"}
DATASET_ORDER = ["DTD", "EuroSat", "MinC-2500", "SuroTex"]
METHOD_COLOR = {
    "Random Init.":     "#2a78d6",
    "Scattering":       "#eb6834",   # frozen filters = Theta_0
    "Fine-tuned Scat.": "#1baf7a",
}
METHOD_MARKER = {"Fine-tuned Scat.": "o", "Random Init.": "X", "Scattering": "s"}
L2_STYLE = {0.0: "-", 0.1: "--", 1.0: ":"}
L2_LEVELS = (0.0, 0.1, 1.0)
# checkpoint_type -> method; init_random (random filters before training) is drawn as a band
METHOD_OF = {"finetuned": "Fine-tuned Scat.", "trained_random": "Random Init.",
             "init_scattering": "Scattering"}
UNTRAINED_LABEL = "Random Init. (untrained)"
GRID = dict(linewidth=0.4, alpha=0.35)

sns.set_theme(font_scale=2.5, style="white", font="serif")
LEGEND_PT, LEGEND_TITLE_PT = 15, 16
LABEL_PAD_EM, LEGEND_PAD_EM = 0.75, 0.8
REG_TITLE = r"Reg. ($\lambda$)"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

F_KEYS = ["dataset", "lambda", "train_fraction", "seed", "part", "checkpoint_type", "run",
          "layer_kind", "layer", "channel", "filter_index"]
F_METRICS = ["morlet_residual", "d_theta_deg", "d_xi_rel", "d_sigma_rel", "dc_leakage"]


def load(out):
    # read only the columns used here, strings as Arrow strings and metrics as float32
    f = pq.read_table(os.path.join(out, "metrics.parquet"), columns=F_KEYS + F_METRICS
                      ).to_pandas(types_mapper={pa.string(): pd.StringDtype("pyarrow")}.get)
    f[F_METRICS] = f[F_METRICS].astype("float32")
    f["dataset"] = f["dataset"].map(lambda d: DATASET_LABEL.get(d, d))
    return f


def per_checkpoint(f):
    """Median and max Morlet residual and DC leakage of each checkpoint's stored
    band-pass filters."""
    keys = ["dataset", "lambda", "train_fraction", "seed", "part", "checkpoint_type", "run"]
    sel = f[(f.layer_kind == "stored") & (f.filter_index >= 1)]
    return sel.groupby(keys, dropna=False).agg(
        rho=("morlet_residual", "median"), rho_max=("morlet_residual", "max"),
        dc_leakage=("dc_leakage", "median"),
        dc_leakage_max=("dc_leakage", "max")).reset_index()


def attach_drift(ck, ds_root):
    """Add each run's ``filter_delta_relative`` from the summary.csv of its results
    directory (``<ds_root>/<run>/..``); ``run`` is relative to ``ds_root``."""
    frames = []
    for parent in sorted(ck["run"].dropna().map(os.path.dirname).unique()):
        path = os.path.join(ds_root, parent, "summary.csv")
        if not os.path.exists(path):
            print(f"[drift] no {path}; its runs get no filter_delta_relative")
            continue
        d = pd.read_csv(path, usecols=["_path", "filter_delta_relative"])
        d["run"] = d["_path"].map(lambda p: os.path.relpath(
            os.path.dirname(p if os.path.isabs(p) else os.path.join(ds_root, p)), ds_root))
        frames.append(d[["run", "filter_delta_relative"]])
    if not frames:
        ck["filter_delta_relative"] = np.nan
        return ck
    return ck.merge(pd.concat(frames).drop_duplicates("run"), on="run", how="left")


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------

def _datasets(ck):
    present = set(ck.loc[ck.checkpoint_type == "finetuned", "dataset"].dropna())
    return [d for d in DATASET_ORDER if d in present]


def _panels(n_rows, n_cols, h=4.2, w=4.0):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(w * n_cols, h * n_rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.grid(True, **GRID)
    return fig, axes


def _arch_handles(methods, untrained=False):
    h = [mlines.Line2D([], [], color=METHOD_COLOR[m], linewidth=3,
                       marker=METHOD_MARKER[m], markersize=10, markeredgecolor="white",
                       label=m) for m in methods]
    if untrained:
        h.append(plt.Rectangle((0, 0), 1, 1, color=METHOD_COLOR["Random Init."],
                               alpha=0.15, label=UNTRAINED_LABEL))
    return h


def _reg_handles(levels=L2_LEVELS):
    return [mlines.Line2D([], [], color="0.35", linestyle=L2_STYLE[l], linewidth=2.5,
                          label=f"{l:g}") for l in levels]


def _fig_frac(fig, bbox):
    return bbox.transformed(fig.transFigure.inverted())


def _em(fig, pt, vertical=True):
    """``pt`` points as a figure fraction."""
    size = fig.get_figheight() if vertical else fig.get_figwidth()
    return pt / 72.0 / size


def _thin_log_x(ax):
    """At most two decade ticks on a log x axis."""
    if ax.get_xscale() != "log":
        return
    lo, hi = ax.get_xlim()
    decs = np.arange(np.ceil(np.log10(lo)), np.floor(np.log10(hi)) + 1)
    if len(decs) == 0:
        return
    step = int(np.ceil(len(decs) / 2))
    ax.set_xticks(10.0 ** decs[::-1][::step][::-1])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _finish(fig, axes, xlabel, ylabel, sections, row_labels=None):
    """Dataset titles on the top row only, one shared x and y label, optional per-row
    tags, and the legend as a row of titled blocks under the x label.

    ``sections``: [(title, handles, ncol), ...] -- one legend block each."""
    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
        _thin_log_x(ax)
    for ax in axes[1:].ravel():
        ax.set_title("")
    if row_labels:
        for ax, tag in zip(axes[:, 0], row_labels):
            ax.set_ylabel(tag, fontsize=plt.rcParams["xtick.labelsize"])
    fig.tight_layout()
    r = fig.canvas.get_renderer()
    ink = _fig_frac(fig, matplotlib.transforms.Bbox.union(
        [ax.get_tightbbox(r) for ax in axes.ravel()]))
    boxes = matplotlib.transforms.Bbox.union([ax.get_position() for ax in axes.ravel()])
    centre = 0.5 * (boxes.x0 + boxes.x1)
    lab = plt.rcParams["axes.labelsize"]
    xl = fig.text(centre, ink.y0 - _em(fig, lab) * LABEL_PAD_EM, xlabel,
                  ha="center", va="top", fontsize=lab)
    fig.text(ink.x0 - _em(fig, lab, vertical=False) * LABEL_PAD_EM,
             0.5 * (boxes.y0 + boxes.y1), ylabel, ha="right", va="center",
             rotation="vertical", fontsize=lab)
    y = _fig_frac(fig, xl.get_window_extent(r)).y0 - _em(fig, lab) * LEGEND_PAD_EM
    legs = []
    for title, handles, ncol in sections:
        lg = fig.legend(handles=handles, title=title, loc="upper left",
                        bbox_to_anchor=(0, y), frameon=False, ncol=ncol,
                        prop={"size": LEGEND_PT}, title_fontsize=LEGEND_TITLE_PT,
                        labelspacing=0.5, handlelength=2.6, columnspacing=1.2)
        lg._legend_box.align = "left"
        legs.append(lg)
    fig.canvas.draw()
    widths = [_fig_frac(fig, lg.get_window_extent(r)).width for lg in legs]
    gap = 2.5 * _em(fig, LEGEND_TITLE_PT, vertical=False)
    x = centre - (sum(widths) + gap * (len(legs) - 1)) / 2
    for lg, wdt in zip(legs, widths):
        lg.set_bbox_to_anchor((x, y), transform=fig.transFigure)
        x += wdt + gap


def _save(fig, out, name, dpi=200):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {d}/{name}.png")


# ---------------------------------------------------------------------------
# admissibility_vs_train_fraction
# ---------------------------------------------------------------------------

def _mean_sd(d, x, col):
    return d.groupby(x)[col].agg(["mean", "std"]).reset_index().sort_values(x)


def fig_train_size(ck, rows, fname, out, ylabel):
    """Metric(s) vs train size on log-log axes, one row of dataset panels per
    ``(col, row label)`` in ``rows``: Fine-tuned Scat. per lambda (line style),
    Random Init. at lambda = 0, and the untrained random filters as a band.  Mean +/- sd
    over seeds (and DTD partitions).  Theta_0 is exactly 0 here, so it is not drawn."""
    dss = _datasets(ck)
    if not dss:
        return
    fig, axes = _panels(len(rows), len(dss))
    for i, (col, _) in enumerate(rows):
        for ax, ds in zip(axes[i], dss):
            d = ck[ck.dataset == ds]
            u = d[d.checkpoint_type == "init_random"][col].dropna()
            if len(u):
                ax.axhspan(u.quantile(0.1), u.quantile(0.9),
                           color=METHOD_COLOR["Random Init."], alpha=0.15, lw=0, zorder=0)
            series = [("Random Init.", d[(d.checkpoint_type == "trained_random")
                                         & np.isclose(d["lambda"], 0.0)], 0.0)]
            series += [("Fine-tuned Scat.", d[(d.checkpoint_type == "finetuned")
                                              & np.isclose(d["lambda"], lam)], lam)
                       for lam in L2_LEVELS]
            for method, sub, lam in series:
                m = _mean_sd(sub.dropna(subset=["train_fraction", col]), "train_fraction", col)
                if m.empty:
                    continue
                c = METHOD_COLOR[method]
                # a band below 0 is meaningless on a log axis
                lo = (m["mean"] - m["std"].fillna(0)).clip(lower=m["mean"] * 0.5)
                ax.fill_between(m.train_fraction, lo, m["mean"] + m["std"].fillna(0),
                                color=c, alpha=0.15, lw=0)
                ax.plot(m.train_fraction, m["mean"], color=c, linestyle=L2_STYLE[lam],
                        marker=METHOD_MARKER[method], markersize=9, markeredgecolor="white",
                        linewidth=2.2, alpha=0.9, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(ds)
    arch = _arch_handles(["Random Init.", "Fine-tuned Scat."], untrained=True)
    _finish(fig, axes, "Train Size", ylabel,
            [("Arch.", arch, 1), (REG_TITLE, _reg_handles(), 1)],
            row_labels=[lab for _, lab in rows])
    _save(fig, out, fname)


# ---------------------------------------------------------------------------
# rho_vs_drift_checkpoint
# ---------------------------------------------------------------------------

# a few train sizes only, with grey "rungs" joining the lambda levels of one train size
DRIFT_SIZES = (0.01, 0.1, 0.5, 1.0)
DRIFT_RUNG = dict(color="0.45", linewidth=1.0, alpha=0.55)
# Fine-tuned Scat. marker per lambda; Random Init. (lambda = 0 only) keeps 'X'
DRIFT_L2_MARKER = {0.0: "o", 0.1: "s", 1.0: "^"}


def _size_area(sizes):
    """Log-spaced marker area, 70..430 pt^2."""
    lo, hi = np.log10(min(sizes)), np.log10(max(sizes))
    return lambda t: 70 + (np.log10(t) - lo) / (hi - lo + 1e-9) * (430 - 70)


def fig_drift(ck, out, rows=(("rho", "Median filter"), ("rho_max", "Worst filter")),
              fname="rho_vs_drift_checkpoint", sizes=DRIFT_SIZES):
    """Morlet residual vs fine-tuning distance: hue + shape = arm, line style = lambda,
    marker area = train size; one path per (arm, lambda) over train size.  One row of
    panels per statistic of the per-checkpoint rho distribution (median filter, worst
    fitted filter).  Random Init. at lambda = 0 only; Scattering has zero drift and no
    place on the log axis."""
    d = ck[ck.checkpoint_type.isin(["finetuned", "trained_random"])]
    d = d[(d.checkpoint_type == "finetuned") | np.isclose(d["lambda"], 0.0)]
    d = d.dropna(subset=["filter_delta_relative"])
    d = d[d.filter_delta_relative > 0]
    if d.empty:
        print("[drift] no filter_delta_relative; figure skipped")
        return
    d = d[d.train_fraction.round(4).isin([round(x, 4) for x in sizes])]
    agg = (d.assign(method=d.checkpoint_type.map(METHOD_OF))
           .groupby(["dataset", "method", "lambda", "train_fraction"])
           .agg(fd=("filter_delta_relative", "mean"), **{c: (c, "mean") for c, _ in rows})
           .reset_index())
    dss = [x for x in _datasets(ck) if x in set(agg.dataset)]
    sizes = sorted(agg.train_fraction.unique())
    area = _size_area(sizes)
    fig, axes = _panels(len(rows), len(dss))
    for i, (col, _) in enumerate(rows):
        # one y range per row, shared by its datasets
        vals = agg.loc[agg.dataset.isin(dss), col].to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        ylo = 10 ** (np.floor(np.log10(vals.min()) * 2) / 2 - 0.25) if len(vals) else 1e-3
        for ax, ds in zip(axes[i], dss):
            a = agg[agg.dataset == ds]
            # rungs first, underneath: the lambda levels of one train size, in order
            for _, grp in a[a.method == "Fine-tuned Scat."].groupby("train_fraction"):
                grp = grp.sort_values("lambda")
                if len(grp) > 1:
                    ax.plot(grp.fd, grp[col], zorder=1, solid_capstyle="round", **DRIFT_RUNG)
            for method in ("Fine-tuned Scat.", "Random Init."):
                for lam in L2_LEVELS:
                    sub = a[(a.method == method) & np.isclose(a["lambda"], lam)].sort_values(
                        "train_fraction")
                    if sub.empty:
                        continue
                    c = METHOD_COLOR[method]
                    ax.plot(sub.fd, sub[col], color=c, linestyle=L2_STYLE[lam],
                            linewidth=2.0, alpha=0.9, zorder=2)
                    mk = (DRIFT_L2_MARKER[lam] if method == "Fine-tuned Scat."
                          else METHOD_MARKER[method])
                    ax.scatter(sub.fd, sub[col], marker=mk, color=c,
                               s=[0.5 * area(t) for t in sub.train_fraction],
                               edgecolor="white", linewidth=1.0, zorder=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ylo, 1.6)
            ax.margins(x=0.12)
            ax.set_title(ds)
    shown = sorted(set(agg.train_fraction))
    size_h = [
        mlines.Line2D([], [], color="0.35", marker="o", linestyle="none",
                      markersize=np.sqrt(0.5 * area(s)) * 0.55, label=f"{s * 100:g}%")
        for s in shown]
    rung_h = [mlines.Line2D([], [], label="same train size", **DRIFT_RUNG)]
    # one entry per series (arm x lambda), carrying its hue, marker and line style
    arch_h = [mlines.Line2D([], [], color=METHOD_COLOR["Random Init."], linewidth=2.5,
                            marker=METHOD_MARKER["Random Init."], markersize=10,
                            markeredgecolor="white", label="Random Init.")]
    arch_h += [mlines.Line2D([], [], color=METHOD_COLOR["Fine-tuned Scat."],
                             linestyle=L2_STYLE[l], linewidth=2.5, marker=DRIFT_L2_MARKER[l],
                             markersize=10, markeredgecolor="white",
                             label=fr"Fine-tuned Scat. ($\lambda$ = {l:g})")
               for l in L2_LEVELS]
    _finish(fig, axes, "Fine-tuning Distance", r"Morlet residual $\rho$",
            [("Arch.", arch_h, 1), ("Link", rung_h, 1),
             ("Train Size", size_h, 2)],
            row_labels=[lab for _, lab in rows])
    _save(fig, out, fname)


# ---------------------------------------------------------------------------
# param_shifted
# ---------------------------------------------------------------------------

def fig_shifts(f, out):
    """Morlet-parameter shifts from the matched Theta_0 filter: Fine-tuned Scat. at each
    lambda (box edge dash = lambda) next to Random Init. at lambda = 0.  The y axis is
    symlog, linear inside the 90th percentile of the fine-tuned lambda = 0 shifts."""
    st = f[(f.layer_kind == "stored") & (f.filter_index >= 1)].dropna(subset=["morlet_residual"])
    ft = st[st.checkpoint_type == "finetuned"]
    rr = st[(st.checkpoint_type == "trained_random") & np.isclose(st["lambda"], 0.0)]
    dss = _datasets(ft)
    if not dss:
        return
    qs = [("d_theta_deg", r"$\Delta\theta$ (deg)"), ("d_xi_rel", r"$\Delta\xi/\xi_0$"),
          ("d_sigma_rel", r"$\Delta\sigma/\sigma_0$")]
    arms = [("Fine-tuned Scat.", lam) for lam in L2_LEVELS] + [("Random Init.", 0.0)]
    pos = [0, 1, 2, 3.4]
    fig, axes = _panels(len(qs), len(dss), h=3.6, w=4.0)
    for i, (col, lab) in enumerate(qs):
        base = ft[np.isclose(ft["lambda"], 0.0)][col].abs().dropna()
        lin = float(base.quantile(0.9)) if len(base) else 1.0
        for j, ds in enumerate(dss):
            ax = axes[i][j]
            data = []
            for method, lam in arms:
                src = ft if method == "Fine-tuned Scat." else rr
                data.append(src[(src.dataset == ds) & np.isclose(src["lambda"], lam)][col]
                            .dropna().values)
            keep = [k for k, v in enumerate(data) if len(v) > 1]
            if keep:
                # whiskers at the 1st/99th percentile, no fliers
                bp = ax.boxplot([data[k] for k in keep], positions=[pos[k] for k in keep],
                                widths=0.6, whis=(1, 99), showfliers=False,
                                patch_artist=True,
                                medianprops=dict(color="0.15", linewidth=1.6))
                for i_box, k in enumerate(keep):
                    method, lam = arms[k]
                    c = METHOD_COLOR[method]
                    box = bp["boxes"][i_box]
                    box.set_facecolor(c)
                    box.set_alpha(0.55)
                    box.set_edgecolor(c)
                    box.set_linewidth(1.6)
                    box.set_linestyle(L2_STYLE[lam])
                    # two whiskers and two caps per box
                    for art in bp["whiskers"][2 * i_box:2 * i_box + 2] + \
                            bp["caps"][2 * i_box:2 * i_box + 2]:
                        art.set_color(c)
                        art.set_linewidth(1.4)
            ax.axhline(0, color="0.35", lw=0.8)
            ax.set_yscale("symlog", linthresh=lin, linscale=1.5)
            # one decade tick each side of the linear band
            vals = np.concatenate([v for v in data if len(v)]) if keep else np.array([lin])
            top = max(np.abs(vals).max(), lin)
            dec = [10.0 ** k for k in range(int(np.ceil(np.log10(lin))),
                                             int(np.floor(np.log10(top))) + 1)]
            dec = dec[-1:]
            if not dec:          # no full decade beyond lin
                dec = [float(f"{lin:.0e}"), float(f"{0.9 * top:.0e}")]
            ax.set_yticks(sorted(set([0.0] + dec + [-t for t in dec])))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: "0" if v == 0 else f"{v:g}"))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks(pos, ["0", "0.1", "1", "0"])
            ax.set_xlim(-0.6, 4.0)
            ax.set_title(ds)
    _finish(fig, axes, r"Reg. $\lambda$", r"Shift from matched $\Theta_0$ Morlet",
            [("Arch.", _arch_handles(["Random Init.", "Fine-tuned Scat."]), 1),
             (REG_TITLE, _reg_handles(), 1)],
            row_labels=[lab for _, lab in qs])
    _save(fig, out, "param_shifted")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="directory with metrics.parquet")
    p.add_argument("--ds-root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   help="deep_scatter directory that the metrics' run paths (and "
                        "summary.csv's _path) are relative to")
    args = p.parse_args()
    f = load(args.out)
    ck = attach_drift(per_checkpoint(f), args.ds_root)
    fig_train_size(ck, [("dc_leakage", "Median filter"), ("dc_leakage_max", "Worst filter")],
                   "admissibility_vs_train_fraction", args.out,
                   ylabel=r"Rel. DC component $|\hat\psi(0)|/\max|\hat\psi|$")
    fig_shifts(f, args.out)
    fig_drift(ck, args.out)


if __name__ == "__main__":
    main()
