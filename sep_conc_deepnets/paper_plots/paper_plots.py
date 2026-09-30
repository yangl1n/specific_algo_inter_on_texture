import argparse
import contextlib
import os
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter

# Facet order in every figure.
DATASET_ORDER = ['DTD', 'EuroSat', 'MinC-2500', 'SuroTex']

# Colour follows the method, so the same hue means the same thing in every figure.
METHOD_COLOR = {
    'Random Init.':     '#2a78d6',
    'Scattering':       '#eb6834',
    'Fine-tuned Scat.': '#1baf7a',
    'ScatNet B.A.':     '#4a3aa7',
}
METHOD_ORDER = ['Random Init.', 'Scattering', 'Fine-tuned Scat.', 'ScatNet B.A.']

# Skewed NC1: within/between-class variance ratio in the classifier subspace.
NC1_SKEW_COL = 'nc1_skew_proj'
NC1_SKEW_MAHAL_COL = 'nc1_skew_mahal'
NC1_SKEW_COLS = (NC1_SKEW_COL, NC1_SKEW_MAHAL_COL)
NC1_PAPYAN_COL = 'nc1_papyan'
# NC columns beyond nc1/nc2/nc3 carried through the recomputed-NC merge.
NC_EXTRA_COLS = NC1_SKEW_COLS + (NC1_PAPYAN_COL,)

# Display names for the `phase` values produced by extract_nc_stats().
PHASE_LABEL = {
    'a': 'Scattering',
    'b': 'Fine-tuned Scat.',
    'joint': 'Random Init.',
    'scatnetblockanalysis': 'ScatNet B.A.',
}

# Parent of deep_scatter/ and sep_conc_deepnets/; override with WPH_ROOT.
WORKSPACE_ROOT = Path(os.environ.get('WPH_ROOT', Path(__file__).resolve().parents[2]))
_REPO_ROOT = str(WORKSPACE_ROOT)

# =========================================================================
# DATASET_SOURCES -- every run directory this script reads
# =========================================================================
# Each entry maps a source kind to {framework: directory} (relative to WORKSPACE_ROOT):
#   summary         primary sweep; its summary.csv and nc_recomputed.csv are read.
#                   Required for both frameworks.
#   l2_disentangle  random-init runs with an L2 penalty toward zero (lambda > 0).
#   l2_anchor       random-init runs with an L2 penalty toward theta_0 / the wavelet init.
#   low_data        extends the primary below its smallest train size.
#   phase_a_nc      --phase-a-only sweep supplying Phase-A NC with a trained head.
#   arm_filter      arm(s) to keep when a summary.csv holds more than one configuration.
# A dataset is used only once every `summary` summary.csv is on disk (dataset_ready()).
DS, SCD = 'deep_scatter', 'sep_conc_deepnets'
FRAMEWORKS = (DS, SCD)
_L2D = 'deep_scatter/training_logs/clean_updateNC/l2_disentangle'

DATASET_SOURCES = {
    'DTD': {
        'summary': {DS: 'deep_scatter/training_logs/clean_v2/dtd',
                    SCD: 'sep_conc_deepnets/training_logs/clean_zarka/dtd'},
        'l2_disentangle': {DS: f'{_L2D}/dtd_v2'},
        'l2_anchor': {DS: f'{_L2D}/dtd_v2_anchors'},
    },
    'EuroSat': {
        'summary': {DS: 'deep_scatter/training_logs/clean_updateNC/eurosat',
                    SCD: 'sep_conc_deepnets/training_logs/clean_zarka/eurosat_v2'},
        'l2_disentangle': {DS: f'{_L2D}/eurosat'},
        'l2_anchor': {DS: f'{_L2D}/eurosat_anchors'},
        'phase_a_nc': {DS: 'deep_scatter/training_logs/phase_a/eurosat'},
    },
    'MinC-2500': {
        'summary': {DS: 'deep_scatter/training_logs/clean_v2/minc',
                    SCD: 'sep_conc_deepnets/training_logs/clean_zarka/minc2500'},
        'l2_disentangle': {DS: f'{_L2D}/minc2500'},
        'l2_anchor': {DS: f'{_L2D}/minc2500_anchors'},
        'low_data': {DS: 'deep_scatter/training_logs/clean_v2/minc_lowdata'},
        'phase_a_nc': {DS: 'deep_scatter/training_logs/phase_a/minc2500'},
    },
    'SuroTex': {
        'summary': {DS: 'deep_scatter/training_logs/surotex_h2h',
                    SCD: 'sep_conc_deepnets/training_logs/clean_zarka/surotex_h2h'},
        'arm_filter': {DS: ('bn_lr1em2', 'bn_rinit_lr1em3'),
                       SCD: 'blk6_whigh_nlsoftshrink_lr1em3'},
        'l2_disentangle': {DS: f'{_L2D}/surotex'},
        'l2_anchor': {DS: f'{_L2D}/surotex_anchors'},
    },
}

# Change only through set_active_datasets().
ACTIVE_DATASETS = list(DATASET_SOURCES)


def dataset_ready(dataset, verbose=False):
    """True when every `summary` summary.csv of ``dataset`` is on disk."""
    missing = [f'{WORKSPACE_ROOT}/{d}/summary.csv'
               for d in DATASET_SOURCES[dataset]['summary'].values()]
    missing = [p for p in missing if not os.path.exists(p)]
    if missing and verbose:
        for path in missing:
            print(f"  [sources] {dataset}: {os.path.relpath(path, WORKSPACE_ROOT)} absent")
        print(f"  [sources] {dataset} skipped -- every framework's summary is required "
              f"(run analyze.py or --redo-analyze)")
    return not missing


def active_datasets(ready_only=True):
    """The active datasets, in registry order; by default only those dataset_ready()."""
    return [d for d in ACTIVE_DATASETS if not ready_only or dataset_ready(d)]


def dataset_sources(kind, filename, datasets=None, ready_only=True):
    """[(abs path to <dir>/<filename>, dataset, framework)] for one source kind."""
    names = active_datasets(ready_only) if datasets is None else datasets
    return [(f'{WORKSPACE_ROOT}/{DATASET_SOURCES[ds][kind][fw]}/{filename}', ds, fw)
            for fw in FRAMEWORKS for ds in names
            if fw in DATASET_SOURCES[ds].get(kind, {})]


# Derived source lists, mutated in place by _refresh_sources().
SUMMARY_SOURCES = []
L2_DISENTANGLE_SUMMARY_SOURCES = []
L2_ANCHOR_SUMMARY_SOURCES = []
EXTEND_LOW_DATA_SOURCES = []
L2_DISENTANGLE_NC_SOURCES = []
PHASE_A_NC_SOURCES = []
ARM_FILTER = {}             # {(dataset, framework): arm or tuple of arms}

_DERIVED_SOURCES = {
    'SUMMARY_SOURCES': ('summary', 'summary.csv'),
    'L2_DISENTANGLE_SUMMARY_SOURCES': ('l2_disentangle', 'summary.csv'),
    'L2_ANCHOR_SUMMARY_SOURCES': ('l2_anchor', 'summary.csv'),
    'EXTEND_LOW_DATA_SOURCES': ('low_data', 'summary.csv'),
    'L2_DISENTANGLE_NC_SOURCES': ('l2_disentangle', 'nc_recomputed.csv'),
    'PHASE_A_NC_SOURCES': ('phase_a_nc', 'nc_recomputed.csv'),
}


def _refresh_sources(verbose=False):
    """Rebuild every derived source list from DATASET_SOURCES and ACTIVE_DATASETS."""
    if verbose:
        for ds in ACTIVE_DATASETS:
            dataset_ready(ds, verbose=True)
    for name, (kind, filename) in _DERIVED_SOURCES.items():
        globals()[name][:] = dataset_sources(kind, filename)
    ARM_FILTER.clear()
    ARM_FILTER.update({(ds, fw): arms for ds in active_datasets()
                       for fw, arms in DATASET_SOURCES[ds].get('arm_filter', {}).items()})


def set_active_datasets(names=None, verbose=True):
    """Restrict every input of this script to ``names`` (None = every registered dataset)."""
    names = list(DATASET_SOURCES) if names is None else list(names)
    unknown = [n for n in names if n not in DATASET_SOURCES]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; registered: {list(DATASET_SOURCES)}")
    ACTIVE_DATASETS[:] = [d for d in DATASET_SOURCES if d in names]
    _refresh_sources(verbose=verbose)
    if verbose:
        print(f"  [sources] active datasets: {', '.join(active_datasets())}")


_refresh_sources()

sns.set_theme(font_scale=2.5, style='white', font='serif')


def _resolve_datasets(df):
    """Datasets present in ``df``, in DATASET_ORDER."""
    present = set(df['dataset'].dropna().unique())
    order = [d for d in DATASET_ORDER if d in present]
    order += sorted(d for d in present if d not in DATASET_ORDER)
    return order


def _tight_extent(fig, artists):
    """Union of the artists' rendered extents, (x0, y0, x1, y1) in figure coordinates."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = []
    for a in artists:
        try:
            bb = a.get_tightbbox(renderer)
        except TypeError:
            bb = a.get_tightbbox()
        if bb is not None:
            (x0, y0), (x1, y1) = inv.transform(bb.p0), inv.transform(bb.p1)
            boxes.append((x0, y0, x1, y1))
        # Axes.get_tightbbox clamps long axis labels to the axes extent; measure them directly.
        for axis_name in ('xaxis', 'yaxis'):
            label = getattr(getattr(a, axis_name, None), 'label', None)
            if label is not None and label.get_text():
                lb = label.get_window_extent(renderer)
                (x0, y0), (x1, y1) = inv.transform(lb.p0), inv.transform(lb.p1)
                boxes.append((x0, y0, x1, y1))
    if not boxes:
        return 0.0, 0.0, 1.0, 1.0
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


# ---------------------------------------------------------------------------
# Figure type size (--combined-font-scale / --combined-legend-font-scale)
# ---------------------------------------------------------------------------
_TEXT_RC_KEYS = ('font.size', 'axes.labelsize', 'axes.titlesize', 'xtick.labelsize',
                 'ytick.labelsize', 'legend.fontsize', 'legend.title_fontsize')
COMBINED_LEGEND_FONT_PT = 15
COMBINED_LEGEND_TITLE_FONT_PT = 16
COMBINED_FONT_SCALE = 1.0           # set from the CLI
COMBINED_LEGEND_FONT_SCALE = 1.0    # set from the CLI; legend text only
_LEGEND_FONT_SCALE = 1.0


def legend_fonts():
    """(prop dict, title size) for the hand-built fig.legend calls."""
    s = _LEGEND_FONT_SCALE
    return {'size': COMBINED_LEGEND_FONT_PT * s}, COMBINED_LEGEND_TITLE_FONT_PT * s


@contextlib.contextmanager
def combined_style():
    """Apply the font scales to rcParams text and to legend_fonts()."""
    global _LEGEND_FONT_SCALE
    text = COMBINED_FONT_SCALE
    legend = COMBINED_FONT_SCALE * COMBINED_LEGEND_FONT_SCALE
    if text == 1.0 and legend == 1.0:
        yield
        return
    rc = {k: plt.rcParams[k] * (legend if k.startswith('legend.') else text)
          for k in _TEXT_RC_KEYS}
    _LEGEND_FONT_SCALE = legend
    try:
        with plt.rc_context(rc):
            yield
    finally:
        _LEGEND_FONT_SCALE = 1.0


# ---------------------------------------------------------------------------
# Shared figure-level axis labels, spaced in em of the label's own type
# ---------------------------------------------------------------------------
SHARED_LABEL_PAD_EM = 0.75    # panels -> shared axis label
SHARED_LEGEND_PAD_EM = 0.8    # shared label -> legend strip


def _em_fraction(fig, pt, em, vertical=True):
    """`em` ems of `pt`-sized type, as a fraction of the figure's height (or width)."""
    size = fig.get_size_inches()
    return em * pt / 72.0 / (size[1] if vertical else size[0])


def _panel_edges(fig, axes):
    """(left, bottom) of the panels' ink (axes, tick labels, axis labels) in figure fractions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    lefts, bottoms = [], []
    for ax in axes:
        x0, y0 = inv.transform(ax.get_window_extent(renderer).p0)
        lefts.append(x0)
        bottoms.append(y0)
        for ticks, keep in ((ax.get_xticklabels(), bottoms),
                            (ax.get_yticklabels(), lefts)):
            for t in ticks:
                if t.get_visible() and t.get_text():
                    tb = t.get_window_extent(renderer)
                    p0 = inv.transform(tb.p0)
                    keep.append(p0[1] if keep is bottoms else p0[0])
        for axis_name, keep in (('xaxis', bottoms), ('yaxis', lefts)):
            label = getattr(ax, axis_name).label
            if label.get_text():
                lb = inv.transform(label.get_window_extent(renderer).p0)
                keep.append(lb[1] if keep is bottoms else lb[0])
    return (min(lefts) if lefts else 0.0), (min(bottoms) if bottoms else 0.0)


def place_shared_xlabel(fig, text, axes=None, pad_em=None, x=0.5, **kw):
    """One x label for the whole figure, `pad_em` below the lowest panel ink.

    Call after the final tight_layout/subplots_adjust.
    """
    axes = list(np.atleast_1d(fig.axes).flat) if axes is None else list(axes)
    pad_em = SHARED_LABEL_PAD_EM if pad_em is None else pad_em
    kw.setdefault('ha', 'center')
    label = fig.text(x, 0.0, text, va='top', **kw)
    bottom = _panel_edges(fig, axes)[1]
    label.set_position((x, bottom - _em_fraction(fig, label.get_fontsize(), pad_em)))
    return label


def place_shared_ylabel(fig, text, axes=None, pad_em=None, y=0.5, **kw):
    """One y label for the whole figure, `pad_em` left of the leftmost panel ink."""
    axes = list(np.atleast_1d(fig.axes).flat) if axes is None else list(axes)
    pad_em = SHARED_LABEL_PAD_EM if pad_em is None else pad_em
    kw.setdefault('va', 'center')
    label = fig.text(0.0, y, text, ha='right', rotation='vertical', **kw)
    left = _panel_edges(fig, axes)[0]
    label.set_position(
        (left - _em_fraction(fig, label.get_fontsize(), pad_em, vertical=False), y))
    return label


def legend_top_under(fig, artist, pad_em=None):
    """Figure y for a legend hanging `pad_em` under `artist`."""
    pad_em = SHARED_LEGEND_PAD_EM if pad_em is None else pad_em
    bottom = _tight_extent(fig, [artist])[1]
    return bottom - _em_fraction(fig, artist.get_fontsize(), pad_em)


def push_legend_under(fig, legend, artist, pad_em=None):
    """Shift `legend` down until it clears `artist` by `pad_em`; never raise it."""
    if legend is None:
        return
    wanted = legend_top_under(fig, artist, pad_em)
    top = _tight_extent(fig, [legend])[3]
    delta = wanted - top
    if delta >= 0:
        return
    inv = fig.transFigure.inverted()
    box = legend.get_bbox_to_anchor()
    (x0, y0), (x1, y1) = inv.transform(box.p0), inv.transform(box.p1)
    legend.set_bbox_to_anchor((x0, y0 + delta, x1 - x0, y1 - y0),
                              transform=fig.transFigure)


# =========================================================================
# Data loading
# =========================================================================

# `_s<size>[_l2<lambda>]_seed<n>` run-label suffix; stripped to recover the arm name.
_ARM_RUN_SUFFIX = re.compile(r'_s\d+p\d+(?:_l2\d+p\d+)?_seed\d+$')


def apply_arm_filter(df, dataset, framework, verbose=True):
    """Keep only the registered arm(s) when a summary holds more than one."""
    wanted = ARM_FILTER.get((dataset, framework))
    if wanted is None or df.empty or 'label' not in df.columns:
        return df
    wanted = (wanted,) if isinstance(wanted, str) else tuple(wanted)
    config = df['label'].astype(str).str.replace(_ARM_RUN_SUFFIX, '', regex=True)
    kept = df.loc[config.isin(wanted)]
    absent = [w for w in wanted if not (config == w).any()]
    if absent:
        raise SystemExit(
            f"ARM_FILTER[({dataset!r}, {framework!r})] asks for {absent!r}, which is not "
            f"in the summary. Present: {sorted(config.unique())}")
    if verbose and len(kept) < len(df):
        print(f"  [arm] {dataset}/{framework}: kept {len(kept)}/{len(df)} rows "
              f"for arm(s) {', '.join(repr(w) for w in wanted)}")
    return kept


def summary_source(dataset, framework, sources=SUMMARY_SOURCES):
    """The summary.csv registered for ``(dataset, framework)``."""
    for path, ds, fw in sources:
        if ds == dataset and fw == framework:
            return path
    raise KeyError(
        f"no summary source registered for ({dataset!r}, {framework!r}); "
        f"known: {sorted({(d, f) for _, d, f in sources})}")


def read_dataset_method(filename, dataset_name, method_name):
    try:
        df = pd.read_csv(filename)
        df['dataset'] = dataset_name
        df['framework'] = method_name
        return apply_arm_filter(df, dataset_name, method_name)
    except Exception as e:
        print(f"Error: {e}. Please run with --redo-analyze to generate the summary files before plotting.")
        sys.exit(1)


def redo_analyze():
    """Regenerate every summary.csv this script reads."""
    import subprocess

    failures = []
    summaries = dataset_sources('summary', 'summary.csv', ready_only=False)
    for summary_csv, _dataset, framework in summaries:
        results_dir = os.path.dirname(summary_csv)
        repo = 'deep_scatter' if framework == 'deep_scatter' else 'sep_conc_deepnets'
        repo_dir = os.path.join(_REPO_ROOT, repo)
        script = os.path.join(repo_dir, 'analyze.py')
        env = dict(os.environ, PYTHONNOUSERSITE='1')
        print(f"[redo-analyze] {repo}: {results_dir}")
        proc = subprocess.run(['python', script, '--recursive', '--results-dir', results_dir],
                              cwd=repo_dir, env=env)
        if proc.returncode != 0:
            failures.append((repo, results_dir, proc.returncode))

    if failures:
        print("\n[redo-analyze] FAILED for:")
        for repo, results_dir, rc in failures:
            print(f"  ({repo}, rc={rc}) {results_dir}")
    else:
        print(f"\n[redo-analyze] regenerated {len(summaries)} summaries, no failures")


def extract_most_recent_result_per_folder(df):
    """Keep the newest results_*.json per (run directory, seed)."""
    has_path = df[df['_path'].notna()].copy()
    no_path = df[df['_path'].isna()].copy()

    if len(has_path) > 0:
        has_path['result_timestamp'] = pd.to_datetime(has_path._path.str.split('results_').str[-1].str.split('.json').str[0], format = '%Y%m%d_%H%M%S')
        has_path['folder'] = has_path._path.str.split('results_').str[0]

        has_path['folder'] = np.where(
            has_path['seed'].isna(),
            has_path['folder'],
            has_path['folder'] + has_path['seed'].astype(str)
        )
        has_path['folder'] = has_path['folder'].fillna(has_path['label'])
        has_path['result_timestamp'] = has_path['result_timestamp'].fillna(pd.Timestamp.now())
        has_path = has_path.loc[has_path.groupby('folder').result_timestamp.idxmax()]

    df_filtered = pd.concat([has_path, no_path], ignore_index=True)
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered


def make_melted_df(df):
    """One row per (run, accuracy type), with the generalization gap of that phase."""
    id_cols = ['train_size', 'seed', 'l2_penalty', 'random_init', 'framework', 'dataset', '_path']
    melted_df = df.melt(
        id_vars=id_cols,
        value_vars=['joint_acc', 'phase_a_test_acc1', 'finetune_acc'],
        var_name='accuracy_type',
        value_name='accuracy_value'
    )
    melted_df = melted_df.dropna(subset=['accuracy_value'])

    # Phase-A ('Scattering') rows take phase_a_gen_error rather than the fine-tuned gen_error.
    merge_cols = ['gen_error', 'filter_delta_relative']
    has_phase_a_ge = 'phase_a_gen_error' in df.columns
    if has_phase_a_ge:
        merge_cols = merge_cols + ['phase_a_gen_error']
    melted_df = pd.merge(melted_df, df[id_cols + merge_cols], on=id_cols, how = 'left')
    if has_phase_a_ge:
        mask_a = melted_df['accuracy_type'] == 'phase_a_test_acc1'
        melted_df.loc[mask_a, 'gen_error'] = melted_df.loc[mask_a, 'phase_a_gen_error']
        melted_df = melted_df.drop(columns=['phase_a_gen_error'])
    return melted_df


def extend_with_low_data_sources(df_wide, sources=EXTEND_LOW_DATA_SOURCES, verbose=True):
    """Append runs below the primary source's smallest train_size (wide, pre-melt frame)."""
    if df_wide.empty:
        return df_wide
    out = df_wide
    for path, ds, fw in sources:
        if not os.path.exists(path):
            if verbose:
                print(f"  [low-data] {ds}/{fw}: {os.path.relpath(path, _REPO_ROOT)} absent, skipped")
            continue
        primary = out[(out['dataset'] == ds) & (out['framework'] == fw)]
        if primary.empty:
            if verbose:
                print(f"  [low-data] {ds}/{fw}: no primary rows to extend, skipped")
            continue
        floor = pd.to_numeric(primary['train_size'], errors='coerce').min()
        extra = pd.read_csv(path)
        extra['dataset'] = ds
        extra['framework'] = fw
        keep = pd.to_numeric(extra['train_size'], errors='coerce') < floor
        extra = extra[keep]
        if extra.empty:
            if verbose:
                print(f"  [low-data] {ds}/{fw}: nothing below train_size {floor:g}, skipped")
            continue
        if verbose:
            sizes = sorted(pd.to_numeric(extra['train_size'], errors='coerce').unique())
            print(f"  [low-data] {ds}/{fw}: +{len(extra)} runs at train_size "
                  f"{', '.join(f'{s:g}' for s in sizes)} (primary floor {floor:g})")
        out = pd.concat([out, extra], ignore_index=True)
    return out


def load_l2_disentangle_frame(sources=L2_DISENTANGLE_SUMMARY_SOURCES, verbose=True,
                              melt=True):
    """Random-init rows at lambda > 0, melted like ``load_data()``'s frame.

    With ``melt=False`` the rows are returned in wide (one row per run) form.
    lambda = 0 is excluded because the primary source already provides it.
    """
    frames = []
    missing = []
    for path, ds, fw in sources:
        if not os.path.exists(path):
            missing.append((ds, path))
            continue
        raw = pd.read_csv(path)
        raw['dataset'] = ds
        raw['framework'] = fw
        frames.append(raw)
    if verbose and missing:
        for ds, path in missing:
            print(f"  [l2-disentangle] no runs on disk for {ds} "
                  f"({os.path.relpath(path, _REPO_ROOT)})")
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    # Random-init sweeps have no two-phase accuracy columns; add them so the melt works.
    for col in ('phase_a_test_acc1', 'finetune_acc'):
        if col not in raw.columns:
            raw[col] = np.nan
    raw = extract_most_recent_result_per_folder(raw)
    if not melt:
        wide = raw[raw['joint_acc'].notna()]
        wide = wide[pd.to_numeric(wide['l2_penalty'], errors='coerce').fillna(0.0) > 0]
        if verbose:
            for (ds, l2), g in wide.groupby(['dataset', 'l2_penalty']):
                print(f"  [l2-disentangle/wide] {ds:<10} L2={l2:g}  {len(g)} random-init runs")
        return wide.reset_index(drop=True)
    melted = make_melted_df(raw)
    melted = melted[melted['accuracy_type'] == 'joint_acc']
    melted = melted[pd.to_numeric(melted['l2_penalty'], errors='coerce').fillna(0.0) > 0]
    if verbose:
        for (ds, l2), g in melted.groupby(['dataset', 'l2_penalty']):
            print(f"  [l2-disentangle] {ds:<10} L2={l2:g}  {len(g)} random-init runs")
    return melted.reset_index(drop=True)


# ---------------------------------------------------------------------------
# sep_conc_deepnets test accuracy
# ---------------------------------------------------------------------------
# main.py reports best validation accuracy; the figures use the test accuracy of the
# best-val checkpoint, which main.py logs as best_ckpt_test_acc1.

# Must match sep_conc_deepnets/best_ckpt_eval.py:BEST_CKPT_EVAL_VERSION.
SCD_BEST_CKPT_EVAL_VERSION = '2026-08-28'


def _inline_scd_test_acc(df):
    """``best_ckpt_test_acc1`` as logged by main.py; NaN where absent or from an older evaluator."""
    if 'best_ckpt_test_acc1' not in df.columns:
        return pd.Series(np.nan, index=df.index)
    acc = pd.to_numeric(df['best_ckpt_test_acc1'], errors='coerce')
    if 'best_ckpt_eval_version' in df.columns:
        ver = df['best_ckpt_eval_version']
        ok = ver.map(lambda v: isinstance(v, str) and v >= SCD_BEST_CKPT_EVAL_VERSION)
    else:
        ok = pd.Series(False, index=df.index)
    return acc.where(ok)


def apply_scd_test_acc(df, verbose=True):
    """Replace SCD's validation ``best_acc1``/``gen_error`` with test-set values.

    Rows with no test value keep their validation accuracy and are reported.
    """
    if df.empty:
        return df

    df = df.copy()
    new_acc = _inline_scd_test_acc(df)
    matched = new_acc.notna()
    if not matched.any():
        return df

    if 'best_acc1' in df.columns and verbose:
        shift = (pd.to_numeric(df.loc[matched, 'best_acc1'], errors='coerce')
                 - new_acc[matched]).mean()
        print(f"  [scd-testacc] {int(matched.sum())}/{len(df)} SCD row(s) use best-val-checkpoint "
              f"test accuracy logged by main.py (mean val-test {shift:+.2f} pts).")
    df['accuracy_source'] = np.where(
        matched, 'test (logged by main.py, best-val ckpt)',
        'val (published best_acc1) -- NOT COMPARABLE TO DS')
    if 'best_acc1' in df.columns:
        df['best_acc1_val_published'] = df['best_acc1']
        df.loc[matched, 'best_acc1'] = new_acc[matched]

    # gen_error = train accuracy at the selected epoch - test accuracy, as in deep_scatter.
    if 'train_acc1_for_gen' in df.columns:
        tr = pd.to_numeric(df['train_acc1_for_gen'], errors='coerce')
        upd = matched & tr.notna()
        if 'gen_error' in df.columns:
            df['gen_error_val_published'] = df['gen_error']
        df.loc[upd, 'gen_error'] = tr[upd] - new_acc[upd]
        stale = matched & ~tr.notna()
        if stale.any():
            print(f"  [scd-testacc] {int(stale.sum())} row(s) have a test accuracy but no "
                  f"train_acc1_for_gen; their gen_error is blanked.")
            df.loc[stale, 'gen_error'] = np.nan
    elif verbose:
        print("  [scd-testacc] WARNING: no train_acc1_for_gen column; gen_error still "
              "uses validation accuracy.")

    n_unmatched = int((~matched).sum())
    if n_unmatched:
        print(f"  [scd-testacc] {n_unmatched} SCD row(s) kept their validation accuracy "
              f"(no best_ckpt_test_acc1). See `accuracy_source`.")
        cols = [c for c in ('dataset', 'arch', 'train_size', 'seed') if c in df.columns]
        if cols:
            for _, r in df.loc[~matched, cols].head(10).iterrows():
                note = ('' if r.get('arch') == 'scatnetblockanalysis'
                        else '   (arch not plotted)')
                print("      " + "  ".join(f"{c}={r[c]}" for c in cols) + note)
    return df


def cleanup_scd_df(df, use_test_acc=True):
    df = extract_most_recent_result_per_folder(df)
    # Before the rename below, which consumes `_path` and `best_acc1`.
    if use_test_acc:
        df = apply_scd_test_acc(df)
    df.replace({'phase': {'phase_b': 'head_acc', 'phase_c': 'finetune_acc', 'joint': 'joint_acc'}}, inplace=True)
    df.rename(columns = {'random_init_filters': 'random_init',
                         'phase': 'accuracy_type',
                         'best_acc1': 'accuracy_value',
                         }, inplace=True)
    df = df[df['accuracy_type'] != 'phase_a'].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Neural collapse
# ---------------------------------------------------------------------------

def extract_nc_stats(df):
    """One row per (run, phase) with `nc1/2/3`, `gen_error` and `test_acc` for that phase.

        phase 'a'     -> phase_a_test_nc*, phase_a_gen_error, head_acc
        phase 'b'     -> test_nc*,         gen_error,         finetune_acc
        phase 'joint' -> test_nc*,         gen_error,         joint_acc
    """
    staged_df = df[df['joint_acc'].notna() == False].copy()
    joint_df = df[df['joint_acc'].notna()].copy()

    def _phase_metrics(frame, gen_col, acc_col):
        for out_col, src_col in (('gen_error', gen_col), ('test_acc', acc_col)):
            frame[out_col] = (frame[src_col] if src_col in frame.columns
                              else np.nan)
        return frame

    staged_nc_cols = ['phase_a_test_nc1', 'phase_a_test_nc2', 'phase_a_test_nc3']
    final_nc_cols = ['test_nc1', 'test_nc2', 'test_nc3']

    phase_a = staged_df.copy()
    phase_a['phase'] = 'a'
    phase_a = phase_a.rename(columns={
        'phase_a_test_nc1': 'nc1',
        'phase_a_test_nc2': 'nc2',
        'phase_a_test_nc3': 'nc3'
    })
    phase_a = phase_a.drop(columns=final_nc_cols)
    phase_a = _phase_metrics(phase_a, 'phase_a_gen_error', 'head_acc')

    phase_b = staged_df.copy()
    phase_b['phase'] = 'b'
    phase_b = phase_b.rename(columns={
        'test_nc1': 'nc1',
        'test_nc2': 'nc2',
        'test_nc3': 'nc3'
    })
    phase_b = phase_b.drop(columns=staged_nc_cols)
    phase_b = _phase_metrics(phase_b, 'gen_error', 'finetune_acc')

    phase_joint = joint_df.copy()
    phase_joint['phase'] = 'joint'
    phase_joint = phase_joint.rename(columns={
        'test_nc1': 'nc1',
        'test_nc2': 'nc2',
        'test_nc3': 'nc3'
    })
    phase_joint = phase_joint.drop(columns=staged_nc_cols)
    phase_joint = _phase_metrics(phase_joint, 'gen_error', 'joint_acc')

    long_nc_df = pd.concat([phase_a, phase_b, phase_joint], ignore_index=True)
    long_nc_df = long_nc_df.dropna(subset=['nc1', 'nc2', 'nc3'], how='all')
    return long_nc_df


def _nc_csv_sources(summary_sources=SUMMARY_SOURCES,
                    l2_sources=L2_DISENTANGLE_NC_SOURCES,
                    low_data_sources=EXTEND_LOW_DATA_SOURCES):
    """nc_recomputed.csv for every summary, l2_disentangle and low-data source."""
    return ([(os.path.join(os.path.dirname(s), 'nc_recomputed.csv'), ds, fw)
             for (s, ds, fw) in summary_sources]
            + list(l2_sources or [])
            + [(os.path.join(os.path.dirname(s), 'nc_recomputed.csv'), ds, fw)
               for (s, ds, fw) in (low_data_sources or [])])


def _normalize_results_path(series):
    """Repo-relative form ('training_logs/...') of a results-JSON path."""
    return series.astype(str).str.replace(r'^.*?(?=training_logs/)', '', regex=True)


def load_recomputed_nc(sources=None, verbose=True):
    """NC recomputed by nc_report.py at the best-val checkpoint, if available.

    Returns an empty frame when no nc_recomputed.csv exists, in which case the logged NC
    values are used.
    """
    if sources is None:
        sources = _nc_csv_sources()
    frames, missing = [], []
    for path, dataset, framework in sources:
        try:
            sub = pd.read_csv(path)
        except Exception:
            missing.append(path)
            continue
        sub['dataset'] = dataset
        sub['framework'] = framework
        frames.append(sub)
    if missing and verbose:
        print(f"  [nc] {len(missing)}/{len(sources)} nc_recomputed.csv missing "
              f"(run nc_report.py; falling back to logged NC for those)")
        for p in missing:
            print(f"        {p}")
    if not frames:
        return pd.DataFrame(columns=['_path', 'phase', 'nc1', 'dataset', 'framework'])

    nc = pd.concat(frames, ignore_index=True)
    if 'split' in nc.columns:
        nc = nc[nc['split'] == 'test']
    # Join key: the results_*.json path, which analyze.py stores as `_path`.
    nc = nc.rename(columns={'results_json': '_path'})
    nc['_path'] = _normalize_results_path(nc['_path'])
    keep = [c for c in ['_path', 'phase', 'nc1', 'nc2', 'nc3', *NC_EXTRA_COLS,
                        'between_class_trace', 'nc_source', 'nc_error'] if c in nc.columns]
    nc = nc[keep].dropna(subset=['_path', 'phase'])
    return nc.drop_duplicates(subset=['_path', 'phase'], keep='last')


def _apply_phase_a_nc(long_nc, sources=PHASE_A_NC_SOURCES, verbose=True):
    """Fill Phase-A NC from the standalone --phase-a-only sweeps.

    Rows not on the recomputed convention take all NC columns from the sweep. Recomputed
    Phase-A rows rebuilt from config have a random classifier, so they take only the
    head-derived skewed NC1 columns.
    """
    if long_nc.empty or 'phase' not in long_nc.columns:
        return long_nc
    if not {'train_size', 'seed'} <= set(long_nc.columns):
        return long_nc
    for path, dataset, framework in sources:
        try:
            sub = pd.read_csv(path)
        except Exception:
            if verbose:
                print(f"  [nc] missing {path} (run nc_report.py on the phase_a tree); "
                      f"{dataset} Phase-A NC stays empty")
            continue
        sub = sub[(sub.get('phase') == 'a') & sub['nc1'].notna()]
        if 'split' in sub.columns:
            sub = sub[sub['split'] == 'test']
        if sub.empty:
            if verbose:
                print(f"  [nc] {path} has no usable Phase-A rows, skipped")
            continue
        cols = ['nc1'] + [c for c in ('nc2', 'nc3', *NC_EXTRA_COLS) if c in sub.columns]
        sub = (sub[['train_size', 'seed'] + cols]
               .dropna(subset=['train_size', 'seed'])
               .drop_duplicates(subset=['train_size', 'seed'], keep='last'))

        target = ((long_nc['dataset'] == dataset)
                  & (long_nc['framework'] == framework)
                  & (long_nc['phase'] == 'a')
                  & (long_nc['nc_convention'] != 'recomputed_nobn'))
        if target.any():
            keys = long_nc.loc[target, ['train_size', 'seed']]
            filled = keys.merge(sub, on=['train_size', 'seed'], how='left')
            filled.index = keys.index
            got = filled['nc1'].notna()
            for col in cols:
                long_nc.loc[filled.index[got], col] = filled.loc[got, col].values
            long_nc.loc[filled.index[got], 'nc_convention'] = 'recomputed_phase_a_sweep'
            if verbose:
                print(f"  [nc] {dataset}/{framework}: Phase-A filled from the phase_a sweep "
                      f"for {int(got.sum())}/{int(target.sum())} rows")

        for col in NC1_SKEW_COLS:
            if col not in sub.columns or col not in long_nc.columns:
                continue
            head = ((long_nc['dataset'] == dataset)
                    & (long_nc['framework'] == framework)
                    & (long_nc['phase'] == 'a')
                    & (long_nc['nc_convention'] == 'recomputed_nobn')
                    & long_nc['nc1'].notna() & long_nc[col].isna())
            if not head.any():
                continue
            keys = long_nc.loc[head, ['train_size', 'seed']]
            filled = keys.merge(sub[['train_size', 'seed', col]],
                                on=['train_size', 'seed'], how='left')
            filled.index = keys.index
            got = filled[col].notna()
            long_nc.loc[filled.index[got], col] = filled.loc[got, col].values
            if verbose:
                print(f"  [nc] {dataset}/{framework}: Phase-A {col} taken from the "
                      f"phase_a sweep's head for {int(got.sum())}/{int(head.sum())} rows")
    return long_nc


def _drop_stale_logged_nc(long_nc, verbose=True):
    """Blank logged NC in any (dataset, framework) that has recomputed NC.

    Keeps a single NC convention per series; sources with no recomputed NC are untouched.
    """
    if long_nc.empty or 'nc_convention' not in long_nc.columns:
        return long_nc
    nc_cols = [c for c in ('nc1', 'nc2', 'nc3', *NC_EXTRA_COLS) if c in long_nc.columns]
    if not nc_cols:
        return long_nc
    dropped = {}
    for (ds, fw), g in long_nc.groupby(['dataset', 'framework']):
        if not g['nc_convention'].str.startswith('recomputed').any():
            continue
        stale = g.index[(g['nc_convention'] == 'logged') & g[nc_cols].notna().any(axis=1)]
        if len(stale) == 0:
            continue
        long_nc.loc[stale, nc_cols] = np.nan
        long_nc.loc[stale, 'nc_convention'] = 'logged_dropped'
        dropped[(ds, fw)] = len(stale)
    if dropped and verbose:
        total = sum(dropped.values())
        print(f"  [nc] dropped {total} logged NC values in series that have recomputed NC:")
        for (ds, fw), n in sorted(dropped.items()):
            print(f"        {ds:<10} {fw:<20} {n}")
    return long_nc


def apply_recomputed_nc(long_nc, nc_recomputed=None, verbose=True):
    """Overlay recomputed NC onto ``long_nc``; ``nc_convention`` records each row's source."""
    if nc_recomputed is None:
        nc_recomputed = load_recomputed_nc(verbose=verbose)
    long_nc = long_nc.copy()
    long_nc['nc_convention'] = 'logged'
    # Skewed / Papyan NC1 are only produced by nc_report.py.
    for col in NC_EXTRA_COLS:
        if col not in long_nc.columns:
            long_nc[col] = np.nan
    if nc_recomputed.empty or '_path' not in long_nc.columns:
        if verbose:
            print("  [nc] no recomputed NC applied; plots use logged values")
        return _apply_phase_a_nc(long_nc, verbose=verbose)

    long_nc['_path'] = _normalize_results_path(long_nc['_path'])
    merged = long_nc.merge(nc_recomputed.add_suffix('_rc'),
                           left_on=['_path', 'phase'],
                           right_on=['_path_rc', 'phase_rc'], how='left')
    got = merged['nc1_rc'].notna()
    for col in ('nc1', 'nc2', 'nc3', *NC_EXTRA_COLS):
        rc = f'{col}_rc'
        if rc in merged.columns and col in merged.columns:
            merged[col] = merged[rc].where(merged[rc].notna(), merged[col])
    merged.loc[got, 'nc_convention'] = 'recomputed_nobn'
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith('_rc')])
    if verbose:
        print(f"  [nc] recomputed NC applied to {int(got.sum())}/{len(merged)} rows")
        for (ds, fw), g in merged.groupby(['dataset', 'framework']):
            n = int((g['nc_convention'] == 'recomputed_nobn').sum())
            print(f"        {ds:<10} {fw:<20} {n}/{len(g)}")
    merged = _apply_phase_a_nc(merged, verbose=verbose)
    merged = _drop_stale_logged_nc(merged, verbose=verbose)
    if verbose:
        n = int(merged['nc_convention'].str.startswith('recomputed').sum())
        print(f"  [nc] NC on the recomputed convention: {n}/{len(merged)} rows")
    return merged


def load_data():
    """Load every summary.csv and assemble the frames the plots need.

    Returns
    -------
    df : melted per-run frame, both frameworks (deep_scatter + sep_conc_deepnets)
    df_ds : the deep_scatter slice of `df`
    long_nc : one row per (run, phase) with NC metrics, both frameworks
    """
    per_fw = {fw: [] for fw in FRAMEWORKS}
    for dataset in ACTIVE_DATASETS:
        if not dataset_ready(dataset, verbose=True):
            continue
        for framework in DATASET_SOURCES[dataset]['summary']:
            per_fw[framework].append(read_dataset_method(
                summary_source(dataset, framework), dataset, framework))
    if not per_fw[DS] or not per_fw[SCD]:
        raise SystemExit("load_data: no active dataset has its summaries on disk "
                         f"(active: {ACTIVE_DATASETS})")

    df_ds = pd.concat(per_fw[DS], ignore_index=True)
    df_ds = extend_with_low_data_sources(df_ds)
    df_scd = pd.concat(per_fw[SCD], ignore_index=True)

    df_ds = extract_most_recent_result_per_folder(df_ds)
    # The random-init lambda > 0 runs feed the NC figure only; `df` keeps a single
    # random-init series (lambda = 0) per dataset.
    df_ds_wide = df_ds.copy()
    _l2d_wide = load_l2_disentangle_frame(verbose=True, melt=False)
    if not _l2d_wide.empty:
        df_ds_wide = pd.concat([df_ds_wide, _l2d_wide], ignore_index=True)
    df_ds = make_melted_df(df_ds)

    df_scd = cleanup_scd_df(df_scd)
    df_scd = df_scd.loc[df_scd.arch == 'scatnetblockanalysis']
    df = pd.concat([df_ds, df_scd], ignore_index=True)
    # The L2 penalty only applies to fine-tuning, so keep lambda > 0 for fine-tuned rows only.
    df = df.loc[(df.accuracy_type == 'finetune_acc') | (df.l2_penalty == 0.0)]
    df_ds = df.loc[df.framework == 'deep_scatter']

    long_nc = extract_nc_stats(df_ds_wide)
    df_scd_nc = df_scd.loc[df_scd.arch == 'scatnetblockanalysis'].reset_index(drop=True)
    df_scd_nc.rename(columns={'test_nc1': 'nc1', 'test_nc2': 'nc2', 'test_nc3': 'nc3',
                              'arch': 'phase'}, inplace=True)
    df_scd_nc['test_acc'] = df_scd_nc['accuracy_value']
    # ScatNet B.A. has no L2 sweep; repeat it under each lambda so it appears in every facet.
    df_scd_nc = pd.concat(
        [df_scd_nc] + [df_scd_nc.assign(l2_penalty=l2) for l2 in (0.1, 1.0)],
        ignore_index=True)
    long_nc = pd.concat([long_nc, df_scd_nc])
    long_nc = long_nc.sort_values(['dataset'], ignore_index=True)
    long_nc = apply_recomputed_nc(long_nc)

    return df, df_ds, long_nc


# =========================================================================
# Shared plotting helpers
# =========================================================================

# Line style encodes the L2 penalty in every figure.
V2_L2_STYLE = {0.0: '-', 0.1: '--', 1.0: ':'}


def _assign_arch_labels(df):
    """Tag rows with the method label ('Arch.')."""
    out = df.copy()
    arch_col = 'Arch.' if 'Arch.' in out.columns else 'arch'
    source_arch = out[arch_col] if arch_col in out.columns else None

    out['Arch.'] = out['accuracy_type'].replace({
        'finetune_acc': 'Fine-tuned Scat.',
        'joint_acc': 'Random Init.',
        'phase_a_test_acc1': 'Scattering'
    })
    if source_arch is not None:
        out.loc[source_arch.isin(['scatnetblockanalysis', 'ScatNet B.A.']), 'Arch.'] = 'ScatNet B.A.'
    return out


def _train_size_area(subset):
    """Log-spaced marker area over ``subset`` (the shared train-size encoding)."""
    smin, smax = float(min(subset)), float(max(subset))

    def area(ts):
        f = (np.log10(ts) - np.log10(smin)) / (np.log10(smax) - np.log10(smin) + 1e-9)
        return 70 + f * (430 - 70)
    return area


def _equalize_units(ax):
    """1:1 aspect: accuracy and gen. gap are both in percentage points."""
    ax.set_aspect('equal', adjustable='datalim')


def _hide_negative_xticks(ax):
    """Blank x tick labels outside [0, 100]; the tick marks are kept."""
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _pos: '' if (v < -1e-9 or v > 100 + 1e-9) else f'{v:g}'))


def _log_tick_label(v):
    r"""Plain-decimal label for a log tick where short ('0.2', '3'), else '$2\times10^{4}$'."""
    if v <= 0:
        return ''
    if 1e-3 <= v < 1e4:
        return f'{v:g}'
    exp = int(np.floor(np.log10(v)))
    mant = v / 10.0 ** exp
    return (rf'$10^{{{exp}}}$' if abs(mant - 1) < 1e-9
            else rf'${mant:g}\times10^{{{exp}}}$')


def _thin_log_xticks(ax, max_ticks=3):
    """At most `max_ticks` labelled ticks on a log x-axis.

    Spans of two or more decades use decade ticks; narrower spans use a 1-2-5 ladder
    labelled as plain decimals.
    """
    lo, hi = ax.get_xlim()
    if not (lo > 0 and hi > lo):
        return
    d0 = int(np.floor(np.log10(lo)))
    d1 = int(np.ceil(np.log10(hi)))
    decades = [10.0 ** k for k in range(d0, d1 + 1) if lo <= 10.0 ** k <= hi]
    if len(decades) >= 2:
        ax.set_xticks(decades[::2] if len(decades) > max_ticks else decades)
    else:
        def ladder(mantissas):
            return sorted(m * 10.0 ** k for k in range(d0, d1 + 1) for m in mantissas
                          if lo <= m * 10.0 ** k <= hi)
        ticks = ladder((1.0, 2.0, 5.0)) or ladder((1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0))
        if ticks:
            if len(ticks) > max_ticks:
                ticks = ticks[::int(np.ceil(len(ticks) / max_ticks))]
            ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _log_tick_label(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _save_three_part_legend(fig, xlabel_artist, sections, xs):
    """Method / Reg. / Train size legends laid out side by side under the shared x label.

    `sections` is a list of (handles, title, ncol); `xs` the left edge of each, in figure
    fractions.
    """
    legend_y = legend_top_under(fig, xlabel_artist)
    lp, ltp = legend_fonts()
    legends = []
    for (handles, title, ncol), x in zip(sections, xs):
        extra = dict(labelspacing=0.7, columnspacing=1.2) if ncol > 1 else dict(labelspacing=0.5)
        legends.append(fig.legend(handles=handles, title=title, loc='upper left',
                                  bbox_to_anchor=(x, legend_y), frameon=False, ncol=ncol,
                                  prop=lp, title_fontsize=ltp, **extra))
    for lg in legends[:-1]:
        fig.add_artist(lg)


def _size_handles(subset, area):
    return [mlines.Line2D([], [], color='0.35', marker='o', linestyle='None',
                          markersize=np.sqrt(area(s)) * 0.55, label=f"{s*100:g}%")
            for s in subset]


def _reg_handles(l2_levels):
    return [mlines.Line2D([], [], color='0.35', linestyle=V2_L2_STYLE[l2],
                          linewidth=2.5, label=f"{l2:g}") for l2 in l2_levels]


# =========================================================================
# accuracy_value_vs_train_size.png
# =========================================================================

def plot_acc_vs_train_size(melted_df, col_name, out=None):
    """`col_name` (e.g. 'accuracy_value') vs train size, one column per dataset."""
    melted_df = melted_df.copy()
    dataset_order = _resolve_datasets(melted_df)
    if not dataset_order:
        return
    if out is None:
        out = f'./{col_name}_vs_train_size.png'
    melted_df.rename(columns = {'arch': 'Arch.'}, inplace=True)
    melted_df.replace({'accuracy_type': {
        'finetune_acc': 'Fine-tuned Scat.',
        'joint_acc': 'Random Init.',
        'phase_a_test_acc1': 'Scattering'
    },
    'Arch.': {'scatnetblockanalysis': 'ScatNet B.A.'}}, inplace=True)
    melted_df['Arch.'] = melted_df['Arch.'].fillna(melted_df['accuracy_type'])
    melted_df.rename(columns = {'l2_penalty': 'Reg.'}, inplace=True)

    g = sns.relplot(
        data=melted_df,
        x='train_size',
        y=col_name,
        hue='Arch.',
        style='Reg.',
        col='dataset',
        kind='line',
        palette=METHOD_COLOR,
        errorbar='sd',
        markers=True,
        height=4,
        aspect=0.5,
        alpha=0.8,
        markersize=10,
        facet_kws={'sharex': False},
        col_order=dataset_order
    )
    g.set(xscale='log')

    old_legend = g._legend
    g._legend.remove()
    # Draw the Reg. legend entries in neutral gray so they do not read as a method.
    for text, handle in zip(old_legend.texts, old_legend.legend_handles):
        if text.get_text() in ['0.0', '0.1', '1.0']:
            if hasattr(handle, 'set_markerfacecolor'):
                handle.set_markerfacecolor('gray')
                handle.set_markeredgecolor('gray')
                handle.set_color('gray')
    keep = [(t, h) for t, h in zip(old_legend.texts, old_legend.legend_handles)
            if t.get_text() != "Arch."]

    g.set_titles(col_template="{col_name}")
    g.add_legend(handles=[h for t, h in keep], labels=[t.get_text() for t, h in keep])
    sns.move_legend(g, "lower center", ncol=4, bbox_to_anchor=(0.5, - 0.45))
    g.set_axis_labels("", "Test Accuracy")
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    shared_xlabel = place_shared_xlabel(g.figure, 'Train Size (Log Scale)')
    push_legend_under(g.figure, g._legend, shared_xlabel)
    plt.savefig(str(out), bbox_inches='tight')
    plt.close()


# =========================================================================
# gen_error_vs_acc_v2.png -- gen. gap vs test accuracy, method as colour
# =========================================================================
V2_METHOD_ORDER = ['Random Init.', 'Scattering', 'Fine-tuned Scat.', 'ScatNet B.A.']


def _assign_v2_method(df):
    """Tag each row with its Method (Random Init. / Scattering / Fine-tuned Scat. / ScatNet B.A.)."""
    sd = df[df['accuracy_type'].isin(['finetune_acc', 'joint_acc',
                                      'phase_a_test_acc1'])].copy()
    sd['Method'] = 'Fine-tuned Scat.'
    sd.loc[sd['accuracy_type'] == 'phase_a_test_acc1', 'Method'] = 'Scattering'
    sd.loc[sd['random_init'].astype(str) == 'True', 'Method'] = 'Random Init.'
    mask_scat = (sd.get('framework') == 'sep_conc_deepnets') & (sd.get('arch') == 'scatnetblockanalysis')
    sd.loc[mask_scat, 'Method'] = 'ScatNet B.A.'
    return sd


def plot_gen_error_vs_acc_v2(df, subset_sizes=(0.01, 0.1, 0.5, 1.0),
                             out='./gen_error_vs_acc_v2.png'):
    """Gen. gap vs test accuracy per method; Fine-tuned Scat. is drawn at every lambda."""
    sd = _assign_v2_method(df.copy())
    dataset_order = _resolve_datasets(sd)
    if not dataset_order:
        return
    sd['train_size'] = sd['train_size'].astype(float).round(3)
    subset = [round(float(s), 3) for s in subset_sizes]
    sd = sd[sd['train_size'].isin(subset)]
    if sd.empty:
        return
    sd = sd[(sd['Method'] == 'Fine-tuned Scat.') | (sd['l2_penalty'] == 0.0)]

    agg = sd.groupby(['dataset', 'Method', 'l2_penalty', 'train_size']).agg(
        acc=('accuracy_value', 'mean'),
        gen=('gen_error', 'mean'),
    ).reset_index()
    area = _train_size_area(subset)

    def draw(data, **kwargs):
        ax = plt.gca()
        ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
        for method in V2_METHOD_ORDER:
            md = data[data['Method'] == method]
            if md.empty:
                continue
            l2_list = sorted(md['l2_penalty'].unique()) if method == 'Fine-tuned Scat.' else [0.0]
            for l2 in l2_list:
                sub = md[md['l2_penalty'] == l2].sort_values('train_size')
                if sub.empty:
                    continue
                ls = V2_L2_STYLE.get(l2, '-') if method == 'Fine-tuned Scat.' else '-'
                ax.plot(sub['acc'], sub['gen'], color=METHOD_COLOR[method],
                        linestyle=ls, linewidth=2.2, alpha=0.9, zorder=2)
                ax.scatter(sub['acc'], sub['gen'],
                           s=[area(t) for t in sub['train_size']],
                           color=METHOD_COLOR[method], edgecolor='white',
                           linewidth=1.2, zorder=3)
        _equalize_units(ax)
        _hide_negative_xticks(ax)

    g = sns.FacetGrid(data=agg, col='dataset', col_order=dataset_order,
                      height=4.2, aspect=0.82, sharex=False, sharey=False)
    g.map_dataframe(draw)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "")

    method_handles = [mlines.Line2D([], [], color=METHOD_COLOR[m], linewidth=3, label=m)
                      for m in V2_METHOD_ORDER]
    fig = g.figure
    xlabel_artist = place_shared_xlabel(fig, 'Test Accuracy')
    place_shared_ylabel(fig, 'Gen. Gap')
    _save_three_part_legend(
        fig, xlabel_artist,
        [(method_handles, 'Method', 1),
         (_reg_handles((0.0, 0.1, 1.0)), 'Reg. (Fine-tuned)', 1),
         (_size_handles(subset, area), 'Train size', 2)],
        xs=(0.10, 0.40, 0.68))
    plt.savefig(str(out), bbox_inches='tight', dpi=150)
    plt.close()


# =========================================================================
# gen_error_vs_acc_v3.png -- the L2 penalty on the wavelet vs random-init arms
# =========================================================================
# hue = arm, line style = lambda, marker area = train size. `Scattering` (frozen
# extractor) is lambda-independent and drawn once as the reference.
V3_ARMS = ('Scattering', 'Fine-tuned Scat.', 'Random Init.')
V3_L2_ARMS = (0.0, 0.1, 1.0)
# Painted back to front so the Scattering reference stays on top.
V3_PAINT_ORDER = tuple(reversed(V3_ARMS))


def _v3_arm_lambda_frame(df, verbose=True):
    """Mean (accuracy, gen_error) per (dataset, arm, lambda, train_size)."""
    base = df.copy()
    l2d = load_l2_disentangle_frame(verbose=verbose)
    if not l2d.empty:
        base = pd.concat([base, l2d], ignore_index=True)

    base = _assign_arch_labels(base)
    base = base[base['Arch.'].isin(V3_ARMS)]
    base['l2_penalty'] = pd.to_numeric(base['l2_penalty'], errors='coerce').fillna(0.0)
    # The frozen extractor is unaffected by the penalty; draw it once at lambda = 0.
    base.loc[base['Arch.'] == 'Scattering', 'l2_penalty'] = 0.0
    base = base[base['l2_penalty'].isin(V3_L2_ARMS)]
    base = base.dropna(subset=['gen_error', 'accuracy_value'])
    if base.empty:
        return base

    agg = (base.groupby(['dataset', 'Arch.', 'l2_penalty', 'train_size'])
           .agg(acc=('accuracy_value', 'mean'), gen=('gen_error', 'mean'))
           .reset_index().sort_values('train_size'))
    if verbose:
        have = agg[agg['Arch.'] == 'Random Init.'].groupby('dataset')['l2_penalty'].nunique()
        thin = sorted(have[have < len(V3_L2_ARMS)].index)
        if thin:
            print(f"  [v3] Random Init. has fewer than {len(V3_L2_ARMS)} lambda arms on: "
                  f"{', '.join(thin)}")
    return agg


def plot_gen_error_vs_acc_v3(df, subset_sizes=(0.01, 0.1, 0.5, 1.0),
                             out='./gen_error_vs_acc_v3.png'):
    """Gen. gap vs test accuracy for the fine-tuned and random-init arms at each lambda."""
    agg = _v3_arm_lambda_frame(df)
    if agg.empty:
        return
    dataset_order = _resolve_datasets(agg)
    agg = agg.assign(train_size=agg['train_size'].astype(float).round(3))
    agg = agg[agg['train_size'].isin([round(float(s), 3) for s in subset_sizes])]
    if agg.empty:
        return
    # Datasets with no random-init lambda > 0 runs are flagged in the panel title.
    have = agg[agg['Arch.'] == 'Random Init.'].groupby('dataset')['l2_penalty'].nunique()
    uncontrolled = {d for d in dataset_order if have.get(d, 0) < len(V3_L2_ARMS)}

    subset = sorted(agg['train_size'].unique())
    area = _train_size_area(subset)

    def draw(data, **kwargs):
        ax = plt.gca()
        ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
        for depth, arm in enumerate(V3_PAINT_ORDER):
            colour = METHOD_COLOR[arm]
            for l2 in V3_L2_ARMS:
                sub = data[(data['Arch.'] == arm)
                           & (data['l2_penalty'] == l2)].sort_values('train_size')
                if sub.empty:
                    continue
                ax.plot(sub['acc'], sub['gen'], color=colour, alpha=0.9,
                        linestyle=V2_L2_STYLE.get(l2, '-'), linewidth=2.0,
                        zorder=2 + 2 * depth)
                ax.scatter(sub['acc'], sub['gen'],
                           s=[area(t) for t in sub['train_size']],
                           color=colour, edgecolor='white',
                           linewidth=1.2, zorder=3 + 2 * depth)
        _equalize_units(ax)
        _hide_negative_xticks(ax)

    g = sns.FacetGrid(data=agg, col='dataset', col_order=dataset_order,
                      height=4.2, aspect=0.82, sharex=False, sharey=False)
    g.map_dataframe(draw)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "")
    for ax, dataset in zip(np.atleast_1d(g.axes.flat), dataset_order):
        if dataset in uncontrolled:
            ax.set_title(f"{dataset}\n(no $\\lambda$ control)", color='#52514e')

    method_handles = [mlines.Line2D([], [], color=METHOD_COLOR[a], linewidth=3,
                                    marker='o', label=a) for a in V3_ARMS]
    fig = g.figure
    xlabel_artist = place_shared_xlabel(fig, 'Test Accuracy')
    place_shared_ylabel(fig, 'Gen. Gap')
    _save_three_part_legend(
        fig, xlabel_artist,
        [(method_handles, 'Arch.', 1),
         (_reg_handles(V3_L2_ARMS), 'Reg. ($\\lambda$)', 1),
         (_size_handles(subset, area), 'Train size', 2)],
        xs=(0.16, 0.40, 0.58))
    plt.savefig(str(out), bbox_inches='tight', dpi=150)
    plt.close(fig)


# =========================================================================
# gen_error_vs_acc_anchor.png -- where the random-init L2 penalty pulls
# =========================================================================
#   zero        ||theta||^2              classic weight decay
#   init        ||theta - theta_0||^2    toward this run's own random init
#   scattering  ||theta - theta_wav||^2  toward the wavelet init of the same architecture
# Rows = lambda, columns = dataset. Random Init. (lambda = 0) and Fine-tuned Scat. (same
# lambda) are drawn in every panel as hollow-marker references.
ANCHOR_LAMBDAS = (0.1, 1.0)
ANCHOR_ORDER = ('zero', 'init', 'scattering')
ANCHOR_REF_RANDOM = 'Random Init. ($\\lambda$=0)'
ANCHOR_REF_FT = 'Fine-tuned Scat.'
ANCHOR_SERIES = {
    # key: (legend label, colour, marker, is_reference)
    ANCHOR_REF_RANDOM: (ANCHOR_REF_RANDOM, METHOD_COLOR['Random Init.'], 'o', True),
    ANCHOR_REF_FT:     (ANCHOR_REF_FT, METHOD_COLOR['Fine-tuned Scat.'], 'D', True),
    'zero':            ('Rand. + anchor 0', '#eda100', 's', False),
    'init':            ('Rand. + anchor $\\theta_0$', '#e34948', '^', False),
    'scattering':      ('Rand. + anchor scat.', '#ff00ff', 'v', False),
}
ANCHOR_SERIES_ORDER = (ANCHOR_REF_RANDOM, ANCHOR_REF_FT, 'zero', 'init', 'scattering')
_ANCHOR_DIR_TAG = re.compile(r'_l2(scat|init|zero)(?:_|/|$)')


def _anchor_from_path(path):
    """'scattering' / 'init' / 'zero' from a run's dir stem (``..._randomInit_l2scat_...``)."""
    m = _ANCHOR_DIR_TAG.search(str(path))
    if m is None:
        return 'zero'
    return {'scat': 'scattering', 'init': 'init', 'zero': 'zero'}[m.group(1)]


def _anchor_frame(df, sources=None, verbose=True):
    """Mean (acc, gen) per (dataset, series, lambda, train_size) for the anchor figure."""
    sources = L2_ANCHOR_SUMMARY_SOURCES if sources is None else sources
    parts = []
    zero = load_l2_disentangle_frame(verbose=False)
    if not zero.empty:
        parts.append(zero)
    extra = load_l2_disentangle_frame(sources=sources, verbose=False)
    if not extra.empty:
        parts.append(extra)
    if not parts:
        return pd.DataFrame()
    rand = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['dataset', '_path'])
    rand['series'] = rand['_path'].map(_anchor_from_path)

    have_anchor = set(rand.loc[rand['series'] != 'zero', 'dataset'])
    if verbose:
        missing = [d for d in DATASET_ORDER
                   if d in set(rand['dataset']) and d not in have_anchor]
        if missing:
            print(f"  [anchor] no init/scattering anchor runs for: "
                  f"{', '.join(missing)} -- skipped")
    rand = rand[rand['dataset'].isin(have_anchor)]
    if rand.empty:
        return rand

    ds = df[(df['framework'] == 'deep_scatter') & df['dataset'].isin(have_anchor)].copy()
    ds['l2_penalty'] = pd.to_numeric(ds['l2_penalty'], errors='coerce').fillna(0.0)
    is_rand = ds['random_init'].astype(str) == 'True'
    base = ds[(ds['accuracy_type'] == 'joint_acc') & is_rand & (ds['l2_penalty'] == 0.0)]
    ft = ds[(ds['accuracy_type'] == 'finetune_acc') & ~is_rand
            & ds['l2_penalty'].isin(ANCHOR_LAMBDAS)].assign(series=ANCHOR_REF_FT)
    base = pd.concat([base.assign(series=ANCHOR_REF_RANDOM, l2_penalty=l2)
                      for l2 in ANCHOR_LAMBDAS], ignore_index=True)

    rows = pd.concat([rand, ft, base], ignore_index=True)
    rows['l2_penalty'] = pd.to_numeric(rows['l2_penalty'], errors='coerce').fillna(0.0)
    rows = rows[rows['l2_penalty'].isin(ANCHOR_LAMBDAS)]
    rows = rows.dropna(subset=['gen_error', 'accuracy_value'])
    rows['train_size'] = rows['train_size'].astype(float).round(3)
    agg = (rows.groupby(['dataset', 'series', 'l2_penalty', 'train_size'])
           .agg(acc=('accuracy_value', 'mean'), gen=('gen_error', 'mean'),
                n=('accuracy_value', 'size'))
           .reset_index().sort_values('train_size'))
    if verbose:
        for (d, s, l2), g in agg.groupby(['dataset', 'series', 'l2_penalty']):
            if s in ANCHOR_ORDER:
                print(f"  [anchor] {d:<10} {s:<10} L2={l2:g}  sizes={len(g)}  "
                      f"runs/cell={int(g['n'].min())}-{int(g['n'].max())}")
    return agg


def plot_gen_error_vs_acc_anchor(df, subset_sizes=(0.01, 0.1, 0.5, 1.0),
                                 out='./gen_error_vs_acc_anchor.png'):
    """Gen. gap vs test accuracy for the random-init model under each L2 anchor."""
    agg = _anchor_frame(df)
    if agg.empty:
        print("  [anchor] nothing to plot")
        return
    dataset_order = _resolve_datasets(agg)
    agg = agg[agg['train_size'].isin([round(float(s), 3) for s in subset_sizes])]
    dataset_order = [d for d in dataset_order if d in set(agg['dataset'])]
    if not dataset_order:
        return
    subset = sorted(agg['train_size'].unique())
    area = _train_size_area(subset)

    n_rows, n_cols = len(ANCHOR_LAMBDAS), len(dataset_order)
    panel = 6.0
    ms = 0.7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel * n_cols, panel * n_rows),
                             squeeze=False)

    for c, dataset in enumerate(dataset_order):
        dd = agg[agg['dataset'] == dataset]
        # One square window per column, so both lambda rows share a 1:1 scale.
        pad_x = 0.08 * max(dd['acc'].max() - dd['acc'].min(), 1.0)
        pad_y = 0.08 * max(dd['gen'].max() - dd['gen'].min(), 1.0)
        x0, x1 = dd['acc'].min() - pad_x, dd['acc'].max() + pad_x
        y0, y1 = dd['gen'].min() - pad_y, dd['gen'].max() + pad_y
        span = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for r, l2 in enumerate(ANCHOR_LAMBDAS):
            ax = axes[r, c]
            ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
            panel_df = dd[dd['l2_penalty'] == l2]
            for key in ANCHOR_SERIES_ORDER:
                _label, colour, marker, is_ref = ANCHOR_SERIES[key]
                sub = panel_df[panel_df['series'] == key].sort_values('train_size')
                if sub.empty:
                    continue
                ax.plot(sub['acc'], sub['gen'], color=colour,
                        linewidth=1.6 if is_ref else 2.4,
                        linestyle='--' if is_ref else '-',
                        alpha=0.9, zorder=2 if is_ref else 3)
                ax.scatter(sub['acc'], sub['gen'],
                           s=[area(t) * ms for t in sub['train_size']], marker=marker,
                           facecolor='white' if is_ref else colour,
                           edgecolor=colour if is_ref else 'white',
                           linewidth=1.8 if is_ref else 1.2,
                           zorder=2.5 if is_ref else 4)
            ax.set_xlim(cx - span / 2, cx + span / 2)
            ax.set_ylim(cy - span / 2, cy + span / 2)
            ax.set_aspect('equal', adjustable='box')
            _hide_negative_xticks(ax)
            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(f"$\\lambda$ = {l2:g}\nGen. Gap")
            if r == n_rows - 1:
                ax.set_xlabel('Test Accuracy')

    series_handles = [
        mlines.Line2D([], [], color=ANCHOR_SERIES[k][1], marker=ANCHOR_SERIES[k][2],
                      markersize=9, linewidth=1.4 if ANCHOR_SERIES[k][3] else 2.5,
                      linestyle='--' if ANCHOR_SERIES[k][3] else '-',
                      markerfacecolor='white' if ANCHOR_SERIES[k][3] else ANCHOR_SERIES[k][1],
                      markeredgecolor=ANCHOR_SERIES[k][1] if ANCHOR_SERIES[k][3] else 'white',
                      label=ANCHOR_SERIES[k][0])
        for k in ANCHOR_SERIES_ORDER]
    size_handles = [mlines.Line2D([], [], color='0.35', marker='o', linestyle='None',
                                  markersize=np.sqrt(area(s) * ms) * 0.8, label=f"{s*100:g}%")
                    for s in subset]
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    lp, ltp = legend_fonts()
    leg1 = fig.legend(handles=series_handles, title='Model', loc='upper left',
                      bbox_to_anchor=(0.04, 0.0), frameon=False, ncol=1, prop=lp,
                      title_fontsize=ltp, labelspacing=0.5)
    fig.legend(handles=size_handles, title='Train size', loc='upper left',
               bbox_to_anchor=(0.68, 0.0), frameon=False, ncol=1, prop=lp,
               title_fontsize=ltp, labelspacing=0.7)
    fig.add_artist(leg1)
    plt.savefig(str(out), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  [anchor] wrote {out}")


# =========================================================================
# gen_error_vs_filter_delta_v2.png -- gen. gap vs fine-tuning distance
# =========================================================================
# hue + marker shape = arm, line style = lambda, marker area = train size. Grey rungs join
# the lambda levels at a fixed train size.
#
# Random Init. is shown at lambda = 0 only: with --random-init the penalty pulls toward
# the origin, so the relative filter distance from init is ~1 by construction for
# lambda > 0. Scattering is frozen (distance 0) and ScatNet B.A. has no filter distance.
FD_V2_ARMS = ('Fine-tuned Scat.', 'Random Init.')
FD_V2_L2 = (0.0, 0.1, 1.0)
FD_V2_RUNG = dict(color='0.45', linewidth=1.0, alpha=0.55)
FD_V2_ARM_MARKER = {'Fine-tuned Scat.': 'o', 'Random Init.': 'X'}


def _fd_v2_frame(df, verbose=True):
    """Mean (filter_delta_relative, gen_error) per (dataset, arm, lambda, train_size)."""
    base = _assign_arch_labels(df.copy())
    base = base[base['Arch.'].isin(FD_V2_ARMS)]
    base['l2_penalty'] = pd.to_numeric(base['l2_penalty'], errors='coerce').fillna(0.0)
    base = base[base['l2_penalty'].isin(FD_V2_L2)]
    dropped = (base['Arch.'] == 'Random Init.') & (base['l2_penalty'] > 0)
    if dropped.any():
        if verbose:
            print(f"  [fd-v2] omitting {int(dropped.sum())} Random Init. run(s) at lambda>0")
        base = base[~dropped]
    base = base.dropna(subset=['gen_error', 'filter_delta_relative'])
    base = base[base['filter_delta_relative'] > 0]   # log axis
    if base.empty:
        return base
    return (base.groupby(['dataset', 'Arch.', 'l2_penalty', 'train_size'])
            .agg(fd=('filter_delta_relative', 'mean'), gen=('gen_error', 'mean'))
            .reset_index().sort_values('train_size'))


def plot_gen_error_vs_filter_delta_v2(df, subset_sizes=(0.01, 0.1, 0.5, 1.0),
                                      out='./gen_error_vs_filter_delta_v2.png'):
    """Gen. gap vs relative fine-tuning distance of the filters."""
    agg = _fd_v2_frame(df)
    if agg.empty:
        print("WARNING [fd-v2]: no filter-delta rows; plot skipped.")
        return
    dataset_order = _resolve_datasets(agg)
    agg = agg.assign(train_size=agg['train_size'].astype(float).round(3))
    agg = agg[agg['train_size'].isin([round(float(s), 3) for s in subset_sizes])]
    if agg.empty:
        return

    subset = sorted(agg['train_size'].unique())
    area = _train_size_area(subset)
    arm_order = [a for a in FD_V2_ARMS if a in set(agg['Arch.'])]
    l2_order = [l for l in FD_V2_L2 if l in set(agg['l2_penalty'])]

    def draw(data, **kwargs):
        ax = plt.gca()
        ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
        ax.set_xscale('log')
        # Rungs: one per (arm, train size), through that size's lambda levels.
        for arm in arm_order:
            for _ts, grp in data[data['Arch.'] == arm].groupby('train_size'):
                grp = grp.sort_values('l2_penalty')
                if len(grp) < 2:
                    continue
                ax.plot(grp['fd'], grp['gen'], zorder=1,
                        linewidth=FD_V2_RUNG['linewidth'],
                        color=FD_V2_RUNG['color'], alpha=FD_V2_RUNG['alpha'],
                        solid_capstyle='round')
        # Trajectories: one per (arm, lambda), over train size.
        for arm in arm_order:
            colour = METHOD_COLOR[arm]
            marker = FD_V2_ARM_MARKER.get(arm, 'o')
            for l2 in l2_order:
                sub = data[(data['Arch.'] == arm)
                           & (data['l2_penalty'] == l2)].sort_values('train_size')
                if sub.empty:
                    continue
                ax.plot(sub['fd'], sub['gen'], color=colour, alpha=0.9,
                        linestyle=V2_L2_STYLE.get(l2, '-'), linewidth=2.0,
                        solid_capstyle='round', zorder=2)
                ax.scatter(sub['fd'], sub['gen'], marker=marker,
                           s=[area(t) for t in sub['train_size']],
                           color=colour, edgecolor='white',
                           linewidth=1.2, zorder=3)
        _thin_log_xticks(ax)

    g = sns.FacetGrid(data=agg, col='dataset', col_order=dataset_order,
                      height=4.2, aspect=0.95, sharex=False, sharey=False)
    g.map_dataframe(draw)
    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("", "")

    method_handles = [mlines.Line2D([], [], color=METHOD_COLOR[a], linewidth=3,
                                    marker=FD_V2_ARM_MARKER.get(a, 'o'), label=a)
                      for a in arm_order]
    rung_handles = ([mlines.Line2D([], [], label='same train size', **FD_V2_RUNG)]
                    if len(l2_order) > 1 else [])

    fig = g.figure
    xlabel_artist = place_shared_xlabel(fig, 'Fine-tuning Distance')
    place_shared_ylabel(fig, 'Gen. Gap')
    sections = [(method_handles, 'Arch.', 1), (_reg_handles(l2_order), 'Reg. ($\\lambda$)', 1)]
    xs = [0.06, 0.28]
    if rung_handles:
        sections.append((rung_handles, 'Link', 1))
        xs.append(0.40)
    sections.append((_size_handles(subset, area), 'Train size', 2))
    xs.append(0.68)
    _save_three_part_legend(fig, xlabel_artist, sections, xs)
    plt.savefig(str(out), bbox_inches='tight', dpi=150)
    plt.close(fig)


# =========================================================================
# nc1_skew_vs_test_acc_variance_dataCols.png
# =========================================================================
# Skewed NC1 (nc1_skew_proj): within/between-class trace ratio under the orthogonal
# projector onto the classifier subspace. Written only by nc_report.py.
NC1_SKEW_X_LABEL = 'Within-Class Var. / Between-Class Var. (classifier subspace)'


def plot_nc_variance_data_cols(long_nc, out, x, x_label, y, y_label):
    """NC metric `x` vs `y` with seed error bars; datasets across, lambda (0, 1) down."""
    long_nc = long_nc.copy()
    for col in (x, y):
        if col not in long_nc.columns:
            print(f"WARNING [nc-plot]: no '{col}' column in the NC frame; {out} skipped.")
            return
    dataset_order = _resolve_datasets(long_nc)
    if not dataset_order:
        return
    long_nc = long_nc.loc[long_nc.l2_penalty.isin([0.0, 1.0])].reset_index(drop=True)
    long_nc = long_nc.loc[long_nc[x].notna() & long_nc[y].notna()].reset_index(drop=True)
    if long_nc.empty:
        return

    long_nc.replace({'phase': PHASE_LABEL}, inplace=True)
    long_nc.rename(columns={'phase': 'Arch.', 'train_size': 'Train Size'}, inplace=True)
    agg_df = long_nc.groupby(
        ['Train Size', 'l2_penalty', 'Arch.', 'dataset', 'framework']
    ).agg(
        x_mean=(x, 'mean'),
        x_std=(x, 'std'),
        y_mean=(y, 'mean'),
        y_std=(y, 'std')
    ).reset_index()

    g = sns.FacetGrid(
        data=agg_df,
        row='l2_penalty',
        col='dataset',
        height=5.5,
        aspect=.75,
        sharex='col',
        col_order=dataset_order
    )
    arch_order = list(METHOD_ORDER)

    def draw_errorbars(data, **kwargs):
        ax = plt.gca()
        for phase_val in arch_order:
            subset = data[data['Arch.'] == phase_val]
            if not subset.empty:
                ax.errorbar(
                    x=subset['x_mean'],
                    y=subset['y_mean'],
                    xerr=subset['x_std'],
                    yerr=subset['y_std'],
                    fmt='none',
                    ecolor=METHOD_COLOR[phase_val],
                    alpha=0.3,
                    elinewidth=1.5,
                    capsize=3
                )

    g.map_dataframe(draw_errorbars)
    g.map_dataframe(
        sns.scatterplot,
        x='x_mean',
        y='y_mean',
        hue='Arch.',
        hue_order=arch_order,
        size='Train Size',
        sizes=(30, 250),
        palette=METHOD_COLOR,
        alpha=0.9,
        edgecolor='w',
        size_norm=LogNorm()
    )

    # Rebuild the legend: methods first, then a subset of train sizes in ascending order.
    g.add_legend()
    handles = g._legend.legend_handles
    labels = [t.get_text() for t in g._legend.texts]
    g._legend.remove()
    for ax in g.axes.flat:
        old_leg = ax.get_legend()
        if old_leg is not None:
            old_leg.remove()

    arch_handles, arch_labels = [], []
    size_items = []
    train_size_header = None
    keep_labels = ['Train Size', 'Random Init.', 'Scattering', 'Fine-tuned Scat.', 'ScatNet B.A.']
    desired_sizes = [0.005, 0.01, 0.1, 0.2, 0.5, 1.0]
    seen_sizes = set()
    for handle, label in zip(handles, labels):
        if label in keep_labels:
            if label in METHOD_ORDER:
                if hasattr(handle, 'set_sizes'):
                    handle.set_sizes([150])
                if hasattr(handle, 'set_markersize'):
                    handle.set_markersize(12)
            arch_handles.append(handle)
            arch_labels.append(label)
        elif label == 'Train Size':
            train_size_header = (handle, label)
        else:
            clean_label = label.replace('−', '-')
            try:
                val_rounded = round(float(clean_label), 3)
                if val_rounded in desired_sizes and val_rounded not in seen_sizes:
                    size_items.append((val_rounded, handle, f"{val_rounded:g}"))
                    seen_sizes.add(val_rounded)
            except ValueError:
                pass
    size_items.sort(key=lambda item: item[0])

    new_handles = arch_handles.copy()
    new_labels = arch_labels.copy()
    if train_size_header:
        new_handles.append(train_size_header[0])
        new_labels.append(train_size_header[1])
    for _, handle, label in size_items:
        new_handles.append(handle)
        new_labels.append(label)

    legend = g.figure.legend(
        handles=new_handles,
        labels=new_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        frameon=False,
        labelspacing=0.8,
        columnspacing=1.5
    )
    g.set_axis_labels("", "")
    plt.subplots_adjust(bottom=0.15, left=0.15)
    g.set_titles(col_template="Reg: {row_name}", row_template="{col_name}")
    plt.tight_layout()
    xlabel_artist = place_shared_xlabel(g.figure, x_label)
    place_shared_ylabel(g.figure, y_label)
    push_legend_under(g.figure, legend, xlabel_artist)
    plt.savefig(str(out), bbox_inches='tight')
    plt.close()


def plot_nc1_skew_vs_test_acc_variance_data_cols(
        long_nc, out='./nc1_skew_vs_test_acc_variance_dataCols.png'):
    """Skewed (classifier-subspace) within-class variance vs. test accuracy."""
    return plot_nc_variance_data_cols(
        long_nc, out=out, x=NC1_SKEW_COL, x_label=NC1_SKEW_X_LABEL,
        y='test_acc', y_label='Test Accuracy (%)')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Plot results from deep_scatter and sep_conc_deepnets')
    argparser.add_argument('--redo-analyze', action='store_true',
                           help='Regenerate every summary.csv before plotting')
    argparser.add_argument('--datasets', nargs='+', default=None, metavar='NAME',
                           choices=list(DATASET_SOURCES),
                           help='Read only these DATASET_SOURCES entries (default: all)')
    argparser.add_argument('--combined-font-scale', type=float, default=1.0, metavar='FACTOR',
                           help='Scale all text in the figures by this factor')
    argparser.add_argument('--combined-legend-font-scale', type=float, default=1.0,
                           metavar='FACTOR',
                           help='Extra multiplier on legend text, applied on top of '
                                '--combined-font-scale')
    argparser.add_argument('--shared-label-pad-em', type=float, default=SHARED_LABEL_PAD_EM,
                           metavar='EM',
                           help='Gap between the panels and the shared axis label, in em '
                                '(default %(default)s)')
    argparser.add_argument('--shared-legend-pad-em', type=float,
                           default=SHARED_LEGEND_PAD_EM, metavar='EM',
                           help='Gap between the shared axis label and the legend strip, '
                                'in em (default %(default)s)')
    args = argparser.parse_args()

    if args.combined_font_scale <= 0 or args.combined_legend_font_scale <= 0:
        raise SystemExit('--combined-font-scale / --combined-legend-font-scale must be > 0')
    COMBINED_FONT_SCALE = args.combined_font_scale
    COMBINED_LEGEND_FONT_SCALE = args.combined_legend_font_scale
    SHARED_LABEL_PAD_EM = args.shared_label_pad_em
    SHARED_LEGEND_PAD_EM = args.shared_legend_pad_em

    set_active_datasets(args.datasets)

    if args.redo_analyze:
        redo_analyze()

    df, df_ds, long_nc = load_data()

    with combined_style():
        plot_acc_vs_train_size(df, 'accuracy_value')
        plot_gen_error_vs_acc_v2(df)
        plot_gen_error_vs_acc_v3(df)
        plot_gen_error_vs_acc_anchor(df)
        plot_gen_error_vs_filter_delta_v2(df_ds)
        plot_nc1_skew_vs_test_acc_variance_data_cols(long_nc)
