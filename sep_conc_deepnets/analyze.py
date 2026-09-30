"""Analyze experiment results from main.py JSON logs.

Reads all ``results_*.json`` files under a directory, writes ``summary.csv``, and
produces comparison plots:

 1. Architecture comparison (grouped bar, full-data)
 2. Box plot -- accuracy by modulus type
 3. Box plot -- accuracy by depth
 4. Box plot -- fine-tune gain by data fraction
 5. Filter delta vs fine-tune gain (scatter)
 6. Data efficiency (three-line: random < scat-frozen < scat-tuned)
 7. Scattering advantage over random vs data size
 8. Heatmap (architecture x data fraction)
 9. L2 penalty effect
10. Paired difference (scat - random) horizontal bars
11. Mixing horizon comparison (4-block only)
12. Best architecture per data fraction
13. Generalization error by data fraction (paired random vs scatter)
14. Generalization error vs filter delta
15. Paired generalization-error difference (scat - random)
16. Generalization-error data efficiency curves
17. Per-experiment training curves (optional)

Usage
-----
    python analyze.py --results-dir training_logs/<sweep> --recursive
    python analyze.py --results-dir training_logs/<sweep> --recursive --curves --interactive
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FRAC_ORDER = ["10%", "30%", "50%", "100%"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(row):
    """Last component of save_dir, e.g. '3b_prelu_4L_s01'."""
    save_dir = row.get("save_dir")
    if isinstance(save_dir, str) and save_dir.strip():
        return save_dir.rstrip("/").split("/")[-1]

    # Fallback: use the folder containing results_*.json
    path = row.get("_path")
    if isinstance(path, str) and path.strip():
        return os.path.basename(os.path.dirname(path))

    return "unknown"


def _data_fraction_label(row):
    ts = row.get("train_size")
    if isinstance(ts, str):
        try:
            ts = float(ts)
        except ValueError:
            pass

    try:
        if ts is None or (isinstance(ts, float) and np.isnan(ts)) or ((isinstance(ts, (int, float))) & (ts >= 1.0)):
            return "100%"
    except Exception:
        breakpoint()
    return f"{int(ts * 100)}%"


def _is_random(row):
    return bool(row.get("random_init", False))


def _arch_key(row):
    """Canonical architecture tuple (ignores init type, data fraction, l2)."""
    def _to_int_or_none(v):
        if pd.isna(v):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    mh = row.get("mixing_horizon")
    if isinstance(mh, float) and np.isnan(mh):
        mh = None
    mod = row.get("modulus_type")
    if pd.isna(mod):
        mod = None
    return (_to_int_or_none(row.get("n_blocks")), mod,
            _to_int_or_none(row.get("L")), mh)


def _arch_short_label(row):
    """Short label like '3b_prelu_4L' or '4b_cmod_8L_h27'."""
    def _to_int_or_none(v):
        if pd.isna(v):
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    nb = _to_int_or_none(row.get("n_blocks"))
    lv = _to_int_or_none(row.get("L"))
    mod_val = row.get("modulus_type")
    if pd.isna(mod_val):
        mod = "unk"
    else:
        mod = "prelu" if mod_val == "phase_relu" else "cmod"
    mh = row.get("mixing_horizon")
    if isinstance(mh, float) and np.isnan(mh):
        mh = None
    nb_str = str(nb) if nb is not None else "x"
    lv_str = str(lv) if lv is not None else "x"
    if mh is not None:
        return f"{nb_str}b_{mod}_{lv_str}L_h{int(mh)}"
    return f"{nb_str}b_{mod}_{lv_str}L"


def _find_matched_pairs(df):
    """Match each random experiment to the best scattering experiment
    (highest finetune_acc across l2 settings) with the same arch + data_frac.

    Returns list of dicts: arch_key, label, scat, rand, data_frac,
    actual_train_samples.
    """
    needed = {"_is_random", "_arch_key", "data_frac", "finetune_acc"}
    if not needed.issubset(df.columns):
        return []

    pairs = []
    scat = df[~df["_is_random"]]
    rand = df[df["_is_random"]]

    for _, r_row in rand.iterrows():
        key = _arch_key(r_row)
        frac = r_row["data_frac"]
        candidates = scat[(scat["_arch_key"] == key) & (scat["data_frac"] == frac)]
        if candidates.empty:
            continue
        valid = candidates[candidates["finetune_acc"].notna()]
        s_row = (valid.loc[valid["finetune_acc"].idxmax()]
                 if not valid.empty else candidates.iloc[0])
        pairs.append({
            "arch_key": key,
            "label": s_row["_arch_label"],
            "scat": s_row,
            "rand": r_row,
            "data_frac": frac,
            "actual_train_samples": r_row.get("actual_train_samples"),
        })
    return pairs


def _boxplot_with_strip(ax, groups, positions, colors, width=0.6):
    """Box plots with overlaid jittered points."""
    bp = ax.boxplot(groups, positions=positions, widths=width,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(42)
    for data, pos in zip(groups, positions):
        if len(data) == 0:
            continue
        jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(data))
        ax.scatter(pos + jitter, data, s=30, zorder=5,
                   edgecolors="black", linewidth=0.5, alpha=0.85)


def _save(fig, save_dir, name):
    path = os.path.join(save_dir, name)
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    return fig


def _get_available_delta_types(df):
    """Detect which delta types are available in the DataFrame.
    
    Returns dict with keys: 'p', 'f', 'filter', 'classifier' and bool values
    indicating presence (an all-NaN column counts as absent), and the primary delta for
    plotting (filter if available, else f, else p).
    """
    delta_types = {}
    for delta_type in ['p', 'f', 'filter', 'classifier']:
        col = f"{delta_type}_delta_relative"
        delta_types[delta_type] = col in df.columns and df[col].notna().any()

    # The classifier is never primary: it is trained under every method.
    primary = None
    for dtype in ['filter', 'f', 'p']:
        if delta_types.get(dtype, False):
            primary = dtype
            break
    
    return delta_types, primary


def _extract_train_acc_for_phase(curves, phase, source, epoch_num=None):
    """Extract train_acc1 for a phase from epoch curves.

    source='best-val': train_acc1 at epoch with highest val_acc1/val_acc within the phase.
    source='last-epoch': train_acc1 at final epoch in the phase.
    epoch_num=<int>: train_acc1 at the requested epoch. If not present,
    returns NaN (with a compatibility check for 0-based epoch numbering).
    """
    if not phase or not isinstance(curves, dict):
        return np.nan

    entries = curves.get(phase)
    if not entries:
        return np.nan

    # Keep entries of this phase (entries without phase metadata are kept).
    phase_entries = []
    for entry in entries:
        entry_phase = entry.get("phase")
        if entry_phase is None or entry_phase == phase:
            phase_entries.append(entry)
    
    if not phase_entries:
        phase_entries = entries

    if epoch_num is not None:
        chosen = None
        target_epochs = {epoch_num}
        # Compatibility: some logs may use 0-based epoch indexing.
        if epoch_num > 0:
            target_epochs.add(epoch_num - 1)

        for entry in phase_entries:
            raw_epoch = entry.get("epoch")
            try:
                entry_epoch = int(float(raw_epoch))
            except (TypeError, ValueError):
                continue
            if entry_epoch in target_epochs:
                chosen = entry
                break
        if chosen is None:
            return np.nan
    elif source == "last-epoch":
        chosen = phase_entries[-1]
    else:
        # For "best-val", find best within this phase only
        chosen = None
        best_val = -np.inf
        for entry in phase_entries:
            val_acc = entry.get("val_acc1", entry.get("val_acc"))
            if pd.isna(val_acc):
                continue
            if chosen is None or val_acc > best_val:
                chosen = entry
                best_val = val_acc
        if chosen is None:
            chosen = phase_entries[-1]

    train_acc = chosen.get("train_acc1", chosen.get("train_acc"))
    return train_acc if pd.notna(train_acc) else np.nan


def _add_generalization_columns(df, gen_train_source, gen_train_epoch=None):
    """Add phase-aligned train/test columns and generalization error.

    Generalization error = train_acc1 - val_acc1 at the best-validation epoch in
    epoch_curves.
    """
    gen_phase = []
    train_acc = []
    val_acc = []
    missing_epoch_count = 0

    for _, row in df.iterrows():
        is_rand = bool(row.get("_is_random", False))
        curves = row.get("_epoch_curves") or row.get("_curves") or {}

        phase = None
        t_acc = np.nan
        v_acc = np.nan

        # Best validation epoch across all phases.
        best_entry = None
        best_val_acc = -np.inf
        best_phase = None

        for phase_name, entries in curves.items():
            if not entries or not isinstance(entries, list):
                continue
            
            for entry in entries:
                val_acc_entry = entry.get("val_acc1", entry.get("val_acc"))
                if pd.isna(val_acc_entry):
                    continue
                
                if best_entry is None or val_acc_entry > best_val_acc:
                    best_entry = entry
                    best_val_acc = val_acc_entry
                    best_phase = phase_name

        if best_entry is not None:
            phase = best_phase
            v_acc = best_entry.get("val_acc1", best_entry.get("val_acc"))
            t_acc = best_entry.get("train_acc1", best_entry.get("train_acc"))
        elif is_rand:
            # No curves: fall back to joint_acc.
            phase = "joint"
            t_acc = row.get("joint_acc")
            if "joint" in curves and curves["joint"]:
                v_acc = curves["joint"][-1].get("val_acc1", curves["joint"][-1].get("val_acc"))
        else:
            # No curves: fall back to best_acc1 and best_epoch_acc1.
            t_acc = row.get("best_acc1")
            if pd.notna(t_acc):
                best_epoch = row.get("best_epoch_acc1")
                if pd.notna(best_epoch):
                    best_epoch = int(best_epoch)
                    for phase_name, entries in curves.items():
                        if not entries:
                            continue
                        for entry in entries:
                            if int(entry.get("epoch", -1)) == best_epoch:
                                phase = phase_name
                                v_acc = entry.get("val_acc1", entry.get("val_acc"))
                                break
                        if phase:
                            break

        gen_phase.append(phase if phase is not None else np.nan)
        train_acc.append(t_acc)
        val_acc.append(v_acc)

    df["gen_phase"] = gen_phase
    df["train_acc1_for_gen"] = train_acc
    df["val_acc1_for_gen"] = val_acc
    df["gen_error"] = df["train_acc1_for_gen"] - df["val_acc1_for_gen"]

    return df


# Every drift column main.py can emit (NaN-filled when not logged).
DELTA_COLS = [f"{g}_delta{suffix}"
              for g in ("p", "f", "filter", "classifier")
              for suffix in ("_l2", "_relative", "_prev_l2", "_prev_relative")]

# ---------------------------------------------------------------------------
# Load & summarize
# ---------------------------------------------------------------------------

def load_experiments(
    results_dir,
    recursive=False,
    gen_train_source="best-val",
    gen_train_epoch=None,
):
    """Read all results_*.json into a DataFrame."""
    pattern = os.path.join(results_dir, "**" if recursive else "",
                           "results_*.json")
    paths = sorted(glob.glob(pattern, recursive=recursive))

    if not paths:
        print(f"No results_*.json found in {results_dir}"
              f"{' (recursive)' if recursive else ''}")
        sys.exit(1)

    rows = []
    for path in paths:
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                breakpoint()
                print(f"Error decoding JSON in {path}: {e}")
                continue
        row = {}
        row.update(data.get("config", {}))
        row.update(data.get("results", {}))
        row["_path"] = path
        row["_step_curves"] = data.get("step_curves", {})
        row["_epoch_curves"] = data.get("epoch_curves", {})
        row["_curves"] = row["_epoch_curves"] or data.get("curves", {})
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in DELTA_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Map acc1-style result names to the names used below.
    if "finetune_acc1" in df.columns and "finetune_acc" not in df.columns:
        df["finetune_acc"] = df["finetune_acc1"]
    if "joint_acc1" in df.columns and "joint_acc" not in df.columns:
        df["joint_acc"] = df["joint_acc1"]
    if "baseline_acc1" in df.columns and "head_acc" not in df.columns:
        df["head_acc"] = df["baseline_acc1"]
    if "finetune_gain1" in df.columns and "finetune_gain" not in df.columns:
        df["finetune_gain"] = df["finetune_gain1"]
    # main.py's '*_nc1_within_class_variance_by_between_class' keys hold nc1_trace; they are
    # exposed as '*_nc1_trace_legacy'. The within/between ratio (nc1_rel) becomes
    # 'best_nc1' / 'test_nc1'.
    if "best_nc1_within_class_variance_by_between_class" in df.columns and "best_nc1_trace_legacy" not in df.columns:
        df["best_nc1_trace_legacy"] = df["best_nc1_within_class_variance_by_between_class"]
    if "test_nc1_within_class_variance_by_between_class" in df.columns and "test_nc1_trace_legacy" not in df.columns:
        df["test_nc1_trace_legacy"] = df["test_nc1_within_class_variance_by_between_class"]
    if "best_nc1_rel" in df.columns and "best_nc1" not in df.columns:
        df["best_nc1"] = df["best_nc1_rel"]
    if "test_nc1_rel" in df.columns and "test_nc1" not in df.columns:
        df["test_nc1"] = df["test_nc1_rel"]
    if "best_nc2_between_class_separation" in df.columns and "best_nc2" not in df.columns:
        df["best_nc2"] = df["best_nc2_between_class_separation"]
    if "best_nc3_class_means_cosine_similarity" in df.columns and "best_nc3" not in df.columns:
        df["best_nc3"] = df["best_nc3_class_means_cosine_similarity"]
    if "test_nc2_between_class_separation" in df.columns and "test_nc2" not in df.columns:
        df["test_nc2"] = df["test_nc2_between_class_separation"]
    if "test_nc3_class_means_cosine_similarity" in df.columns and "test_nc3" not in df.columns:
        df["test_nc3"] = df["test_nc3_class_means_cosine_similarity"]

    df["label"] = df.apply(_make_label, axis=1)
    df["data_frac"] = df.apply(_data_fraction_label, axis=1)
    df["_is_random"] = df.apply(_is_random, axis=1)
    df["_arch_key"] = df.apply(_arch_key, axis=1)
    df["_arch_label"] = df.apply(_arch_short_label, axis=1)
    df = _add_generalization_columns(df, gen_train_source, gen_train_epoch)

    if "joint_acc" in df.columns and "finetune_acc" in df.columns:
        df["best_acc"] = df["joint_acc"].fillna(df["finetune_acc"])
    elif "joint_acc" in df.columns:
        df["best_acc"] = df["joint_acc"]
    elif "finetune_acc" in df.columns:
        df["best_acc"] = df["finetune_acc"]

    n_scat = (~df["_is_random"]).sum()
    n_rand = df["_is_random"].sum()
    print(f"Loaded {len(df)} experiment(s) from {results_dir}"
          f" ({n_scat} scattering, {n_rand} random)\n")
    return df


# main.py's best-val checkpoint scored on test and val (see best_ckpt_eval.py).
# `best_acc1` is the max validation accuracy over epochs and `test_acc1` the last-epoch
# weights on test; `best_ckpt_test_acc1` is comparable to deep_scatter's test accuracy.
BEST_CKPT_COLS = [
    "best_ckpt_eval_version", "best_ckpt_source", "best_ckpt_epoch",
    "best_ckpt_test_acc1", "best_ckpt_test_acc5",
    "best_ckpt_test_nc1_rel", "best_ckpt_test_nc2", "best_ckpt_test_nc3",
    "best_ckpt_val_acc1", "best_ckpt_val_acc5",
]

SUMMARY_COLS = [
    "label", "arch", "random_init", "joint", "n_blocks", "L", "modulus_type", "use_batch_norm",
    "mixing_horizon", "l2_penalty", "actual_train_samples", "train_size", "epochs", "seed",
    "phase", "phase_epochs", "p_init", "f_init", "wavelet_finetune", "random_init_filters", "_path",
    "best_acc1", "best_epoch_acc1", "best_phase_epoch_acc1",
    "best_acc5", "best_epoch_acc5", "best_phase_epoch_acc5",
    "best_train_loss", "best_val_loss", "best_nc1", "best_nc1_rel", "best_nc2", "best_nc3",
    "best_nc1_trace_legacy",
    "test_loss", "test_acc1", "test_nc1", "test_nc2", "test_nc3",
    "test_nc1_trace_legacy",
    "phase_a_test_acc1", "phase_a_test_acc5", "phase_a_test_loss",
    "phase_a_test_nc1", "phase_a_test_nc2", "phase_a_test_nc3", "phase_a_test_nc1_trace",
    "gen_phase", "train_acc1_for_gen", "test_acc1_for_gen", "gen_error",
    "head_acc", "finetune_acc", "finetune_gain", "joint_acc",
    # Drift of each parameter group (NaN when not logged).
    "p_delta_l2", "p_delta_relative", "p_delta_prev_l2", "p_delta_prev_relative",
    "f_delta_l2", "f_delta_relative", "f_delta_prev_l2", "f_delta_prev_relative",
    "filter_delta_l2", "filter_delta_relative", "filter_delta_prev_l2", "filter_delta_prev_relative",
    "classifier_delta_l2", "classifier_delta_relative",
    "classifier_delta_prev_l2", "classifier_delta_prev_relative",
    "n_params_extractor", "n_params_classifier",
    "n_params_total", "n_params_trainable",
] + BEST_CKPT_COLS


def _best_phase_acc(entries):
    """Return best val acc1/acc5 and the epoch metadata for a phase curve."""
    if not entries:
        return {
            "phase_epochs": 0,
            "best_acc1": np.nan,
            "best_epoch_acc1": np.nan,
            "best_phase_epoch_acc1": np.nan,
            "best_acc5": np.nan,
            "best_epoch_acc5": np.nan,
            "best_phase_epoch_acc5": np.nan,
            "best_train_loss": np.nan,
            "best_val_loss": np.nan,
            "best_nc1": np.nan,
            "best_nc1_rel": np.nan,
            "best_nc2": np.nan,
            "best_nc3": np.nan,
        }

    def _best_for(metric_key, fallback_key):
        best_entry = None
        best_val = -np.inf
        for e in entries:
            val = e.get(metric_key, e.get(fallback_key))
            if pd.isna(val):
                continue
            if best_entry is None or val > best_val:
                best_entry = e
                best_val = val
        if best_entry is None:
            return np.nan, np.nan, np.nan
        return (
            best_val,
            best_entry.get("epoch", np.nan),
            best_entry.get("phase_epoch", np.nan),
        )

    best_entry = None
    best_val = -np.inf
    for e in entries:
        val = e.get("val_acc1", e.get("val_acc"))
        if pd.isna(val):
            continue
        if best_entry is None or val > best_val:
            best_entry = e
            best_val = val

    best_acc1, best_epoch_acc1, best_phase_epoch_acc1 = _best_for("val_acc1", "val_acc")
    best_acc5, best_epoch_acc5, best_phase_epoch_acc5 = _best_for("val_acc5", "val_acc5")
    return {
        "phase_epochs": len(entries),
        "best_acc1": best_acc1,
        "best_epoch_acc1": best_epoch_acc1,
        "best_phase_epoch_acc1": best_phase_epoch_acc1,
        "best_acc5": best_acc5,
        "best_epoch_acc5": best_epoch_acc5,
        "best_phase_epoch_acc5": best_phase_epoch_acc5,
        "best_train_loss": best_entry.get("train_loss", np.nan) if best_entry is not None else np.nan,
        "best_val_loss": best_entry.get("val_loss", np.nan) if best_entry is not None else np.nan,
        # Epoch curves log nc1_trace under "nc1"; "nc1_rel" is the within/between ratio.
        "best_nc1": best_entry.get("nc1", np.nan) if best_entry is not None else np.nan,
        "best_nc1_rel": best_entry.get("nc1_rel", np.nan) if best_entry is not None else np.nan,
        "best_nc2": best_entry.get("nc2", np.nan) if best_entry is not None else np.nan,
        "best_nc3": best_entry.get("nc3", np.nan) if best_entry is not None else np.nan,
    }


def _build_phase_summary_rows(df):
    """Expand to one summary row per (experiment label, phase)."""
    base_cols = [
        c for c in SUMMARY_COLS
        if c in df.columns and c not in {
            "phase", "phase_epochs",
            "best_acc1", "best_epoch_acc1", "best_phase_epoch_acc1",
            "best_acc5", "best_epoch_acc5", "best_phase_epoch_acc5",
            "best_train_loss", "best_val_loss", "best_nc1", "best_nc1_rel", "best_nc2", "best_nc3",
        }
    ]

    phase_rows = []
    phase_order = ["phase_a", "phase_b", "phase_c", "joint"]

    for _, row in df.iterrows():
        base = {c: row.get(c) for c in base_cols}
        curves = row.get("_epoch_curves") or row.get("_curves") or {}
        added = False

        if isinstance(curves, dict):
            for phase in phase_order:
                entries = curves.get(phase)
                if not entries:
                    continue
                stats = _best_phase_acc(entries)
                rec = dict(base)
                rec["phase"] = phase
                rec.update(stats)
                phase_rows.append(rec)
                added = True

        if added:
            continue

        # Fallback if curves are missing: use top-level results fields.
        fallback = []
        if pd.notna(row.get("head_acc")):
            fallback.append(("phase_a", row.get("head_acc"), row.get("baseline_acc5")))
        if pd.notna(row.get("finetune_acc")):
            fallback.append(("phase_b", row.get("finetune_acc"), row.get("finetune_acc5")))
        if pd.notna(row.get("joint_acc")):
            fallback.append(("joint", row.get("joint_acc"), row.get("joint_acc5")))

        for phase, acc1, acc5 in fallback:
            rec = dict(base)
            rec["phase"] = phase
            rec["phase_epochs"] = np.nan
            rec["best_acc1"] = acc1
            rec["best_epoch_acc1"] = np.nan
            rec["best_phase_epoch_acc1"] = np.nan
            rec["best_acc5"] = acc5
            rec["best_epoch_acc5"] = np.nan
            rec["best_phase_epoch_acc5"] = np.nan
            rec["best_train_loss"] = np.nan
            rec["best_val_loss"] = np.nan
            rec["best_nc1"] = np.nan
            rec["best_nc1_rel"] = np.nan
            rec["best_nc2"] = np.nan
            rec["best_nc3"] = np.nan
            phase_rows.append(rec)

    return pd.DataFrame(phase_rows)


def print_summary(df, save_dir=None, summary_epoch=None):
    view = _build_phase_summary_rows(df)
    if view.empty:
        print("No phase rows available for summary.")
        return

    cols = [c for c in SUMMARY_COLS if c in view.columns]
    view = view[cols].copy()
    sort_keys = [c for c in ["label", "phase", "n_blocks", "modulus_type", "actual_train_samples"]
                 if c in view.columns]
    if sort_keys:
        view = view.sort_values(sort_keys)
    print("=" * 120)
    print("EXPERIMENT SUMMARY")
    print("=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(view.to_string(index=False))
    print()
    if save_dir:
        filename = (
            f"summary_ep{summary_epoch}.csv"
            if summary_epoch is not None
            else "summary.csv"
        )
        path = os.path.join(save_dir, filename)
        view.to_csv(path, index=False)
        print(f"Saved {path}\n")


def save_epoch_metrics(df, save_dir):
    """Save one row per experiment-phase-epoch from logged training curves."""
    rows = []
    base_cols = [
        c for c in [
            "label", "random_init", "joint", "n_blocks", "L", "modulus_type",
            "mixing_horizon", "l2_penalty", "actual_train_samples", "lr_head",
            "epochs", "head_acc", "finetune_acc", "finetune_gain", "joint_acc",
            "best_acc1", "best_acc5", "best_train_loss", "best_val_loss",
            "best_nc1", "best_nc2", "best_nc3", "test_loss", "test_acc1",
            "test_nc1", "test_nc2", "test_nc3",
            "p_delta_l2", "p_delta_relative", "f_delta_l2", "f_delta_relative",
            "filter_delta_l2", "filter_delta_relative",
            "classifier_delta_l2", "classifier_delta_relative",
            "n_params_extractor", "n_params_classifier",
            "n_params_total", "n_params_trainable", "data_frac", "_path",
        ]
        if c in df.columns
    ]

    for _, row in df.iterrows():
        curves = row.get("_curves") or {}
        if not isinstance(curves, dict):
            continue

        base = {c: row.get(c) for c in base_cols}
        for phase, entries in curves.items():
            if not entries:
                continue
            for entry in entries:
                rec = dict(base)
                rec["curve_phase"] = phase
                for k, v in entry.items():
                    rec[f"curve_{k}"] = v
                rows.append(rec)

    if not rows:
        print("No epoch-curve entries found; skipped per-epoch metrics export.")
        return None

    out = pd.DataFrame(rows)
    path = os.path.join(save_dir, "summary_all_epochs.csv")
    out.to_csv(path, index=False)
    print(f"Saved {path}\n")
    return path


# ---------------------------------------------------------------------------
# Plot 1: Architecture comparison (grouped bar, full-data)
# ---------------------------------------------------------------------------

def plot_arch_comparison(df, save_dir):
    needed = {
        "data_frac", "_is_random", "_arch_label", "finetune_acc",
        "n_blocks", "modulus_type"
    }
    if not needed.issubset(df.columns):
        return None

    full = df[df["data_frac"] == "100%"].copy()
    if full.empty:
        return None

    scat_all = full[~full["_is_random"]]
    if scat_all.empty:
        return None

    # Pick the best l2 variant per architecture
    best_idx = scat_all.groupby("_arch_label")["finetune_acc"].idxmax()
    scat = scat_all.loc[best_idx].sort_values(["n_blocks", "modulus_type"])
    rand = full[full["_is_random"]].sort_values(["n_blocks", "modulus_type"])

    labels = scat["_arch_label"].values
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), 6))

    if "head_acc" in scat.columns:
        ax.bar(x - width, scat["head_acc"].fillna(0), width,
               label="Scat head (Phase A)", color="tab:blue", alpha=0.7)
    if "finetune_acc" in scat.columns:
        bars = ax.bar(x, scat["finetune_acc"].fillna(0), width,
                      label="Scat fine-tuned (Phase B)", color="tab:orange", alpha=0.7)
        if "finetune_gain" in scat.columns:
            for bar, gain in zip(bars, scat["finetune_gain"]):
                if pd.notna(gain):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}",
                            ha="center", va="bottom", fontsize=7)

    if "joint_acc" in rand.columns and not rand.empty:
        rand_by_arch = rand.set_index("_arch_label")
        rand_vals = [rand_by_arch.loc[al, "joint_acc"]
                     if al in rand_by_arch.index else 0
                     for al in labels]
        ax.bar(x + width, rand_vals, width,
               label="Random (joint)", color="tab:red", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Architecture Comparison (full training data)")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_arch_comparison.png")


# ---------------------------------------------------------------------------
# Plot 2: Box plot -- accuracy by modulus type
# ---------------------------------------------------------------------------

def plot_box_modulus(df, save_dir):
    needed = {"_is_random", "modulus_type", "head_acc", "finetune_acc"}
    if not needed.issubset(df.columns):
        return None

    scat = df[~df["_is_random"]]
    types = sorted(scat["modulus_type"].dropna().unique())
    if len(types) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    groups_head = [scat.loc[scat["modulus_type"] == t, "head_acc"].dropna().values
                   for t in types]
    groups_ft = [scat.loc[scat["modulus_type"] == t, "finetune_acc"].dropna().values
                 for t in types]

    pos_h = np.arange(len(types)) * 2.5
    pos_f = pos_h + 0.8
    _boxplot_with_strip(ax, groups_head, pos_h, ["tab:blue"] * len(types))
    _boxplot_with_strip(ax, groups_ft, pos_f, ["tab:orange"] * len(types))
    ax.set_xticks(pos_h + 0.4)
    ax.set_xticklabels(types)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
    ], labels=["Head (A)", "Fine-tuned (B)"])
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy by Modulus Type (scattering init)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_box_modulus.png")


# ---------------------------------------------------------------------------
# Plot 3: Box plot -- accuracy by depth
# ---------------------------------------------------------------------------

def plot_box_depth(df, save_dir):
    needed = {"_is_random", "n_blocks", "head_acc", "finetune_acc"}
    if not needed.issubset(df.columns):
        return None

    scat = df[~df["_is_random"]]
    depths = sorted(scat["n_blocks"].dropna().unique())
    if len(depths) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    groups_head = [scat.loc[scat["n_blocks"] == d, "head_acc"].dropna().values
                   for d in depths]
    groups_ft = [scat.loc[scat["n_blocks"] == d, "finetune_acc"].dropna().values
                 for d in depths]

    pos_h = np.arange(len(depths)) * 2.5
    pos_f = pos_h + 0.8
    _boxplot_with_strip(ax, groups_head, pos_h, ["tab:blue"] * len(depths))
    _boxplot_with_strip(ax, groups_ft, pos_f, ["tab:orange"] * len(depths))
    ax.set_xticks(pos_h + 0.4)
    ax.set_xticklabels([str(int(d)) for d in depths])
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, fc="tab:orange", alpha=0.5),
    ], labels=["Head (A)", "Fine-tuned (B)"])
    ax.set_xlabel("Number of blocks")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy by Network Depth (scattering init)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_box_depth.png")


# ---------------------------------------------------------------------------
# Plot 4: Box plot -- fine-tune gain by data fraction
# ---------------------------------------------------------------------------

def plot_box_gain(df, save_dir):
    sub = df.dropna(subset=["finetune_gain"])
    if sub.empty:
        return None

    fracs = [f for f in FRAC_ORDER if f in sub["data_frac"].values]
    if not fracs:
        return None

    groups = [sub.loc[sub["data_frac"] == f, "finetune_gain"].values for f in fracs]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(fracs))
    _boxplot_with_strip(ax, groups, positions, ["tab:green"] * len(fracs))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(fracs)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Fine-tune gain (%)")
    ax.set_title("Fine-Tuning Gain by Data Fraction")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_box_gain.png")


# ---------------------------------------------------------------------------
# Plot 5: Delta vs fine-tune gain (scatter) - handles multiple delta types
# ---------------------------------------------------------------------------

def plot_delta_vs_gain(df, save_dir, args):
    """Scatter plots of delta vs finetune gain. Supports p, f, and filter deltas."""
    delta_types, primary = _get_available_delta_types(df)
    
    # If no deltas available, skip
    if not any(delta_types.values()):
        return None
    
    # Generate plots for each available delta type
    for delta_type in ['filter', 'f', 'p']:
        if not delta_types.get(delta_type, False):
            continue
            
        delta_col = f"{delta_type}_delta_relative"
        sub = df.dropna(subset=[delta_col, "finetune_gain"])
        if sub.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sub[delta_col], sub["finetune_gain"],
                   s=60, edgecolors="black", linewidth=0.5, zorder=5,
                   cmap="viridis")
        if args.filter_delta_log_scale:
            ax.set_xscale('log')

        for _, row in sub.iterrows():
            ax.annotate(row["label"],
                        (row[delta_col], row["finetune_gain"]),
                        fontsize=6, alpha=0.75,
                        xytext=(4, 4), textcoords="offset points")

        delta_label_map = {'p': 'P filter', 'f': 'F filter', 'filter': 'Main filter'}
        delta_label = delta_label_map.get(delta_type, delta_type)
        ax.set_xlabel(f"Relative {delta_label} delta (L2)")
        ax.set_ylabel("Fine-tune gain (%)")
        ax.set_title(f"{delta_label} Change vs. Accuracy Improvement")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3)

        if len(sub) >= 3:
            r = sub[[delta_col, "finetune_gain"]].corr().iloc[0, 1]
            ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.95),
                        xycoords="axes fraction", fontsize=11)

        fig.tight_layout()
        filename = f"plot_delta_vs_gain_{delta_type}.png"
        _save(fig, save_dir, filename)
    
    # Return without explicit return value, all plots saved separately
    return None


# ---------------------------------------------------------------------------
# Plot 6: Data efficiency (three-line per architecture family)
# ---------------------------------------------------------------------------

def _build_init_families(df):
    """Group matched (scat, random) pairs by architecture into families.

    Returns dict: arch_label -> {scat_samples, scat_head, scat_ft,
    rand_samples, rand_joint}, sorted by sample count.
    """
    pairs = _find_matched_pairs(df)
    if not pairs:
        return {}

    families = {}
    for p in pairs:
        arch = p["label"]
        families.setdefault(arch, {
            "scat_samples": [], "scat_head": [], "scat_ft": [],
            "rand_samples": [], "rand_joint": [],
        })
        f = families[arch]
        s = p["actual_train_samples"]
        f["scat_samples"].append(s)
        f["scat_head"].append(p["scat"].get("head_acc"))
        f["scat_ft"].append(p["scat"].get("finetune_acc"))
        f["rand_samples"].append(s)
        f["rand_joint"].append(p["rand"].get("joint_acc"))

    result = {}
    for name, f in families.items():
        if len(f["scat_samples"]) < 2:
            continue
        order = np.argsort(f["scat_samples"])
        for key in f:
            f[key] = [f[key][i] for i in order]
        result[name] = f
    return result


def plot_data_efficiency(df, save_dir):
    """Three-line per family: random joint, scat head, scat fine-tuned."""
    families = _build_init_families(df)
    if not families:
        return None
    if len(families.keys()) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, f) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        ax.plot(f["scat_samples"], f["scat_head"], "^--",
                color=color, alpha=0.6, label=f"{name} scat head")
        ax.plot(f["scat_samples"], f["scat_ft"], "s-",
                color=color, label=f"{name} scat fine-tuned")
        ax.plot(f["rand_samples"], f["rand_joint"], "x:",
                color=color, alpha=0.6, markersize=8,
                label=f"{name} random joint")

    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Data Efficiency: Scattering Init vs Random Init")
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_data_efficiency.png")


# ---------------------------------------------------------------------------
# Plot 7: Scattering advantage over random vs data size
# ---------------------------------------------------------------------------

def plot_init_advantage(df, save_dir):
    """Line chart: (scat fine-tuned - random joint) vs training samples."""
    families = _build_init_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, f) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        advantage = [(ft or 0) - (rj or 0)
                     for ft, rj in zip(f["scat_ft"], f["rand_joint"])]
        ax.plot(f["scat_samples"], advantage, "o-", color=color, label=name)
        for s, adv in zip(f["scat_samples"], advantage):
            ax.annotate(f"{adv:+.1f}", (s, adv), fontsize=7,
                        textcoords="offset points", xytext=(0, 6), ha="center")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy advantage (%)")
    ax.set_title("Scattering Init Advantage over Random Init")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_init_advantage.png")


# ---------------------------------------------------------------------------
# Plot 8: Heatmap -- architecture x data fraction
# ---------------------------------------------------------------------------

def plot_heatmap(df, save_dir):
    """Side-by-side heatmaps: scattering best finetune_acc and random joint_acc."""
    needed = {"data_frac", "_is_random", "_arch_label"}
    if not needed.issubset(df.columns):
        return None

    fracs = [f for f in FRAC_ORDER if f in df["data_frac"].values]
    if not fracs:
        return None

    scat = df[~df["_is_random"]]
    rand = df[df["_is_random"]]

    def _arch_label_sort_key(label):
        label = str(label)
        try:
            return (int(label.split("b", 1)[0]), label)
        except (TypeError, ValueError):
            return (10**9, label)

    arch_labels = sorted(scat["_arch_label"].unique(),
                         key=_arch_label_sort_key)
    if not arch_labels:
        return None

    heatmaps = []
    if "finetune_acc" in scat.columns:
        scat_best = scat.groupby(["_arch_label", "data_frac"])["finetune_acc"].max()
        heatmaps.append(("Scattering (best fine-tuned)", scat_best))
    if "joint_acc" in rand.columns:
        rand_best = rand.groupby(["_arch_label", "data_frac"])["joint_acc"].max()
        heatmaps.append(("Random (joint)", rand_best))
    if not heatmaps:
        return None

    fig, axes = plt.subplots(
        1,
        len(heatmaps),
        figsize=(7 * len(heatmaps), max(5, len(arch_labels) * 0.4)),
    )
    if len(heatmaps) == 1:
        axes = [axes]

    for ax, (title, data) in zip(axes, heatmaps):
        mat = np.full((len(arch_labels), len(fracs)), np.nan)
        for i, al in enumerate(arch_labels):
            for j, fr in enumerate(fracs):
                try:
                    mat[i, j] = data.loc[(al, fr)]
                except KeyError:
                    pass

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels(fracs)
        ax.set_yticks(range(len(arch_labels)))
        ax.set_yticklabels(arch_labels, fontsize=8)
        ax.set_xlabel("Data fraction")
        ax.set_title(title)

        vmax = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if v < vmax * 0.8 else "white")

        fig.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Accuracy Heatmap: Architecture x Data Fraction", fontsize=13)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 9: L2 penalty effect (scattering init only)
# ---------------------------------------------------------------------------

def plot_l2_effect(df, save_dir):
    """Box plot: accuracy delta (l2=0.005 minus l2=0) grouped by data fraction."""
    scat = df[~df["_is_random"]]
    needed = {"l2_penalty", "finetune_acc", "data_frac", "_arch_label"}
    if not needed.issubset(scat.columns):
        return None

    fracs = [f for f in FRAC_ORDER if f in scat["data_frac"].values]
    if not fracs:
        return None

    deltas_by_frac = {}
    for frac in fracs:
        sub = scat[scat["data_frac"] == frac]
        deltas = []
        for al in sub["_arch_label"].unique():
            arch_sub = sub[sub["_arch_label"] == al]
            l2_0 = arch_sub[arch_sub["l2_penalty"] == 0]["finetune_acc"]
            l2_on = arch_sub[arch_sub["l2_penalty"] > 0]["finetune_acc"]
            if not l2_0.empty and not l2_on.empty:
                deltas.append(l2_on.max() - l2_0.max())
        deltas_by_frac[frac] = np.array(deltas) if deltas else np.array([])

    non_empty = [f for f in fracs if len(deltas_by_frac[f]) > 0]
    if not non_empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [deltas_by_frac[f] for f in non_empty]
    positions = np.arange(len(non_empty))
    _boxplot_with_strip(ax, groups, positions, ["tab:purple"] * len(non_empty))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(non_empty)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Accuracy delta (l2=0.005 minus l2=0)")
    ax.set_title("L2 Anchor Penalty Effect on Scattering Init")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_l2_effect.png")


# ---------------------------------------------------------------------------
# Plot 10: Paired difference (scattering - random)
# ---------------------------------------------------------------------------

def plot_paired_diff(df, save_dir):
    """Horizontal bar: scat_best - random for every arch x fraction pair."""
    pairs = _find_matched_pairs(df)
    if not pairs:
        return None

    entries = []
    for p in pairs:
        scat_acc = p["scat"].get("finetune_acc", 0) or 0
        rand_acc = p["rand"].get("joint_acc", 0) or 0
        entries.append({
            "label": f"{p['label']} ({p['data_frac']})",
            "diff": scat_acc - rand_acc,
        })

    entries.sort(key=lambda e: e["diff"])
    labels = [e["label"] for e in entries]
    diffs = [e["diff"] for e in entries]
    colors = ["tab:blue" if d >= 0 else "tab:red" for d in diffs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(entries) * 0.3)))
    y = np.arange(len(entries))
    ax.barh(y, diffs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Accuracy difference (scattering - random)")
    ax.set_title("Scattering vs Random Init: Paired Differences")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_paired_diff.png")


# ---------------------------------------------------------------------------
# Plot 11: Mixing horizon comparison (4-block only)
# ---------------------------------------------------------------------------

def plot_mixing_horizon(df, save_dir):
    """Grouped bar: mixing_horizon=27 vs 243 for 4-block models."""
    needed = {
        "n_blocks", "_is_random", "mixing_horizon", "data_frac",
        "modulus_type", "L", "finetune_acc", "joint_acc"
    }
    if not needed.issubset(df.columns):
        return None

    sub = df[df["n_blocks"] == 4].copy()
    if sub.empty or "mixing_horizon" not in sub.columns:
        return None

    horizons = sorted(sub["mixing_horizon"].dropna().unique())
    if len(horizons) < 2:
        return None

    fracs = [f for f in FRAC_ORDER if f in sub["data_frac"].values]
    if not fracs:
        return None

    scat = sub[~sub["_is_random"]]
    rand = sub[sub["_is_random"]]

    combos = []
    for mod in sorted(scat["modulus_type"].dropna().unique()):
        mod_short = "prelu" if mod == "phase_relu" else "cmod"
        for lv in sorted(scat["L"].dropna().unique()):
            for frac in fracs:
                label = f"4b_{mod_short}_{int(lv)}L ({frac})"
                vals = {}
                for h in horizons:
                    s_sub = scat[(scat["modulus_type"] == mod) &
                                 (scat["L"] == lv) &
                                 (scat["mixing_horizon"] == h) &
                                 (scat["data_frac"] == frac)]
                    r_sub = rand[(rand["modulus_type"] == mod) &
                                 (rand["L"] == lv) &
                                 (rand["mixing_horizon"] == h) &
                                 (rand["data_frac"] == frac)]
                    vals[h] = {
                        "scat": s_sub["finetune_acc"].max() if not s_sub.empty else np.nan,
                        "rand": r_sub["joint_acc"].max() if not r_sub.empty else np.nan,
                    }
                combos.append({"label": label, "vals": vals})

    if not combos:
        return None

    x = np.arange(len(combos))
    h0, h1 = horizons[0], horizons[1]
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(combos) * 0.8), 6))

    ax.bar(x - 1.5 * width,
           [c["vals"][h0]["scat"] for c in combos], width,
           label=f"Scat h={int(h0)}", color="tab:blue", alpha=0.7)
    ax.bar(x - 0.5 * width,
           [c["vals"][h1]["scat"] for c in combos], width,
           label=f"Scat h={int(h1)}", color="tab:cyan", alpha=0.7)
    ax.bar(x + 0.5 * width,
           [c["vals"][h0]["rand"] for c in combos], width,
           label=f"Rand h={int(h0)}", color="tab:red", alpha=0.7)
    ax.bar(x + 1.5 * width,
           [c["vals"][h1]["rand"] for c in combos], width,
           label=f"Rand h={int(h1)}", color="tab:orange", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in combos], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Mixing Horizon Comparison (4-block models)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_mixing_horizon.png")


# ---------------------------------------------------------------------------
# Plot 12: Best architecture per data fraction
# ---------------------------------------------------------------------------

def plot_best_summary(df, save_dir):
    """Bar chart: best scattering accuracy per data fraction, labelled with arch."""
    needed = {"_is_random", "data_frac", "finetune_acc", "_arch_label"}
    if not needed.issubset(df.columns):
        return None

    scat = df[~df["_is_random"]]

    fracs = [f for f in FRAC_ORDER if f in scat["data_frac"].values]
    if not fracs:
        return None

    best_accs, best_labels = [], []
    for frac in fracs:
        sub = scat[scat["data_frac"] == frac]
        if sub.empty:
            best_accs.append(0)
            best_labels.append("n/a")
            continue
        valid = sub["finetune_acc"].dropna()
        if valid.empty:
            best_accs.append(0)
            best_labels.append("n/a")
            continue
        idx = valid.idxmax()
        best_accs.append(sub.loc[idx, "finetune_acc"])
        best_labels.append(sub.loc[idx, "_arch_label"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(fracs))
    bars = ax.bar(x, best_accs, color="tab:green", alpha=0.7,
                  edgecolor="black", linewidth=0.5)

    for bar, lbl in zip(bars, best_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                lbl, ha="center", va="bottom", fontsize=8, rotation=20)

    ax.set_xticks(x)
    ax.set_xticklabels(fracs)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Best val accuracy (%)")
    ax.set_title("Best Scattering Architecture per Data Fraction")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_best_summary.png")


# ---------------------------------------------------------------------------
# Plot 13: Generalization error by data fraction
# ---------------------------------------------------------------------------

def plot_gen_error_by_fraction(df, save_dir):
    """Paired box plots: scattering vs random generalization error by fraction."""
    pairs = _find_matched_pairs(df)
    if not pairs:
        return None

    paired_rows = []
    for p in pairs:
        s_ge = p["scat"].get("gen_error")
        r_ge = p["rand"].get("gen_error")
        if pd.notna(s_ge) and pd.notna(r_ge):
            paired_rows.append((p["data_frac"], s_ge, r_ge))

    if not paired_rows:
        return None

    fracs = [f for f in FRAC_ORDER if any(fr == f for fr, _, _ in paired_rows)]
    if not fracs:
        return None

    groups_s = [np.array([s for fr, s, _ in paired_rows if fr == f]) for f in fracs]
    groups_r = [np.array([r for fr, _, r in paired_rows if fr == f]) for f in fracs]

    fig, ax = plt.subplots(figsize=(10, 5))
    pos_s = np.arange(len(fracs)) * 2.5
    pos_r = pos_s + 0.8

    _boxplot_with_strip(ax, groups_s, pos_s, ["tab:blue"] * len(fracs))
    _boxplot_with_strip(ax, groups_r, pos_r, ["tab:red"] * len(fracs))

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(pos_s + 0.4)
    ax.set_xticklabels(fracs)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Generalization error (test - train, %)")
    ax.set_title("Generalization Error by Data Fraction (paired random vs scatter)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc="tab:blue", alpha=0.5),
        plt.Rectangle((0, 0), 1, 1, fc="tab:red", alpha=0.5),
    ], labels=["Scattering init", "Random init"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_gen_error_by_fraction.png")


# ---------------------------------------------------------------------------
# Plot 14: Generalization error vs delta - handles multiple delta types
# ---------------------------------------------------------------------------

def plot_gen_error_vs_delta(df, save_dir, args):
    """Scatter of delta vs generalization error by init type. Supports p, f, and filter deltas."""
    delta_types, primary = _get_available_delta_types(df)
    
    # If no deltas available, skip
    if not any(delta_types.values()):
        return None
    
    # Generate plots for each available delta type
    for delta_type in ['filter', 'f', 'p']:
        if not delta_types.get(delta_type, False):
            continue
        
        delta_col = f"{delta_type}_delta_relative"
        sub = df.dropna(subset=[delta_col, "gen_error"])
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        if args.filter_delta_log_scale:
            ax.set_xscale('log')
        
        scat = sub[~sub["_is_random"]]
        rand = sub[sub["_is_random"]]

        if not scat.empty:
            ax.scatter(scat[delta_col], scat["gen_error"],
                       s=65, marker="o", color="tab:blue", alpha=0.85,
                       edgecolors="black", linewidth=0.5, label="Scattering init")
        if not rand.empty:
            ax.scatter(rand[delta_col], rand["gen_error"],
                       s=65, marker="x", color="tab:red", alpha=0.85,
                       linewidth=1.0, label="Random init")

        if len(sub) >= 3:
            r_all = sub[[delta_col, "gen_error"]].corr().iloc[0, 1]
            ax.annotate(f"overall r = {r_all:.3f}", xy=(0.03, 0.95),
                        xycoords="axes fraction", fontsize=10)
        if len(scat) >= 3:
            r_scat = scat[[delta_col, "gen_error"]].corr().iloc[0, 1]
            ax.annotate(f"scattering r = {r_scat:.3f}", xy=(0.03, 0.90),
                        xycoords="axes fraction", fontsize=10)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        delta_label_map = {'p': 'P filter', 'f': 'F filter', 'filter': 'Main filter'}
        delta_label = delta_label_map.get(delta_type, delta_type)
        ax.set_xlabel(f"Relative {delta_label} delta (L2)")
        ax.set_ylabel("Generalization error (test - train, %)")
        ax.set_title(f"Generalization Error vs {delta_label} Delta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        filename = f"plot_gen_error_vs_delta_{delta_type}.png"
        _save(fig, save_dir, filename)
    
    # Return without explicit return value, all plots saved separately
    return None


# ---------------------------------------------------------------------------
# Plot 15: Paired generalization difference (scat - random)
# ---------------------------------------------------------------------------

def plot_paired_gen_error_diff(df, save_dir):
    """Horizontal bar: paired (scattering gen error - random gen error)."""
    pairs = _find_matched_pairs(df)
    if not pairs:
        return None

    entries = []
    for p in pairs:
        scat_ge = p["scat"].get("gen_error")
        rand_ge = p["rand"].get("gen_error")
        if pd.notna(scat_ge) and pd.notna(rand_ge):
            entries.append({
                "label": f"{p['label']} ({p['data_frac']})",
                "diff": scat_ge - rand_ge,
            })

    if not entries:
        return None

    entries.sort(key=lambda e: e["diff"])
    labels = [e["label"] for e in entries]
    diffs = [e["diff"] for e in entries]
    colors = ["tab:blue" if d >= 0 else "tab:red" for d in diffs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(entries) * 0.3)))
    y = np.arange(len(entries))
    ax.barh(y, diffs, color=colors, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Gen-error difference (scattering - random, %)")
    ax.set_title("Paired Generalization Error Differences")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_paired_gen_error_diff.png")


# ---------------------------------------------------------------------------
# Plot 16: Generalization error data efficiency
# ---------------------------------------------------------------------------

def _build_gen_error_families(df):
    """Group paired generalization-error series by architecture label."""
    pairs = _find_matched_pairs(df)
    if not pairs:
        return {}

    families = {}
    for p in pairs:
        s_ge = p["scat"].get("gen_error")
        r_ge = p["rand"].get("gen_error")
        if pd.isna(s_ge) or pd.isna(r_ge):
            continue

        arch = p["label"]
        families.setdefault(arch, {
            "samples": [],
            "scat_gen": [],
            "rand_gen": [],
        })
        families[arch]["samples"].append(p["actual_train_samples"])
        families[arch]["scat_gen"].append(s_ge)
        families[arch]["rand_gen"].append(r_ge)

    result = {}
    for name, f in families.items():
        if len(f["samples"]) < 2:
            continue
        order = np.argsort(f["samples"])
        result[name] = {
            "samples": [f["samples"][i] for i in order],
            "scat_gen": [f["scat_gen"][i] for i in order],
            "rand_gen": [f["rand_gen"][i] for i in order],
        }
    return result


def plot_gen_error_efficiency(df, save_dir):
    """Line plot: generalization error vs samples for random and scattering."""
    families = _build_gen_error_families(df)
    if not families:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    for i, (name, f) in enumerate(sorted(families.items())):
        color = cmap(i % 10)
        ax.plot(f["samples"], f["scat_gen"], "o-", color=color,
                label=f"{name} scatter")
        ax.plot(f["samples"], f["rand_gen"], "x--", color=color, alpha=0.75,
                label=f"{name} random")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Generalization error (test - train, %)")
    ax.set_title("Generalization Error Data Efficiency")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_dir, "plot_gen_error_efficiency.png")


# ---------------------------------------------------------------------------
# Plot 17: Per-experiment training curves (optional)
# ---------------------------------------------------------------------------

def plot_training_curves(df, save_dir):
    curves_dir = os.path.join(save_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    count = 0

    for _, row in df.iterrows():
        curves = row.get("_step_curves") or row.get("_curves", {})
        if not curves:
            continue

        phases = [p for p in ["phase_a", "phase_b", "phase_c", "joint"] if curves.get(p)]
        if not phases:
            continue

        fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4))
        if len(phases) == 1:
            axes = [axes]

        for phase, ax in zip(phases, axes):
            entries = curves[phase]
            if not entries:
                continue
            
            epochs = [e.get("epoch") for e in entries]
            
            # Detect which column name is used for training accuracy (backwards compatible)
            train_col = None
            for col in ["train_acc1", "train_acc"]:
                if any(col in e for e in entries):
                    train_col = col
                    break
            
            if train_col:
                train_accs = [e.get(train_col, np.nan) for e in entries]
            else:
                train_accs = [np.nan] * len(entries)
            
            # Detect which column name is used for validation accuracy (backwards compatible)
            val_col = None
            for col in ["val_acc1", "val_acc"]:
                if any(col in e for e in entries):
                    val_col = col
                    break
            
            if val_col:
                val_accs = [e.get(val_col, np.nan) for e in entries]
            else:
                val_accs = [np.nan] * len(entries)
            
            ax.plot(epochs, train_accs, label="train_acc", color="tab:orange", marker="o", markersize=3)
            ax.plot(epochs, val_accs, label="val_acc", color="tab:blue", marker="s", markersize=3)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(phase.replace("_", " ").title())
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

        label = row.get("label", "unknown")
        samples = row.get("actual_train_samples", "?")
        fig.suptitle(f"{label} ({samples} samples)", fontsize=11)
        fig.tight_layout()

        path = os.path.join(curves_dir, f"curves_{label}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        count += 1

    print(f"  Saved {count} training curve(s) -> {curves_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Analyze experiment results")
    p.add_argument("--results-dir", default="cifar_exps",
                   help="directory containing results_*.json files")
    p.add_argument("--recursive", action="store_true",
                   help="search subdirectories for results_*.json")
    p.add_argument("--interactive", action="store_true",
                   help="show plots interactively (plt.show)")
    p.add_argument("--curves", action="store_true",
                   help="also plot per-experiment training curves")
    p.add_argument("--gen-train-source", default="best-val",
                   choices=["best-val", "last-epoch"],
                   help=("train accuracy source for generalization error: "
                         "best-val or last-epoch"))
    p.add_argument(
        "--gen-train-epoch",
        nargs="?",
        const=-1,
        type=int,
        default=None,
        help=(
            "extract train accuracy for generalization error from a specific "
            "epoch number; if provided without a value, uses the last epoch"
        ),
    )
    p.add_argument("--filter-delta-log-scale", action="store_true",
                   help="plot filter delta on log scale (for plot_delta_vs_gain)")
    p.add_argument(
        "--save-all-epoch-metrics",
        action="store_true",
        help=(
            "save one CSV row per experiment-phase-epoch using logged curves "
            "(summary_all_epochs.csv)"
        ),
    )
    args = p.parse_args()

    gen_train_epoch = None
    if args.gen_train_epoch is not None:
        gen_train_epoch = None if args.gen_train_epoch == -1 else args.gen_train_epoch

    df = load_experiments(args.results_dir, recursive=args.recursive,
                          gen_train_source=args.gen_train_source,
                          gen_train_epoch=gen_train_epoch)

    print_summary(df, save_dir=args.results_dir,
                  summary_epoch=gen_train_epoch)
    if args.save_all_epoch_metrics:
        save_epoch_metrics(df, save_dir=args.results_dir)

    plot_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("Generating plots...")
    plot_arch_comparison(df, plot_dir)
    plot_box_modulus(df, plot_dir)
    plot_box_depth(df, plot_dir)
    plot_box_gain(df, plot_dir)
    plot_delta_vs_gain(df, plot_dir, args)
    plot_data_efficiency(df, plot_dir)
    plot_init_advantage(df, plot_dir)
    plot_heatmap(df, plot_dir)
    plot_l2_effect(df, plot_dir)
    plot_paired_diff(df, plot_dir)
    plot_mixing_horizon(df, plot_dir)
    plot_best_summary(df, plot_dir)
    plot_gen_error_by_fraction(df, plot_dir)
    plot_gen_error_vs_delta(df, plot_dir, args)
    plot_paired_gen_error_diff(df, plot_dir)
    plot_gen_error_efficiency(df, plot_dir)

    if args.curves:
        plot_training_curves(df, plot_dir)

    if args.interactive:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
