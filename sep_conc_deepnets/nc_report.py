"""Offline Neural-Collapse computation for sep_conc_deepnets (SCD) runs.

The counterpart of ``deep_scatter/nc_report.py``; the two write the same columns so their
CSVs concatenate. Uses the best-val checkpoint (``checkpoints/<dir>/<arch>_batchsize_<bs>
_lrfreq_<f>_best.pth.tar``) and ``training_utils.compute_neural_collapse_metrics``.

Runs whose ``results_*.json`` already carries main.py's best-checkpoint NC (keys in
``best_ckpt_eval``) are read from the log instead of recomputed; ``--no-inline``
recomputes anyway and ``--check-inline`` reports coverage and exits.

Usage
-----
    python nc_report.py --results-dir training_logs/<sweep> --recursive --device cuda
    python nc_report.py --results-dir training_logs/<sweep> --check-inline
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as tvsets

import best_ckpt_eval as bce
import run_loading as rl  # imported before deep_scatter is added to sys.path
from training_utils import compute_neural_collapse_metrics


NC_IDENT_COLS = ["dataset", "framework", "method", "phase", "split", "arch",
                 "l2_penalty", "seed", "train_size", "actual_train_samples",
                 "n_blocks", "results_json", "checkpoint", "ckpt_mtime", "nc_source"]
NC_VALUE_COLS = ["nc1", "nc1_rel", "nc1_old", "nc1_trace", "between_class_trace",
                 "nc2", "nc3", "n_samples", "n_features", "nc_seconds", "nc_error"]
NC_SKEW_COLS = ["nc1_skew", "nc1_skew_proj", "nc1_skew_mahal", "skew_rank"]
NC_PAPYAN_COLS = ["nc1_papyan"]
NC_VALUE_COLS = NC_VALUE_COLS + NC_SKEW_COLS + NC_PAPYAN_COLS

_ARCH_LABEL = {"scatnetblockanalysis": "ScatNet B.A.", "scatnetblock": "ScatNet B.",
               "scatnet": "ScatNet", "analysis": "Analysis"}

CACHE_NAME = "nc.json"
# Bump when the feature path, the metric, or the split reconstruction changes, so stale
# sidecar caches are ignored. Same value as deep_scatter/nc_report.py.
NC_CODE_VERSION = "2026-09-17-papyan"

_LOADER_CACHE = {}


def _ensure_deep_scatter_on_path():
    """Append deep_scatter to sys.path (as main.py does) for data_utils' dataset builders."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    for p in (root, os.path.join(root, "deep_scatter")):
        if p not in sys.path:
            sys.path.append(p)


def _data_dir(config):
    return config.get("data") or config.get("data_dir") or "./data"


def _split_seed(config):
    """The seed the dataset builders saw: main.py passes ``args.seed or 0``."""
    return config.get("seed") or 0


def _loader_identity(config):
    """Config fields that determine which images/labels a split contains."""
    return (config.get("dataset"), _data_dir(config), config.get("train_size"),
            config.get("val_frac", 0.2) or 0.2, _split_seed(config),
            config.get("dtd_partition", 1) or 1,
            config.get("input_size"))


def build_split_loader(config, split="test", batch_size=64, workers=4):
    """Rebuild one real-data split loader from a run's persisted config.

    Uses main.py's dataset builders via ``data_utils``. Returns ``None`` on failure.
    """
    idx = {"train": 0, "val": 1, "test": 2}[split]
    dataset = config.get("dataset")
    key = _loader_identity(config) + (split, batch_size, workers)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]

    loader = None
    try:
        _ensure_deep_scatter_on_path()
        import data_utils as du
        ts = config.get("train_size", 1.0)
        try:
            ts = float(ts)
        except Exception:
            pass
        vf = config.get("val_frac", 0.2) or 0.2
        seed = _split_seed(config)
        dd = _data_dir(config)
        if dataset == "cifar10":
            out = du._build_cifar_legacy_datasets(tvsets.CIFAR10, "cifar10", ts, vf, seed)
        elif dataset == "cifar100":
            out = du._build_cifar_legacy_datasets(tvsets.CIFAR100, "cifar100", ts, vf, seed)
        elif dataset == "dtd":
            out = du._build_dtd_legacy_datasets(ts, vf, seed,
                                                partition=config.get("dtd_partition", 1) or 1)
        elif dataset in ("minc2500", "minc"):
            out = du._build_minc_legacy_datasets(ts, vf, seed, data_dir=dd)
        elif dataset == "eurosat":
            out = du._build_eurosat_legacy_datasets(ts, vf, seed, data_dir=dd)
        elif dataset == "alot":
            out = du._build_alot_legacy_datasets(ts, vf, seed, data_dir=dd)
        elif dataset == "surotex":
            out = du._build_surotex_legacy_datasets(ts, vf, seed, data_dir=dd,
                                                    input_size=config.get("input_size"))
        else:
            out = None
        if out is not None:
            loader = torch.utils.data.DataLoader(out[idx], batch_size=batch_size,
                                                 shuffle=False, num_workers=workers)
    except Exception as e:
        print(f"    [loader] could not build {split} loader for {dataset}: "
              f"{type(e).__name__}: {e}")
        loader = None
    _LOADER_CACHE[key] = loader
    return loader


class _TruncatedLoader:
    """First ``n`` batches only (debugging)."""

    def __init__(self, loader, n):
        self._loader, self._n = loader, n
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                return
            yield batch


def _nc_row(config, ckpt_path, device, split, batch_size, workers, n_blocks,
            num_classes, max_batches=None):
    """Rebuild the model, load the checkpoint, compute NC. Never raises."""
    out = {"split": split, "checkpoint": ckpt_path}
    t0 = time.time()
    try:
        if ckpt_path is None:
            out["nc_error"] = "no checkpoint found under checkpoints/<dir>/"
            return out
        loader = build_split_loader(config, split=split, batch_size=batch_size,
                                    workers=workers)
        if loader is None:
            out["nc_error"] = f"could not build {split} loader"
            return out
        n_ds = len(getattr(loader, "dataset", []) or [])
        if max_batches:
            loader = _TruncatedLoader(loader, max_batches)

        model = rl.build_model_from_config(config, device=device)
        rl.load_state_into(model, ckpt_path, device=device)
        model.eval()
        out["ckpt_mtime"] = os.path.getmtime(ckpt_path)

        m = compute_neural_collapse_metrics(model=model, loader=loader,
                                            n_blocks=n_blocks, num_classes=num_classes,
                                            device=device)

        # `nc1` is nc1_rel, matching analyze.py's *_nc1_rel -> *_nc1 mapping.
        out["nc1"] = m["nc1_rel"]
        out["nc1_rel"] = m["nc1_rel"]
        out["nc1_old"] = m["nc1_old"]
        out["nc1_trace"] = m["nc1_trace"]
        out["between_class_trace"] = m["between_class_trace"]
        out["nc2"] = m["nc2"]
        out["nc3"] = m["nc3"]
        for k in NC_SKEW_COLS + NC_PAPYAN_COLS:
            out[k] = m.get(k)
        out["n_samples"] = n_ds
        if m["nc1_old"]:
            out["n_features"] = int(round(m["nc1_trace"] / m["nc1_old"]))
    except Exception as e:
        out["nc_error"] = f"{type(e).__name__}: {e}"
    out["nc_seconds"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Which run does a shared *_best.pth.tar actually belong to?
# ---------------------------------------------------------------------------

# Cached per checkpoint path (one extra torch.load per shared checkpoint).
_ATTRIBUTION_CACHE = {}

# Tolerance on accuracy percentages: float32 in the checkpoint vs 4 dp in the JSON.
_ACC_TOL = 1e-3


def _run_identity(results_json):
    """(seed, best_acc1, best_acc5, best_epoch_acc1) as recorded in a results JSON."""
    try:
        with open(results_json) as f:
            payload = json.load(f)
    except Exception:
        return None
    res = payload.get("results", {})
    cfg = payload.get("config", {})
    return (cfg.get("seed"), res.get("best_acc1"), res.get("best_acc5"),
            res.get("best_epoch_acc1"))


def _checkpoint_identity(ckpt_path):
    """``((best_acc1, best_acc5, best_epoch_acc1), error)`` from a checkpoint header.

    main.py stores these alongside the weights. A load error is returned so an unreadable
    checkpoint is reported as such.
    """
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    return (blob.get("best_acc1"), blob.get("best_acc5"),
            blob.get("best_epoch_acc1")), None


def _identities_match(ck, run):
    """``ck`` is a checkpoint identity, ``run`` a results-JSON identity (seed first).

    Matches on ``best_acc1`` and ``best_epoch_acc1``. ``best_acc5`` is excluded: the
    checkpoint's value is the best as of the acc1-best epoch, the JSON's the best overall.
    """
    if ck is None or run is None:
        return False
    _seed, acc1, _acc5, epoch = run
    ck_acc1, _ck_acc5, ck_epoch = ck
    if any(v is None for v in (ck_acc1, ck_epoch, acc1, epoch)):
        return False
    return abs(ck_acc1 - acc1) <= _ACC_TOL and int(ck_epoch) == int(epoch)


def attribute_shared_checkpoint(results_json, ckpt_path, verbose=True):
    """Does ``ckpt_path`` hold *this* run's weights?  Returns ``(owns_it, reason)``.

    Checkpoint names do not include the seed, so runs of several seeds in one run dir
    share one ``*_best.pth.tar``. The checkpoint is attributed by matching its recorded
    ``best_acc1``/``best_epoch_acc1`` against each co-located run; if that does not resolve
    to exactly one run, no value is produced for any of them.
    """
    if not ckpt_path:
        return False, ""
    run_dir = os.path.dirname(results_json)
    co_tenants, _ = drop_superseded(sorted(glob.glob(os.path.join(run_dir, "results_*.json"))),
                                    verbose=False)
    if len(co_tenants) <= 1:
        return True, ""

    if ckpt_path not in _ATTRIBUTION_CACHE:
        ck, ck_error = _checkpoint_identity(ckpt_path)
        matches = [p for p in co_tenants if _identities_match(ck, _run_identity(p))]
        winner = matches[0] if len(matches) == 1 else None
        _ATTRIBUTION_CACHE[ckpt_path] = (winner, len(co_tenants), ck_error)
        if verbose:
            if winner:
                seed = (_run_identity(winner) or [None])[0]
                print(f"    [attrib] {os.path.basename(ckpt_path)} shared by "
                      f"{len(co_tenants)} runs -> seed {seed}")
            else:
                why = f"unreadable ({ck_error})" if ck_error \
                    else f"{len(matches)} candidates matched"
                print(f"    [attrib] {os.path.basename(ckpt_path)} shared by "
                      f"{len(co_tenants)} runs -> UNRESOLVED: {why}; "
                      f"all will be marked as errors")

    winner, n_tenants, ck_error = _ATTRIBUTION_CACHE[ckpt_path]
    if ck_error:
        return False, (f"checkpoint is unreadable ({ck_error}); shared by {n_tenants} "
                       f"co-located runs, so it cannot be attributed either")
    if winner is None:
        return False, (f"checkpoint shared by {n_tenants} co-located runs and could not be "
                       f"attributed (no unique best_acc1/best_epoch_acc1 match -- the "
                       f"owning run's results JSON is probably missing); "
                       f"sep_conc_deepnets writes one *_best.pth.tar per run dir")
    if os.path.abspath(winner) == os.path.abspath(results_json):
        # Recorded in nc_source.
        return True, f"attributed: seed {(_run_identity(winner) or [None])[0]}"
    winner_seed = (_run_identity(winner) or [None])[0]
    return False, (f"checkpoint overwritten by co-located seed {winner_seed} "
                   f"(sep_conc_deepnets writes one *_best.pth.tar per run dir, so this "
                   f"run's weights no longer exist on disk)")


def load_run(results_json):
    """``(config, results)`` for a run; shape fields are taken from the results block."""
    with open(results_json) as f:
        payload = json.load(f)
    config = dict(payload.get("config", payload))
    results = payload.get("results", {})
    # The results block holds the values resolved at run time (config may hold defaults).
    for k in ("nb_classes", "in_channels", "in_spatial"):
        if results.get(k) is not None:
            config[k] = results[k]
    for k in ("actual_train_samples", "n_params_total"):
        if k in results and k not in config:
            config[k] = results[k]
    return config, results


def base_row(results_json, config, framework="sep_conc_deepnets"):
    """The identity columns every NC row carries, however the values were obtained."""
    arch = config.get("arch")
    return {
        "dataset": config.get("dataset"), "framework": framework, "arch": arch,
        "method": _ARCH_LABEL.get(arch, arch),
        # SCD runs are single-phase; paper_plots keys the phase on the arch name.
        "phase": arch,
        "l2_penalty": config.get("l2_penalty"), "seed": config.get("seed"),
        "train_size": config.get("train_size"),
        "actual_train_samples": config.get("actual_train_samples"),
        "n_blocks": config.get("n_blocks"),
        "results_json": results_json,
    }


# ---------------------------------------------------------------------------
# Inline metrics -- written by main.py's best-checkpoint evaluation
# ---------------------------------------------------------------------------

def inline_rows(results_json, with_val=False, framework="sep_conc_deepnets"):
    """NC rows read from the run's JSON (``best_ckpt_eval`` keys), or ``None``.

    Returns ``None`` unless every requested split is complete, so the run is recomputed.
    """
    config, results = load_run(results_json)
    splits = ["test", "val"] if with_val else ["test"]
    got = {split: bce.inline_nc(results, split) for split in splits}
    if any(v is None for v in got.values()):
        return None
    # Runs without the skewed NC1 keys are recomputed.
    skew = {split: bce.inline_nc_skew(results, split) for split in splits}
    if any(v is None for v in skew.values()):
        return None

    base = base_row(results_json, config, framework=framework)
    rows = []
    for split in splits:
        m = got[split]
        row = dict(base)
        row["split"] = split
        row["nc_source"] = "inline: " + bce.source_label(results)
        row["checkpoint"] = None      # evaluated in memory at training time
        row["nc1"] = m["nc1_rel"]
        row["nc1_rel"] = m["nc1_rel"]
        row["nc1_old"] = m["nc1_old"]
        row["nc1_trace"] = m["nc1_trace"]
        row["between_class_trace"] = m["between_class_trace"]
        row["nc2"] = m["nc2"]
        row["nc3"] = m["nc3"]
        row.update(skew[split])
        n_key = bce.acc_keys(split)["n_samples"]
        row["n_samples"] = results.get(n_key)
        if m["nc1_old"]:
            row["n_features"] = int(round(m["nc1_trace"] / m["nc1_old"]))
        row["nc_seconds"] = 0.0
        rows.append(row)
    return rows


def report_run(results_json, device="cpu", batch_size=64, workers=4, with_val=False,
               max_batches=None, framework="sep_conc_deepnets"):
    """One NC row per split for the run's best-val checkpoint."""
    config, results = load_run(results_json)
    base = base_row(results_json, config, framework=framework)

    ckpt, _args_json = rl.find_checkpoint(results_json, config=config, prefer="best")
    n_blocks = config.get("n_blocks")
    num_classes = results.get("nb_classes") or config.get("nb_classes")
    # Only the run that wrote a shared checkpoint is scored from it.
    owns_ckpt, attrib_reason = attribute_shared_checkpoint(results_json, ckpt)

    rows = []
    for split in (["test", "val"] if with_val else ["test"]):
        row = dict(base)
        if not owns_ckpt and attrib_reason:
            row["nc_source"] = "unavailable (checkpoint not attributable to this run)"
            row["split"] = split
            row["checkpoint"] = ckpt
            row["nc_error"] = attrib_reason
            rows.append(row)
            continue
        row["nc_source"] = "best.pth.tar (best-val)" + (
            f" [{attrib_reason}]" if attrib_reason else "")
        row.update(_nc_row(config, ckpt, device, split, batch_size, workers,
                           n_blocks, num_classes, max_batches=max_batches))
        rows.append(row)

    if max_batches:
        for r in rows:
            r["nc_source"] = f"TRUNCATED({max_batches}b) " + str(r.get("nc_source", ""))
    return rows


# ---------------------------------------------------------------------------
# Sidecar cache
# ---------------------------------------------------------------------------

def _cache_path(results_json):
    """One sidecar per results JSON (a run dir can hold several seeds)."""
    stem = os.path.splitext(os.path.basename(results_json))[0]
    return os.path.join(os.path.dirname(results_json), f"nc_{stem}.json")


def _signature(results_json, config, device, batch_size, with_val):
    try:
        ckpt, _ = rl.find_checkpoint(results_json, config=config, prefer="best")
    except Exception:
        ckpt = None
    try:
        mt = os.path.getmtime(ckpt) if ckpt else None
    except OSError:
        mt = None
    return {"code_version": NC_CODE_VERSION, "device": device, "batch_size": batch_size,
            "with_val": bool(with_val), "ckpt": ckpt, "ckpt_mtime": mt,
            "results_json": os.path.realpath(results_json)}


def _cache_read(results_json, sig):
    try:
        with open(_cache_path(results_json)) as f:
            blob = json.load(f)
    except Exception:
        return None
    if blob.get("signature") != sig:
        return None
    rows = blob.get("rows")
    # Never reuse a cached failure.
    if rows and any(r.get("nc_error") for r in rows):
        return None
    return rows


def _cache_write(results_json, sig, rows):
    try:
        tmp = _cache_path(results_json) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"signature": sig, "rows": rows}, f)
        os.replace(tmp, _cache_path(results_json))
    except Exception as e:
        print(f"    [cache] could not write {CACHE_NAME}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------

def drop_superseded(jsons, verbose=True):
    """Keep only the newest ``results_*.json`` per (run dir, seed).

    A re-run leaves the older results JSON next to a checkpoint it did not produce.
    Grouping includes the seed because several seeds can share one run dir.
    """
    groups = {}
    for j in jsons:
        try:
            with open(j) as f:
                seed = json.load(f).get("config", {}).get("seed")
        except Exception:
            seed = ("unreadable", j)
        groups.setdefault((os.path.dirname(j), seed), []).append(j)

    keep, dropped = [], []
    for members in groups.values():
        members = sorted(members, key=lambda p: (os.path.getmtime(p), p))
        keep.append(members[-1])
        dropped.extend(members[:-1])
    if dropped and verbose:
        print(f"[dedup] {len(dropped)} superseded results JSON(s) skipped "
              f"(same run dir, same seed, older mtime); {len(keep)} kept")
        for p in dropped[:5]:
            print(f"        {p}")
        if len(dropped) > 5:
            print(f"        ... and {len(dropped) - 5} more")
    return sorted(keep), dropped


def report_dir(results_dir, recursive=True, device="cpu", out_name="nc_recomputed.csv",
               batch_size=64, workers=4, with_val=False, force=False, limit=None,
               threads=8, verbose=True, max_batches=None, use_inline=True):
    """Write ``nc_recomputed.csv`` for every run under ``results_dir``.

    Superseded result JSONs are dropped first (``drop_superseded``). Runs whose JSON
    carries main.py's best-checkpoint metrics are read from it (``use_inline``) instead of
    recomputed; ``--force`` or ``--no-inline`` recomputes regardless.
    """
    torch.set_num_threads(threads)
    pattern = os.path.join(results_dir, "**", "results_*.json") if recursive \
        else os.path.join(results_dir, "results_*.json")
    jsons = sorted(glob.glob(pattern, recursive=recursive))
    if not jsons:
        print(f"No results_*.json under {results_dir}")
        return None
    jsons, _ = drop_superseded(jsons, verbose=verbose)
    if limit:
        jsons = jsons[:limit]

    all_rows, n_cached, n_inline = [], 0, 0
    for i, j in enumerate(jsons):
        # The run's own log first, then the sidecar cache, then a recompute.
        if use_inline and not force and not max_batches:
            inline = inline_rows(j, with_val=with_val)
            if inline is not None:
                all_rows.extend(inline)
                n_inline += 1
                if verbose:
                    print(f"[{i+1}/{len(jsons)}] {os.path.relpath(j, results_dir)} "
                          f"(inline -- logged at training time)")
                continue
        try:
            with open(j) as f:
                cfg = json.load(f).get("config", {})
        except Exception:
            cfg = {}
        sig = _signature(j, cfg, device, batch_size, with_val)
        cached = None if (force or max_batches) else _cache_read(j, sig)
        if cached is not None:
            all_rows.extend(cached)
            n_cached += 1
            if verbose:
                print(f"[{i+1}/{len(jsons)}] {os.path.relpath(j, results_dir)} (cached)")
            continue
        if verbose:
            print(f"[{i+1}/{len(jsons)}] {os.path.relpath(j, results_dir)}", flush=True)
        try:
            rows = report_run(j, device=device, batch_size=batch_size, workers=workers,
                              with_val=with_val, max_batches=max_batches)
            if not max_batches:
                _cache_write(j, sig, rows)
            all_rows.extend(rows)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            all_rows.append({"results_json": j, "nc_error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(all_rows)
    cols = [c for c in NC_IDENT_COLS + NC_VALUE_COLS if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df = df[cols]
    out_path = os.path.join(results_dir, out_name)
    df.to_csv(out_path, index=False)

    n_ok = int(df["nc1"].notna().sum()) if "nc1" in df.columns else 0
    print(f"\nSaved {len(df)} rows to {out_path}  "
          f"({n_ok} with a value, {len(df) - n_ok} blank, {n_cached} runs from cache, "
          f"{n_inline} runs from inline training-time metrics)")
    if n_inline == len(jsons):
        print(f"[inline] all {n_inline} run(s) already recorded their own best-checkpoint NC "
              f"at training time -- nothing was recomputed and no GPU was needed.\n"
              f"         Pass --force (or --no-inline) to recompute anyway.")
    elif n_inline:
        print(f"[inline] {n_inline} of {len(jsons)} run(s) carried training-time NC; the other "
              f"{len(jsons) - n_inline} were recomputed from checkpoints.")
    if "nc_error" in df.columns and df["nc_error"].notna().any():
        print("nc_error breakdown:")
        for msg, n in df["nc_error"].value_counts().items():
            print(f"  {n:5d}  {str(msg)[:110]}")
    return out_path


def compare_to_logged(results_dir, out_name="nc_recomputed.csv"):
    """Diff recomputed NC against each run's logged ``test_nc1_rel``.

    The logged value is from the last-epoch model and the recomputed one from the best-val
    checkpoint, so small differences are expected.
    """
    df = pd.read_csv(os.path.join(results_dir, out_name))
    if "split" in df.columns:
        df = df[df["split"] == "test"]

    recs = []
    for _, r in df.iterrows():
        if pd.isna(r.get("nc1")):
            continue
        try:
            with open(r["results_json"]) as f:
                res = json.load(f).get("results", {})
        except Exception:
            continue
        logged = res.get("test_nc1_rel")
        if logged is None:
            continue
        recs.append({"results_json": r["results_json"], "train_size": r.get("train_size"),
                     "seed": r.get("seed"), "logged": float(logged),
                     "recomputed": float(r["nc1"]),
                     "ratio": float(r["nc1"]) / float(logged) if float(logged) else np.nan})
    cmp_df = pd.DataFrame(recs)
    if cmp_df.empty:
        print("Nothing comparable: no run here logged test_nc1_rel.")
        return cmp_df

    cmp_df["abs_diff"] = (cmp_df["recomputed"] - cmp_df["logged"]).abs()
    cmp_df["rel_diff"] = cmp_df["abs_diff"] / cmp_df["logged"].abs()
    out = os.path.join(results_dir, "nc_recompute_vs_logged.csv")
    cmp_df.to_csv(out, index=False)
    print(f"\n=== recomputed vs logged ({len(cmp_df)} rows) ===")
    print(f"  ratio median={cmp_df['ratio'].median():.4f} "
          f"min={cmp_df['ratio'].min():.4f} max={cmp_df['ratio'].max():.4f}  "
          f"rel_diff median={cmp_df['rel_diff'].median():.2e}")
    # Per seed: a split-reconstruction error shows up as only one seed agreeing.
    if "seed" in cmp_df.columns and cmp_df["seed"].nunique() > 1:
        print("  by seed (a split-reconstruction bug shows up as ONE seed agreeing):")
        for seed, grp in cmp_df.groupby("seed", dropna=False):
            print(f"    seed {str(seed):>5}: {len(grp):3d} rows, "
                  f"rel_diff median={grp['rel_diff'].median():.2e}")
    print(f"  wrote {out}")
    return cmp_df


def check_inline(results_dir, recursive=True, with_val=False):
    """Print inline coverage and return (n_inline, n_total). No GPU, no CSV written."""
    pattern = os.path.join(results_dir, "**", "results_*.json") if recursive \
        else os.path.join(results_dir, "results_*.json")
    jsons = sorted(glob.glob(pattern, recursive=recursive))
    if not jsons:
        print(f"No results_*.json under {results_dir}")
        return 0, 0
    jsons, _ = drop_superseded(jsons, verbose=False)
    covered = [j for j in jsons if inline_rows(j, with_val=with_val) is not None]
    splits = "test+val" if with_val else "test"
    print(f"{len(covered)}/{len(jsons)} run(s) under {results_dir} already carry "
          f"training-time best-checkpoint NC on {splits}.")
    if len(covered) == len(jsons):
        print("Nothing to recompute -- this directory needs no GPU pass.")
    else:
        missing = [j for j in jsons if j not in set(covered)]
        print(f"{len(missing)} run(s) still need it, e.g.:")
        for j in missing[:5]:
            print(f"    {os.path.relpath(j, results_dir)}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    return len(covered), len(jsons)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-name", default="nc_recomputed.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--with-val", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="recompute everything: ignore both the nc.json sidecar cache and "
                        "any training-time metrics in the run's JSON")
    p.add_argument("--no-inline", dest="use_inline", action="store_false",
                   help="ignore main.py's own best-checkpoint NC and recompute from the "
                        "checkpoints")
    p.add_argument("--check-inline", action="store_true",
                   help="report how many runs already carry training-time NC, then exit "
                        "without loading torch or writing anything")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None,
                   help="debugging only: stop after N batches; rows are tagged TRUNCATED and "
                        "never cached")
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()

    if args.check_inline:
        check_inline(args.results_dir, recursive=args.recursive, with_val=args.with_val)
        return

    report_dir(args.results_dir, recursive=args.recursive, device=args.device,
               out_name=args.out_name, batch_size=args.batch_size, workers=args.workers,
               with_val=args.with_val, force=args.force, limit=args.limit,
               threads=args.threads, max_batches=args.max_batches,
               use_inline=args.use_inline)
    if args.compare:
        compare_to_logged(args.results_dir, out_name=args.out_name)


if __name__ == "__main__":
    main()
