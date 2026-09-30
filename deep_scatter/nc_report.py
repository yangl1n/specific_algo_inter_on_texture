"""Offline Neural-Collapse computation for deep_scatter runs.

Rebuilds each run's model and data split from its ``results_*.json`` config, loads
``checkpoints/best.pth`` (and ``phase_a.pth`` for the Phase-A model, which can also be
rebuilt from config -- see ``phase_a_reconstructible``), and computes NC with
``train.compute_neural_collapse_metrics``. Writes ``nc_recomputed.csv`` beside
``summary.csv``; ``paper_plots.py`` joins it on ``(_path, phase)``.

Usage
-----
    python nc_report.py --results-dir <dir> --recursive --device cuda
    python nc_report.py --results-dir <dir> --recursive --limit 5      # smoke test
    python nc_report.py --results-dir <dir> --recursive --with-val     # also nc*_for_gen

A full pass loads every checkpoint and forward-passes a whole split, so it is far too
slow for a login node -- submit it as a batch job or use ``--limit``.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import run_loading as rl
from run_loading import build_split_loader
from train import compute_neural_collapse_metrics


# Identity of the run + which model instance the row describes.
NC_IDENT_COLS = ["dataset", "framework", "method", "phase", "split", "random_init",
                 "l2_penalty", "seed", "train_size", "actual_train_samples",
                 "n_blocks", "L", "use_batchnorm", "global_avg_pool", "use_analysis",
                 "results_json", "checkpoint", "ckpt_mtime", "nc_source"]
# The recomputed values. `nc1` is nc1_rel -- see _nc_row().
NC_VALUE_COLS = ["nc1", "nc1_rel", "nc1_old", "nc1_trace", "between_class_trace",
                 "nc2", "nc3", "n_samples", "n_features", "nc_seconds", "nc_error"]
# Head-derived skewed NC1 (blank on config-rebuilt Phase-A rows, whose head is random).
NC_SKEW_COLS = ["nc1_skew", "nc1_skew_proj", "nc1_skew_mahal", "skew_rank"]
# Papyan's tr(Sw Sb^+)/K; features only, so valid on config-rebuilt Phase-A rows.
NC_PAPYAN_COLS = ["nc1_papyan"]
NC_VALUE_COLS = NC_VALUE_COLS + NC_SKEW_COLS + NC_PAPYAN_COLS

# The result key each (phase, split) pair corresponds to in the original results JSON.
# Used by --compare to diff recomputed vs logged.
LOGGED_KEY = {
    ("a", "test"): "phase_a_test_nc1",
    ("b", "test"): "test_nc1",
    ("joint", "test"): "test_nc1",
}

CACHE_NAME = "nc.json"
# Bump when the feature path, the metric, or the split reconstruction changes, so
# stale sidecar caches are ignored.
NC_CODE_VERSION = "2026-09-17-papyan"


# ---------------------------------------------------------------------------
# Which runs can have their Phase-A model reconstructed without a checkpoint
# ---------------------------------------------------------------------------

def phase_a_reconstructible(config):
    """Can the end-of-Phase-A NC features be rebuilt from ``config`` alone?

    The NC feature path is ``blocks + standardization -> [analysis] -> [avg_pool] ->
    flatten``. During Phase A the extractor is frozen at its deterministic Morlet init, so
    the features match a freshly built model unless the path holds BatchNorm running
    statistics (``use_batchnorm``, or a projected concat block).

    Returns (bool, reason).
    """
    if config.get("random_init"):
        return False, "random-init run has no Phase A"
    if config.get("use_batchnorm"):
        return False, ("phase_a not reconstructible: use_batchnorm=True, so the extractor's "
                       "standardization BatchNorms accumulated running stats during Phase A "
                       "that no surviving checkpoint records")
    if config.get("use_concat_projection") and config.get("use_projection"):
        return False, ("phase_a not reconstructible: ConcatProjectedScatteringBlock carries "
                       "its own BatchNorm")
    return True, ""


# ---------------------------------------------------------------------------
# Per-model NC computation
# ---------------------------------------------------------------------------

class _TruncatedLoader:
    """Yield only the first ``n`` batches of ``loader`` (debugging only).

    Rows produced this way are tagged ``nc_source='TRUNCATED(...)'``.
    """

    def __init__(self, loader, n):
        self._loader, self._n = loader, n
        self.dataset = getattr(loader, "dataset", None)

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                return
            yield batch


def _nc_row(config, results_json, ckpt_path, device, split, batch_size, workers,
            load_weights=True, max_batches=None):
    """Rebuild one model instance and compute NC on ``split``. Never raises."""
    out = {"split": split, "checkpoint": ckpt_path}
    t0 = time.time()
    try:
        loader = build_split_loader(config, split=split, batch_size=batch_size,
                                    workers=workers)
        if loader is None:
            out["nc_error"] = f"could not build {split} loader"
            return out
        if max_batches:
            loader = _TruncatedLoader(loader, max_batches)

        model = rl.build_model_from_config(config, device=device)
        if load_weights:
            if ckpt_path is None:
                out["nc_error"] = "no checkpoint"
                return out
            rl.load_state_into(model, ckpt_path, device=device)
            out["ckpt_mtime"] = os.path.getmtime(ckpt_path)
        model.eval()

        m = compute_neural_collapse_metrics(model=model, loader=loader, device=device)

        # train.py logs nc1_rel under every *_nc1 result key.
        out["nc1"] = m["nc1_rel"]
        out["nc1_rel"] = m["nc1_rel"]
        out["nc1_old"] = m["nc1_old"]
        out["nc1_trace"] = m["nc1_trace"]
        out["between_class_trace"] = m["between_class_trace"]
        out["nc2"] = m["nc2"]
        out["nc3"] = m["nc3"]
        for k in NC_SKEW_COLS + NC_PAPYAN_COLS:
            out[k] = m.get(k)
        if not load_weights:
            # Config-rebuilt Phase-A model: random head, so head-derived metrics are blanked.
            for k in NC_SKEW_COLS:
                out[k] = None
        out["n_samples"] = len(getattr(loader, "dataset", []) or [])
        # Feature dimension d = nc1_trace / nc1_old (sum vs mean over dimensions).
        if m["nc1_old"]:
            out["n_features"] = int(round(m["nc1_trace"] / m["nc1_old"]))
    except Exception as e:
        out["nc_error"] = f"{type(e).__name__}: {e}"
    out["nc_seconds"] = round(time.time() - t0, 2)
    return out


def report_run(results_json, device="cpu", batch_size=128, workers=4, with_val=False,
               framework="deep_scatter", max_batches=None):
    """Return the NC rows for one run JSON: one per (model instance, split)."""
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

    random_init = bool(config.get("random_init", False))
    base = {
        "dataset": config.get("dataset"), "framework": framework,
        "random_init": random_init,
        "l2_penalty": config.get("l2_penalty"), "seed": config.get("seed"),
        "train_size": config.get("train_size"),
        "actual_train_samples": config.get("actual_train_samples"),
        "n_blocks": config.get("n_blocks"), "L": config.get("L"),
        "use_batchnorm": bool(config.get("use_batchnorm", False)),
        "global_avg_pool": bool(config.get("global_avg_pool", False)),
        "use_analysis": bool(config.get("use_analysis", False)),
        "results_json": results_json,
    }

    splits = ["test", "val"] if with_val else ["test"]
    rows = []

    # End-of-training model (best.pth): Fine-tuned Scat. (phase b) or Random Init. (joint).
    best = rl.find_checkpoint(results_json, prefer="best")
    for split in splits:
        row = dict(base)
        row["method"] = "Random Init." if random_init else "Fine-tuned Scat."
        row["phase"] = "joint" if random_init else "b"
        row["nc_source"] = "best.pth"
        row.update(_nc_row(config, results_json, best, device, split, batch_size, workers,
                           max_batches=max_batches))
        if best is None:
            row["nc_error"] = "no best.pth"
        rows.append(row)

    # Phase-A model: frozen-init extractor + head ("Scattering").
    if not random_init:
        phase_a = rl.find_checkpoint(results_json, prefer="phase_a")
        ok, reason = phase_a_reconstructible(config)
        for split in splits:
            row = dict(base)
            row["method"] = "Scattering"
            row["phase"] = "a"
            if phase_a is not None:
                row["nc_source"] = "phase_a.pth"
                row.update(_nc_row(config, results_json, phase_a, device, split,
                                   batch_size, workers, max_batches=max_batches))
            elif ok:
                row["nc_source"] = "config-rebuilt init (no phase_a.pth needed)"
                row.update(_nc_row(config, results_json, None, device, split,
                                   batch_size, workers, load_weights=False,
                                   max_batches=max_batches))
            else:
                row["nc_source"] = "unavailable"
                row["split"] = split
                row["nc_error"] = reason
            rows.append(row)

    if max_batches:
        for r in rows:
            r["nc_source"] = f"TRUNCATED({max_batches}b) " + str(r.get("nc_source", ""))
    return rows


# ---------------------------------------------------------------------------
# Sidecar cache
# ---------------------------------------------------------------------------

def _cache_path(results_json):
    """One sidecar per results JSON (a run dir can hold several)."""
    stem = os.path.splitext(os.path.basename(results_json))[0]
    return os.path.join(os.path.dirname(results_json), f"nc_{stem}.json")


def _signature(results_json, device, batch_size, with_val):
    best = rl.find_checkpoint(results_json, prefer="best")
    phase_a = rl.find_checkpoint(results_json, prefer="phase_a")
    def _mt(p):
        try:
            return os.path.getmtime(p) if p else None
        except OSError:
            return None
    return {
        "code_version": NC_CODE_VERSION, "device": device,
        "batch_size": batch_size, "with_val": bool(with_val),
        "best": best, "best_mtime": _mt(best),
        "phase_a": phase_a, "phase_a_mtime": _mt(phase_a),
        "results_json": os.path.realpath(results_json),
    }


def _cache_read(results_json, sig):
    try:
        with open(_cache_path(results_json)) as f:
            blob = json.load(f)
    except Exception:
        return None
    if blob.get("signature") != sig:
        return None
    rows = blob.get("rows")
    # Never replay a transient failure; deterministic "unavailable" rows are kept.
    if rows and any(_is_transient_error(r) for r in rows):
        return None
    return rows


def _is_transient_error(row):
    err = row.get("nc_error")
    if not err:
        return False
    return not (row.get("nc_source") == "unavailable" or err in ("no best.pth", "no checkpoint"))


def _cache_write(results_json, sig, rows):
    try:
        tmp = _cache_path(results_json) + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"signature": sig, "rows": rows}, f)
        os.replace(tmp, _cache_path(results_json))
    except Exception as e:
        print(f"    [cache] could not write {CACHE_NAME}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Directory driver
# ---------------------------------------------------------------------------

def drop_superseded(jsons, verbose=True):
    """Keep only the newest ``results_*.json`` per (run dir, seed).

    A re-run writes a new results JSON next to the same ``checkpoints/best.pth``, so
    older JSONs in the directory no longer match the checkpoint.
    """
    groups = {}
    for j in jsons:
        try:
            with open(j) as f:
                seed = json.load(f).get("config", {}).get("seed")
        except Exception:
            # Unique key: never deduplicated, reported as an error downstream.
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
               batch_size=128, workers=4, with_val=False, force=False, limit=None,
               threads=8, verbose=True, max_batches=None):
    """Write ``nc_recomputed.csv`` for every run under ``results_dir``.

    Superseded result JSONs are dropped first -- see ``drop_superseded``.
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

    all_rows, n_cached = [], 0
    for i, j in enumerate(jsons):
        sig = _signature(j, device, batch_size, with_val)
        # Truncated (debug) runs never use the cache.
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
          f"({n_ok} with a value, {len(df) - n_ok} blank, {n_cached} runs from cache)")
    if "nc_error" in df.columns and df["nc_error"].notna().any():
        print("nc_error breakdown:")
        for msg, n in df["nc_error"].value_counts().items():
            print(f"  {n:5d}  {str(msg)[:110]}")
    return out_path


# ---------------------------------------------------------------------------
# Validation: recomputed vs logged
# ---------------------------------------------------------------------------

# Relative tolerance for "the recompute reproduced the logged value" (well above
# float32 nondeterminism across GPUs).
AGREE_RTOL = 1e-3


def compare_to_logged(results_dir, out_name="nc_recomputed.csv"):
    """Diff recomputed NC against the value each run logged during training."""
    csv_path = os.path.join(results_dir, out_name)
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "test"] if "split" in df.columns else df

    recs = []
    for _, r in df.iterrows():
        key = LOGGED_KEY.get((r.get("phase"), "test"))
        if key is None or pd.isna(r.get("nc1")):
            continue
        try:
            with open(r["results_json"]) as f:
                logged = json.load(f).get("results", {}).get(key)
        except Exception:
            continue
        if logged is None:
            continue
        recs.append({"results_json": r["results_json"], "phase": r["phase"],
                     "nc_source": r.get("nc_source"), "train_size": r.get("train_size"),
                     "seed": r.get("seed"), "l2_penalty": r.get("l2_penalty"),
                     "logged": float(logged), "recomputed": float(r["nc1"]),
                     "ratio": float(r["nc1"]) / float(logged) if float(logged) else np.nan})
    cmp_df = pd.DataFrame(recs)
    if cmp_df.empty:
        print("Nothing comparable (no logged NC values for these phases).")
        return cmp_df

    cmp_df["abs_diff"] = (cmp_df["recomputed"] - cmp_df["logged"]).abs()
    cmp_df["rel_diff"] = cmp_df["abs_diff"] / cmp_df["logged"].abs()
    cmp_df["agrees"] = cmp_df["rel_diff"] <= AGREE_RTOL
    out = os.path.join(results_dir, "nc_recompute_vs_logged.csv")
    cmp_df.to_csv(out, index=False)

    print(f"\n=== recomputed vs logged ({len(cmp_df)} comparable rows, "
          f"rtol={AGREE_RTOL:g}) ===")
    for phase, grp in cmp_df.groupby("phase"):
        agree = int(grp["agrees"].sum())
        print(f"  phase {phase}: {len(grp)} rows, {agree} agree "
              f"({100.0 * agree / len(grp):.0f}%)  "
              f"rel_diff median={grp['rel_diff'].median():.2e} "
              f"p95={grp['rel_diff'].quantile(.95):.2e} "
              f"max={grp['rel_diff'].max():.2e}  "
              f"ratio median={grp['ratio'].median():.4f}")
    # Per seed: a split-reconstruction error shows up as only one seed agreeing.
    if "seed" in cmp_df.columns and cmp_df["seed"].nunique() > 1:
        print("  by seed (a split-reconstruction bug shows up as ONE seed agreeing):")
        for seed, grp in cmp_df.groupby("seed", dropna=False):
            agree = int(grp["agrees"].sum())
            print(f"    seed {str(seed):>5}: {len(grp):3d} rows, {agree:3d} agree  "
                  f"rel_diff median={grp['rel_diff'].median():.2e}")
    print(f"  wrote {out}")
    return cmp_df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-name", default="nc_recomputed.csv")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--with-val", action="store_true",
                   help="also compute on the val split (reproduces analyze.py's "
                        "nc1_for_gen/nc2_for_gen/nc3_for_gen, which use the best-val epoch)")
    p.add_argument("--force", action="store_true", help="ignore the nc.json sidecar cache")
    p.add_argument("--limit", type=int, default=None,
                   help="only process the first N run JSONs (smoke test)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None,
                   help="debugging only: stop after N batches per split; rows are tagged "
                        "nc_source='TRUNCATED(...)' and never cached.")
    p.add_argument("--compare", action="store_true",
                   help="after writing the CSV, diff recomputed vs logged NC")
    args = p.parse_args()

    report_dir(args.results_dir, recursive=args.recursive, device=args.device,
               out_name=args.out_name, batch_size=args.batch_size, workers=args.workers,
               with_val=args.with_val, force=args.force, limit=args.limit,
               threads=args.threads, max_batches=args.max_batches)
    if args.compare:
        compare_to_logged(args.results_dir, out_name=args.out_name)


if __name__ == "__main__":
    main()
