"""Per-filter wavelet-likeness metrics over checkpoints -> Parquet shards -> metrics.parquet.

    python -m wavelet_likeness.run_metrics --out <results dir> --workers 8
    python -m wavelet_likeness.run_metrics --out <results dir> --combine

Run from ``deep_scatter/``.  One job per checkpoint: ``finetuned`` (wavelet-init run,
best.pth), ``trained_random`` / ``init_random`` (random-init run, best.pth /
param_snapshot.data), and one ``init_scattering`` (Theta_0) per dataset.  Each job
writes ``shards/<id>.parquet`` atomically; an existing shard is skipped, so the command
is resumable.  ``--combine`` concatenates the shards into ``metrics.parquet``.

One row per stored band-pass filter (the trainable 5x5 bank of each block):

    keys      dataset, lambda, train_fraction, seed, part, checkpoint_type, run,
              layer_kind ('stored'), layer (block), channel, filter_index (1..L =
              orientation l = filter_index - 1, native WaveletConv order)
    metrics   morlet_residual   relative residual of the best Morlet fit (M1)
              d_theta_deg, d_xi_rel, d_sigma_rel
                                fitted parameters relative to the matched Theta_0 filter
              dc_leakage        |g_hat(0)| / max |g_hat|
              flagged_degenerate  filter norm < 1e-3 of its init (no fit)

The Morlet fit runs on every channel of the first ``--m1-full-blocks`` blocks and on a
fixed seeded subset of ``--m1-sample`` channels of each deeper block.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd

from . import DS_ROOT
from . import extract as ex
from . import localization as loc
from . import matching as mt
from . import morlet_fit as mf
from .fourier import PAD_STORED

DEGENERATE_REL = 1e-3
# The deep_scatter sweeps used in the paper.
_TL = os.path.join(DS_ROOT, "training_logs")
DEFAULT_RUNS = [os.path.join(_TL, *p) for p in (
    ("clean_v2", "dtd", "*"),
    ("clean_updateNC", "eurosat", "*"),
    ("clean_v2", "minc", "*"),
    ("clean_v2", "minc_lowdata", "*"),
    ("surotex_h2h", "bn_lr1em2_*"),
    ("surotex_h2h", "bn_rinit_lr1em3_*"),
)]


# ---------------------------------------------------------------------------
# job enumeration
# ---------------------------------------------------------------------------

def _run_config(run_dir):
    res = sorted(glob.glob(os.path.join(run_dir, "results_*.json")))
    if not res:
        return None
    cfg = json.load(open(res[-1]))["config"]
    m = re.search(r"_part(\d+)$", os.path.basename(run_dir.rstrip("/")))
    return dict(dataset=cfg.get("dataset"),
                **{"lambda": float(cfg.get("l2_penalty") or 0.0)},
                train_fraction=float(cfg.get("train_size")),
                seed=int(cfg.get("seed")),
                part=int(m.group(1)) if m else np.nan,
                random_init=bool(cfg.get("random_init")),
                run=os.path.relpath(os.path.abspath(run_dir), DS_ROOT))


def enumerate_jobs(patterns):
    jobs, init_seen = [], {}
    dirs = sorted({d for p in patterns for d in glob.glob(p) if os.path.isdir(d)})
    for d in dirs:
        meta = _run_config(d)
        ck = os.path.join(d, "checkpoints")
        if meta is None or not os.path.exists(os.path.join(ck, "best.pth")):
            continue
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", meta["run"])
        if meta["random_init"]:
            jobs.append(dict(meta, checkpoint_type="trained_random",
                             path=os.path.join(ck, "best.pth"),
                             init_path=os.path.join(ck, "param_snapshot.data"),
                             id=f"trained_random__{tag}"))
            jobs.append(dict(meta, checkpoint_type="init_random",
                             path=os.path.join(ck, "param_snapshot.data"),
                             init_path=None, id=f"init_random__{tag}"))
        else:
            jobs.append(dict(meta, checkpoint_type="finetuned",
                             path=os.path.join(ck, "best.pth"), init_path=None,
                             id=f"finetuned__{tag}"))
            snap = os.path.join(ck, "param_snapshot.data")
            if meta["dataset"] not in init_seen and os.path.exists(snap):
                init_seen[meta["dataset"]] = dict(
                    dataset=meta["dataset"], **{"lambda": np.nan},
                    train_fraction=np.nan, seed=np.nan, part=np.nan,
                    run=meta["run"], checkpoint_type="init_scattering", path=snap,
                    init_path=None, id=f"init_scattering__{meta['dataset']}")
    return list(init_seen.values()) + jobs


# ---------------------------------------------------------------------------
# reference bank (cached per process)
# ---------------------------------------------------------------------------

_REF = {}


def _nominal_start(theta, xi, sigma):
    return dict(theta=theta, xi=xi, sigma=sigma, s=1.0, x0=0.0, y0=0.0)


def stored_reference(L, k):
    """Init band-pass bank + its own Morlet fits (which reproduce the WaveletConv params)."""
    key = (L, k)
    if key not in _REF:
        _, bp = ex.reference_bank(L, k)
        th = ex.nominal_theta(L)
        fits = [mf.fit(bp[l], [list(_nominal_start(th[l], 3 * np.pi / 4, 0.8).values())])
                for l in range(L)]
        _REF[key] = dict(bp=bp, bp_fit=fits,
                         bp_norm=np.linalg.norm(bp.reshape(L, -1), axis=1))
    return _REF[key]


# ---------------------------------------------------------------------------
# per-checkpoint work
# ---------------------------------------------------------------------------

def m1_channels(block, C, full_blocks, n_sample, seed=0):
    """Fixed channel subset for the Morlet fit: all channels in shallow blocks, otherwise
    the same seeded draw in every checkpoint (so fits are paired across checkpoints)."""
    if block < full_blocks or C <= n_sample:
        return np.arange(C)
    return np.sort(np.random.default_rng(seed + 1000 * block).choice(C, n_sample, replace=False))


def _fit_rows(G, ref_fits, matched, m2, chans, max_nfev):
    """Morlet fits of the band-pass filters G[chans] (G: (C, L, n, n))."""
    rows = {}
    for c in chans:
        for l, g in enumerate(G[c]):
            if not np.isfinite(m2["norm_ok"][c, l]) or not m2["norm_ok"][c, l]:
                continue
            rf = ref_fits[matched[c, l]]
            m2i = {k: v[c, l] for k, v in m2.items()}
            f = mf.fit(g, mf.start_set(rf, m2i), max_nfev=max_nfev)
            f["_ref"] = rf
            rows[(c, l)] = f
    return rows


def _fit_columns(f):
    rf = f["_ref"]
    return dict(morlet_residual=f["rho"],
                d_theta_deg=float(mf.wrap_deg_pi(np.degrees(f["theta"] - rf["theta"]))),
                d_xi_rel=(f["xi"] - rf["xi"]) / rf["xi"] if rf["xi"] else np.nan,
                d_sigma_rel=(f["sigma"] - rf["sigma"]) / rf["sigma"])


def _process_bank(rows, key, G, ref, init_norm, chans, max_nfev):
    """Rows for the band-pass bank G (C, L, n, n) of one block."""
    C, L = G.shape[:2]
    # orientation assignment and conjugate gauge against Theta_0
    matched, conj, _ = mt.match(G, ref["bp"])
    Ga = np.where(conj[..., None, None], G.conj(), G)
    m2 = loc.m2_metrics(Ga.reshape(C * L, *G.shape[2:]), PAD_STORED)
    m2 = {k: v.reshape(C, L) for k, v in m2.items()}
    denom = init_norm if init_norm is not None else ref["bp_norm"][matched]
    rel = m2["filter_norm"] / np.maximum(denom, 1e-30)
    m2["norm_ok"] = (rel >= DEGENERATE_REL).astype(float)
    fits = _fit_rows(Ga, ref["bp_fit"], matched, m2, chans, max_nfev)
    for c in range(C):
        for l in range(L):
            row = dict(key, channel=c, filter_index=l + 1, dc_leakage=m2["dc_leakage"][c, l],
                       flagged_degenerate=bool(rel[c, l] < DEGENERATE_REL))
            if (c, l) in fits:
                row.update(_fit_columns(fits[(c, l)]))
            rows.append(row)


def process(job, args):
    import torch
    torch.set_num_threads(1)
    state = ex.load_block_state(job["path"])
    init = ex.load_block_state(job["init_path"]) if job.get("init_path") else None
    return process_state(state, init, job, args)


def process_state(state, init, job, args):
    """All rows for one checkpoint's block state.  ``init`` is the run's own starting
    weights for random-init runs (None when Theta_0 is the reference)."""
    blocks = ex.extract_filters(state)
    init_blocks = ex.extract_filters(init) if init is not None else None
    L = next(b["L"] for b in blocks if b["L"])
    ref = stored_reference(L, blocks[0]["k"])
    base = {c: job[c] for c in ("dataset", "lambda", "train_fraction", "seed", "part",
                                "checkpoint_type", "run")}
    rows = []
    for b, blk in enumerate(blocks):
        if blk["bandpass"] is None:
            continue
        key = dict(base, layer_kind="stored", layer=b)
        chans = m1_channels(b, blk["C"], args.m1_full_blocks, args.m1_sample)
        ib = init_blocks[b] if init_blocks is not None else None
        if job["checkpoint_type"] == "init_random":
            ib = blk
        init_n = None if ib is None else np.linalg.norm(
            ib["bandpass"].reshape(blk["C"], L, -1), axis=2)
        _process_bank(rows, key, blk["bandpass"], ref, init_n, chans, args.max_nfev)
    return pd.DataFrame(rows)


def _write_atomic(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_job(job, args):
    shard = os.path.join(args.out, "shards", job["id"] + ".parquet")
    if os.path.exists(shard):
        return job["id"], "skip", 0.0
    t = time.time()
    try:
        df = process(job, args)
    except Exception:
        return job["id"], "ERROR\n" + traceback.format_exc(), time.time() - t
    _write_atomic(df, shard)
    return job["id"], f"ok {len(df)} rows", time.time() - t


def combine(args):
    """Concatenate the shards of the current --runs set into metrics.parquet."""
    wanted = {j["id"] for j in enumerate_jobs(args.runs)}
    found = sorted(glob.glob(os.path.join(args.out, "shards", "*.parquet")))
    files = [f for f in found if os.path.basename(f)[:-len(".parquet")] in wanted]
    if len(files) < len(found):
        print(f"ignoring {len(found) - len(files)} shards outside --runs")
    if len(files) < len(wanted):
        print(f"WARNING {len(wanted) - len(files)} of {len(wanted)} jobs have no shard")
    if not files:
        print(f"no shards in {args.out}/shards")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dest = os.path.join(args.out, "metrics.parquet")
    _write_atomic(df, dest)
    print(f"{dest}: {len(df)} rows from {len(files)} shards")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="globs of run directories (each with results_*.json + checkpoints/)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--m1-full-blocks", type=int, default=3,
                   help="blocks below this get the Morlet fit on every channel")
    p.add_argument("--m1-sample", type=int, default=64,
                   help="channels per deeper block that get the Morlet fit")
    p.add_argument("--max-nfev", type=int, default=200)
    p.add_argument("--only", nargs="*", default=None,
                   help="checkpoint types to run (default all)")
    p.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke test)")
    p.add_argument("--list", action="store_true", help="print jobs and exit")
    p.add_argument("--combine", action="store_true")
    args = p.parse_args()
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)

    if args.combine:
        combine(args)
        return
    jobs = enumerate_jobs(args.runs)
    if args.only:
        jobs = [j for j in jobs if j["checkpoint_type"] in args.only]
    if args.limit:
        jobs = jobs[:args.limit]
    if args.list:
        for j in jobs:
            print(j["id"])
        print(f"{len(jobs)} jobs")
        return
    print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    n_err = 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(args.workers) as pool:
            futs = [pool.submit(run_job, j, args) for j in jobs]
            for i, fu in enumerate(as_completed(futs)):
                try:
                    jid, status, dt = fu.result()
                except Exception as exc:   # e.g. a worker killed: rerunning resumes
                    jid, status, dt = "?", f"ERROR {exc!r}", 0.0
                n_err += status.startswith("ERROR")
                print(f"[{i + 1}/{len(jobs)}] {jid}: {status} ({dt:.0f}s)", flush=True)
    else:
        for i, j in enumerate(jobs):
            jid, status, dt = run_job(j, args)
            n_err += status.startswith("ERROR")
            print(f"[{i + 1}/{len(jobs)}] {jid}: {status} ({dt:.0f}s)", flush=True)
    print(f"done, {n_err} errors", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
