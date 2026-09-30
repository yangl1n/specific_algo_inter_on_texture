"""Rebuild a trained run (model and checkpoint) from its results JSON."""

import argparse
import contextlib
import copy
import os
import re

import torch


def find_checkpoint(results_json_path, config=None, prefer="best"):
    """Locate the checkpoint that goes with a results JSON.

    ``main.py`` writes logs to ``./training_logs/<args.dir>/`` and checkpoints to
    ``./checkpoints/<args.dir>/<arch>_batchsize_<bs>_lrfreq_<f>[_best].pth.tar``.
    Returns ``(ckpt_path, args_json_path)``, either of which may be ``None``.
    """
    def _logs_to_ckpts(d):
        """./training_logs/<rel>/ -> ./checkpoints/<rel>/, or None if not under logs."""
        parts = d.split(os.sep)
        if "training_logs" not in parts:
            return None
        idx = len(parts) - 1 - parts[::-1].index("training_logs")
        return os.sep.join(parts[:idx] + ["checkpoints"] + parts[idx + 1:])

    run_dir = os.path.abspath(os.path.dirname(results_json_path))
    config = config or {}

    candidates = []
    # realpath as well as abspath, in case the run dir is a symlink.
    for d in (run_dir, os.path.realpath(run_dir)):
        c = _logs_to_ckpts(d)
        if c and c not in candidates:
            candidates.append(c)
    # Fall back to the run's recorded `dir`.
    cfg_dir = config.get("dir")
    root = _logs_to_ckpts(run_dir) or _logs_to_ckpts(os.path.realpath(run_dir))
    if cfg_dir and root:
        head = root.split(os.sep + "checkpoints" + os.sep)[0]
        c = os.path.join(head, "checkpoints", cfg_dir)
        if c not in candidates:
            candidates.append(c)
    candidates.append(os.path.join(run_dir, "checkpoints"))

    names = []
    arch = config.get("arch")
    bs = config.get("batch_size")
    lrfreq = config.get("learning_rate_adjust_frequency")
    if arch is not None and bs is not None and lrfreq is not None:
        stem = f"{arch}_batchsize_{bs}_lrfreq_{lrfreq}"
        names = [f"{stem}_best.pth.tar", f"{stem}.pth.tar"]
        if prefer != "best":
            names.reverse()

    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        args_json = os.path.join(cand, "args.json")
        args_json = args_json if os.path.exists(args_json) else None
        for name in names:
            p = os.path.join(cand, name)
            if os.path.exists(p):
                return p, args_json
        # Otherwise any checkpoint in the directory, preferring *_best.
        tars = sorted(f for f in os.listdir(cand) if f.endswith(".pth.tar"))
        best = [f for f in tars if f.endswith("_best.pth.tar")]
        pick = (best or tars)
        if pick:
            return os.path.join(cand, pick[0]), args_json
    return None, None


def build_model_from_config(config, device="cpu"):
    """Reconstruct the ``nn.Sequential(ScatNetAnalysis...)`` model from a run's args."""
    import main as scd_main  # heavy import (Kymatio)

    cfg = copy.deepcopy(dict(config))
    args = argparse.Namespace(**cfg)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        model = scd_main.load_model(args, None, None)
    return model.to(device).eval()


_DP_PREFIX = re.compile(r"^(\d+\.)?module\.")


def _strip_dataparallel(key):
    return _DP_PREFIX.sub(lambda m: m.group(1) or "", key)


def load_state_into(model, ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # DataParallel inserts a ".module." level ("0.module.linear_proj..." -> "0.linear_proj...").
    state = {_strip_dataparallel(k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return model, {"missing": list(missing), "unexpected": list(unexpected)}
