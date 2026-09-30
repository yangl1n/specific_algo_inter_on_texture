"""Rebuild a trained run (model, checkpoint, data split) from its results JSON."""

import os
from types import SimpleNamespace

import torch

from scattering_net import ScatteringNet


_SCATNET_ARGS = [
    "n_blocks", "in_channels", "in_spatial", "nb_classes", "L", "kernel_size",
    "modulus_type", "mixing_horizon", "global_avg_pool", "lowpass_last",
    "use_batchnorm", "use_concat_projection", "use_skip_connections",
    "use_projection", "projection_out_channels", "use_analysis",
    "analysis_frame_width", "analysis_frame_kernel_size", "analysis_frame_stride",
    "analysis_non_linearity", "analysis_zero_bias", "analysis_frame_init_mode",
]


def build_model_from_config(config, device="cpu"):
    """Reconstruct a ScatteringNet from a results-JSON ``config`` dict."""
    kwargs = {k: config[k] for k in _SCATNET_ARGS if k in config}
    # ``learnable`` only toggles requires_grad; irrelevant for eval-time weights.
    kwargs.setdefault("learnable", True)
    return ScatteringNet(**kwargs).to(device).eval()


def find_checkpoint(results_json_path, prefer="best"):
    """Checkpoint in ``<run dir>/checkpoints/`` for a ``results_*.json`` file.

    ``prefer="phase_a"`` returns the end-of-Phase-A snapshot (``phase_a.pth``) or None.
    """
    ckpt_dir = os.path.join(os.path.dirname(results_json_path), "checkpoints")
    if prefer == "phase_a":
        p = os.path.join(ckpt_dir, "phase_a.pth")
        return p if os.path.exists(p) else None
    for name in ([f"{prefer}.pth", "best.pth", "curr.pth"]):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_state_into(model, ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
    # Strip DataParallel "module." prefixes.
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items()}
    model.load_state_dict(state)
    return model


_LOADER_CACHE = {}


def _loader_identity(config):
    """Config fields that determine which images/labels a split contains."""
    return (config.get("dataset"), config.get("data_dir"), config.get("train_size"),
            config.get("val_frac"), config.get("seed"), config.get("dtd_partition"),
            config.get("input_size"))


def build_split_loader(config, split="train", batch_size=8, workers=2):
    """Rebuild one split loader ("train", "val" or "test") from a run's config.

    Uses ``data.build_dataloaders``, so the split matches training exactly. Returns
    ``None`` if the loader cannot be built.
    """
    idx = {"train": 0, "val": 1, "test": 2}[split]
    key = _loader_identity(config) + (split, batch_size, workers)
    if key in _LOADER_CACHE:
        return _LOADER_CACHE[key]

    try:
        from data import build_dataloaders
        args = SimpleNamespace(**dict(config))
        args.distributed = False
        args.workers = workers
        args.batch_size = batch_size
        args.batch_size_per_device = batch_size
        if getattr(args, "val_frac", None) is None:
            args.val_frac = 0.2
        if getattr(args, "dtd_partition", None) is None:
            args.dtd_partition = 1
        _LOADER_CACHE[key] = build_dataloaders(args)[idx]
    except Exception as e:
        print(f"    [loader] could not build {split} loader for "
              f"{config.get('dataset')}: {type(e).__name__}: {e}")
        _LOADER_CACHE[key] = None
    return _LOADER_CACHE[key]
