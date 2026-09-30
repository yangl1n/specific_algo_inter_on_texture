
import argparse
import math
import os
import shutil
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from data import build_dataloaders
from scattering_net import build_scattering_net
from engine import (run_epoch, EarlyStopping,
                    snapshot_extractor_params, compute_filter_delta,
                    snapshot_classifier_params, compute_classifier_delta,
                    make_anchor_penalty, resolve_l2_anchor, l2_mode_label,
                    L2_ANCHOR_CHOICES, CLASSIFIER_PHASE_A_SNAPSHOT_FILE)
from utils import (get_raw_model, save_checkpoint, save_phase_a_checkpoint, load_model, auto_resume, JOINT, PHASE_A, PHASE_B,
                   ExperimentLogger, init_distributed, cleanup_distributed, is_distributed,
                   get_world_size, is_main_process, console_logging, get_local_rank)
import time
import random
import numpy as np
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Skewed (classifier-metric) neural collapse
# ---------------------------------------------------------------------------
# nc1_rel measures within-class spread with the Euclidean metric over all d feature
# dimensions, so spread in directions the classifier cannot see is charged in full. The
# skewed variants measure the same within/between ratio under a metric built from the
# classifier itself. With W_eff the Jacobian of the (affine, eval-mode) head w.r.t. the NC
# features and W~ = W_eff minus its mean row (softmax only sees logit differences, so a row
# shared by every class carries no decision information):
#
#   nc1_skew        M = W~^T W~       NC1 of the centered logits; scale-invariant in W.
#   nc1_skew_proj   M = P_row(W~)     orthogonal projector onto the classifier subspace;
#                                     ignores how strongly the head weights each direction.
#   nc1_skew_mahal  tr(Sw Sb^+)/rank  Papyan-style Mahalanobis NC1 inside that subspace;
#                                     invariant to any invertible map within it.
#   skew_rank       numerical rank of W~ (<= min(K-1, d)).
#
# nc1_papyan (below) needs no head: Papyan's tr(Sw Sb^+)/K over the full feature space,
# i.e. the Mahalanobis form of nc1_rel. nc1_skew_mahal is the same formula restricted to
# the head's row space (and divided by rank rather than K).
#
# Everything is computed in <= K-dimensional coordinates, so no d x d matrix is formed.
# Identical to sep_conc_deepnets/training_utils.py.

SKEW_NC_KEYS = ('nc1_skew', 'nc1_skew_proj', 'nc1_skew_mahal', 'skew_rank')
# Relative singular-value / eigenvalue cutoff for the classifier subspace and the pinv.
_SKEW_RTOL = 1e-6


def _nan_skew():
    return {k: float('nan') for k in SKEW_NC_KEYS}


def _project_rows(features, basis):
    """``features @ basis.T`` in float64, chunked so no (N, d) float64 copy is made."""
    basis = basis.to(torch.float64)
    rows = max(1, int(2e8 // max(1, features.shape[1])))
    return torch.cat([chunk.to(torch.float64) @ basis.t() for chunk in features.split(rows)],
                     dim=0)


def _class_means_and_devs(z, labels, num_classes):
    """Per present class: (mean (1, r), deviations (n_c, r))."""
    out = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        zc = z[mask]
        mu = zc.mean(dim=0, keepdim=True)
        out.append((mu, zc - mu))
    return out


def _ratio_nc1(features, labels, num_classes, basis=None):
    """Within/between trace ratio with nc1_rel's conventions, under an optional basis.

    Within = per-class mean squared deviation, averaged over classes; between = mean squared
    norm of the class means about the mean of the class means. ``basis=None`` is the
    Euclidean metric and reproduces ``nc1_rel``; a (r, d) ``basis`` measures ``||basis x||``.
    """
    z = features.to(torch.float64) if basis is None else _project_rows(features, basis)
    stats = _class_means_and_devs(z, labels, num_classes)
    if len(stats) < 2:
        return float('nan')
    within = sum(dev.pow(2).sum(dim=1).mean().item() for _, dev in stats) / len(stats)
    means = torch.cat([mu for mu, _ in stats], dim=0)
    between = (means - means.mean(dim=0, keepdim=True)).pow(2).sum(dim=1).mean().item()
    return within / (between + 1e-8)


def _effective_head_weight(head_fn, feat_shape, device, dtype=torch.float32):
    """(K, d) Jacobian of an affine head w.r.t. the flattened NC features, one VJP per class.

    Exact for any eval-mode affine head, so it folds in what the raw ``Linear.weight`` misses:
    the head BatchNorm's gamma/sqrt(running_var + eps) scale and any pooling between the
    features and the Linear. Shifts (BN running_mean/beta, the bias) drop out of a Jacobian.
    """
    d = math.prod(feat_shape)
    with torch.enable_grad():
        x = torch.zeros(1, d, device=device, dtype=dtype, requires_grad=True)
        y = head_fn(x.view(1, *feat_shape))
        rows = [torch.autograd.grad(y[0, k], x, retain_graph=True)[0][0].detach().cpu()
                for k in range(y.shape[1])]
    return torch.stack(rows)


def _skewed_nc(features, labels, W_eff, num_classes):
    """The SKEW_NC_KEYS for (N, d) ``features`` under the classifier metric of ``W_eff``."""
    if W_eff is None or W_eff.dim() != 2 or W_eff.shape[1] != features.shape[1]:
        return _nan_skew()
    W = W_eff.detach().cpu().to(torch.float64)
    W = W - W.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    if S.numel() == 0 or not S[0] > 0:
        return _nan_skew()
    rank = int((S > _SKEW_RTOL * S[0]).sum())
    V = Vh[:rank]
    # ||W~ x|| = ||diag(S) Vh x||, so one projection serves both metrics.
    z = _project_rows(features, V)
    stats = _class_means_and_devs(z, labels, num_classes)
    if len(stats) < 2:
        return _nan_skew()

    out = {'skew_rank': rank,
           'nc1_skew_proj': _ratio_nc1(z, labels, num_classes),
           'nc1_skew': _ratio_nc1(z * S[:rank], labels, num_classes)}

    means = torch.cat([mu for mu, _ in stats], dim=0)
    centered = means - means.mean(dim=0, keepdim=True)
    sigma_b = centered.t() @ centered / len(stats)
    sigma_w = sum(dev.t() @ dev / dev.shape[0] for _, dev in stats) / len(stats)
    rank_b = int(torch.linalg.matrix_rank(sigma_b, hermitian=True, rtol=_SKEW_RTOL))
    if rank_b == 0:
        out['nc1_skew_mahal'] = float('nan')
    else:
        pinv_b = torch.linalg.pinv(sigma_b, hermitian=True, rtol=_SKEW_RTOL)
        out['nc1_skew_mahal'] = torch.trace(sigma_w @ pinv_b).item() / rank_b
    return out


def _papyan_nc1(features, labels, num_classes):
    """Papyan et al. (2020) NC1, tr(Sw Sb^+) / K, over the full feature space (no head).

    Sb has rank <= K-1 and its pseudo-inverse is supported on the span of the centered class
    means, so with Q an orthonormal basis of that span, Sb^+ = Q^T (Q Sb Q^T)^+ Q and the
    trace is computed exactly in <= K-1 coordinates. Conventions follow nc1_rel: Sw is
    averaged over classes and the means are centered on the mean of the class means; K
    counts the classes present in the loader.
    """
    means = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            means.append(features[mask].to(torch.float64).mean(dim=0, keepdim=True))
    if len(means) < 2:
        return float('nan')
    means = torch.cat(means, dim=0)
    _, S, Vh = torch.linalg.svd(means - means.mean(dim=0, keepdim=True), full_matrices=False)
    if not S[0] > 0:
        return float('nan')
    z = _project_rows(features, Vh[:int((S > _SKEW_RTOL * S[0]).sum())])
    stats = _class_means_and_devs(z, labels, num_classes)
    zm = torch.cat([mu for mu, _ in stats], dim=0)
    centered = zm - zm.mean(dim=0, keepdim=True)
    sigma_b = centered.t() @ centered / len(stats)
    sigma_w = sum(dev.t() @ dev / dev.shape[0] for _, dev in stats) / len(stats)
    pinv_b = torch.linalg.pinv(sigma_b, hermitian=True, rtol=_SKEW_RTOL)
    return torch.trace(sigma_w @ pinv_b).item() / len(stats)


@torch.no_grad()
def compute_neural_collapse_metrics(model, loader, device='cuda'):
    """Neural collapse statistics of the features feeding the classifier.

    Features are the extractor output *before* the classifier's BatchNorm, matching
    sep_conc_deepnets (``ScatNetAnalysis.extract_features``). The inter-block
    ``standardization`` layers and the parameter-free ``avg_pool`` are included.
    """
    model.eval()
    raw_model = get_raw_model(model)
    all_features = []
    all_labels = []
    
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        x = inputs
        for block, standardize in zip(raw_model.blocks, raw_model.standardization):
            x = standardize(block(x))
        if raw_model.use_analysis and raw_model.analysis is not None:
            x = raw_model.analysis(x)
        if raw_model.avg_pool is not None:
            x = raw_model.avg_pool(x)
        feat_shape = tuple(x.shape[1:])  # pre-flatten, for the head Jacobian below
        x = x.flatten(1)
        all_features.append(x.detach().cpu())
        all_labels.append(targets.detach().cpu())
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    num_classes = raw_model.classifier.out_features
    
    class_means = []
    within_class_vars = []
    total_within_class_trace = 0.0
    valid_classes = 0
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        valid_classes += 1
        class_features = features[mask]
        class_mean = class_features.mean(dim=0, keepdim=True)
        class_means.append(class_mean)
        
        centered = class_features - class_mean
        within_class_vars.append(float((centered ** 2).mean()))
        total_within_class_trace += torch.norm(centered, p=2, dim=1).pow(2).mean().item()
    
    if valid_classes < num_classes:
        warnings.warn(f"Warning: Only {valid_classes}/{num_classes} classes have samples in the loader. ")
    if not class_means:
        return {
            'nc1_old': float('nan'),
            'nc1_trace': float('nan'), 
            'between_class_trace': float('nan'),
            'nc1_rel': float('nan'), 
            'nc2': float('nan'), 
            'nc3': float('nan'),
            'nc1_papyan': float('nan'),
            **_nan_skew(),
        }
    
    # Element-wise within-class variance.
    nc1_old = float(torch.tensor(within_class_vars).mean())

    # NC1: average within-class trace, and its ratio to the between-class trace.
    nc1_trace = total_within_class_trace / valid_classes
    class_means = torch.cat(class_means, dim=0)
    global_mean = class_means.mean(dim=0, keepdim=True)
    centered_class_means = class_means - global_mean
    between_class_trace = torch.norm(centered_class_means, p=2, dim=1).pow(2).mean().item()
    nc1_rel = nc1_trace / (between_class_trace + 1e-8)
    nc1_papyan = _papyan_nc1(features, labels, num_classes)

    # NC2: ETF coherence
    class_means_norm = centered_class_means / (torch.norm(centered_class_means, dim=1, keepdim=True) + 1e-8)
    cosine_sims = torch.mm(class_means_norm, class_means_norm.t())
    mask_tri = torch.triu(torch.ones_like(cosine_sims), diagonal=1).bool()
    nc2_etf_coherence = float(cosine_sims[mask_tri].mean()) if mask_tri.sum() > 0 else 0.0

    # NC3: classifier weight alignment
    try:
        W = raw_model.classifier.weight.detach().cpu()
        W_centered = W - W.mean(dim=0, keepdim=True)
        W_norm = W_centered / (torch.norm(W_centered, dim=1, keepdim=True) + 1e-8)
        alignment = torch.sum(W_norm * class_means_norm, dim=1).mean().item()
    except (AttributeError, RuntimeError):  # RuntimeError: a class absent from the loader
        alignment = float('nan')

    # Skewed NC1 under the classifier metric; head = bn -> flatten -> classifier.
    try:
        W_eff = _effective_head_weight(
            lambda z: raw_model.classifier(raw_model.bn(z).flatten(1)), feat_shape, device,
            dtype=features.dtype)
        skew = _skewed_nc(features, labels, W_eff, num_classes)
    except Exception as e:
        warnings.warn(f"skewed NC unavailable: {type(e).__name__}: {e}")
        skew = _nan_skew()

    return {
        'nc1_old': nc1_old,
        'nc1_trace': nc1_trace,
        'between_class_trace': between_class_trace, 
        'nc1_rel': nc1_rel,
        'nc2': nc2_etf_coherence, 
        'nc3': alignment,
        'nc1_papyan': nc1_papyan,
        **skew,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="ScatteringNet training pipelines: Two-phase and joint training")

    # Dataset
    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset", default="cifar10",
                   choices=["cifar10", "cifar100", "imagenet", "dtd", "minc2500", "eurosat", "fmd", "alot", "surotex", "pathmnist", "octmnist", "so2sat_rgb", "so2sat_s2"])
    g.add_argument("--data-dir", default="./data",
                   help="root directory for dataset (ImageNet: parent of "
                        "train/ and val/; DTD: parent of dtd/; MINC2500: Kaggle cache root)")
    g.add_argument("--input-size", default=None, type=int, choices=[28, 64, 128, 224],
                   help="override the input resolution. For MedMNIST (pathmnist/octmnist) this "
                        "selects the MedMNIST+ variant (28/64/128/224, auto-downloaded). When "
                        "unset, the per-dataset default in data.py's config is used.")
    g.add_argument("--dtd-partition", default=1, type=int,
                   help="DTD partition (fold) number for k-fold cross-validation (1-10). "
                        "Default: 1")
    g.add_argument("--imagenet-source", default="pytorch", choices=["pytorch", "huggingface"],
                   help="ImageNet data source. 'pytorch' uses local train/ and val/ folders. "
                        "'huggingface' supports both folder layout and parquet snapshot layout.")
    g.add_argument("--hf-download", action="store_true",
                   help="when --imagenet-source huggingface, call huggingface_hub.snapshot_download "
                        "before building ImageNet datasets")
    g.add_argument("--hf-repo-id", default="imagenet-1k", type=str,
                   help="Hugging Face dataset repo id used with --imagenet-source huggingface")
    g.add_argument("--hf-local-dir", default=None, type=str,
                   help="local dir for Hugging Face snapshot. Defaults to --data-dir")
    g.add_argument("--hf-token", default=None, type=str,
                   help="Hugging Face token override. If unset, uses HF_TOKEN env var")
    g.add_argument("--tar-cache-size", default=128, type=int,
                   help="LRU cache size for tar handles per worker (ImageNet shards)")
    g.add_argument("--imagenet-shards", action="store_true",
                   help="use tar shards (train/*.tar) for ImageNet. "
                        "Run build_imagenet_tar_index.py once first.")
    g.add_argument("--imagenet-sampler", default="global", choices=["sharded", "global"],
                   help="sampling mode for ImageNet tar shards: 'sharded' improves I/O locality; "
                        "'global' gives stronger class mixing per batch")
    g.add_argument("--train-size", default=None, type=float,
                   help="training subset size: float for fraction (e.g. 0.1),"
                        " int for absolute count (e.g. 5000). "
                        "Default: use full training set")
    g.add_argument("--val-frac", default=0.2, type=float,
                   help="fraction of (possibly subset) train data for validation "
                        "(e.g. 0.2 = 20%%). Used for early stopping.")

    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--n-blocks", default=2, type=int)
    g.add_argument("--L", default=8, type=int,
                   help="wavelet orientations per block")
    g.add_argument("--kernel-size", default=5, type=int,
                   help="wavelet kernel size (must be odd)")
    g.add_argument("--modulus-type", default="complex_modulus",
                   choices=["complex_modulus", "phase_relu"])
    g.add_argument("--mixing-horizon", default=None, type=int,
                   help="max input channels each wavelet filter can see; "
                        "None=full mixing. Recommended 27 for >=4 blocks")
    g.add_argument("--global-avg-pool", action="store_true",
                   help="apply adaptive avg pool to 1x1 before classifier")
    g.add_argument("--lowpass-last", action="store_true",
                   help="final block outputs only low-pass (phi) path")
    g.add_argument("--use-batchnorm", dest="use_batchnorm",
                   action="store_true",
                   help="enable BatchNorm2d standardization layers")
    g.add_argument("--random-init", action="store_true",
                   help="skip wavelet initialization, use Kaiming random init."
                   "single-phase joint training (skip Phase A/B split)")
    g.add_argument("--use-concat-projection", action="store_true",
                   help="use ConcatProjectedScatteringBlock instead of ScatteringBlock; "
                        "allows independent skip connections and/or 1x1 projection layers")
    g.add_argument("--use-skip-connections", dest="use_skip_connections",
                   action="store_true", default=True,
                   help="when using concat-projection, include skip connections (default: True)")
    g.add_argument("--no-skip-connections", dest="use_skip_connections",
                   action="store_false",
                   help="when using concat-projection, disable skip connections")
    g.add_argument("--use-projection", dest="use_projection",
                   action="store_true", default=True,
                   help="when using concat-projection, apply 1x1 projection (default: True)")
    g.add_argument("--no-projection", dest="use_projection",
                   action="store_false",
                   help="when using concat-projection, disable 1x1 projection")
    g.add_argument("--projection-out-channels", default=None, type=str,
                   help="comma-separated list of output channels for each projection layer "
                        "(e.g., '16,32,64'). If None, defaults to keeping input channels. "
                        "Must match n_blocks if provided. Only applies when --use-projection is enabled.")

    # Analysis layer (optional post-processing)
    g.add_argument("--use-analysis", action="store_true",
                   help="add optional Analysis tight frame layer after scattering blocks")
    g.add_argument("--analysis-frame-width", default=-1, type=int,
                   help="Analysis frame width (-1 for 2x overcomplete default)")
    g.add_argument("--analysis-frame-kernel-size", default=1, type=int,
                   help="Analysis frame kernel size")
    g.add_argument("--analysis-frame-stride", default=1, type=int,
                   help="Analysis frame stride")
    g.add_argument("--analysis-non-linearity", default="relu",
                   choices=["relu", "absolute", "softshrink"],
                   help="Analysis non-linearity type")
    g.add_argument("--analysis-zero-bias", action="store_true",
                   help="ReLU with zero bias for Analysis layer")
    g.add_argument("--analysis-frame-init-mode", default="orthogonal",
                   choices=["orthogonal", "random", "i_minus_i"],
                   help="Analysis frame initialization mode")

    # Parseval regularization
    g.add_argument("--pars-beta", default=0.5, type=float,
                   help="Parseval frame update step size (0=disabled)")
    g.add_argument("--parseval-analysis", action="store_true",
                   help="apply Parseval update to analysis frames during classifier training")

    # Training phases
    g = p.add_argument_group("Training phases")
    g.add_argument("--epochs", default=100, type=int,
                   help="max epochs for phase B or random init training")
    g.add_argument("--lr-epochs", default=20, type=int,
                   help="max epochs for Phase A (head-only)")
    g.add_argument("--phase-a-only", action='store_true',
                   help="train Phase A, save checkpoints/phase_a.pth (the 'Scattering "
                        "(no FT)' model), and exit without running Phase B. Ignored with "
                        "--random-init. Phase A does not use --l2-penalty.")
    g.add_argument("--lr-head", default=1e-3, type=float,
                   help="classifier head learning rate (both phases); "
                   "For random init training, it is the learning rate for all parameters")
    g.add_argument("--lr-extractor", default=1e-4, type=float,
                   help="feature extractor learning rate (Phase B)")
    g.add_argument("--l2-penalty", default=0, type=float,
                   help="L2 penalty lambda on extractor filters (0=off). "
                        "What the penalty pulls toward is set by --l2-anchor.")
    g.add_argument("--l2-anchor", default="auto", choices=list(L2_ANCHOR_CHOICES),
                   help="what --l2-penalty anchors the extractor filters to. "
                        "'zero': classic weight decay, ||theta||^2 (isolates "
                        "regularisation strength from any prior). "
                        "'scattering': ||theta - theta_wavelet||^2, i.e. pull "
                        "toward the wavelet initialization of this same "
                        "architecture -- with --random-init this injects the "
                        "scattering prior purely through the penalty. "
                        "'init': ||theta - theta_0||^2 for whatever weights "
                        "this run started from (L2-SP: the wavelet prior for a "
                        "two-phase run, the random draw under --random-init). "
                        "'auto' (default): 'zero' with --random-init, 'init' otherwise.")

    # Common training
    g = p.add_argument_group("Training")
    g.add_argument("--batch-size-per-device", default=128, type=int,
                   help="batch size per device")
    g.add_argument("--weight-decay", default=1e-4, type=float)
    g.add_argument("--patience", default=50, type=int,
                   help="early-stopping patience (epochs)")
    g.add_argument("--min-delta", default=1e-4, type=float,
                   help="minimum improvement in validation accuracy to reset patience")
    g.add_argument("--workers", default=4, type=int,
                   help="number of workers for data loading")
    g.add_argument("--no-persistent-workers", action="store_true",
                   help="disable persistent DataLoader workers")
    g.add_argument("--seed", default=123, type=int)
    g.add_argument("--save-dir", default="run", type=str)
    g.add_argument("--print-freq", default=50, type=int,
                   help="print frequency over gradient update steps")
    g.add_argument("--save-every-sec", default=15 * 60, type=int,
                   help="save checkpoint every X seconds")
    g.add_argument("--save-every-epoch", default=5, type=int, 
                   help="save checkpoint every X epochs")
    g.add_argument("--grad-accum-steps", default=1, type=int, 
                   help="number of gradient accumulation steps")
    g.add_argument("--retrain", action='store_true',
                   help="force training from scratch by removing existing checkpoints")
    args = p.parse_args()

    # Coerce train_size: if it looks like an int greater than 1, cast it
    if args.train_size is not None and args.train_size == int(args.train_size) and args.train_size > 1:
        args.train_size = int(args.train_size)

    # Parse projection_out_channels if provided
    if args.projection_out_channels is not None:
        try:
            args.projection_out_channels = [int(x.strip()) for x in args.projection_out_channels.split(',')]
        except ValueError:
            raise ValueError(
                f"--projection-out-channels must be a comma-separated list of integers, "
                f"got: {args.projection_out_channels}"
            )
        # Validate length matches n_blocks
        if len(args.projection_out_channels) != args.n_blocks:
            raise ValueError(
                f"--projection-out-channels length ({len(args.projection_out_channels)}) "
                f"must match --n-blocks ({args.n_blocks})"
            )
        # Require use_projection=True
        if not args.use_projection:
            raise ValueError(
                "--projection-out-channels requires --use-projection to be enabled"
            )
    
    # Validate concat projection parameters
    if args.use_concat_projection:
        if not args.use_skip_connections and not args.use_projection:
            raise ValueError(
                "When using --use-concat-projection, at least one of "
                "--use-skip-connections or --use-projection must be enabled"
            )
    else:
        # Warn if skip/projection flags are set but concat-projection is not enabled
        if not args.use_skip_connections or not args.use_projection:
            if not args.use_skip_connections:
                print("Warning: --no-skip-connections only applies with --use-concat-projection (ignored)")
            if not args.use_projection:
                print("Warning: --no-projection only applies with --use-concat-projection (ignored)")

    # Validate DTD partition
    if args.dataset == "dtd":
        if not (1 <= args.dtd_partition <= 10):
            raise ValueError(f"DTD partition must be in [1, 10], got {args.dtd_partition}")

    if args.l2_anchor != "auto" and args.l2_penalty <= 0:
        print(f"Warning: --l2-anchor {args.l2_anchor} has no effect with "
              f"--l2-penalty {args.l2_penalty} (no penalty term is added)")
    if args.l2_anchor == "scattering" and args.use_concat_projection:
        raise ValueError(
            "--l2-anchor scattering is not supported with --use-concat-projection: "
            "the 1x1 projection layers have no wavelet initialization to anchor to."
        )

    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    distributed = init_distributed()

    # Relative save dirs go under ./training_logs unless given as './...'.
    if not args.save_dir.startswith('./'):
        args.save_dir = os.path.join('./training_logs', args.save_dir)

    with console_logging(args.save_dir):
        local_rank = get_local_rank()
        args.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if distributed:
            print(f"Running on {get_world_size()} GPUs")
        print(f"Local Device: {args.device}")

        # Also sets args used later (sample counts, steps per epoch, ...).
        train_loader, val_loader, test_loader = build_dataloaders(args)
        loaders = SimpleNamespace(train_loader=train_loader, val_loader=val_loader,
                                 test_loader=test_loader)

        # ---- Model ----
        model = build_scattering_net(args, device=args.device)
        logger = _print_summary(args, model)    
        if distributed:
            model = nn.parallel.DistributedDataParallel(
                    model, device_ids=[local_rank])

        if args.random_init:
            _run_joint(args, model, logger, loaders)
        else:
            _run_two_phase(args, model, logger, loaders)

    cleanup_distributed()

# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def _run_phase(tag, args, save_dir, 
               logger, train_state, loaders):
    """Generic epoch loop with early stopping and checkpoint saving.
    After the loop finishes, loads the best checkpoint and evaluates on test set.
    Returns test_acc (top-1 on held-out test set).
    """
    if logger is not None:
        logger.set_phase(tag)
    train_state.phase = tag

    start_epoch = train_state.curr_epoch
    model = train_state.model
    step = train_state.step
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    patience = args.patience
    min_delta = args.min_delta
    device = args.device
    if tag == PHASE_A:
        epochs = args.lr_epochs
    else:
        epochs = args.epochs
    stopper = EarlyStopping(patience=patience, min_delta=min_delta)
    last_save_time = time.time()
    # On resume, skip the batches already completed in the current epoch.
    start_index = ((step - 1) % args.steps_per_epoch) * args.grad_accum_steps
    if start_index == 0 and step != 1:
        start_epoch += 1
    for epoch in range(start_epoch, epochs + 1):
        print(f"[{tag}] Epoch {epoch}/{epochs}")
        train_state.curr_epoch = epoch
        if logger is not None:
            logger.set_epoch(epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if epoch != start_epoch:
            start_index = 0
        topk = (1,5)
        if args.dataset in ['octmnist']:
            topk = (1,)
        train_loss, train_acc1, train_acc5 = run_epoch(
            train_loader, train_state, device,
            is_training=True, logger=logger, grad_accum_steps=args.grad_accum_steps,
            start_index=start_index, print_freq=args.print_freq, pars_beta=args.pars_beta,
            topk=topk)
        val_loss, val_acc1, val_acc5 = run_epoch(
            val_loader, train_state, device,
            is_training=False, logger=logger,
            topk=topk)

        is_best = val_acc1 > train_state.best_val_acc
        train_state.best_val_acc = max(val_acc1, train_state.best_val_acc)

        marker = " *" if is_best else ""
        print(f"  Train  loss={train_loss:.4f}  acc@1={train_acc1:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  acc@1={val_acc1:.2f}%  "
                f"acc@5={val_acc5:.2f}%{marker}")

        # Compute neural collapse metrics on validation set
        nc_metrics = {'nc1': float('nan'), 'nc2': float('nan'), 'nc3': float('nan'), 'nc1_rel': float('nan'), 'between_class_trace': float('nan')}
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            nc_metrics = compute_neural_collapse_metrics(
                model=model,
                loader=val_loader,
                device=args.device
            )
        except Exception as exc:
            print(f"Warning: could not compute neural collapse metrics for epoch {epoch}: {exc}")

        if logger is not None:
            logger.log_epoch(epoch, train_loss, train_acc1, train_acc5,
                             val_loss, val_acc1, val_acc5,
                             nc1=nc_metrics['nc1_rel'], nc2=nc_metrics['nc2'], nc3=nc_metrics['nc3'],
                             extra = {'nc1_rel': nc_metrics['nc1_rel'], 'nc_between_class_trace':nc_metrics['between_class_trace']})

        if  is_main_process() and (
            epoch % args.save_every_epoch == 0 or
            is_best or
            time.time() - last_save_time > args.save_every_sec
        ):
            train_state.nc_metrics = nc_metrics
            save_checkpoint(train_state, is_best, save_dir=save_dir)
            last_save_time = time.time()

        if stopper.step(val_acc1):
            print(f"  Early stopping triggered after {epoch} epochs.")
            break

    # Wait for rank 0 to finish any final writes before loading
    if is_distributed():
        dist.barrier()

    # Reload best checkpoint on all ranks so model state stays in sync
    best_path = os.path.join(save_dir, "best.pth")
    if os.path.exists(best_path):
        load_model(best_path, model)

    # Evaluate best checkpoint on test set
    _, test_acc1, test_acc5 = run_epoch(
        loaders.test_loader, train_state, device,
        is_training=False, logger=logger, topk=(1,) if args.dataset in ['octmnist'] else (1,5))
    return test_acc1, test_acc5

def _run_joint(args, model, logger, loaders):
    """Single-phase joint training — all parameters trainable from the start."""
    # ---- Prepare training state ----
    save_dir = os.path.join(args.save_dir, "checkpoints")
    if getattr(args, 'retrain', False):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"--retrain: cleared existing checkpoints at {save_dir}")
    else:
        os.makedirs(save_dir, exist_ok=True)

    train_state = auto_resume(save_dir, model, args, logger)
    # Initial parameters, for filter / classifier drift.
    param_snapshot = snapshot_extractor_params(model, save_dir)
    train_state.param_snapshot = param_snapshot
    train_state.classifier_snapshot = snapshot_classifier_params(model, save_dir)
    train_state.anchor_reg = make_anchor_penalty(args, param_snapshot, save_dir)
    # ---- Run training ----
    print("\n" + "=" * 60)
    print("Joint training (all parameters)")
    print("=" * 60 + "\n")
    joint_acc1, joint_acc5 = _run_phase(
        JOINT, args, save_dir,
        logger, train_state, loaders)
    
    # ---- Test set evaluation ----
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60 + "\n")
    model = train_state.model
    device = args.device
    loss, test_acc1, test_acc5 = run_epoch(
        loaders.test_loader, train_state, device,
        is_training=False, logger=logger,
            topk=(1,) if args.dataset in ['octmnist'] else (1,5))

    test_nc_metrics = {'nc1': float('nan'), 'nc2': float('nan'), 'nc3': float('nan'), 'nc1_rel': float('nan'), 'between_class_trace': float('nan')}
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        test_nc_metrics = compute_neural_collapse_metrics(
            model=model,
            loader=loaders.test_loader,
            device=device
        )
    except Exception as exc:
        print(f"Warning: could not compute neural collapse metrics on test set: {exc}")

    print(f"  Test  acc@1={test_acc1:.2f}%  acc@5={test_acc5:.2f}%")
    print(f"  NC1={test_nc_metrics['nc1_rel']:.6f}, NC2={test_nc_metrics['nc2']:.6f}, NC3={test_nc_metrics['nc3']:.6f}")

    delta_l2, delta_rel = compute_filter_delta(train_state)
    cls_delta_l2, cls_delta_rel = compute_classifier_delta(train_state)
    print(f"  filter_delta   = {delta_l2:.4f} "
            f"(relative: {delta_rel:.4f})")
    print(f"  classifier_delta = {cls_delta_l2:.4f} "
            f"(relative: {cls_delta_rel:.4f})")

    # ---- Save results ----
    if logger is not None:
        logger.set_result("joint_acc1", joint_acc1)
        logger.set_result("joint_err1", round(100.0 - joint_acc1, 4))
        logger.set_result("joint_acc5", joint_acc5)
        logger.set_result("joint_err5", round(100.0 - joint_acc5, 4))
        logger.set_result("filter_delta_l2", delta_l2)
        logger.set_result("filter_delta_relative", delta_rel)
        logger.set_result("classifier_delta_l2", cls_delta_l2)
        logger.set_result("classifier_delta_relative", cls_delta_rel)
        logger.set_result("test_acc1", test_acc1)
        logger.set_result("test_err1", round(100.0 - test_acc1, 4))
        logger.set_result("test_acc5", test_acc5)
        logger.set_result("test_err5", round(100.0 - test_acc5, 4))
        logger.set_result("test_nc1", test_nc_metrics['nc1_rel'])
        logger.set_result("test_nc2", test_nc_metrics['nc2'])
        logger.set_result("test_nc3", test_nc_metrics['nc3'])
        logger.set_result("test_between_class_trace", test_nc_metrics['between_class_trace'])
        logger.save()
    print(f"\nJoint training complete. test_acc1 = {test_acc1:.2f}%  test_acc5 = {test_acc5:.2f}%")


def _run_two_phase(args, model, logger, loaders):
    """Standard two-phase pipeline: freeze extractor, then fine-tune."""
    save_dir = os.path.join(args.save_dir, "checkpoints")
    if getattr(args, 'retrain', False):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        print(f"--retrain: cleared existing checkpoints at {save_dir}")
    else:
        os.makedirs(save_dir, exist_ok=True)

    train_state = auto_resume(save_dir, model, args, logger)
    # Initial parameters, for filter / classifier drift (head reference = random init).
    param_snapshot = snapshot_extractor_params(model, save_dir)
    train_state.param_snapshot = param_snapshot
    train_state.classifier_snapshot = snapshot_classifier_params(model, save_dir)
    # ================================================================
    # Phase A: train classifier head (feature extractor frozen)
    # ================================================================
    if train_state.phase == PHASE_A:
        print("=" * 60)
        print("Phase A: Train classifier head (extractor frozen)")
        print("=" * 60 + "\n")

        baseline_acc1, baseline_acc5 = _run_phase(
            PHASE_A, args, save_dir,
            logger, train_state, loaders)

        if logger is not None:
            logger.set_result("baseline_acc1", baseline_acc1)
            logger.set_result("baseline_err1", round(100.0 - baseline_acc1, 4))
            logger.set_result("baseline_acc5", baseline_acc5)
            logger.set_result("baseline_err5", round(100.0 - baseline_acc5, 4))
        print("\nPhase A complete.")
        print(f"Baseline acc (before fine-tune): {baseline_acc1:.2f}%  baseline_acc5 = {baseline_acc5:.2f}%")
        test_nc_metrics = {'nc1': float('nan'), 'nc2': float('nan'), 'nc3': float('nan'), 'nc1_rel': float('nan'), 'between_class_trace': float('nan')}
        try:
            model = train_state.model
            device = args.device
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            test_nc_metrics = compute_neural_collapse_metrics(
                model=model,
                loader=loaders.test_loader,
                device=device
            )
        except Exception as exc:
            print(f"Warning: could not compute neural collapse metrics on Phase A test set: {exc}")

        print(f"  Phase A Test  acc@1={baseline_acc1:.2f}%  acc@5={baseline_acc5:.2f}%")
        print(f"  Phase A NC1={test_nc_metrics['nc1_rel']:.6f}, NC2={test_nc_metrics['nc2']:.6f}, NC3={test_nc_metrics['nc3']:.6f}")
        # The extractor is frozen in Phase A, so filter_delta is 0 by construction.
        delta_l2, delta_rel = compute_filter_delta(train_state)
        cls_delta_l2, cls_delta_rel = compute_classifier_delta(train_state)
        print(f"  Phase A filter_delta   = {delta_l2:.4f} "
                f"(relative: {delta_rel:.4f})")
        print(f"  Phase A classifier_delta = {cls_delta_l2:.4f} "
                f"(relative: {cls_delta_rel:.4f})")

        if logger is not None:
            logger.set_result("phase_a_test_acc1", baseline_acc1)
            logger.set_result("phase_a_test_err1", round(100.0 - baseline_acc1, 4))
            logger.set_result("phase_a_test_acc5", baseline_acc5)
            logger.set_result("phase_a_test_err5", round(100.0 - baseline_acc5, 4))
            logger.set_result("phase_a_test_nc1", test_nc_metrics['nc1_rel'])
            logger.set_result("phase_a_test_nc2", test_nc_metrics['nc2'])
            logger.set_result("phase_a_test_nc3", test_nc_metrics['nc3'])
            logger.set_result("phase_a_test_between_class_trace", test_nc_metrics['between_class_trace'])
            logger.set_result("phase_a_filter_delta_l2", delta_l2)
            logger.set_result("phase_a_filter_delta_relative", delta_rel)
            logger.set_result("phase_a_classifier_delta_l2", cls_delta_l2)
            logger.set_result("phase_a_classifier_delta_relative", cls_delta_rel)
            logger.save()
        # "Scattering (no fine-tune)" model: frozen extractor + Phase-A head.
        save_phase_a_checkpoint(train_state, save_dir)

        if getattr(args, "phase_a_only", False):
            print("\n" + "=" * 60)
            print("--phase-a-only: Phase A complete, skipping Phase B")
            print(f"  wrote {os.path.join(save_dir, 'phase_a.pth')}")
            print("=" * 60 + "\n")
            if logger is not None:
                logger.set_result("phase_a_only", True)
                logger.save()
            return

        train_state = _prepare_phase_b(model, args, train_state, save_dir, baseline_acc1, baseline_acc5)


    # ================================================================
    # Phase B: joint fine-tune (different LR)
    # ================================================================
    print("=" * 60)
    print("Phase B: Joint fine-tune")
    print("=" * 60 + "\n")
 
    finetune_acc1, finetune_acc5 = _run_phase(
        PHASE_B, args, save_dir,
        logger, train_state, loaders)
    baseline_acc1 = train_state.baseline_acc1
    baseline_acc5 = train_state.baseline_acc5

    finetune_gain1 = finetune_acc1 - baseline_acc1
    finetune_gain5 = finetune_acc5 - baseline_acc5
    delta_l2, delta_rel = compute_filter_delta(train_state)
    cls_delta_l2, cls_delta_rel = compute_classifier_delta(train_state)
    # Head drift since random init, and during fine-tuning alone.
    cls_fa_delta_l2, cls_fa_delta_rel = compute_classifier_delta(
        train_state, reference="phase_a")

    print("\nPhase B complete.")
    print(f"  finetune_acc1   = {finetune_acc1:.2f}%")
    print(f"  finetune_acc5   = {finetune_acc5:.2f}%")
    print(f"  finetune_gain1  = {finetune_gain1:+.2f}%")
    print(f"  finetune_gain5  = {finetune_gain5:+.2f}%")
    print(f"  filter_delta   = {delta_l2:.4f} "
            f"(relative: {delta_rel:.4f})")
    print(f"  classifier_delta = {cls_delta_l2:.4f} "
            f"(relative: {cls_delta_rel:.4f})  [from random init]")
    print(f"  classifier_delta_from_phase_a = {cls_fa_delta_l2:.4f} "
            f"(relative: {cls_fa_delta_rel:.4f})")

    # ---- Test set evaluation ----
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60 + "\n")
    model = train_state.model
    device = args.device
    loss, test_acc1, test_acc5 = run_epoch(
        loaders.test_loader, train_state, device,
        is_training=False, logger=logger,
            topk=(1,) if args.dataset in ['octmnist'] else (1,5))

    test_nc_metrics = {'nc1': float('nan'), 'nc2': float('nan'), 'nc3': float('nan'), 'nc1_rel': float('nan'), 'between_class_trace': float('nan')}
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        test_nc_metrics = compute_neural_collapse_metrics(
            model=model,
            loader=loaders.test_loader,
            device=device
        )
    except Exception as exc:
        print(f"Warning: could not compute neural collapse metrics on test set: {exc}")
    
    print(f"  Test  acc@1={test_acc1:.2f}%  acc@5={test_acc5:.2f}%")
    print(f"  NC1={test_nc_metrics['nc1_rel']:.6f}, NC2={test_nc_metrics['nc2']:.6f}, NC3={test_nc_metrics['nc3']:.6f}")
    if logger is not None:
        logger.set_result("finetune_acc1", finetune_acc1)
        logger.set_result("finetune_acc5", finetune_acc5)
        logger.set_result("finetune_err1", round(100.0 - finetune_acc1, 4))
        logger.set_result("finetune_err5", round(100.0 - finetune_acc5, 4))
        logger.set_result("finetune_gain1", finetune_gain1)
        logger.set_result("finetune_gain5", finetune_gain5)
        logger.set_result("filter_delta_l2", delta_l2)
        logger.set_result("filter_delta_relative", delta_rel)
        logger.set_result("classifier_delta_l2", cls_delta_l2)
        logger.set_result("classifier_delta_relative", cls_delta_rel)
        logger.set_result("classifier_delta_from_phase_a_l2", cls_fa_delta_l2)
        logger.set_result("classifier_delta_from_phase_a_relative", cls_fa_delta_rel)
        logger.set_result("test_acc1", test_acc1)
        logger.set_result("test_err1", round(100.0 - test_acc1, 4))
        logger.set_result("test_acc5", test_acc5)
        logger.set_result("test_err5", round(100.0 - test_acc5, 4))
        logger.set_result("test_nc1", test_nc_metrics['nc1_rel'])
        logger.set_result("test_nc2", test_nc_metrics['nc2'])
        logger.set_result("test_nc3", test_nc_metrics['nc3'])
        logger.set_result("test_nc1_rel", test_nc_metrics['nc1_rel'])
        logger.set_result("test_between_class_trace", test_nc_metrics['between_class_trace'])
        logger.save()

def _prepare_phase_b(model, args, train_state, save_dir, baseline_acc1, baseline_acc5):
    """Unfreeze the extractor and set up the Phase-B optimizer, scheduler and snapshots."""
    raw = get_raw_model(model)
    print("Unfreezing extractor blocks")
    for block in raw.blocks:
        block.unfreeze()
    head_params = list(raw.bn.parameters()) + list(raw.classifier.parameters())
    extractor_params = list(raw.blocks.parameters())
    train_params = [
        {"params": extractor_params,
            "lr": args.lr_extractor},
        {"params": head_params, 
            "lr": args.lr_head * 0.5}
        ]
    total_steps = args.b_total_steps
    optimizer = optim.AdamW(
        train_params, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)  

    param_snapshot = snapshot_extractor_params(model, save_dir)
    # Head reference at the handover, for drift during fine-tuning.
    classifier_phase_a_snapshot = snapshot_classifier_params(
        model, save_dir, filename=CLASSIFIER_PHASE_A_SNAPSHOT_FILE,
        label="classifier (phase A)")
    anchor_reg = make_anchor_penalty(args, param_snapshot, save_dir)
    train_state.phase = PHASE_B
    train_state.curr_epoch = 1
    train_state.step = 1
    train_state.optimizer = optimizer
    train_state.scheduler = scheduler
    train_state.anchor_reg = anchor_reg
    train_state.baseline_acc1 = baseline_acc1
    train_state.baseline_acc5 = baseline_acc5
    train_state.param_snapshot = param_snapshot
    train_state.classifier_phase_a_snapshot = classifier_phase_a_snapshot
    return train_state

def _print_summary(args, model):
    model = get_raw_model(model)
    ps = model.param_summary()
    bs = model.block_summary()

    init_tag = "random" if args.random_init else "wavelet"
    mode_tag = "joint" if args.random_init else "two-phase"
    concat_proj_tag = ", concat_projection=True" if args.use_concat_projection else ""
    print(f"Model: {args.n_blocks} blocks, L={args.L}, "
        f"modulus={args.modulus_type}, "
        f"mixing_horizon={args.mixing_horizon}, "
        f"gap={args.global_avg_pool}, lowpass_last={args.lowpass_last}{concat_proj_tag}")
    if args.use_concat_projection:
        skip_tag = "skip" if args.use_skip_connections else "no_skip"
        proj_tag = "proj" if args.use_projection else "no_proj"
        print(f"  concat_projection modes: {skip_tag}, {proj_tag}")
        if args.projection_out_channels is not None:
            print(f"    projection_out_channels: {args.projection_out_channels}")
    print(f"  init={init_tag}, training={mode_tag}")
    print(f"Parameters: {ps['total']:,} total, {ps['trainable']:,} trainable")
    print(f"  Extractor (blocks): {ps['extractor']:,}")
    print(f"  Classifier (head):  {ps['classifier']:,}")
    print(f"  {'Block':<6} {'Type':<18} {'In_ch':>8} {'Out_ch':>8} {'Horizon':>8}")
    for row in bs:
        tags = []
        if row.get("mixing_horizon") is not None and row.get("in_channels") is not None:
            if row["mixing_horizon"] == row["in_channels"]:
                tags.append("dense")
        req = row.get("requested_horizon")
        if req is not None and row.get("mixing_horizon") is not None and req != row["mixing_horizon"]:
            tags.append(f"requested {req}")
        if row.get("lowpass_only"):
            tags.append("lowpass only")
        if row.get("type") == "ConcatProjected":
            tags.append("skip+proj")
        suffix = f"  ({', '.join(tags)})" if tags else ""
        block_type = row.get("type", "ScatteringBlock")[:16]
        horizon_str = f"{row['mixing_horizon']:>8}" if row.get("mixing_horizon") is not None else "       N/A"
        print(f"  {row['block']:<6} {block_type:<18} {row['in_channels']:>8} "
            f"{row['out_channels']:>8} {horizon_str}{suffix}")
    
    val_str = f"val {args.val_samples}  " if hasattr(args, "val_samples") else ""
    print(f"Dataset: {args.dataset}  |  "
            f"train_size {args.train_size if args.train_size is not None else 'full'}  "
            f"train {args.actual_train_samples}  "
            f"{val_str}test {args.test_samples}  "
            f"effective batch size {args.effective_batch_size}")

    logger = None
    if is_main_process():
        logger = ExperimentLogger(args.save_dir, args)
        logger.set_result("world_size", get_world_size())
        logger.set_result("lr_head", args.lr_head)
        logger.set_result("lr_extractor", args.lr_extractor)
        logger.set_result("block_summary", bs)
        logger.set_result("l2_penalty", args.l2_penalty)
        logger.set_result("l2_mode", l2_mode_label(args))
        logger.set_result("l2_anchor", resolve_l2_anchor(args))
        logger.set_result("use_concat_projection", args.use_concat_projection)
        if args.use_concat_projection:
            logger.set_result("use_skip_connections", args.use_skip_connections)
            logger.set_result("use_projection", args.use_projection)
        if args.projection_out_channels is not None:
            logger.set_result("projection_out_channels", args.projection_out_channels)

        logger.set_result("actual_train_samples", args.actual_train_samples)
        if hasattr(args, "val_samples"):
            logger.set_result("val_samples", args.val_samples)
        logger.set_result("test_samples", args.test_samples)
        logger.set_result("in_channels", args.in_channels)
        logger.set_result("in_spatial", args.in_spatial)
        logger.set_result("nb_classes", args.nb_classes)
        logger.set_result("effective_batch_size", args.effective_batch_size)
        for k, v in ps.items():
            logger.set_result(f"n_params_{k}", v)
    return logger

if __name__ == "__main__":
    main()