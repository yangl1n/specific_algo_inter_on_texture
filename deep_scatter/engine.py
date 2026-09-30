"""Training / evaluation loops, early stopping, and parameter-delta metrics."""

import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from utils import is_distributed, get_raw_model, is_main_process


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------

class AverageMeter:
    """Tracks running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Return list of top-k accuracies (in %) for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


# ---------------------------------------------------------------------------
# Parseval frame update
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_parseval_update_to_analysis(model, pars_beta):
    """Parseval frame update of the analysis layer: F := (1+beta)*F - beta*F*F^T*F.

    Parameters
    ----------
    model : nn.Module
        Model (possibly wrapped with DistributedDataParallel)
    pars_beta : float
        Update step size (0 = disabled)
    """
    if pars_beta <= 0:
        return
    
    raw_model = get_raw_model(model)
    if not hasattr(raw_model, 'analysis') or raw_model.analysis is None:
        return
    
    frame_weight = raw_model.analysis.frame_weight
    if not frame_weight.requires_grad:
        return

    frame = frame_weight.data.reshape((frame_weight.shape[0], -1))
    N, C = frame.shape  # N atoms, C input dimension
    # F F^T F, multiplied in the cheaper order.
    if N > C:
        prod = torch.matmul(frame, torch.matmul(frame.t(), frame))
    else:
        prod = torch.matmul(torch.matmul(frame, frame.t()), frame)

    frame_weight.data = ((1 + pars_beta) * frame - pars_beta * prod).reshape(frame_weight.shape)


# ---------------------------------------------------------------------------
# Single-epoch train / eval
# ---------------------------------------------------------------------------

def run_epoch(loader, train_state, device,
              is_training, logger, grad_accum_steps=1, 
              start_index=0, print_freq=0, pars_beta=0.0, topk = (1, 5)):
    """Run one epoch of training or evaluation.

    Parameters
    ----------
    train_state : TrainState
        Training state containing the optimizer, scheduler, and anchor regularization.
    pars_beta : float
        Parseval frame update step size. If > 0, applies Parseval update to analysis
        frames after each gradient update during training.
    Returns
    -------
    avg_loss, top1_acc, top5_acc : float
    """
    model = train_state.model
    optimizer = train_state.optimizer
    scheduler = train_state.scheduler
    criterion = nn.CrossEntropyLoss()

    if is_training:
        model.train()
    else:
        model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    start = time.time()

    ctx = torch.enable_grad() if is_training else torch.inference_mode()
    with ctx:
        for i, (inputs, targets) in enumerate(loader):
            if is_training and i < start_index:
                continue
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if is_training:
                loss = loss / grad_accum_steps
                if (i + 1) % grad_accum_steps == 0:
                    if train_state.anchor_reg is not None:
                        raw = get_raw_model(model)
                        loss = loss + train_state.anchor_reg(raw)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    train_state.step += 1
                    apply_parseval_update_to_analysis(model, pars_beta)
                else:
                    loss.backward()
            bsz = targets.size(0)
            if topk == (1,5):
                acc1, acc5 = accuracy(outputs, targets, topk=topk)
                top5.update(acc5, bsz)
            else:
                acc1 = accuracy(outputs, targets, topk=topk)[0]
            losses.update(loss.item(), bsz)
            top1.update(acc1, bsz)

            if is_training and (i + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
                lr = optimizer.param_groups[0]["lr"]
                if logger is not None:
                    logger.log_step(train_state.step, losses.avg,
                                   top1.avg, top5.avg, lr, grad_norm)

                if print_freq > 0 and (i + 1) // grad_accum_steps % print_freq == 0:
                    elapsed = time.time() - start
                    total_batches = len(loader)
                    progress = f"[{i+1}/{total_batches}]  lr={lr:.7f}"
                    print(f"  {progress}  "
                        f"loss {losses.avg:.4f}  "
                        f"acc@1 {top1.avg:.2f}%  "
                        f"acc@5 {top5.avg:.2f}%  "
                        f"({elapsed:.1f}s)")

    # All-reduce so every rank makes the same early-stopping / checkpoint decisions.
    if is_distributed():
        stats = torch.tensor(
            [losses.sum, losses.count,
             top1.sum, top1.count,
             top5.sum, top5.count],
            device=device, dtype=torch.float64)
        dist.all_reduce(stats)
        losses.avg = (stats[0] / stats[1]).item()
        top1.avg = (stats[2] / stats[3]).item()
        top5.avg = (stats[4] / stats[5]).item()

    return losses.avg, top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        How many epochs to wait after the last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def step(self, metric):
        """Update with the latest metric value.

        Returns ``True`` if training should stop.
        """
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# L2 anchor penalty (L2-SP regularization)
# ---------------------------------------------------------------------------

class AnchorPenalty:
    """L2 penalty on the extractor weights toward a fixed anchor.

    Computes ``lam * sum_i ||theta_i - theta_i^0||^2`` with theta^0 given by
    ``named_params_snapshot`` (selected by ``--l2-anchor``, see
    :func:`make_anchor_penalty`):

    * ``init``       -- the weights this run started from;
    * ``scattering`` -- the wavelet-initialised weights of the same architecture.

    With ``anchor_to_zero=True`` the anchor is the origin (classic L2,
    ``--l2-anchor zero``).

    Parameters
    ----------
    named_params_snapshot : dict
        ``{name: CPU tensor}`` keyed to match ``raw_model.blocks``
        ``named_parameters()``. Ignored when ``anchor_to_zero=True``.
    lam : float
        Penalty strength (lambda).
    device : torch.device
        Device to store anchor tensors on (should match model device).
    anchor_to_zero : bool
        If True, penalise distance from 0 (classic L2) instead of distance
        from the snapshot.
    """

    def __init__(self, named_params_snapshot, lam, device, anchor_to_zero=False):
        self.anchor_to_zero = anchor_to_zero
        self.lam = lam
        if anchor_to_zero:
            self.anchor = None
        else:
            self.anchor = {k: v.to(device) for k, v in named_params_snapshot.items()}

    def __call__(self, raw_model):
        if self.anchor_to_zero:
            penalty = sum(
                p.pow(2).sum()
                for _, p in raw_model.blocks.named_parameters()
            )
        else:
            penalty = sum(
                (p - self.anchor[n]).pow(2).sum()
                for n, p in raw_model.blocks.named_parameters()
                if n in self.anchor
            )
        return self.lam * penalty


# ---------------------------------------------------------------------------
# Parameter snapshots and delta computation
# ---------------------------------------------------------------------------

def _snapshot_named_params(modules, save_dir, filename, label):
    """Snapshot ``{name: cpu tensor}`` for ``modules``, writing it once per run.

    An existing snapshot is loaded rather than overwritten, so a resumed run keeps
    its initial weights as the reference.

    ``modules`` is a list of ``(prefix, module)``; the prefix keeps names unique
    and matches the model's own ``state_dict`` keys.
    """
    save_path = os.path.join(save_dir, filename)
    if os.path.exists(save_path):
        try:
            snap = torch.load(save_path, map_location="cpu", weights_only=True)
            print(f"Reusing existing {label} snapshot from {save_path}")
            return snap
        except Exception as exc:
            warnings.warn(f"Unreadable {label} snapshot '{save_path}' "
                          f"({exc}); re-snapshotting from current weights")
    snap = {}
    for prefix, module in modules:
        if module is None:
            continue
        for name, param in module.named_parameters():
            snap[f"{prefix}{name}" if prefix else name] = param.detach().cpu().clone()
    if is_main_process():
        print(f"Saving {label} snapshot to {save_path}")
        torch.save(snap, save_path)
    return snap


def snapshot_extractor_params(model, save_dir):
    """Deep-copy feature extractor (``model.blocks``) parameters.

    Returns
    -------
    dict : ``{name: tensor}`` on CPU.
    """
    model = get_raw_model(model)
    # Unprefixed names, matching raw_model.blocks.named_parameters().
    return _snapshot_named_params([("", model.blocks)], save_dir,
                                  "param_snapshot.data", "parameter")


SCATTERING_ANCHOR_FILE = "scattering_anchor.data"

L2_ANCHOR_CHOICES = ("auto", "zero", "scattering", "init")


def resolve_l2_anchor(args):
    """Resolve ``--l2-anchor`` to a concrete anchor mode.

    ``auto``: a random-init (joint) run is regularised toward 0, a wavelet-init
    two-phase run toward its own starting weights (the scattering prior).
    """
    mode = getattr(args, "l2_anchor", "auto") or "auto"
    if mode == "auto":
        mode = "zero" if getattr(args, "random_init", False) else "init"
    if mode not in ("zero", "scattering", "init"):
        raise ValueError(f"unknown l2 anchor mode: {mode!r}")
    return mode


def l2_mode_label(args):
    """Tag for the resolved anchor, logged as ``l2_mode``.

    ``init`` is ``l2_sp`` for wavelet-init runs and ``l2_random_init`` under
    ``--random-init``.
    """
    mode = resolve_l2_anchor(args)
    if mode == "zero":
        return "l2_zero"
    if mode == "scattering":
        return "l2_scattering"
    return "l2_random_init" if getattr(args, "random_init", False) else "l2_sp"


def scattering_anchor_params(args, save_dir):
    """``{name: CPU tensor}`` for a *wavelet-initialised* extractor of this arch.

    Builds a throw-away :class:`ScatteringNet` with ``random_init=False`` and
    snapshots ``blocks``; cached in ``save_dir`` so resumed runs reuse it.
    """
    from scattering_net import build_scattering_net

    if getattr(args, "use_concat_projection", False):
        raise ValueError(
            "--l2-anchor scattering is not supported with "
            "--use-concat-projection: the projection layers have no wavelet "
            "initialization to anchor to."
        )

    save_path = os.path.join(save_dir, SCATTERING_ANCHOR_FILE)
    if os.path.exists(save_path):
        try:
            snap = torch.load(save_path, map_location="cpu", weights_only=True)
            print(f"Reusing existing scattering anchor from {save_path}")
            return snap
        except Exception as exc:
            warnings.warn(f"Unreadable scattering anchor '{save_path}' "
                          f"({exc}); rebuilding from the wavelet initialization")

    ref = build_scattering_net(args, random_init=False, learnable=True)
    snap = {name: p.detach().cpu().clone()
            for name, p in ref.blocks.named_parameters()}
    del ref
    if is_main_process():
        print(f"Saving scattering anchor to {save_path}")
        torch.save(snap, save_path)
    return snap


def make_anchor_penalty(args, param_snapshot, save_dir):
    """Build the :class:`AnchorPenalty` selected by ``--l2-anchor``.

    Returns ``None`` when ``--l2-penalty`` is 0 or the ``init`` snapshot is missing.
    """
    lam = getattr(args, "l2_penalty", 0)
    if not lam or lam <= 0:
        return None

    mode = resolve_l2_anchor(args)
    if mode == "zero":
        penalty = AnchorPenalty(None, lam, args.device, anchor_to_zero=True)
        desc = "distance-from-0 (classic L2)"
    elif mode == "scattering":
        penalty = AnchorPenalty(scattering_anchor_params(args, save_dir),
                                lam, args.device)
        desc = "L2 toward the wavelet/scattering initialization"
    else:
        if param_snapshot is None:
            warnings.warn("--l2-anchor init requested but no parameter snapshot "
                          "is available; L2 penalty disabled for this phase")
            return None
        penalty = AnchorPenalty(param_snapshot, lam, args.device)
        desc = ("L2-SP (drift from random init)"
                if getattr(args, "random_init", False)
                else "L2-SP (drift from wavelet init)")
    print(f"L2 penalty on extractor: lambda={lam} [{desc}]")
    return penalty


CLASSIFIER_SNAPSHOT_FILE = "classifier_snapshot.data"
CLASSIFIER_PHASE_A_SNAPSHOT_FILE = "classifier_phase_a_snapshot.data"


def snapshot_classifier_params(model, save_dir,
                               filename=CLASSIFIER_SNAPSHOT_FILE,
                               label="classifier"):
    """Deep-copy the classifier head (``model.bn`` + ``model.classifier``).

    With the default ``filename`` this is taken at the head's random
    initialization; ``filename=CLASSIFIER_PHASE_A_SNAPSHOT_FILE`` is used at the
    Phase A -> Phase B handover to measure drift during fine-tuning.
    """
    model = get_raw_model(model)
    return _snapshot_named_params(
        [("bn.", getattr(model, "bn", None)),
         ("classifier.", getattr(model, "classifier", None))],
        save_dir, filename, label)


def _named_param_delta(snapshot_before, named_params):
    """L2 distance and relative change of ``named_params`` from ``snapshot_before``.

    Returns ``(nan, nan)`` when the snapshot is missing or covers none of the
    parameters (unknown, as opposed to zero drift).
    """
    if not snapshot_before:
        return float("nan"), float("nan")
    delta_sq = 0.0
    norm_sq = 0.0
    matched = 0
    for name, param in named_params:
        if name not in snapshot_before:
            continue
        before = snapshot_before[name]
        after = param.detach().cpu()
        delta_sq += (after - before).pow(2).sum().item()
        norm_sq += before.pow(2).sum().item()
        matched += 1
    if matched == 0:
        return float("nan"), float("nan")
    delta_l2 = delta_sq ** 0.5
    norm_l2 = norm_sq ** 0.5
    relative = delta_l2 / norm_l2 if norm_l2 > 0 else 0.0
    return delta_l2, relative


def compute_filter_delta(train_state):
    """L2 distance and relative change of the extractor from its snapshot.

    Parameters
    ----------
    train_state : TrainState
        Training state containing the snapshot of the extractor parameters.

    Returns
    -------
    filter_delta_l2, filter_delta_relative : float
        ``(nan, nan)`` if no snapshot was taken for this run.
    """
    snapshot_before = train_state.param_snapshot
    if snapshot_before is None:
        print("No parameter snapshot found; skipping filter delta computation")
        return float("nan"), float("nan")
    model_after = get_raw_model(train_state.model)
    return _named_param_delta(snapshot_before,
                              model_after.blocks.named_parameters())


def compute_classifier_delta(train_state, reference="init"):
    """L2 distance and relative change of the classifier head from a reference.

    Parameters
    ----------
    reference : {"init", "phase_a"}
        ``"init"``: from the head's random initialization (whole run).
        ``"phase_a"``: from the head at the end of Phase A (fine-tuning only).

    Returns
    -------
    classifier_delta_l2, classifier_delta_relative : float
        ``(nan, nan)`` if the snapshot is missing (e.g. ``"phase_a"`` for joint runs).
    """
    attr = {"init": "classifier_snapshot",
            "phase_a": "classifier_phase_a_snapshot"}[reference]
    snapshot_before = getattr(train_state, attr, None)
    if snapshot_before is None:
        return float("nan"), float("nan")
    raw = get_raw_model(train_state.model)
    named = []
    for prefix, attr in (("bn.", "bn"), ("classifier.", "classifier")):
        module = getattr(raw, attr, None)
        if module is not None:
            named += [(f"{prefix}{n}", p) for n, p in module.named_parameters()]
    return _named_param_delta(snapshot_before, named)
