"""Checkpoint I/O, experiment JSON logger, distributed helpers."""

import json
import multiprocessing
import os
from datetime import datetime
import torch.optim as optim
import torch
import torch.distributed as dist
import torch.nn as nn
import sys
from contextlib import contextmanager
import signal
import warnings


def _safe_torch_load(path, map_location="cpu", weights_only=False):
    """Best-effort torch.load that returns None on corruption or read errors."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as exc:
        warnings.warn(f"Failed to load checkpoint '{path}': {exc}")
        return None


def _atomic_torch_save_with_validation(state, final_path):
    """Atomically save a checkpoint and validate it before replacing the target file.

    The existing target file is never replaced unless the newly written file can be
    loaded successfully.
    """
    tmp_path = f"{final_path}.tmp"

    with open(tmp_path, "wb") as fh:
        torch.save(state, fh)
        fh.flush()
        os.fsync(fh.fileno())

    validated = _safe_torch_load(tmp_path, map_location="cpu", weights_only=False)
    if validated is None:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Refusing to replace '{final_path}' with an invalid checkpoint")

    os.replace(tmp_path, final_path)


# ---------------------------------------------------------------------------
# Model wrapper helpers
# ---------------------------------------------------------------------------

def get_raw_model(model):
    """Unwrap DDP to get the underlying ScatteringNet."""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class TrainState:
    def __init__(self, phase, curr_epoch, step, best_val_acc, 
    optimizer, scheduler, param_snapshot, anchor_reg, model,
    baseline_acc1, baseline_acc5, logger, classifier_snapshot=None,
    classifier_phase_a_snapshot=None):
        # ---- Save ----
        self.phase = phase
        self.curr_epoch = curr_epoch
        self.step = step
        self.best_val_acc = best_val_acc
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.param_snapshot = param_snapshot
        self.classifier_snapshot = classifier_snapshot
        # Head at the Phase A -> Phase B handover; None for joint runs.
        self.classifier_phase_a_snapshot = classifier_phase_a_snapshot
        self.model = model
        self.baseline_acc1 = baseline_acc1
        self.baseline_acc5 = baseline_acc5
        self.logger = logger
        # ---- Not save ----
        self.anchor_reg = anchor_reg

def register_signal_handlers(state, save_dir):
    """Register signal handlers to save the checkpoint when the program is interrupted"""
    def save_and_exit(signum, frame):
        if multiprocessing.current_process().name != 'MainProcess' or not is_main_process():
            os._exit(0)
        print(f"Signal {signum} received")
        save_checkpoint(state, False, save_dir)
        print(f"Checkpoint saved to {save_dir} successfully")
        os._exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

def save_checkpoint(train_state, is_best, save_dir):
    """Keep the current checkpoint and the best checkpoint"""
    if not is_main_process():
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "curr.pth")
    if is_best:
        path = os.path.join(save_dir, "best.pth")
    state = {
        "phase": train_state.phase,
        "step": train_state.step,
        "curr_epoch": train_state.curr_epoch,
        "model": get_raw_model(train_state.model).state_dict(),
        "optimizer": train_state.optimizer.state_dict(),
        "scheduler": train_state.scheduler.state_dict(),
        "log_history": train_state.logger.payload(),
        "best_val_acc": train_state.best_val_acc,
        "baseline_acc1": train_state.baseline_acc1,
        "baseline_acc5": train_state.baseline_acc5,
    }
    try:
        _atomic_torch_save_with_validation(state, path)
    except Exception as exc:
        warnings.warn(f"Checkpoint save skipped for '{path}': {exc}")


def save_phase_a_checkpoint(train_state, save_dir):
    """Snapshot the end-of-Phase-A model (frozen-init extractor + trained head).

    This is the "Scattering (no fine-tune)" model: the extractor still holds the
    Morlet initialization. Best-effort; a failure never aborts training.
    """
    if not is_main_process():
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "phase_a.pth")
    state = {
        "phase": PHASE_A,
        "curr_epoch": getattr(train_state, "curr_epoch", None),
        "model": get_raw_model(train_state.model).state_dict(),
        "baseline_acc1": getattr(train_state, "baseline_acc1", None),
        "baseline_acc5": getattr(train_state, "baseline_acc5", None),
    }
    try:
        _atomic_torch_save_with_validation(state, path)
        print(f"  Saved Phase-A (no-fine-tune) checkpoint to {path}")
    except Exception as exc:
        warnings.warn(f"Phase-A checkpoint save skipped for '{path}': {exc}")


def load_params_snapshot(save_dir):
    """Load the extractor parameter snapshot from the checkpoint dir."""
    return _load_snapshot(save_dir, "param_snapshot.data", "Parameter")


def load_classifier_snapshot(save_dir):
    """Load the classifier-head snapshot, or None for runs that never took one."""
    return _load_snapshot(save_dir, "classifier_snapshot.data", "Classifier")


def load_classifier_phase_a_snapshot(save_dir):
    """Load the Phase-A classifier snapshot (needed when resuming into Phase B)."""
    return _load_snapshot(save_dir, "classifier_phase_a_snapshot.data",
                          "Classifier (phase A)")


def _load_snapshot(save_dir, filename, label):
    path = os.path.join(save_dir, filename)
    if os.path.exists(path):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        print(f"{label} snapshot loaded from {path}")
        return snapshot
    return None

def load_model(path, model):
    """Load a checkpoint into *model*.

    Returns the checkpoint dict.
    """
    ckpt = _safe_torch_load(path, map_location="cpu", weights_only=False)
    if ckpt is None:
        return None
    get_raw_model(model).load_state_dict(ckpt["model"])
    return ckpt


def list_checkpoints_newest_first(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return []

    candidates = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth"):
            path = os.path.join(ckpt_dir, f)
            candidates.append(path)

    if not candidates:
        return []

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates

def auto_resume(
    ckpt_dir,
    model,
    args,
    logger
):
    # ---- Initialize phase ----
    if args.random_init:
        phase = JOINT
    else:
        phase = PHASE_A
    # ---- Load checkpoint model ----
    ckpt_candidates = list_checkpoints_newest_first(ckpt_dir)
    ckpt = None
    ckpt_path = None
    if ckpt_candidates:
        for candidate in ckpt_candidates:
            try:
                maybe_ckpt = _safe_torch_load(candidate, map_location="cpu", weights_only=False)
                if maybe_ckpt is None:
                    continue
                get_raw_model(model).load_state_dict(maybe_ckpt["model"])
                ckpt = maybe_ckpt
                ckpt_path = candidate
                phase = ckpt["phase"]
                print(f"Resuming from checkpoint: {ckpt_path} with phase {phase}")
                break
            except Exception as exc:
                warnings.warn(f"Skipping unreadable checkpoint '{candidate}': {exc}")
                continue

        if ckpt is None:
            warnings.warn("No valid checkpoint found; starting from scratch")
            print(f"No valid checkpoint found — starting from scratch with phase {phase}")
    else:
        print(f"No checkpoint found — starting from scratch with phase {phase}")
    raw = get_raw_model(model)
    if phase == PHASE_A:
        print("Freezing extractor blocks")
        for block in raw.blocks:
            block.freeze()
        head_params = list(raw.bn.parameters()) + list(raw.classifier.parameters())
        train_params = [
            {"params": head_params,
            "lr": args.lr_head}
        ]
        total_steps = args.a_total_steps
    elif phase == PHASE_B:
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
    else:
        train_params = [
           {"params": model.parameters(),
            "lr": args.lr_head}
        ]
        total_steps = args.total_steps

    # ---- Initialize other training states ----
    optimizer = optim.AdamW(
        train_params, weight_decay=args.weight_decay)
    eta_min = args.lr_extractor if phase == PHASE_A else 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=eta_min)  
    curr_epoch = 1
    step = 1
    best_val_acc = 0.0  
    baseline_acc1 = 0.0
    baseline_acc5 = 0.0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        curr_epoch = ckpt["curr_epoch"]
        step = ckpt["step"]
        best_val_acc = ckpt["best_val_acc"]
        baseline_acc1 = ckpt["baseline_acc1"]
        baseline_acc5 = ckpt["baseline_acc5"]
        # ---- Load logger state ----
        if logger is not None:
            logger.load_state_dict(ckpt["log_history"])
    
    # ---- Load parameter snapshots ----
    param_snapshot = load_params_snapshot(ckpt_dir)
    classifier_snapshot = load_classifier_snapshot(ckpt_dir)
    classifier_phase_a_snapshot = load_classifier_phase_a_snapshot(ckpt_dir)
    # On a fresh run the phase runner builds the anchor; here only on resume.
    anchor_reg = None
    if param_snapshot is not None:
        from engine import make_anchor_penalty
        anchor_reg = make_anchor_penalty(args, param_snapshot, ckpt_dir)
    train_state = TrainState(
        phase=phase,
        curr_epoch=curr_epoch,
        step=step,
        best_val_acc=best_val_acc,
        optimizer=optimizer,
        scheduler=scheduler,
        param_snapshot=param_snapshot,
        classifier_snapshot=classifier_snapshot,
        classifier_phase_a_snapshot=classifier_phase_a_snapshot,
        anchor_reg=anchor_reg,
        model=model,
        baseline_acc1=baseline_acc1,
        baseline_acc5=baseline_acc5,
        logger=logger
    )

    register_signal_handlers(train_state, ckpt_dir)

    return train_state

# ---------------------------------------------------------------------------
# Experiment JSON logger
# ---------------------------------------------------------------------------
PHASE_A = "phase_a"
PHASE_B = "phase_b"
JOINT = "joint"
class ExperimentLogger:
    """Accumulate per-step and per-epoch metrics; write to JSON."""

    def __init__(self, save_dir, args):
        self.save_dir = save_dir
        self.config = vars(args) if args is not None else {}
        self.results = {}
        self.step_curves = {PHASE_A: [], PHASE_B: [], JOINT: []}
        self.epoch_curves = {PHASE_A: [], PHASE_B: [], JOINT: []}
        os.makedirs(self.save_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.save_dir, f"results_{stamp}.json")

    def set_phase(self, phase):
        if phase not in [PHASE_A, PHASE_B, JOINT]:
            raise ValueError(f"Invalid phase: {phase}")
        self.phase = phase

    def set_epoch(self, epoch):
        self.epoch = epoch

    def log_step(self, step, train_loss, train_acc1, train_acc5, lr, grad_norm):
        """Log one optimizer step's training metrics (no val — computed per epoch)."""
        entry = {
            "step": step,
            "epoch": self.epoch,
            "train_loss": round(train_loss, 5),
            "train_acc1": round(train_acc1, 3),
            "train_acc5": round(train_acc5, 3),
            "lr": round(lr, 7),
            "grad_norm": round(grad_norm, 6),
        }
        self.step_curves[self.phase].append(entry)

    def log_epoch(self, epoch, train_loss, train_acc1, train_acc5,
                  val_loss, val_acc1, val_acc5, nc1=None, nc2=None, nc3=None, extra=None):
        """Log one epoch's train and val metrics, plus optional neural collapse stats."""
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc1": round(train_acc1, 3),
            "train_acc5": round(train_acc5, 3),
            "val_loss": round(val_loss, 5),
            "val_acc1": round(val_acc1, 3),
            "val_acc5": round(val_acc5, 3),
        }
        if nc1 is not None:
            entry["nc1"] = round(nc1, 6)
        if nc2 is not None:
            entry["nc2"] = round(nc2, 6)
        if nc3 is not None:
            entry["nc3"] = round(nc3, 6)
        if extra is not None:
            for key, value in extra.items():
                entry[key] = value
        self.epoch_curves[self.phase].append(entry)

    def set_result(self, key, value):
        """Store a final metric (e.g. ``head_acc``, ``finetune_gain``)."""
        if isinstance(value, float):
            value = round(value, 4)
        self.results[key] = value

    def payload(self):
        return {
            "config": self.config,
            "results": self.results,
            "step_curves": self.step_curves,
            "epoch_curves": self.epoch_curves,
        }

    def load_state_dict(self, state_dict):
        self.config = state_dict["config"]
        self.results = state_dict["results"]
        self.step_curves = state_dict["step_curves"]
        self.epoch_curves = state_dict["epoch_curves"]
        return self

    def save(self):
        """Write everything to the same JSON file for this training run."""
        os.makedirs(self.save_dir, exist_ok=True)
        payload = self.payload()
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, self.path)
        print(f"Experiment log saved -> {self.path}")
        return self.path


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed():
    """Auto-detect and initialize DDP from ``torchrun`` env vars.

    Returns True if a distributed process group was created, False
    otherwise (single-GPU or CPU run).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        return True
    return False


def cleanup_distributed():
    """Destroy the process group if it was initialized."""
    if is_distributed():
        dist.destroy_process_group()


def get_world_size():
    """Return the number of processes (1 when not distributed)."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process():
    """True on rank 0 or when not in distributed mode."""
    if not is_distributed():
        return True
    return dist.get_rank() == 0

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

# ---------------------------------------------------------------------------
# Logging context
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror writes to multiple streams (stdout/stderr + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

@contextmanager
def console_logging(save_dir):
    """Print only on the main process and tee its output to save_dir/output_<time>.txt.

    Errors in other processes are still printed.
    """
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(save_dir, f"output_{timestamp}.txt")

        log_file = open(log_path, "w", encoding="utf-8")

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        sys.stdout = _Tee(old_stdout, log_file)
        sys.stderr = sys.stdout

        try:
            print(f"Logging also saved to {log_path}")
            yield log_file
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            log_file.close()
    else:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        try:
            yield None
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout