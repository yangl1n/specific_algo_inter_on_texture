from datetime import datetime
import json
import os

import torch

PHASE_A = "phase_a"
PHASE_B = "phase_b"
PHASE_C = "phase_c"
JOINT = "joint"


class ExperimentLogger:
    """Accumulate per-step and per-epoch metrics and write one JSON artifact."""

    def __init__(self, save_dir, args):
        self.save_dir = save_dir
        self.config = vars(args).copy() if args is not None else {}
        self.results = {}
        self.step_curves = {PHASE_A: [], PHASE_B: [], PHASE_C: [], JOINT: []}
        self.epoch_curves = {PHASE_A: [], PHASE_B: [], PHASE_C: [], JOINT: []}
        self.phase = JOINT
        self.epoch = 0
        os.makedirs(self.save_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.save_dir, f"results_{stamp}.json")

    def refresh_config(self, args):
        """Re-snapshot ``args`` once dataset-derived fields (nb_classes, in_channels,
        in_spatial) are set, so the logged config matches the model that ran.
        """
        self.config.update(vars(args))

    def set_phase(self, phase):
        if phase not in [PHASE_A, PHASE_B, PHASE_C, JOINT]:
            raise ValueError(f"Invalid phase: {phase}")
        self.phase = phase

    def set_epoch(self, epoch):
        self.epoch = epoch

    def log_step(self, step, train_loss, train_acc1, train_acc5, lr, grad_norm):
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

    def log_epoch(self, epoch, train_loss, train_acc1, train_acc5, val_loss, val_acc1, val_acc5, extra=None):
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc1": round(train_acc1, 3),
            "train_acc5": round(train_acc5, 3),
            "val_loss": round(val_loss, 5),
            "val_acc1": round(val_acc1, 3),
            "val_acc5": round(val_acc5, 3),
        }
        if extra:
            for key, value in extra.items():
                if value is None:
                    entry[key] = None
                elif isinstance(value, float):
                    entry[key] = round(value, 6)
                else:
                    entry[key] = value
        self.epoch_curves[self.phase].append(entry)

    def set_result(self, key, value):
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
        self.config = state_dict.get("config", self.config)
        self.results = state_dict.get("results", self.results)
        self.step_curves = state_dict.get("step_curves", self.step_curves)
        self.epoch_curves = state_dict.get("epoch_curves", self.epoch_curves)
        return self

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        payload = self.payload()
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, self.path)
        print(f"Experiment log saved -> {self.path}")
        return self.path


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def grad_norm_l2(model):
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(torch.sum(grad * grad).item())
    return total ** 0.5


def dataset_meta(dataset_name):
    if dataset_name == "cifar10":
        return 3, 32, 10
    if dataset_name == "cifar100":
        return 3, 32, 100
    if dataset_name == "mnist":
        return 1, 28, 10
    if dataset_name == "imagenet":
        return 3, 224, 1000
    if dataset_name == "dtd":
        return 3, 224, 47
    if dataset_name == "minc2500":
        return 3, 224, 23
    if dataset_name == "eurosat":
        return 3, 64, 10
    if dataset_name == 'fmd':
        return 3, 128, 10
    if dataset_name == 'alot':
        return 3, 128, 250
    if dataset_name == 'surotex':
        return 3, 128, 50
    if dataset_name == 'pathmnist':
        return 3, 64, 9
    if dataset_name == 'octmnist':
        return 1, 28, 4
    if dataset_name == 'so2sat_rgb':
        return 3, 32, 17
    if dataset_name == 'so2sat_s2':
        return 10, 32, 17
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def init_logger_results(
    logger,
    *,
    world_size,
    actual_train_samples,
    val_samples,
    test_samples,
    in_channels,
    in_spatial,
    nb_classes,
    effective_batch_size,
    n_params_total,
    n_params_extractor,
    n_params_classifier,
):
    logger.set_result("world_size", world_size)
    logger.set_result("actual_train_samples", actual_train_samples)
    logger.set_result("val_samples", val_samples)
    logger.set_result("test_samples", test_samples)
    logger.set_result("in_channels", in_channels)
    logger.set_result("in_spatial", in_spatial)
    logger.set_result("nb_classes", nb_classes)
    logger.set_result("effective_batch_size", effective_batch_size)
    logger.set_result("n_params_total", n_params_total)
    logger.set_result("n_params_extractor", n_params_extractor)
    logger.set_result("n_params_classifier", n_params_classifier)

    logger.set_result("lr_head", None)
    logger.set_result("lr_extractor", None)
    logger.set_result("l2_penalty", None)
    logger.set_result("block_summary", None)
    logger.set_result("baseline_acc1", None)
    logger.set_result("baseline_err1", None)
    logger.set_result("baseline_acc5", None)
    logger.set_result("baseline_err5", None)
    logger.set_result("finetune_acc1", None)
    logger.set_result("finetune_err1", None)
    logger.set_result("finetune_acc5", None)
    logger.set_result("finetune_err5", None)
    logger.set_result("finetune_gain1", None)
    logger.set_result("finetune_gain5", None)
    logger.set_result("filter_delta_l2", None)
    logger.set_result("filter_delta_relative", None)
    logger.set_result("p_delta_l2", None)
    logger.set_result("p_delta_relative", None)
    logger.set_result("f_delta_l2", None)
    logger.set_result("f_delta_relative", None)


def finalize_joint_results(logger, best_acc1, best_acc5, best_epoch_acc1, best_epoch_acc5):
    logger.set_result("best_acc1", best_acc1)
    logger.set_result("best_acc5", best_acc5)
    logger.set_result("best_epoch_acc1", best_epoch_acc1)
    logger.set_result("best_epoch_acc5", best_epoch_acc5)
    logger.set_result("joint_acc1", best_acc1)
    logger.set_result("joint_err1", 100.0 - best_acc1)
    logger.set_result("joint_acc5", best_acc5)
    logger.set_result("joint_err5", 100.0 - best_acc5)
