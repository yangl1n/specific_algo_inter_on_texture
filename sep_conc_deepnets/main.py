import argparse
import json
import os
import random
import signal
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from kymatio.torch import Scattering2D
from phase_scattering2d_torch import ScatteringTorch2D_wph, ScatteringTorch2D_wphTrainable
from kymatio_trainable import Scattering2DTrainable, Scattering2DRandomFilters
from deep_scatter_adapter import DeepScatterParity2D
from models.Analysis import Analysis
from models.LinearProj import LinearProj
from models.Classifier import Classifier
from models.ScatNetAnalysis import ScatNetAnalysis
import best_ckpt_eval as bce
from experiment_logging import (
    PHASE_A,
    PHASE_C,
    JOINT,
    ExperimentLogger,
    dataset_meta,
    finalize_joint_results,
    grad_norm_l2,
    init_logger_results,
    to_float,
)
from training_utils import (
    _build_optimizer_for_phase,
    _collect_named_param_groups,
    _compute_delta,
    _normalize_validate_args,
    _phase_anchor_penalty,
    _phase_plan,
    _snapshot_named_params,
    AverageMeter,
    ProgressMeter,
    accuracy,
    adjust_learning_rate,
    compute_neural_collapse_metrics,
    save_checkpoint,
)

from data_utils import (
    _apply_imagenet_class_subset,
    _base_dataset,
    _build_cifar_legacy_datasets,
    _build_dtd_legacy_datasets,
    _build_imagenet_sharded_datasets,
    _build_minc_legacy_datasets,
    _build_mnist_legacy_datasets,
    _build_eurosat_legacy_datasets,
    _build_fmd_legacy_datasets,
    _build_alot_legacy_datasets,
    _build_surotex_legacy_datasets,
    _build_medmnist_legacy_datasets,
    _build_so2sat_legacy_datasets,
    _build_stratified_train_val,
    _maybe_subset_train,
    _parse_train_size,
    _ensure_deep_scatter_available,
)


# Optional deep_scatter reuse layer.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(_THIS_DIR)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(_WORKSPACE_ROOT, 'deep_scatter'))

DEEP_SCATTER_AVAILABLE = True
DEEP_SCATTER_IMPORT_ERROR = None

try:
    from deep_scatter.data import (
        DATASET_CONFIGS,
        ImageNetTarDataset,
        ShardedSampler,
        _cifar_transforms,
        _imagenet_transforms,
    )
except Exception as exc:  # deep_scatter dependencies unavailable
    DEEP_SCATTER_AVAILABLE = False
    DEEP_SCATTER_IMPORT_ERROR = str(exc)
    DATASET_CONFIGS = {}
    ImageNetTarDataset = None
    ShardedSampler = None
    _cifar_transforms = None
    _imagenet_transforms = None


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--dataset', default='imagenet', choices=['mnist', 'cifar10', 'cifar100', 'imagenet', 'dtd', 'minc2500', 'eurosat', 'fmd', 'alot', 'surotex', 'pathmnist', 'octmnist', 'so2sat_rgb', 'so2sat_s2'],
                    help='dataset to train on (default: imagenet)')
parser.add_argument('--input-size', default=None, type=int, choices=[28, 64, 128, 224],
                    help='override the input resolution (sets in_spatial). For MedMNIST '
                         '(pathmnist/octmnist) this selects the MedMNIST+ variant. Unset = dataset '
                         'default. Same as deep_scatter/train.py --input-size.')
parser.add_argument('-a', '--arch', default='scatnetblockanalysis',
                    choices=['analysis', 'scatnet', 'scatnetblock', 'scatnetblockanalysis'],
                    help='model architecture (default: scatnetblockanalysis)')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run (default: 150)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate (default: 0.01)', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int, help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--learning-rate-adjust-frequency', default=30, type=int,
                    help='number of epoch after which learning rate is decayed by 10 (default: 30)')
parser.add_argument('--dir', default='default_dir', type=str, help='directory for training logs and checkpoints')
parser.add_argument('--pars-beta', default=0.0005, type=float, help='learning rate for pars reg (default: 0.0005)')
parser.add_argument('--n-blocks', default=4, type=int, help='number of blocks in the pipeline (default: 4)')
parser.add_argument('--checkpoint-save-frequency', default=10, type=int, help='save checkpoint every N epochs (default: 10)')

# Scattering parameters
parser.add_argument('--scat-angles', default=8, type=int, nargs='+', help='number of orientations for wavelet frame(s) (default: 8)')
parser.add_argument('--backend', default='torch', type=str, help='scattering backend')
parser.add_argument('--scattering-backend', default='legacy', choices=['legacy', 'legacy_trainable', 'deep_scatter', 'deep_scatter_4L'],
                    help='scattering implementation backend (default: legacy). legacy_trainable: phase scattering with trainable filters; deep_scatter_4L: deep_scatter without phase condensing; deep_scatter: deep_scatter with phase condensing')
parser.add_argument('--scattering-J', default=4, type=int,
                    help='maximum scale for the scattering transform - for scatnet arch only (default: 4)')
parser.add_argument('--wavelet-finetune', action='store_true',
                    help='enable training of deep_scatter wavelet filters (default: frozen)')
parser.add_argument('--random-init-filters', action='store_true',
                    help='initialize wavelet filters randomly instead of from Kymatio (for legacy_trainable backend only)')

# Linear projection parameters
parser.add_argument('--P-proj-size', default=256, type=int, nargs='+',
                    help='output dimension of the linear projection(s) P(s) (default: 256)')
parser.add_argument('--P-kernel-size', default=1, type=int, nargs='+', help='kernel size of P(s) (default: 1)')
parser.add_argument('--p-init', default='orthogonal', choices=['orthogonal', 'identity', 'random'],
                    help='initialization mode for linear projections P (default: orthogonal)')
parser.add_argument('--no-batchnorm', dest='use_batchnorm', action='store_false',
                    help='disable BatchNorm2d standardization layers')

# Analysis parameters
parser.add_argument('--non-linearity', default='relu', type=str, choices=['absolute', 'softshrink', 'relu'],
                    help='non linearity for analysis (default: relu)')
parser.add_argument('--zero-bias', action='store_true', help='force zero bias for ReLU')
parser.add_argument('--frame-width', default=2048, type=int, nargs='+', help='size(s) of tight frame(s) (default: 2048)')
parser.add_argument('--frame-kernel-size', default=1, type=int, nargs='+', help='kernel size of frame(s) (default: 1)')
parser.add_argument('--frame-stride', default=1, type=int, nargs='+', help='stride of frame(s) (default: 1)')
parser.add_argument('--f-init', default='orthogonal', choices=['orthogonal', 'i_minus_i', 'random'],
                    help='initialization mode for frame filters F (default: orthogonal)')

# Classifier parameters
parser.add_argument('--classifier-type', default='fc', type=str, choices=['fc', 'mlp'], help='classifier type (default: fc)')
parser.add_argument('--avg-ker-size', default=1, type=int, help='size of averaging kernel (default: 1)')
parser.add_argument('--nb-hidden-units', default=2048, type=int, help='number of hidden units for mlp classifier '
                                                                      '(default: 2048)')
parser.add_argument('--dropout-p-mlp', default=0.3, type=float, help='dropout probability in mlp (default: 0.3)')
parser.add_argument('--nb-l-mlp', default=2, type=int, help='number of hidden layers in mlp (default: 2)')

# Train on a subsample of classes
parser.add_argument('--nb-classes', default=1000, type=int, help='ImageNet only - number of classes randomly chosen '
                    'used for training and validation (default: 1000 = whole train/val dataset)')
parser.add_argument('--class-indices', default=None,
                    help='ImageNet only - numpy array of indices used in case nb-classes < 1000')
parser.add_argument('--train-size', default=None, type=str,
                    help='optional train subset size: fraction in (0, 1] or absolute integer count')
parser.add_argument('--val-frac', default=0.2, type=float,
                    help='validation split fraction for deep_scatter split mode (default: 0.2)')
parser.add_argument('--data-split-mode', default='legacy', choices=['legacy', 'deep_scatter'],
                    help='legacy keeps official validation split; deep_scatter uses train/val split from train set')
parser.add_argument('--imagenet-shards', action='store_true',
                    help='ImageNet only - use train/*.tar shard catalog built by deep_scatter utilities')
parser.add_argument('--imagenet-sampler', default='sharded', choices=['sharded', 'global'],
                    help='ImageNet tar mode sampler choice (default: sharded)')
parser.add_argument('--tar-cache-size', default=128, type=int,
                    help='ImageNet tar mode LRU cache size per worker (default: 128)')
parser.add_argument('--dtd-partition', default=1, type=int,
                    help='DTD only - partition (fold) number 1-10 for k-fold cross-validation (default: 1)')

# Schedule controls
parser.add_argument('--train-schedule', default='legacy', choices=['legacy', 'staged'],
                    help='legacy: existing joint training, staged: separate PF/filter phases')
parser.add_argument('--phase-a-epochs', default=None, type=int,
                    help='staged schedule: epochs for phase A (head only)')
parser.add_argument('--phase-b-epochs', default=0, type=int,
                    help='staged schedule: epochs for phase B (P/F + optional head)')
parser.add_argument('--phase-c-epochs', default=0, type=int,
                    help='staged schedule: epochs for optional phase C (filters + optional head)')
parser.add_argument('--freeze-head-bc', action='store_true',
                    help='freeze classifier head during phases B and C (default: head trainable)')
parser.add_argument('--freeze-pf-c', action='store_true',
                    help='freeze projection and frame filters during phase C (default: filters trainable)')
parser.add_argument('--phase-a-lr-head', default=None, type=float)
parser.add_argument('--phase-b-lr-head', default=None, type=float)
parser.add_argument('--phase-b-lr-pf', default=None, type=float)
parser.add_argument('--phase-c-lr-head', default=None, type=float)
parser.add_argument('--phase-c-lr-filter', default=None, type=float)
parser.add_argument('--l2-penalty', default=0.0, type=float,
                    help='phase-C filter anchor penalty lambda (L2-SP style)')
parser.add_argument('--no-best-ckpt-eval', dest='best_ckpt_eval', action='store_false',
                    help='skip the end-of-training evaluation of the best-val weights on the test '
                         'and val splits (the evaluation that matches deep_scatter, which reloads '
                         'best.pth before scoring test)')
parser.set_defaults(parseval_p=True, parseval_f=True, best_ckpt_eval=True)


def main():
    args = parser.parse_args()
    _normalize_validate_args(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # fill missing arguments
    num_args = ["scat_angles", "frame_kernel_size", "frame_width", "frame_stride", "P_proj_size", "P_kernel_size"]
    if args.arch in ['scatnet', 'analysis']: #override default in case not specified and ensure a single block is used
        setattr(args, "n_blocks", 1)

    n_blocks = args.n_blocks

    for item in num_args:
        value = getattr(args, item)
        if type(value) == int or type(value) == float:  # value repeated
            setattr(args, item, [value] * n_blocks)
        elif len(value) < n_blocks:  # default value added
            setattr(args, item, value + [parser.get_default(item)] * (n_blocks - len(value)))

    main_worker(args)


def load_model(args, logfile, summaryfile):
    n_blocks = args.n_blocks

    # Create building blocks
    scattering = [nn.Identity()] * n_blocks

    standardization = [nn.Identity()] * n_blocks
    proj = [nn.Identity()] * n_blocks
    linear_proj = [nn.Identity()] * n_blocks

    analysis = [nn.Identity()] * n_blocks

    model = [nn.Identity()] * n_blocks

    # Create model log
    ###########################################################################################

    if args.arch == 'scatnetblockanalysis':
        arch_log = "=> creating model ScatNetBlockAnalysis with {} blocks, scat angles {}, linear projection " \
                   "dimensions {}, frame widths {}, non linearity {}, frame kernel sizes {}, frame strides {} " \
                 "pars beta {}, classifier {} pipeline, scattering backend {}".\
                   format(n_blocks, args.scat_angles, args.P_proj_size, args.frame_width,  args.non_linearity,
                     args.frame_kernel_size, args.frame_stride, args.pars_beta, args.classifier_type,
                     args.scattering_backend)

    elif args.arch == 'scatnetblock':
        arch_log = "=> creating model ScatNetBlock with {} blocks, scat angles {}, linear projection dimensions {}, " \
                   "pars beta {}, classifier {} pipeline, scattering backend {}". \
                   format(n_blocks, args.scat_angles, args.P_proj_size, args.pars_beta, args.classifier_type,
                          args.scattering_backend)

    elif args.arch == 'scatnet':
        arch_log = "=> creating model ScatNet with scattering J {}, scat angles {}, " \
                 "linear projection dimension {}, pars beta {}, classifier {} pipeline, scattering backend {}". \
                   format(args.scattering_J, args.scat_angles[0], args.P_proj_size[0],
                     args.pars_beta, args.classifier_type, args.scattering_backend)

    elif args.arch == 'analysis':
        arch_log = "=> creating model Analysis with linear projection dimension {}, " \
                    "frame width {}, non linearity {}, frame kernel size {}, frame stride {} pars beta {}, " \
                    "classifier {} pipeline". \
                    format(args.P_proj_size[0], args.frame_width[0], args.non_linearity,
                           args.frame_kernel_size[0], args.frame_stride[0], args.pars_beta, args.classifier_type)

    # Model creation
    ###########################################################################################

    if hasattr(args, 'in_channels') and hasattr(args, 'in_spatial') and hasattr(args, 'nb_classes'):
        nb_channels_in = args.in_channels
        n_space = args.in_spatial
        nb_classes = args.nb_classes
    else:
        nb_channels_in, n_space, nb_classes = dataset_meta(args.dataset)

    nb_params, nb_proj_params, nb_frame_params, nb_classifier_params = compute_model_size(
        args, n_space, nb_channels_in, nb_classes)

    print_and_write('Total number of params:{:.2f}M, proj:{:.2f}M, frame:{:.2f}M, classifier:{:.2f}M'.
                    format(nb_params / 1e6, nb_proj_params / 1e6, nb_frame_params / 1e6,
                           nb_classifier_params / 1e6), logfile, summaryfile)

    for i in range(n_blocks):
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            # create scattering
            if args.arch == 'scatnet':
                J = args.scattering_J
            else:
                J = 1  # one block per scale
            L_ang = args.scat_angles[i]

            # For scatnet, choose between frozen Kymatio or trainable wrapper
            if args.arch == 'scatnet':
                if args.scattering_backend == 'legacy_trainable':
                    # Use trainable Kymatio wrapper for full order-2 scattering with learnable filters
                    if args.random_init_filters:
                        scattering[i] = Scattering2DRandomFilters(
                            J=J,
                            shape=(n_space, n_space),
                            L=L_ang,
                            max_order=2,
                            backend=args.backend,
                            init_std=0.1,
                            seed=args.seed
                        )
                    else:
                        scattering[i] = Scattering2DTrainable(
                            J=J,
                            shape=(n_space, n_space),
                            L=L_ang,
                            max_order=2,
                            backend=args.backend,
                            learnable_filters=args.wavelet_finetune
                        )
                    scattering[i] = nn.Sequential(scattering[i], nn.Flatten(1,2))
                else:
                    # Use frozen Kymatio Scattering2D (default for scatnet)
                    scattering[i] = Scattering2D(
                        J=J,
                        shape=(n_space, n_space),
                        L=L_ang,
                        max_order=2,
                        backend=args.backend
                    )
                    scattering[i] = nn.Sequential(scattering[i], nn.Flatten(1,2))
            # For scatnetblock/scatnetblockanalysis, use our wrapped implementations
            elif args.scattering_backend == 'legacy':
                scattering[i] = ScatteringTorch2D_wph(J=J, shape=(n_space, n_space), L=L_ang, backend=args.backend)
                # Flatten scattering
                scattering[i] = nn.Sequential(scattering[i], nn.Flatten(1, 2))
            elif args.scattering_backend == 'legacy_trainable':
                scattering[i] = ScatteringTorch2D_wphTrainable(
                    J=J, 
                    shape=(n_space, n_space), 
                    L=L_ang, 
                    backend=args.backend,
                    learnable_filters=args.wavelet_finetune
                )
                # Flatten scattering
                scattering[i] = nn.Sequential(scattering[i], nn.Flatten(1, 2))
            else:
                scattering[i] = DeepScatterParity2D(
                    J=J,
                    shape=(n_space, n_space),
                    in_channels=nb_channels_in,
                    L=L_ang,
                    learnable_filters=args.wavelet_finetune,
                    backend=args.scattering_backend
                )

            if args.arch == 'scatnet':
                factor_channels = 1 + L_ang * J + (L_ang ** 2) * J * (J - 1) // 2  # scattering of order 2
            elif args.scattering_backend == 'deep_scatter':
                factor_channels = 1 + L_ang  # condense 4 phases
            elif args.scattering_backend == 'legacy_trainable':
                factor_channels = 1 + 4 * L_ang  # 4 phases used (like legacy)
            else:
                factor_channels = 1 + 4 * L_ang # 4 phases used
            nb_channels_in *= factor_channels

            n_space = n_space // (2 ** J)

        ###########################################################################################
        # create linear proj
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            linear_proj[i] = LinearProj.create_projection_layer(
                in_channels=nb_channels_in,
                proj_size=args.P_proj_size[i],
                p_kernel_size=args.P_kernel_size[i],
                p_init=args.p_init,
                frame_kernel_size=args.frame_kernel_size[i],
                frame_stride=args.frame_stride[i],
                use_batchnorm=args.use_batchnorm,
            )
            nb_channels_in = args.P_proj_size[i]
        else:
            # no need to standardize on two-layer network since input is already standardized, but normalize
            linear_proj[i] = LinearProj(standardization=nn.Identity(), proj=nn.Identity(), P_kernel_size=args.P_kernel_size[i],
                                        frame_kernel_size=args.frame_kernel_size[i], frame_stride=args.frame_stride[i],
                                        use_batchnorm=False)

        ###########################################################################################
        # Create tight frame analysis block

        if args.arch in ['scatnetblockanalysis', 'analysis']:
            if args.non_linearity == 'relu' and not args.zero_bias:
                lambda_ = 1.
            elif args.non_linearity == 'softshrink':
                lambda_ = 1.5
            else:
                lambda_ = 0  # ReLU with zero bias or absolute value (ignores lambda anyway)

            analysis[i] = Analysis(nb_channels_in, frame_width=args.frame_width[i],
                                   lambda_=lambda_/np.sqrt(args.frame_width[i]),
                                   frame_kernel_size=args.frame_kernel_size[i], frame_stride=args.frame_stride[i],
                                   non_linearity=args.non_linearity, n_space=n_space,
                                   frame_init_mode=args.f_init)

    ###########################################################################################
    # Create classifier
    ###########################################################################################

    classifier = Classifier(n_space, nb_channels_in, classifier_type=args.classifier_type,
                            nb_classes=nb_classes, nb_hidden_units=args.nb_hidden_units,
                            nb_l_mlp=args.nb_l_mlp, dropout_p_mlp=args.dropout_p_mlp,
                            avg_ker_size=args.avg_ker_size)

    # Create model
    ###########################################################################################
    for i in range(n_blocks):
        # scattering[i] and/or analysis[[i] can be nn.Identity()
        model[i] = ScatNetAnalysis(scattering[i], linear_proj[i], analysis[i],
                                   nn.Identity() if i < n_blocks - 1 else classifier)

    print_and_write('Model architecture: {}'.format(model), logfile, summaryfile)
    print_and_write(arch_log, logfile, summaryfile)
    if args.train_schedule == 'staged':
        msg = 'Number of epochs {} (A={}, B={}, C={}), learning rate decay epochs {}, learning rate {}'.format(
            args.epochs,
            args.phase_a_epochs,
            args.phase_b_epochs,
            args.phase_c_epochs,
            args.learning_rate_adjust_frequency,
            args.lr,
        )
        print_and_write(
            msg,
            logfile,
            summaryfile,
        )
    else:
        print_and_write(
            'Number of epochs {}, learning rate decay epochs {}, learning rate {}'.format(
                args.epochs,
                args.learning_rate_adjust_frequency,
                args.lr,
            ),
            logfile,
            summaryfile,
        )

    model = nn.Sequential(*model)
    return model


def main_worker(args):
    best_acc1 = 0
    best_acc5 = 0
    best_epoch_acc1 = 0
    best_epoch_acc5 = 0
    best_nc_metrics = {'nc1_trace': float('nan'), 'between_class_trace': float('nan'), 'nc1_rel': float('nan'), 'nc2': float('nan'), 'nc3': float('nan')}
    # CPU copy of the best-val weights for the end-of-training evaluation (kept in memory
    # because a run dir without the seed in its name can be shared by several seeds).
    best_state = None
    run_status = "failed"

    checkpoint_savedir = os.path.join('./checkpoints', args.dir)
    if not os.path.exists(checkpoint_savedir):
        os.makedirs(checkpoint_savedir, exist_ok=True)
    checkpoint_savefile = os.path.join(checkpoint_savedir, '{}_batchsize_{}_lrfreq_{}.pth.tar'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency))
    best_checkpoint_savefile = os.path.join(checkpoint_savedir,
                                            '{}_batchsize_{}_lrfreq_{}_best.pth.tar'.format(
                                                args.arch, args.batch_size, args.learning_rate_adjust_frequency))

    logs_dir = os.path.join('./training_logs', args.dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    logfile = open(os.path.join(logs_dir, 'training_{}_b_{}_lrfreq_{}.log'.format(
        args.arch, args.batch_size, args.learning_rate_adjust_frequency)), 'a')
    summaryfile = open(os.path.join(logs_dir, 'summary_file.txt'), 'a')
    writer = SummaryWriter(logs_dir)
    logger = ExperimentLogger(logs_dir, args)
    logger.set_phase(JOINT)

    # Also save args.
    with open(os.path.join(checkpoint_savedir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Data loading code
    ###########################################################################################
    print_and_write(f"Working on {args.dataset.upper()}", logfile, summaryfile)

    train_size = _parse_train_size(args.train_size)
    print_and_write(f"Parsed train size: {train_size}", logfile, summaryfile)
    split_mode = args.data_split_mode
    if split_mode == 'deep_scatter':
        _ensure_deep_scatter_available("--data-split-mode deep_scatter")

    train_dataset = None
    val_dataset = None
    test_dataset = None
    effective_nb_classes = None
    train_samples_before_split = None
    topk = (1,5)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        ds_cls = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100

        if split_mode == 'deep_scatter':
            cfg = DATASET_CONFIGS[args.dataset]
            train_tf, val_tf = _cifar_transforms(cfg['mean'], cfg['std'])         
            train_ds_aug = ds_cls(root='./data', train=True, download=False, transform=train_tf)
            train_ds_eval = ds_cls(root='./data', train=True, download=False, transform=val_tf)
            train_dataset, val_dataset, train_samples_before_split = _build_stratified_train_val(
                train_ds_aug, train_ds_eval, train_size, args.val_frac, args.seed or 0
            )
            # Use official test set for test_dataset
            test_dataset = ds_cls(root='./data', train=False, download=False, transform=val_tf)
        else:
            # Use helper to create proper train/val/test splits
            train_dataset, val_dataset, test_dataset, train_samples_before_split = _build_cifar_legacy_datasets(
                ds_cls, args.dataset, train_size, args.val_frac, args.seed or 0
            )

    elif args.dataset == 'mnist':
        if split_mode == 'deep_scatter':
            normalize = transforms.Normalize((0.1307,), (0.3081,))
            train_ds_aug = datasets.MNIST(root='./data', train=True, download=True,
                                          transform=transforms.Compose([transforms.ToTensor(), normalize]))
            train_ds_eval = datasets.MNIST(root='./data', train=True, download=True,
                                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
            train_dataset, val_dataset, train_samples_before_split = _build_stratified_train_val(
                train_ds_aug, train_ds_eval, train_size, args.val_frac, args.seed or 0
            )
            # Use official test set for test_dataset
            test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                          transform=transforms.Compose([transforms.ToTensor(), normalize]))
        else:
            # Use helper to create proper train/val/test splits
            train_dataset, val_dataset, test_dataset, train_samples_before_split = _build_mnist_legacy_datasets(
                train_size, args.val_frac, args.seed or 0
            )

    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        if args.imagenet_shards:
            train_ds_aug, train_ds_eval, official_val_ds = _build_imagenet_sharded_datasets(args)
        else:
            if split_mode == 'deep_scatter':
                _ensure_deep_scatter_available("--data-split-mode deep_scatter")
                train_tf, val_tf = _imagenet_transforms((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            else:
                train_tf = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
                val_tf = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
            train_ds_aug = datasets.ImageFolder(traindir, train_tf)
            train_ds_eval = datasets.ImageFolder(traindir, val_tf)
            official_val_ds = datasets.ImageFolder(valdir, val_tf)

        if args.nb_classes < 1000:
            if split_mode == 'deep_scatter':
                train_ds_aug, train_ds_eval, effective_nb_classes = _apply_imagenet_class_subset(
                    train_ds_aug, train_ds_eval, args, logs_dir, logfile, summaryfile
                )
            else:
                train_ds_aug, official_val_ds, effective_nb_classes = _apply_imagenet_class_subset(
                    train_ds_aug, official_val_ds, args, logs_dir, logfile, summaryfile
                )

        if split_mode == 'deep_scatter':
            train_dataset, val_dataset, train_samples_before_split = _build_stratified_train_val(
                train_ds_aug, train_ds_eval, train_size, args.val_frac, args.seed or 0
            )
            # Use official val as test dataset
            test_dataset = official_val_ds
        else:
            train_dataset = _maybe_subset_train(train_ds_aug, train_size, args.seed or 0)
            val_dataset = official_val_ds
            # Use same val_dataset as test_dataset for ImageNet (typical practice)
            test_dataset = official_val_ds

    elif args.dataset == 'dtd':
        # DTD dataset with k-fold cross-validation by partition
        partition = getattr(args, 'dtd_partition', 1)
        train_dataset, val_dataset, test_dataset, train_samples_before_split = _build_dtd_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, partition=partition
        )

    elif args.dataset == 'minc2500':
        # MINC2500 dataset (downloaded via KaggleHub)
        train_dataset, val_dataset, test_dataset, train_samples_before_split = _build_minc_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, data_dir=args.data
        )

    elif args.dataset == 'eurosat':
        # eurosat dataset, builtin from torchvision
        train_dataset, val_dataset, test_dataset, train_samples_before_split = _build_eurosat_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, data_dir=args.data
        )
    elif args.dataset == 'fmd':
        # fmd dataset, borrowed from deep_scatter
        train_dataset, val_dataset, test_dataset = _build_fmd_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, data_dir=args.data,
            input_size=getattr(args, 'input_size', None)
        )
        train_samples_before_split = len(train_dataset) + len(val_dataset)
    elif args.dataset == 'alot':
        # alot dataset, borrowed from deep_scatter
        train_dataset, val_dataset, test_dataset = _build_alot_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, data_dir=args.data,
            input_size=getattr(args, 'input_size', None)
        )
        train_samples_before_split = len(train_dataset) + len(val_dataset)
    elif args.dataset == 'surotex':
        # surotex dataset, borrowed from deep_scatter
        train_dataset, val_dataset, test_dataset = _build_surotex_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, data_dir=args.data,
            input_size=getattr(args, 'input_size', None)
        )
        train_samples_before_split = len(train_dataset) + len(val_dataset)
    elif args.dataset in ['pathmnist', 'octmnist']:
        # medmnist datasets, borrowed from deep_scatter
        train_dataset, val_dataset, test_dataset = _build_medmnist_legacy_datasets(
            train_size, args.val_frac, args.seed or 0, dataset_name=args.dataset, data_dir=args.data,
            input_size=getattr(args, 'input_size', None)
        )
        train_samples_before_split = len(train_dataset)  # official train/val splits
    elif args.dataset in ['so2sat_rgb', 'so2sat_s2']:
        # so2sat dataset, borrowed from deep_scatter
        train_dataset, val_dataset, test_dataset = _build_so2sat_legacy_datasets(
           train_size= train_size, seed=args.seed or 0, data_dir=args.data,
           dataset=args.dataset
        )
        train_samples_before_split = len(train_dataset)  # official train/val/test splits
    else:
        assert False

    if args.dataset in ['octmnist']:
        topk = (1,)  # octmnist has 4 classes

    args.in_channels, args.in_spatial, default_nb_classes = dataset_meta(args.dataset)
    # in_spatial sets the scattering shape, so it follows --input-size.
    if getattr(args, 'input_size', None):
        args.in_spatial = args.input_size
    args.nb_classes = effective_nb_classes if effective_nb_classes is not None else default_nb_classes

    # Re-snapshot args now that in_channels / in_spatial / nb_classes are resolved.
    logger.refresh_config(args)

    train_sampler = None
    val_sampler = None
    if args.dataset == 'imagenet' and args.imagenet_shards and args.imagenet_sampler == 'sharded':
        _ensure_deep_scatter_available("--imagenet-sampler sharded")
        if isinstance(_base_dataset(train_dataset), ImageNetTarDataset):
            train_sampler = ShardedSampler(train_dataset, num_replicas=1, rank=0, shuffle=True, seed=args.seed or 0)
        if isinstance(_base_dataset(val_dataset), ImageNetTarDataset):
            val_sampler = ShardedSampler(val_dataset, num_replicas=1, rank=0, shuffle=False, seed=args.seed or 0)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.workers,
        pin_memory=True,
    )

    if train_samples_before_split is None:
        train_samples_before_split = len(train_dataset)
    args.train_samples_before_val_split = int(train_samples_before_split)
    args.actual_train_samples = len(train_dataset)
    args.train_samples_optimized = len(train_dataset)
    args.val_samples = len(val_dataset)
    args.test_samples = len(test_dataset)

    print_and_write(
        f"Train samples before val split: {args.train_samples_before_val_split}, "
        f"optimized train samples: {args.train_samples_optimized}, val/eval samples: {args.val_samples}, "
        f"test samples: {args.test_samples}",
        logfile,
        summaryfile,
    )

    # Save args after data setup so derived metadata is persisted.
    with open(os.path.join(checkpoint_savedir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    n_params, n_proj_params, n_frame_params, n_classifier_params = compute_model_size(
        args, args.in_spatial, args.in_channels, args.nb_classes
    )
    init_logger_results(
        logger,
        world_size=1,
        actual_train_samples=args.actual_train_samples,
        val_samples=len(val_dataset),
        test_samples=len(test_dataset),
        in_channels=args.in_channels,
        in_spatial=args.in_spatial,
        nb_classes=args.nb_classes,
        effective_batch_size=args.batch_size,
        n_params_total=int(n_params),
        n_params_extractor=int(n_proj_params + n_frame_params),
        n_params_classifier=int(n_classifier_params),
    )
    logger.set_result("evaluation_split", "validation")
    logger.set_result("eval_samples", args.val_samples)
    logger.set_result("test_samples", args.test_samples)
    logger.set_result("train_samples_optimized", args.train_samples_optimized)
    logger.set_result("train_samples_before_val_split", args.train_samples_before_val_split)
    logger.set_result("train_schedule", args.train_schedule)
    logger.set_result("scattering_backend", args.scattering_backend)
    logger.set_result("wavelet_finetune", args.wavelet_finetune)
    logger.set_result("phase_a_epochs", args.phase_a_epochs)
    logger.set_result("phase_b_epochs", args.phase_b_epochs)
    logger.set_result("phase_c_epochs", args.phase_c_epochs)
    logger.set_result("l2_penalty", args.l2_penalty)
    ###########################################################################################

    model = load_model(args,  logfile, summaryfile)
    model = torch.nn.Sequential(*[torch.nn.DataParallel(submodel) for submodel in model]).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    all_named_groups = _collect_named_param_groups(model)
    drift_initial = {k: _snapshot_named_params(v) for k, v in all_named_groups.items()}
    drift_prev = {k: _snapshot_named_params(v) for k, v in all_named_groups.items()}

    # optionally resume from a checkpoint
    checkpoint = None
    resume_phase = JOINT
    resume_phase_epoch = 0
    resume_global_epoch = int(args.start_epoch)
    resume_optimizer_state = None
    if args.resume:
        if os.path.isfile(args.resume):
            print_and_write("=> loading checkpoint '{}'".format(args.resume), logfile, summaryfile)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc5 = checkpoint.get('best_acc5', best_acc5)
            best_epoch_acc1 = checkpoint.get('best_epoch_acc1', best_epoch_acc1)
            best_epoch_acc5 = checkpoint.get('best_epoch_acc5', best_epoch_acc5)
            model.load_state_dict(checkpoint['state_dict'])
            all_named_groups = _collect_named_param_groups(model)
            drift_initial = checkpoint.get('drift_initial_snapshot', drift_initial)
            drift_prev = checkpoint.get('drift_prev_snapshot', drift_prev)
            resume_phase = checkpoint.get('phase_id', JOINT)
            resume_phase_epoch = int(checkpoint.get('phase_epoch', 0))
            resume_global_epoch = int(checkpoint.get('global_epoch', args.start_epoch))
            resume_optimizer_state = checkpoint.get('optimizer')
            if 'log_history' in checkpoint:
                logger.load_state_dict(checkpoint['log_history'])
                # Keep the dataset-resolved args rather than the checkpoint's config snapshot.
                logger.refresh_config(args)
            print_and_write("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), logfile,
                            summaryfile)
        else:
            print_and_write("=> no checkpoint found at '{}'".format(args.resume), logfile, summaryfile)

    global_step = 0
    if args.resume and checkpoint is not None:
        global_step = int(checkpoint.get('global_step', 0))
    global_epoch = resume_global_epoch

    cudnn.benchmark = True

    print_model_info(args, model, logfile, summaryfile, phase_name=resume_phase)

    # Setup signal handlers and cleanup function early so they can be used by all code paths
    interrupted = False
    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        try:
            print_and_write("\n*** Training interrupted by signal ***", logfile, summaryfile)
        except:
            print("\n*** Training interrupted by signal ***")
    
    def cleanup_and_exit():
        """Save results and close resources gracefully on any exit"""
        nonlocal logfile, summaryfile, writer, logger
        try:
            logger.save()
        except Exception as e:
            try:
                print_and_write(f"Warning: could not save logger: {e}", logfile, summaryfile)
            except:
                pass
        try:
            writer.close()
        except Exception as e:
            try:
                print_and_write(f"Warning: could not close writer: {e}", logfile, summaryfile)
            except:
                pass
        try:
            if logfile is not None:
                logfile.close()
        except:
            pass
        try:
            if summaryfile is not None:
                summaryfile.close()
        except:
            pass

    signal.signal(signal.SIGINT, handle_interrupt)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_interrupt)  # SLURM timeout

    if args.evaluate:
        eval_optimizer, _ = _build_optimizer_for_phase(args, model, JOINT)
        print_and_write("Evaluating model at epoch {}...".format(args.start_epoch), logfile)
        logger.set_epoch(args.start_epoch)
        val_loss, val_acc1, val_acc5, global_step, _ = one_epoch(
            loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=eval_optimizer,
            epoch=args.start_epoch,
            args=args,
            logfile=logfile,
            summaryfile=summaryfile,
            writer=writer,
            is_training=False,
            logger=logger,
            global_step=global_step,
            phase_name="eval", phase_epoch=args.start_epoch,
            topk=topk
        )
        logger.log_epoch(args.start_epoch, val_loss, val_acc1, val_acc5, val_loss, val_acc1, val_acc5)
        finalize_joint_results(logger, val_acc1, val_acc5, args.start_epoch, args.start_epoch)
        run_status = "success"
        logger.set_result('run_status', run_status)
        cleanup_and_exit()
        return

    try:
        phase_items = [(name, epochs) for name, epochs in _phase_plan(args) if epochs > 0]
        phase_order = [name for name, _ in phase_items]
        resume_phase_idx = 0
        if resume_phase in phase_order:
            resume_phase_idx = phase_order.index(resume_phase)
        latest_delta_info = None
        final_phase_name = resume_phase
        previous_optimizer = None

        for idx, (phase_name, phase_epochs) in enumerate(phase_items):
            final_phase_name = phase_name
            if args.resume and idx < resume_phase_idx:
                continue

            if interrupted:
                print_and_write(f"Skipping remaining phases due to interruption", logfile, summaryfile)
                break

            logger.set_phase(phase_name)
        optimizer, phase_named_groups = _build_optimizer_for_phase(args, model, phase_name, previous_optimizer=previous_optimizer)
        if args.resume and idx == resume_phase_idx and resume_optimizer_state is not None:
            try:
                optimizer.load_state_dict(resume_optimizer_state)
            except Exception as exc:
                print_and_write(f"Warning: could not load optimizer state for {phase_name}: {exc}", logfile, summaryfile)

        phase_start_epoch = resume_phase_epoch if (args.resume and idx == resume_phase_idx) else 0
        filter_anchor_snapshot = None
        if phase_name == PHASE_C and args.l2_penalty > 0:
            filter_anchor_snapshot = _snapshot_named_params(all_named_groups['filter'])

        for phase_epoch in range(phase_start_epoch, phase_epochs):
            if interrupted:
                print_and_write(f"Breaking epoch loop due to interruption at epoch {global_epoch}", logfile, summaryfile)
                break

            epoch_start_time = time.time()
            logger.set_epoch(global_epoch)
            if args.train_schedule == 'legacy' and phase_name == JOINT:
                adjust_learning_rate(optimizer, global_epoch, args)

            train_loss, train_acc1, train_acc5, global_step, anchor_penalty_avg = one_epoch(
                loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=global_epoch,
                args=args,
                logfile=logfile,
                summaryfile=summaryfile,
                writer=writer,
                is_training=True,
                logger=logger,
                global_step=global_step,
                filter_named_params=all_named_groups['filter'],
                filter_anchor_snapshot=filter_anchor_snapshot,
                filter_anchor_lambda=(args.l2_penalty if phase_name == PHASE_C else 0.0),
                phase_name = phase_name, phase_epoch=phase_epoch,
                topk=topk
            )
            val_loss, acc1, acc5, global_step, _ = one_epoch(
                loader=val_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=global_epoch,
                args=args,
                logfile=logfile,
                summaryfile=summaryfile,
                writer=writer,
                is_training=False,
                logger=logger,
                global_step=global_step,
                phase_name = phase_name, phase_epoch=phase_epoch,
                topk=topk
            )
            
            # Compute neural collapse metrics on validation set
            # Note: Synchronize CUDA before try-except to catch async errors properly
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            try:
                nc_metrics = compute_neural_collapse_metrics(
                    model=model,
                    loader=val_loader,
                    n_blocks=args.n_blocks,
                    num_classes=args.nb_classes,
                    device='cuda'
                )
            except Exception as exc:
                print_and_write(f"Warning: could not compute neural collapse metrics for {phase_name} epoch {phase_epoch}: {exc}", logfile, summaryfile)
                nc_metrics = {'nc1_trace': float('nan'), 'between_class_trace': float('nan'), 'nc1_rel': float('nan'), 'nc2': float('nan'), 'nc3': float('nan')}

            # 'nc1_within_class_var_by_between_class' holds nc1_trace (within-class
            # variance); nc1_rel is the within/between ratio.
            if writer is not None:
                writer.add_scalar('neural_collapse/nc1_within_class_var_by_between_class', nc_metrics['nc1_trace'], global_step=global_epoch)
                writer.add_scalar('neural_collapse/nc1_rel_within_over_between', nc_metrics['nc1_rel'], global_step=global_epoch)
                writer.add_scalar('neural_collapse/nc2_between_class_sep', nc_metrics['nc2'], global_step=global_epoch)
                writer.add_scalar('neural_collapse/nc3_class_means_cosine_sim', nc_metrics['nc3'], global_step=global_epoch)
            
            
            p_init_l2, p_init_rel = _compute_delta(all_named_groups['p'], drift_initial['p'])
            p_prev_l2, p_prev_rel = _compute_delta(all_named_groups['p'], drift_prev['p'])
            f_init_l2, f_init_rel = _compute_delta(all_named_groups['f'], drift_initial['f'])
            f_prev_l2, f_prev_rel = _compute_delta(all_named_groups['f'], drift_prev['f'])
            filt_init_l2, filt_init_rel = _compute_delta(all_named_groups['filter'], drift_initial['filter'])
            filt_prev_l2, filt_prev_rel = _compute_delta(all_named_groups['filter'], drift_prev['filter'])
            # 'head' = every parameter that is not a P projection, F frame or scattering filter.
            head_init_l2, head_init_rel = _compute_delta(all_named_groups['head'], drift_initial['head'])
            head_prev_l2, head_prev_rel = _compute_delta(all_named_groups['head'], drift_prev['head'])
            extra = {
                'phase': phase_name,
                'phase_epoch': phase_epoch,
                'nc1': nc_metrics['nc1_trace'],  # within-class trace, not the ratio
                'nc1_rel': nc_metrics['nc1_rel'],  # within/between-class variance ratio
                'nc_between_class_trace': nc_metrics['between_class_trace'],
                'nc2': nc_metrics['nc2'],
                'nc3': nc_metrics['nc3'],
                'p_delta_l2': p_init_l2,
                'p_delta_relative': p_init_rel,
                'p_delta_prev_l2': p_prev_l2,
                'p_delta_prev_relative': p_prev_rel,
                'f_delta_l2': f_init_l2,
                'f_delta_relative': f_init_rel,
                'f_delta_prev_l2': f_prev_l2,
                'f_delta_prev_relative': f_prev_rel,
                'filter_delta_l2': filt_init_l2,
                'filter_delta_relative': filt_init_rel,
                'filter_delta_prev_l2': filt_prev_l2,
                'filter_delta_prev_relative': filt_prev_rel,
                'classifier_delta_l2': head_init_l2,
                'classifier_delta_relative': head_init_rel,
                'classifier_delta_prev_l2': head_prev_l2,
                'classifier_delta_prev_relative': head_prev_rel,
                'filter_anchor_penalty': anchor_penalty_avg,
            }
            logger.log_epoch(global_epoch, train_loss, train_acc1, train_acc5, val_loss, acc1, acc5, extra=extra)
            latest_delta_info = {
                'global_epoch': global_epoch,
                'phase_epoch': phase_epoch,
                'p_init_l2': p_init_l2,
                'p_init_rel': p_init_rel,
                'f_init_l2': f_init_l2,
                'f_init_rel': f_init_rel,
                'filt_init_l2': filt_init_l2,
                'filt_init_rel': filt_init_rel,
            }

            print_model_info(args, model, logfile, summaryfile, phase_name=phase_name, delta_info=latest_delta_info)

            logger.set_result('p_delta_l2', p_init_l2)
            logger.set_result('p_delta_relative', p_init_rel)
            logger.set_result('f_delta_l2', f_init_l2)
            logger.set_result('f_delta_relative', f_init_rel)
            logger.set_result('filter_delta_l2', filt_init_l2)
            logger.set_result('filter_delta_relative', filt_init_rel)
            logger.set_result('classifier_delta_l2', head_init_l2)
            logger.set_result('classifier_delta_relative', head_init_rel)


            drift_prev['p'] = _snapshot_named_params(all_named_groups['p'])
            drift_prev['f'] = _snapshot_named_params(all_named_groups['f'])
            drift_prev['filter'] = _snapshot_named_params(all_named_groups['filter'])
            drift_prev['head'] = _snapshot_named_params(all_named_groups['head'])

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch_acc1 = global_epoch
                best_nc_metrics = nc_metrics.copy()
                if args.best_ckpt_eval:
                    best_state = {k: v.detach().to('cpu', copy=True)
                                  for k, v in model.state_dict().items()}
            if acc5 > best_acc5:
                best_acc5 = acc5
                best_epoch_acc5 = global_epoch

            if phase_epoch % args.checkpoint_save_frequency == 0 or is_best:
                save_checkpoint({
                    'epoch': global_epoch + 1,
                    'global_epoch': global_epoch + 1,
                    'phase_id': phase_name,
                    'phase_epoch': phase_epoch + 1,
                    'phase_total_epochs': phase_epochs,
                    'has_filter_anchor_snapshot': filter_anchor_snapshot is not None,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'best_epoch_acc1': best_epoch_acc1,
                    'best_epoch_acc5': best_epoch_acc5,
                    'global_step': global_step,
                    'log_history': logger.payload(),
                    'optimizer': optimizer.state_dict(),
                    'drift_initial_snapshot': drift_initial,
                    'drift_prev_snapshot': drift_prev,
                }, is_best, checkpoint_filename=checkpoint_savefile, best_checkpoint_filename=best_checkpoint_savefile)

            global_epoch += 1

        # Phase A (frozen extractor, head-only) test evaluation, as in deep_scatter/train.py.
        if phase_name == PHASE_A:
            print_and_write("\nEvaluating Phase A (frozen extractor) on test set...", logfile, summaryfile)
            phase_a_test_loss, phase_a_test_acc1, phase_a_test_acc5, global_step, _ = one_epoch(
                loader=test_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=global_epoch,
                args=args,
                logfile=logfile,
                summaryfile=summaryfile,
                writer=writer,
                is_training=False,
                logger=logger,
                global_step=global_step,
                phase_name="phase_a_test",
                phase_epoch=0,
                topk=topk
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                phase_a_test_nc_metrics = compute_neural_collapse_metrics(
                    model=model,
                    loader=test_loader,
                    n_blocks=args.n_blocks,
                    num_classes=args.nb_classes,
                    device='cuda'
                )
            except Exception as exc:
                print_and_write(f"Warning: could not compute neural collapse metrics on Phase A test set: {exc}", logfile, summaryfile)
                phase_a_test_nc_metrics = {'nc1_trace': float('nan'), 'between_class_trace': float('nan'), 'nc1_rel': float('nan'), 'nc2': float('nan'), 'nc3': float('nan')}
            print_and_write(
                f"Phase A Test Results: Loss {phase_a_test_loss:.4f}, Acc@1 {phase_a_test_acc1:.2f}, Acc@5 {phase_a_test_acc5:.2f}",
                logfile, summaryfile)
            print_and_write(
                f"Phase A Test NC Metrics: NC1_trace={phase_a_test_nc_metrics['nc1_trace']:.6f}, "
                f"NC1_rel={phase_a_test_nc_metrics['nc1_rel']:.6f}, NC2={phase_a_test_nc_metrics['nc2']:.6f}, "
                f"NC3={phase_a_test_nc_metrics['nc3']:.6f}",
                logfile, summaryfile)
            logger.set_result('phase_a_test_loss', phase_a_test_loss)
            logger.set_result('phase_a_test_acc1', phase_a_test_acc1)
            logger.set_result('phase_a_test_acc5', phase_a_test_acc5)
            logger.set_result('phase_a_test_nc1', phase_a_test_nc_metrics['nc1_rel'])
            logger.set_result('phase_a_test_nc1_trace', phase_a_test_nc_metrics['nc1_trace'])
            logger.set_result('phase_a_test_between_class_trace', phase_a_test_nc_metrics['between_class_trace'])
            logger.set_result('phase_a_test_nc2', phase_a_test_nc_metrics['nc2'])
            logger.set_result('phase_a_test_nc3', phase_a_test_nc_metrics['nc3'])

        # Preserve optimizer momentum for next phase transition
        previous_optimizer = optimizer

    except Exception as e:
        print_and_write(f"\n*** Training interrupted or encountered error: {e} ***", logfile, summaryfile)
        import traceback
        print_and_write(traceback.format_exc(), logfile, summaryfile)
        # Determine failure reason
        if interrupted:
            run_status = "interrupted"
        else:
            run_status = "error"
        # Save partial results
        try:
            if 'best_acc1' in locals() and best_acc1 > 0:
                logger.set_result('interrupted', True)
                logger.set_result('interrupt_epoch', global_epoch if 'global_epoch' in locals() else 0)
                logger.set_result('best_acc1_at_interrupt', best_acc1)
                logger.set_result('run_status', run_status)
        except Exception as save_err:
            print_and_write(f"Could not save partial results: {save_err}", logfile, summaryfile)

    finally:
        # Ensure logger is saved before exiting (files will be closed at the very end)
        try:
            logger.save()
        except Exception as e:
            try:
                print_and_write(f"Warning: could not save logger: {e}", logfile, summaryfile)
            except:
                pass

    # If training was interrupted, skip post-training evaluation and exit cleanly
    if interrupted:
        print_and_write("\n*** Training was interrupted. Skipping post-training evaluation. ***", logfile, summaryfile)
        cleanup_and_exit()
        return

    print_model_info(args, model, logfile, summaryfile, phase_name=final_phase_name, delta_info=latest_delta_info)
    print_and_write(
        "Best top 1 accuracy {:.2f} at epoch {}, best top 5 accuracy {:.2f} at epoch {}".
        format(best_acc1, best_epoch_acc1, best_acc5, best_epoch_acc5), logfile, summaryfile)
    
    # NC metrics at the best epoch. '..._within_class_variance_by_between_class' holds
    # nc1_trace; 'best_nc1_rel' is the within/between ratio.
    logger.set_result('best_nc1_within_class_variance_by_between_class', best_nc_metrics['nc1_trace'])
    logger.set_result('best_nc1_rel', best_nc_metrics['nc1_rel'])
    logger.set_result('best_nc_between_class_trace', best_nc_metrics['between_class_trace'])
    logger.set_result('best_nc2_between_class_separation', best_nc_metrics['nc2'])
    logger.set_result('best_nc3_class_means_cosine_similarity', best_nc_metrics['nc3'])
    
    # Evaluate on test set once at the end of training
    print_and_write("\n" + "="*80, logfile, summaryfile)
    print_and_write("Evaluating on test set...", logfile, summaryfile)
    print_and_write("="*80, logfile, summaryfile)
    
    eval_optimizer, _ = _build_optimizer_for_phase(args, model, JOINT)
    test_loss, test_acc1, test_acc5, _, _ = one_epoch(
        loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=eval_optimizer,
        epoch=global_epoch,
        args=args,
        logfile=logfile,
        summaryfile=summaryfile,
        writer=writer,
        is_training=False,
        logger=logger,
        global_step=global_step,
        phase_name="test", 
        phase_epoch=0,
        topk=topk
    )
    
    # Compute neural collapse metrics on test set
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    try:
        test_nc_metrics = compute_neural_collapse_metrics(
            model=model,
            loader=test_loader,
            n_blocks=args.n_blocks,
            num_classes=args.nb_classes,
            device='cuda'
        )
    except Exception as exc:
        print_and_write(f"Warning: could not compute neural collapse metrics on test set: {exc}", logfile, summaryfile)
        test_nc_metrics = {'nc1_trace': float('nan'), 'between_class_trace': float('nan'), 'nc1_rel': float('nan'), 'nc2': float('nan'), 'nc3': float('nan')}

    # '..._within_class_var(iance)_by_between_class' names hold nc1_trace; the
    # 'nc1_rel' names hold the within/between ratio.
    if writer is not None:
        writer.add_scalar('test/loss', test_loss, global_step=global_epoch)
        writer.add_scalar('test/acc1', test_acc1, global_step=global_epoch)
        writer.add_scalar('test/acc5', test_acc5, global_step=global_epoch)
        writer.add_scalar('test_neural_collapse/nc1_within_class_var_by_between_class', test_nc_metrics['nc1_trace'], global_step=global_epoch)
        writer.add_scalar('test_neural_collapse/nc1_rel_within_over_between', test_nc_metrics['nc1_rel'], global_step=global_epoch)
        writer.add_scalar('test_neural_collapse/nc2_between_class_sep', test_nc_metrics['nc2'], global_step=global_epoch)
        writer.add_scalar('test_neural_collapse/nc3_class_means_cosine_sim', test_nc_metrics['nc3'], global_step=global_epoch)

    logger.set_result('test_loss', test_loss)
    logger.set_result('test_acc1', test_acc1)
    logger.set_result('test_acc5', test_acc5)
    logger.set_result('test_nc1_within_class_variance_by_between_class', test_nc_metrics['nc1_trace'])
    logger.set_result('test_nc1_rel', test_nc_metrics['nc1_rel'])
    logger.set_result('test_nc_between_class_trace', test_nc_metrics['between_class_trace'])
    logger.set_result('test_nc2_between_class_separation', test_nc_metrics['nc2'])
    logger.set_result('test_nc3_class_means_cosine_similarity', test_nc_metrics['nc3'])
    
    print_and_write(
        f"Test Results: Loss {test_loss:.4f}, Acc@1 {test_acc1:.2f}, Acc@5 {test_acc5:.2f}",
        logfile, summaryfile)
    print_and_write(
        f"Test NC Metrics: NC1_trace={test_nc_metrics['nc1_trace']:.6f}, NC1_rel={test_nc_metrics['nc1_rel']:.6f}, "
        f"NC2={test_nc_metrics['nc2']:.6f}, NC3={test_nc_metrics['nc3']:.6f}",
        logfile, summaryfile)
    print_and_write("="*80 + "\n", logfile, summaryfile)

    # =========================================================================
    # Best-val checkpoint evaluation (test + val)
    # =========================================================================
    # The block above scores the last-epoch weights. Here the best-val weights are
    # restored and scored on test and val, matching deep_scatter's train.py; keys are
    # defined in best_ckpt_eval. Weights come from the in-memory snapshot; the best
    # checkpoint file is only used for a resumed run whose val accuracy never improved,
    # and only if its bookkeeping matches this run.
    if args.best_ckpt_eval:
        print_and_write("\n" + "="*80, logfile, summaryfile)
        print_and_write("Evaluating BEST-VAL checkpoint on test and val splits...", logfile, summaryfile)
        print_and_write("="*80, logfile, summaryfile)

        best_state_source = None
        if best_state is not None:
            best_state_source = "in-memory snapshot (epoch {})".format(best_epoch_acc1)
        elif os.path.exists(best_checkpoint_savefile):
            # Resumed run that never beat its restored best_acc1.
            try:
                ck = torch.load(best_checkpoint_savefile, map_location='cpu')
                ck_best = ck.get('best_acc1')
                ck_epoch = ck.get('best_epoch_acc1')
                if ck_best is not None and abs(float(ck_best) - float(best_acc1)) < 1e-6 \
                        and (ck_epoch is None or int(ck_epoch) == int(best_epoch_acc1)):
                    best_state = ck['state_dict']
                    best_state_source = "best.pth.tar (reloaded; matches this run's best_acc1)"
                else:
                    print_and_write(
                        "Warning: {} reports best_acc1={} epoch={} but this run has {} / {}; "
                        "it belongs to a co-located seed. Skipping the best-checkpoint eval "
                        "rather than publishing another run's weights as this one's.".format(
                            best_checkpoint_savefile, ck_best, ck_epoch, best_acc1, best_epoch_acc1),
                        logfile, summaryfile)
            except Exception as exc:
                print_and_write("Warning: could not read {}: {}".format(
                    best_checkpoint_savefile, exc), logfile, summaryfile)

        if best_state is None:
            print_and_write(
                "Warning: no best-val weights available; best_ckpt_* metrics not recorded.",
                logfile, summaryfile)
            logger.set_result(bce.SOURCE_KEY, "unavailable (no attributable best-val weights)")
        else:
            model.load_state_dict(best_state)
            model.eval()
            logger.set_result(bce.VERSION_KEY, bce.BEST_CKPT_EVAL_VERSION)
            logger.set_result(bce.SOURCE_KEY, best_state_source)
            logger.set_result(bce.EPOCH_KEY, best_epoch_acc1)

            best_eval_optimizer, _ = _build_optimizer_for_phase(args, model, JOINT)
            for split_name, split_loader, n_split in (
                    ("test", test_loader, args.test_samples),
                    ("val", val_loader, args.val_samples)):
                bc_loss, bc_acc1, bc_acc5, _, _ = one_epoch(
                    loader=split_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=best_eval_optimizer,
                    epoch=global_epoch,
                    args=args,
                    logfile=logfile,
                    summaryfile=summaryfile,
                    writer=writer,
                    is_training=False,
                    logger=logger,
                    global_step=global_step,
                    phase_name="best_ckpt_{}".format(split_name),
                    phase_epoch=0,
                    topk=topk
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                try:
                    bc_nc = compute_neural_collapse_metrics(
                        model=model,
                        loader=split_loader,
                        n_blocks=args.n_blocks,
                        num_classes=args.nb_classes,
                        device='cuda'
                    )
                except Exception as exc:
                    print_and_write(
                        "Warning: could not compute neural collapse metrics on the best-val "
                        "checkpoint's {} split: {}".format(split_name, exc), logfile, summaryfile)
                    bc_nc = {'nc1_trace': float('nan'), 'between_class_trace': float('nan'),
                             'nc1_rel': float('nan'), 'nc1_old': float('nan'),
                             'nc2': float('nan'), 'nc3': float('nan')}

                akeys = bce.acc_keys(split_name)
                logger.set_result(akeys['acc1'], bc_acc1)
                logger.set_result(akeys['acc5'], bc_acc5)
                logger.set_result(akeys['loss'], bc_loss)
                logger.set_result(akeys['n_samples'], n_split)
                nkeys = bce.nc_keys(split_name)
                for field, key in nkeys.items():
                    logger.set_result(key, bc_nc[field])

                if writer is not None:
                    writer.add_scalar('best_ckpt_{}/loss'.format(split_name), bc_loss, global_step=global_epoch)
                    writer.add_scalar('best_ckpt_{}/acc1'.format(split_name), bc_acc1, global_step=global_epoch)
                    writer.add_scalar('best_ckpt_{}/acc5'.format(split_name), bc_acc5, global_step=global_epoch)

                print_and_write(
                    "Best-ckpt {} Results: Loss {:.4f}, Acc@1 {:.2f}, Acc@5 {:.2f} "
                    "(n={})".format(split_name, bc_loss, bc_acc1, bc_acc5, n_split),
                    logfile, summaryfile)
                print_and_write(
                    "Best-ckpt {} NC Metrics: NC1_trace={:.6f}, NC1_rel={:.6f}, NC2={:.6f}, "
                    "NC3={:.6f}".format(split_name, bc_nc['nc1_trace'], bc_nc['nc1_rel'],
                                        bc_nc['nc2'], bc_nc['nc3']),
                    logfile, summaryfile)

            # Gap between best_acc1 (max val over epochs) and best-ckpt test accuracy.
            bc_test = logger.payload().get('results', {}).get(bce.acc_keys('test')['acc1'])
            if bc_test is not None:
                print_and_write(
                    "best_acc1 (max val over epochs) {:.2f} - best-ckpt test acc1 {:.2f} = "
                    "{:+.2f} pts".format(best_acc1, bc_test, best_acc1 - bc_test),
                    logfile, summaryfile)
            print_and_write("="*80 + "\n", logfile, summaryfile)

    run_status = "success"
    logger.set_result('run_status', run_status)
    finalize_joint_results(logger, best_acc1, best_acc5, best_epoch_acc1, best_epoch_acc5)
    cleanup_and_exit()


@torch.no_grad()
def print_model_info(args, model, logfile, summaryfile, phase_name=None, delta_info=None, print_structure=True):
    if delta_info is not None and phase_name != PHASE_A:
        print_and_write(
            "Epoch {} [{}:{}] deltas | P: {:.4e} ({:.4e}) F: {:.4e} ({:.4e}) Filters: {:.4e} ({:.4e})".format(
                delta_info['global_epoch'],
                phase_name,
                delta_info['phase_epoch'],
                delta_info['p_init_l2'],
                delta_info['p_init_rel'],
                delta_info['f_init_l2'],
                delta_info['f_init_rel'],
                delta_info['filt_init_l2'],
                delta_info['filt_init_rel'],
            ),
            logfile,
            summaryfile,
        )

    if not print_structure:
        return

    n_blocks = len(model)

    for i in range(n_blocks):
        module = model[i].module

        if args.arch in ['analysis', 'scatnetblockanalysis'] and phase_name not in [PHASE_A, PHASE_C]:  # only analyze frames when they are trained
            count = 0
            for k in range(args.frame_width[i]):
                if module.analysis.frame_weight.data[k].norm(p=2) < module.analysis.frame_norm_mean - 0.01 \
                        or module.analysis.frame_weight.data[k].norm(p=2) > module.analysis.frame_norm_mean + 0.01:
                    count += 1

            if count == 0:
                print_and_write("frame {} atoms well normalized".format(i), logfile, summaryfile)
            else:
                print_and_write("{} frame {} atoms not well normalized".format(count, i), logfile,
                                summaryfile)
            frame_norms = module.analysis.frame_weight.data.norm(p=2, dim=(1, 2, 3))
            print_and_write("Max/min norms ratio {:.2f}, mean {:.2f} {} for frame {} atoms".
                            format(torch.max(frame_norms) / torch.min(frame_norms), torch.mean(frame_norms),
                                   '(norm ref {:.2f})'.format(module.analysis.frame_norm_mean),
                                   i), logfile, summaryfile)

            F_reshaped = module.analysis.frame_weight.data.reshape((module.analysis.frame_weight.shape[0], -1))
            N, C = F_reshaped.shape  # N: number of atoms, C input dimension
            if N > C:  # typical case, then better to compute F^T F (C x C)
                frame_singular_values = torch.linalg.eigvals(F_reshaped.t() @ F_reshaped)
            else:  # better to compute F F^T (N X N)
                frame_singular_values = torch.linalg.eigvals(F_reshaped @ F_reshaped.t())
            min_sing, max_sing = frame_singular_values[0], frame_singular_values[-1]
            print_and_write(
                "Parseval regularization: min {:.3f} and max {:.3f} singular values for frame {}".
                format(min_sing, max_sing, i), logfile)

            gram = torch.matmul(F_reshaped, F_reshaped.t())  # (N, N)
            idx = torch.triu_indices(*gram.shape, offset=1, device=gram.device)  # (2, N(N-1)/2)
            gram_triu = gram[idx[0], idx[1]].abs()
            print_and_write("Frame {} max coherence {:.3f}, median coherence {:.3f}".
                            format(i, gram_triu.max().item(), gram_triu.median().item()), logfile, summaryfile)

        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis'] and phase_name not in [PHASE_A, PHASE_C]:  # no proj for analysis
            lin_p = module.linear_proj.proj.weight.data
            lin_p_reshaped = lin_p.reshape((lin_p.shape[0], -1))
            lin_singular_values = torch.linalg.eigvals(lin_p_reshaped @ lin_p_reshaped.t())
            min_sing, max_sing = lin_singular_values[0], lin_singular_values[-1]
            print_and_write(
                "Parseval regularization: min {:.3f} and max {:.3f} singular values for linear projector {}".
                format(min_sing, max_sing, i), logfile, summaryfile)


def one_epoch(loader, model, criterion, optimizer, epoch, phase_name, phase_epoch, args, logfile, summaryfile, writer, is_training,
              logger=None, global_step=0, filter_named_params=None, filter_anchor_snapshot=None,
              filter_anchor_lambda=0.0,
              topk = (1,5)):
    batch_time = AverageMeter('Time', ':.1f')
    data_time = AverageMeter('Data', ':.1f')
    losses = AverageMeter('Loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.1f')
    top5 = AverageMeter('Acc@5', ':.1f')
    anchor_penalty_meter = AverageMeter('AnchorPenalty', ':.4f')
    name_epoch = "Train" if is_training else "Validation"
    progress = ProgressMeter(
        len(loader), [batch_time, data_time, losses, top1, top5],
        prefix="{} Epoch: [{}] ({} [{}])".format(name_epoch, epoch, phase_name, phase_epoch))

    n_blocks = len(model)

    if is_training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_training):
        end = time.time()
        for i, (input, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            for j in range(n_blocks):
                input = model[j](input, j=j)
            output = input

            loss = criterion(output, target)
            if loss.isnan():
                breakpoint()
            anchor_penalty = None
            if is_training and filter_anchor_lambda > 0 and filter_named_params:
                anchor_penalty = _phase_anchor_penalty(filter_named_params, filter_anchor_snapshot, filter_anchor_lambda)
                if anchor_penalty is not None:
                    loss = loss + anchor_penalty

            # measure accuracy and record loss
            if topk == (1,5):
                acc1, acc5 = accuracy(output, target, topk=topk)
            elif topk == (1,):
                acc1 = accuracy(output, target, topk=topk)[0]
                acc5 = torch.tensor(0.0).unsqueeze(0)  # dummy value when top5 is not computed
            else:
                raise ValueError("Unsupported topk value: {}".format(topk))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            if is_training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                if logger is not None:
                    lr = optimizer.param_groups[0]['lr']
                    logger.log_step(
                        global_step,
                        to_float(losses.avg),
                        to_float(top1.avg),
                        to_float(top5.avg),
                        to_float(lr),
                        grad_norm_l2(model),
                    )
                global_step += 1
                optimizer.step()
                # Parseval step
                pars_update(model, args)
                if anchor_penalty is not None:
                    anchor_penalty_meter.update(to_float(anchor_penalty), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print_and_write('\n', logfile)
                progress.display(i, logfile, print_and_write)

    # Print statistics summary
    logfiles = [logfile, summaryfile]
    if not is_training and epoch == 0:
        epoch_text = ' (First epoch)'
    elif not is_training and epoch == args.epochs - 1:
        epoch_text = ' (Final epoch)'
    elif not is_training and (epoch % args.learning_rate_adjust_frequency) == (args.learning_rate_adjust_frequency - 1):
        epoch_text = ' (before learning rate adjustment n° {})'.format(1 + epoch // args.learning_rate_adjust_frequency)
    else:
        epoch_text = ''
        logfiles = [logfile, None]
    print_and_write('\n{} Epoch {}{}, * Acc@1 {:.2f} Acc@5 {:.2f}'.
                    format(name_epoch, epoch, epoch_text, top1.avg, top5.avg), *logfiles)

    if writer is not None:
        suffix = "train" if is_training else "val"
        writer.add_scalar(f'top5_{suffix}', top5.avg, global_step=epoch)
        writer.add_scalar(f'top1_{suffix}', top1.avg, global_step=epoch)
        writer.add_scalar(f'loss_{suffix}', losses.avg, global_step=epoch)

    return to_float(losses.avg), to_float(top1.avg), to_float(top5.avg), global_step, to_float(anchor_penalty_meter.avg)


def compute_model_size(args, n_space, nb_channels=3, nb_classes=1000):
    nb_proj_params = 0
    nb_frame_params = 0
    nb_classifier_params = 0

    for i in range(args.n_blocks):
        # Compute proj parameters
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis']:
            if args.arch != 'scatnet':
                J = 1
                A = 4
                max_order = 1
            else:
                J = args.scattering_J
                A = 1
                max_order = 2
            L_ang = args.scat_angles[i]

            factor_channels = 1 + A * L_ang * J
            if max_order == 2:
                factor_channels += (L_ang ** 2) * J * (J - 1) // 2
            nb_channels *= factor_channels
            n_space = n_space // 2 ** J

            nb_proj_params += args.P_kernel_size[i] ** 2 * nb_channels * args.P_proj_size[i]
            nb_channels = args.P_proj_size[i]

        # Compute frame parameters
        if args.arch in ['analysis', 'scatnetblockanalysis']:
            nb_frame_params += args.frame_kernel_size[i] ** 2 * nb_channels * args.frame_width[i]

    # Compute classifier parameters
    if args.use_batchnorm:
        nb_classifier_params += 2 * nb_channels  # Batch norm
    if args.avg_ker_size > 1:
        n = n_space - args.avg_ker_size + 1
    else:
        n = n_space

    in_planes = nb_channels * (n ** 2)
    if args.classifier_type == 'fc':
        nb_classifier_params += in_planes * nb_classes
    else: # mlp
        nb_classifier_params += in_planes * args.nb_hidden_units
        for i in range(args.nb_l_mlp - 1):
            nb_classifier_params += args.nb_hidden_units**2
        nb_classifier_params += args.nb_hidden_units * nb_classes


    nb_params = nb_classifier_params + nb_proj_params + nb_frame_params
    return nb_params, nb_proj_params, nb_frame_params, nb_classifier_params


@torch.no_grad()
def pars_update(model, args):
    beta = args.pars_beta

    def update(frame_param):
        frame = frame_param.data.reshape((frame_param.shape[0], -1))
        N, C = frame.shape  # N: number of atoms, C input dimension

        # F = (1 + beta) F - beta F F^T F ; F F^T is N x N, F^T F is C x C
        if N > C:  # typical case, better to compute F^T F (C x C)
            prod = torch.matmul(frame, torch.matmul(frame.t(), frame))
        else:  # better to compute F F^T (N X N)
            prod = torch.matmul(torch.matmul(frame, frame.t()), frame)

        frame_param.data = ((1 + beta) * frame - beta * prod).reshape(frame_param.shape)

    for i in range(len(model)):
        if args.arch in ['analysis', 'scatnetblockanalysis'] and args.parseval_f:
            frame_weight = model[i].module.analysis.frame_weight
            if frame_weight.requires_grad:
                update(frame_weight)
        if args.arch in ['scatnet', 'scatnetblock', 'scatnetblockanalysis'] and args.parseval_p:
            proj_weight = model[i].module.linear_proj.proj.weight
            if proj_weight.requires_grad:
                update(proj_weight)


def print_and_write(log, *logfiles):
    """ Prints log (a string) and writes it to logfiles, which should have a write method (open files) or None. """
    print(log)
    for logfile in logfiles:
        if logfile is not None:
            logfile.write(log + '\n')


if __name__ == '__main__':
    main()
