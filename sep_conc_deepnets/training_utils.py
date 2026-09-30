import math
import warnings

import torch

from experiment_logging import PHASE_A, PHASE_B, PHASE_C, JOINT


def _phase_lr(args, phase, group_name):
    value = getattr(args, f"phase_{phase}_lr_{group_name}", None)
    return args.lr if value is None else value


def _normalize_validate_args(args):
    if args.scattering_backend == 'deep_scatter' and args.arch == 'scatnet':
        raise ValueError("--scattering-backend deep_scatter is not supported for --arch scatnet")

    if args.phase_a_epochs is None:
        args.phase_a_epochs = args.epochs

    for key in ["phase_a_epochs", "phase_b_epochs", "phase_c_epochs"]:
        if getattr(args, key) < 0:
            raise ValueError(f"--{key.replace('_', '-')} must be >= 0")

    if args.train_schedule == 'legacy':
        args.phase_a_epochs = args.epochs
        args.phase_b_epochs = 0
        args.phase_c_epochs = 0
    else:
        if args.phase_a_epochs <= 0:
            raise ValueError("staged schedule requires --phase-a-epochs > 0")
        if args.phase_c_epochs > 0 and args.scattering_backend not in ['deep_scatter', 'deep_scatter_4L', 'legacy_trainable']:
            raise ValueError("phase-C filter finetune requires --scattering-backend to be deep_scatter, deep_scatter_4L, or legacy_trainable")
        if args.phase_c_epochs > 0 and not args.wavelet_finetune:
            raise ValueError("phase-C filter finetune requires --wavelet-finetune")
        if args.l2_penalty > 0 and args.phase_c_epochs <= 0:
            raise ValueError("--l2-penalty > 0 requires --phase-c-epochs > 0")

        # In staged mode, keep one source of truth for total training epochs.
        phased_total = args.phase_a_epochs + args.phase_b_epochs + args.phase_c_epochs
        if args.epochs != phased_total:
            warnings.warn(
                f"Overriding --epochs={args.epochs} with staged total {phased_total} "
                f"(A={args.phase_a_epochs}, B={args.phase_b_epochs}, C={args.phase_c_epochs})."
            )
        args.epochs = phased_total

    if args.wavelet_finetune and args.scattering_backend not in ['deep_scatter', 'deep_scatter_4L', 'legacy_trainable']:
        raise ValueError("--wavelet-finetune is only supported with --scattering-backend deep_scatter, deep_scatter_4L, or legacy_trainable")


def _collect_named_param_groups(model):
    groups = {
        'p': [],
        'f': [],
        'filter': [],
        'head': [],
    }
    for name, param in model.named_parameters():
        if '.linear_proj.proj.weight' in name:
            groups['p'].append((name, param))
        elif '.analysis.frame_weight' in name:
            groups['f'].append((name, param))
        elif '.scattering.wavelets.' in name or \
             ('.scattering.' in name and ('.phis' in name or '.psis' in name)) or \
             ('.scattering.' in name and 'tensor' in name):
            # Deep scatter uses .wavelets., legacy_trainable uses .phis/.psis, Scattering2DTrainable uses tensorN
            groups['filter'].append((name, param))
        else:
            groups['head'].append((name, param))
    return groups


def _set_trainable(named_params, enabled):
    for _, param in named_params:
        param.requires_grad = enabled


def _snapshot_named_params(named_params):
    return {name: param.detach().cpu().clone() for name, param in named_params}


def _compute_delta(named_params, baseline_snapshot):
    """L2 drift of ``named_params`` from ``baseline_snapshot``.

    Returns ``(nan, nan)`` when there is nothing to measure -- an absent parameter
    group (e.g. no F frame in this architecture), a missing snapshot, or a snapshot
    that shares no names with the model. Zero is a *real* measurement here (a frozen
    group genuinely does not move), so "not measured" must not collapse onto it: the
    generalization bound reads drift as its r term and would silently report a
    degenerate zero instead of an honest NaN.
    """
    if not named_params or baseline_snapshot is None:
        return float('nan'), float('nan')
    delta_sq = 0.0
    norm_sq = 0.0
    matched = 0
    for name, param in named_params:
        if name not in baseline_snapshot:
            continue
        before = baseline_snapshot[name].to(param.device)
        diff = (param.detach() - before)
        delta_sq += float(torch.sum(diff * diff).item())
        norm_sq += float(torch.sum(before * before).item())
        matched += 1
    if matched == 0:
        return float('nan'), float('nan')
    delta_l2 = delta_sq ** 0.5
    baseline_l2 = norm_sq ** 0.5
    relative = delta_l2 / max(baseline_l2, 1e-12)
    return delta_l2, relative


def _phase_anchor_penalty(named_params, anchor_snapshot, lam):
    if lam <= 0 or anchor_snapshot is None:
        return None
    penalty = None
    for name, param in named_params:
        if name not in anchor_snapshot:
            continue
        anchor = anchor_snapshot[name].to(param.device)
        term = (param - anchor).pow(2).sum()
        penalty = term if penalty is None else (penalty + term)
    if penalty is None:
        return None
    return lam * penalty


def _phase_plan(args):
    if args.train_schedule == 'legacy':
        return [(JOINT, args.epochs)]
    return [
        (PHASE_A, args.phase_a_epochs),
        (PHASE_B, args.phase_b_epochs),
        (PHASE_C, args.phase_c_epochs),
    ]


def _build_optimizer_for_phase(args, model, phase_name, previous_optimizer=None):
    groups = _collect_named_param_groups(model)
    head_trainable_bc = not getattr(args, "freeze_head_bc", False)
    pf_trainable_c = not getattr(args, "freeze_pf_c", False)

    if phase_name == JOINT:
        _set_trainable(groups['p'], True)
        _set_trainable(groups['f'], True)
        _set_trainable(groups['filter'], args.wavelet_finetune)
        _set_trainable(groups['head'], True)
        # Keep legacy behavior isolated from staged per-phase LR knobs.
        lr_head = args.lr
        lr_pf = args.lr
        lr_filter = _phase_lr(args, 'c', 'filter')
    elif phase_name == PHASE_A:
        _set_trainable(groups['p'], False)
        _set_trainable(groups['f'], False)
        _set_trainable(groups['filter'], False)
        _set_trainable(groups['head'], True)
        lr_head = _phase_lr(args, 'a', 'head')
        lr_pf = args.lr  # Define for consistency, won't be used
        lr_filter = args.lr  # Define for consistency, won't be used
    elif phase_name == PHASE_B:
        _set_trainable(groups['p'], True)
        _set_trainable(groups['f'], True)
        _set_trainable(groups['filter'], False)
        _set_trainable(groups['head'], head_trainable_bc)
        lr_head = _phase_lr(args, 'b', 'head')
        lr_pf = _phase_lr(args, 'b', 'pf')
        lr_filter = args.lr  # Define for consistency, won't be used
    elif phase_name == PHASE_C:
        _set_trainable(groups['p'], pf_trainable_c)
        _set_trainable(groups['f'], pf_trainable_c)
        _set_trainable(groups['filter'], args.wavelet_finetune)
        _set_trainable(groups['head'], head_trainable_bc)
        lr_head = _phase_lr(args, 'c', 'head')
        lr_pf = _phase_lr(args, 'c', 'pf')
        lr_filter = _phase_lr(args, 'c', 'filter')
    else:
        raise ValueError(f"Unsupported phase: {phase_name}")

    param_groups = []
    if groups['head']:
        param_groups.append({'params': [p for _, p in groups['head'] if p.requires_grad], 'lr': lr_head})
    pf_params = [p for _, p in groups['p'] + groups['f'] if p.requires_grad]
    if pf_params:
        param_groups.append({'params': pf_params, 'lr': lr_pf, 'weight_decay': 0.0})
    filter_params = [p for _, p in groups['filter'] if p.requires_grad]
    if filter_params:
        param_groups.append({'params': filter_params, 'lr': lr_filter, 'weight_decay': 0.0})

    if not param_groups:
        raise RuntimeError(f"No trainable parameters configured for phase {phase_name}")

    optimizer = torch.optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    
    # Preserve momentum state across phase boundaries when reusing the same optimizer
    if previous_optimizer is not None and hasattr(previous_optimizer, 'state'):
        for param_group, old_param_group in zip(optimizer.param_groups, previous_optimizer.param_groups):
            for param in param_group['params']:
                # Only restore state for params that existed in previous optimizer
                for old_param in old_param_group['params']:
                    if param is old_param and param in previous_optimizer.state:
                        optimizer.state[param] = previous_optimizer.state[param]
                        break
    
    return optimizer, groups


def save_checkpoint(state, is_best, checkpoint_filename='checkpoint.pth.tar',
                    best_checkpoint_filename='model_best.pth.tar'):
    torch.save(state, checkpoint_filename)
    if is_best:
        import shutil
        shutil.copyfile(checkpoint_filename, best_checkpoint_filename)


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
# Identical to deep_scatter/train.py.

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


def _resolve_head(model, n_blocks):
    """``(classifier_module, linear)`` of an SCD model, either possibly ``None``.

    The model is an ``nn.Sequential`` of blocks whose last element (possibly DataParallel-
    wrapped) is a ``ScatNetAnalysis``; its ``.classifier`` is the ``Classifier`` module
    (bn -> optional avg_pool -> Linear or MLP) and *that* module's ``.classifier`` is the
    Linear. ``linear`` is ``None`` for an MLP head, which has no single weight matrix.
    """
    last = model[n_blocks - 1] if isinstance(model, torch.nn.Sequential) else model
    last = getattr(last, 'module', last)
    head = getattr(last, 'classifier', None)
    head = getattr(head, 'module', head)
    linear = getattr(head, 'classifier', None)
    if not isinstance(linear, torch.nn.Linear):
        linear = None
    return head, linear


@torch.no_grad()
def compute_neural_collapse_metrics(model, loader, n_blocks, num_classes, device='cuda'):
    """Neural collapse statistics of the pre-classifier features.

    Args:
        model: Sequential model with n_blocks
        loader: DataLoader for computing features
        n_blocks: Number of blocks in the model
        num_classes: Number of classes
        device: Device to use
        
    Returns:
        Dictionary of NC metrics (element-wise ``nc1_old`` and trace-based ``nc1_rel``, ...).
    """
    model.eval()

    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # All blocks but the last, then the last block's pre-classifier features.
            for j in range(n_blocks - 1):
                inputs = model[j](inputs, j=j)

            if hasattr(model[n_blocks - 1], 'module'):
                last_block = model[n_blocks - 1].module
            else:
                last_block = model[n_blocks - 1]
                
            inputs = last_block.extract_features(inputs)
            
            feat_shape = tuple(inputs.shape[1:])  # pre-flatten, for the head Jacobian
            all_features.append(inputs.detach().cpu())
            all_labels.append(targets.detach().cpu())
    
    features = torch.cat(all_features, dim=0)  # (N, C, H, W) or (N, d)
    labels = torch.cat(all_labels, dim=0)      # (N,)
    
    if features.dim() > 2:
        features = features.reshape(features.shape[0], -1)  # (N, d)

    class_means = []
    within_class_vars_old = []
    total_within_class_trace = 0.0
    valid_classes = 0
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        
        valid_classes += 1
        class_features = features[mask]  # (n_c, d)
        class_mean = class_features.mean(dim=0, keepdim=True)  # (1, d)
        class_means.append(class_mean)
        
        centered = class_features - class_mean
        within_class_vars_old.append(float((centered ** 2).mean()))
        total_within_class_trace += torch.norm(centered, p=2, dim=1).pow(2).mean().item()
    
    if not class_means:
        return {
            'nc1_old': float('nan'),
            'nc1_per_class_old': [],
            'nc1_trace': float('nan'), 
            'between_class_trace': float('nan'),
            'nc1_rel': float('nan'), 
            'nc2': float('nan'), 
            'nc3': float('nan'),
            'nc1_papyan': float('nan'),
            **_nan_skew(),
        }
    
    # Element-wise within-class variance.
    nc1_old = float(torch.tensor(within_class_vars_old).mean())

    # NC1: average within-class trace, and its ratio to the between-class trace.
    nc1_trace = total_within_class_trace / valid_classes
    class_means = torch.cat(class_means, dim=0)  # (num_classes, d)
    global_mean = class_means.mean(dim=0, keepdim=True)  # (1, d)
    centered_class_means = class_means - global_mean
    between_class_trace = torch.norm(centered_class_means, p=2, dim=1).pow(2).mean().item()
    nc1_rel = nc1_trace / (between_class_trace + 1e-8)
    nc1_papyan = _papyan_nc1(features, labels, num_classes)

    # NC2: mean cosine similarity of centered class means
    class_means_norm = centered_class_means / (torch.norm(centered_class_means, dim=1, keepdim=True) + 1e-8)
    cosine_sims = torch.mm(class_means_norm, class_means_norm.t())
    mask_tri = torch.triu(torch.ones_like(cosine_sims), diagonal=1).bool()
    nc2_etf_coherence = float(cosine_sims[mask_tri].mean()) if mask_tri.sum() > 0 else 0.0
    
    # NC3: classifier weight / class mean alignment
    nc3_alignment = float('nan')
    try:
        classifier_layer = None
        if hasattr(model, 'classifier'):
            classifier_layer = model.classifier
        elif hasattr(model, 'fc'):
            classifier_layer = model.fc
        elif isinstance(model, torch.nn.Sequential):
            classifier_layer = model[-1]
        # model[-1] is a ScatNetAnalysis; its Linear is two levels further down.
        if classifier_layer is None or not hasattr(classifier_layer, 'weight'):
            classifier_layer = _resolve_head(model, n_blocks)[1]
            
        if classifier_layer is not None and hasattr(classifier_layer, 'weight'):
            W = classifier_layer.weight.detach().cpu()
            if hasattr(classifier_layer, 'module'):
                W = classifier_layer.module.weight.detach().cpu()
                
            W_centered = W - W.mean(dim=0, keepdim=True)
            W_norm = W_centered / (torch.norm(W_centered, dim=1, keepdim=True) + 1e-8)
            nc3_alignment = torch.sum(W_norm * class_means_norm, dim=1).mean().item()
    except Exception:
        pass  # non-standard classifier: NaN

    # Skewed NC1; head = Classifier module (bn -> optional avg_pool -> Linear).
    # MLP heads are not affine and give NaN.
    try:
        head, linear = _resolve_head(model, n_blocks)
        if head is None or linear is None:
            skew = _nan_skew()
        else:
            W_eff = _effective_head_weight(head, feat_shape, device, dtype=features.dtype)
            skew = _skewed_nc(features, labels, W_eff, num_classes)
    except Exception as e:
        warnings.warn(f"skewed NC unavailable: {type(e).__name__}: {e}")
        skew = _nan_skew()
        
    return {
        'nc1_old': nc1_old,
        'nc1_per_class_old': within_class_vars_old,
        'nc1_trace': nc1_trace,
        'between_class_trace': between_class_trace,
        'nc1_rel': nc1_rel,
        'nc2': nc2_etf_coherence,
        'nc3': nc3_alignment,
        'nc1_papyan': nc1_papyan,
        **skew,
    }


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logfile, print_and_write_fn):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_and_write_fn('\t'.join(entries), logfile)

    def add(self, *meters):
        for meter in meters:
            self.meters.append(meter)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets learning rate to initial LR decayed by 10 every adjust-frequency epochs."""
    lr = args.lr * (0.1 ** (epoch // args.learning_rate_adjust_frequency))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes accuracy over the k top predictions for specified k values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
