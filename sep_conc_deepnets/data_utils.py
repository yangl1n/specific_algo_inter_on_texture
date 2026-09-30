import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
deep_scatter_data_path = project_root / 'deep_scatter' / 'data'

DEEP_SCATTER_AVAILABLE = True
DEEP_SCATTER_IMPORT_ERROR = None
try:
    from deep_scatter.data import (
        DATASET_CONFIGS,
        ImageNetTarDataset,
        _imagenet_transforms,
        _load_tar_catalog,
        stratified_subset,
        stratified_train_val_split,
        _build_ImageFolder_dataset,
        _build_medmnist_dataset,
        _build_so2sat_hf_dataset,
    )
except Exception as exc:  # deep_scatter dependencies unavailable
    DEEP_SCATTER_AVAILABLE = False
    DEEP_SCATTER_IMPORT_ERROR = str(exc)
    DATASET_CONFIGS = {}
    ImageNetTarDataset = None
    _imagenet_transforms = None
    _load_tar_catalog = None
    stratified_subset = None
    stratified_train_val_split = None


# MINC2500 constants
MINC2500_DATASET_ID = "liewyousheng/minc2500"
MINC2500_DEEPER_PATH = "minc-2500/images"
MINC2500_TEST_FRAC = 0.15


def print_and_write(log, *logfiles):
    """Prints log (a string) and writes it to logfiles, which should have a write method (open files) or None."""
    print(log)
    for logfile in logfiles:
        if logfile is not None:
            logfile.write(log + '\n')


def _ensure_deep_scatter_available(feature_name):
    """Raise error if deep_scatter is not available."""
    if DEEP_SCATTER_AVAILABLE:
        return
    raise RuntimeError(
        f"{feature_name} requires deep_scatter.data utilities, but import failed: {DEEP_SCATTER_IMPORT_ERROR}"
    )



class RemappedTargetsDataset(torch.utils.data.Dataset):
    """Wrap a dataset and remap labels to a compact contiguous label space."""

    def __init__(self, dataset, label_map):
        self.dataset = dataset
        self.label_map = {int(k): int(v) for k, v in label_map.items()}
        base_targets = _extract_targets(dataset)
        self.targets = np.asarray([self.label_map[int(t)] for t in base_targets], dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.label_map[int(y)]


def _parse_train_size(train_size_raw):
    if train_size_raw in (None, "None", "none", ""):
        return None
    token = str(train_size_raw).strip()
    if "." in token:
        value = float(token)
        if value <= 0:
            raise ValueError("--train-size as fraction must be > 0")
        return value
    value = int(token)
    if value <= 0:
        raise ValueError("--train-size as count must be > 0")
    return value


def _extract_targets(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base_targets = _extract_targets(dataset.dataset)
        return np.asarray(base_targets)[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, 'targets'):
        return np.asarray(dataset.targets)
    if hasattr(dataset, 'labels'):
        return np.asarray(dataset.labels)
    raise AttributeError(f"Dataset {type(dataset).__name__} has no targets/labels")


def _build_stratified_train_val(train_ds_aug, train_ds_eval, train_size, val_frac, seed):
    _ensure_deep_scatter_available("Stratified train/val split")
    subset = stratified_subset(train_ds_aug, train_size, seed=seed) if train_size is not None else train_ds_aug
    train_idx, val_idx = stratified_train_val_split(subset, val_frac=val_frac, seed=seed)
    train_ds = torch.utils.data.Subset(train_ds_aug, train_idx)
    val_ds = torch.utils.data.Subset(train_ds_eval, val_idx)
    return train_ds, val_ds, len(subset)


def _maybe_subset_train(train_dataset, train_size, seed):
    if train_size is None:
        return train_dataset
    _ensure_deep_scatter_available("--train-size")
    return stratified_subset(train_dataset, train_size, seed=seed)


def _filter_dataset_by_classes(dataset, class_indices):
    targets = _extract_targets(dataset)
    class_indices = np.asarray(class_indices, dtype=np.int64)
    selected = np.where(np.isin(targets, class_indices))[0].tolist()
    return torch.utils.data.Subset(dataset, selected)


def _apply_imagenet_class_subset(train_dataset, val_dataset, args, logs_dir, logfile, summaryfile):
    if args.dataset != 'imagenet' or args.nb_classes >= 1000:
        return train_dataset, val_dataset, None

    classes_names = torch.load('utils_sampling/labels_frame')
    if args.class_indices is not None:
        class_indices = torch.load(args.class_indices)
    else:
        perm = torch.randperm(1000)
        class_indices = perm[:args.nb_classes].tolist()

    class_indices = sorted({int(i) for i in class_indices})
    if len(class_indices) != args.nb_classes:
        raise ValueError(
            f"Expected {args.nb_classes} unique class ids, got {len(class_indices)} from --class-indices"
        )

    selected_classes_names = [classes_names[i] for i in class_indices]
    classes_indices_file = os.path.join(logs_dir, 'classes_indices_selected')
    torch.save(class_indices, classes_indices_file)

    print_and_write('Selected {} classes indices: {}'.format(args.nb_classes, class_indices), logfile, summaryfile)
    print_and_write('Selected {} classes names: {}'.format(args.nb_classes, selected_classes_names), logfile,
                    summaryfile)
    if args.seed is not None:
        print_and_write('Random seed used {}'.format(args.seed), logfile, summaryfile)

    train_dataset = _filter_dataset_by_classes(train_dataset, class_indices)
    val_dataset = _filter_dataset_by_classes(val_dataset, class_indices)
    label_map = {cls: idx for idx, cls in enumerate(class_indices)}
    train_dataset = RemappedTargetsDataset(train_dataset, label_map)
    val_dataset = RemappedTargetsDataset(val_dataset, label_map)
    return train_dataset, val_dataset, len(class_indices)


def _base_dataset(dataset):
    while isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    if hasattr(dataset, 'dataset'):
        return _base_dataset(dataset.dataset)
    return dataset


def _build_imagenet_sharded_datasets(args):
    _ensure_deep_scatter_available("--imagenet-shards")
    imagenet_root = args.data
    valdir = os.path.join(imagenet_root, 'val')
    train_tf, val_tf = _imagenet_transforms((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_ds = datasets.ImageFolder(valdir, transform=val_tf)

    tar_paths, path_ids, offsets, sizes, class_ids = _load_tar_catalog(imagenet_root)
    train_eval_tf = val_tf
    train_ds_aug = ImageNetTarDataset(
        tar_paths, path_ids, offsets, sizes, class_ids, train_tf, tar_cache_size=args.tar_cache_size
    )
    train_ds_eval = ImageNetTarDataset(
        tar_paths, path_ids, offsets, sizes, class_ids, train_eval_tf, tar_cache_size=args.tar_cache_size
    )
    return train_ds_aug, train_ds_eval, test_ds


def _build_cifar_legacy_datasets(ds_cls, dataset_name, train_size, val_frac, seed):
    """Build CIFAR train/val/test datasets in legacy mode.
    
    Creates validation split from training data and uses official test set.
    Uses stratified split if deep_scatter is available, otherwise random split.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Load training data with augmentation
    train_ds_aug = ds_cls(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize,
    ]))
    
    # Load training data without augmentation (for validation sampling)
    train_ds_eval = ds_cls(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    
    # Load official test set
    test_dataset = ds_cls(root='./data', train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    # Create train/val split from training data
    if DEEP_SCATTER_AVAILABLE:
        # Use stratified split if available
        subset = stratified_subset(train_ds_aug, train_size, seed=seed) if train_size is not None else train_ds_aug
        train_idx, val_idx = stratified_train_val_split(subset, val_frac=val_frac, seed=seed)
        train_dataset = torch.utils.data.Subset(train_ds_aug, train_idx)
        val_dataset = torch.utils.data.Subset(train_ds_eval, val_idx)
        train_samples_before_split = len(subset)
    else:
        # Use random split if deep_scatter not available
        n_train = len(train_ds_aug)
        if train_size is not None:
            if isinstance(train_size, float):
                n_train = int(n_train * train_size)
            else:
                n_train = int(train_size)
        n_val = int(n_train * val_frac)
        n_train = n_train - n_val
        
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(train_ds_aug))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        
        train_dataset = torch.utils.data.Subset(train_ds_aug, train_idx)
        val_dataset = torch.utils.data.Subset(train_ds_eval, val_idx)
        train_samples_before_split = n_train + n_val
    
    return train_dataset, val_dataset, test_dataset, train_samples_before_split


def _build_mnist_legacy_datasets(train_size, val_frac, seed):
    """Build MNIST train/val/test datasets in legacy mode.
    
    Creates validation split from training data and uses official test set.
    Uses stratified split if deep_scatter is available, otherwise random split.
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    
    # Load training data with augmentation
    train_ds_aug = datasets.MNIST(root='./data', train=True, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    # Load training data without augmentation (for validation sampling)
    train_ds_eval = datasets.MNIST(root='./data', train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    # Load official test set
    test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    # Create train/val split from training data
    if DEEP_SCATTER_AVAILABLE:
        # Use stratified split if available
        subset = stratified_subset(train_ds_aug, train_size, seed=seed) if train_size is not None else train_ds_aug
        train_idx, val_idx = stratified_train_val_split(subset, val_frac=val_frac, seed=seed)
        train_dataset = torch.utils.data.Subset(train_ds_aug, train_idx)
        val_dataset = torch.utils.data.Subset(train_ds_eval, val_idx)
        train_samples_before_split = len(subset)
    else:
        # Use random split if deep_scatter not available
        n_train = len(train_ds_aug)
        if train_size is not None:
            if isinstance(train_size, float):
                n_train = int(n_train * train_size)
            else:
                n_train = int(train_size)
        n_val = int(n_train * val_frac)
        n_train = n_train - n_val
        
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(train_ds_aug))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        
        train_dataset = torch.utils.data.Subset(train_ds_aug, train_idx)
        val_dataset = torch.utils.data.Subset(train_ds_eval, val_idx)
        train_samples_before_split = n_train + n_val
    
    return train_dataset, val_dataset, test_dataset, train_samples_before_split


def _dtd_transforms(mean, std):
    """Transform pipeline for DTD (minimal augmentation, similar to ImageNet eval)."""
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((224, 224)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((224, 224)),
    ])
    return train_tf, val_tf


def _build_dtd_legacy_datasets(train_size, val_frac, seed, partition=1):
    """Build DTD train/val/test datasets in legacy mode.
    
    Creates validation split from partition's train+val merged data.
    Uses stratified split if available, otherwise random split.
    
    Parameters
    ----------
    train_size : float | int | None
        If provided, subsample training set to this fraction (0, 1] or absolute count
    val_frac : float
        Fraction of merged train data to use for validation (default 0.2)
    seed : int
        Random seed for reproducibility
    partition : int
        DTD partition (fold) number, 1-10 (default 1)
    """
    try:
        from dtd_dataset import (
            build_partition_datasets,
            stratified_subset as dtd_stratified_subset,
            stratified_train_val_split as dtd_stratified_train_val_split,
        )
    except ImportError:
        raise RuntimeError(
            "DTD dataset support requires dtd_dataset module. "
            "Ensure it is available in the current directory or Python path."
        )
    
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet normalization
    train_tf, val_tf = _dtd_transforms(mean, std)
    
    # Load partition (train+val merged) and test split
    train_merged_ds, test_ds = build_partition_datasets(
        deep_scatter_data_path, partition, train_tf, download=False
    )
    
    # For validation evaluation, use same partition but with eval transform
    _, test_ds_eval = build_partition_datasets(
        deep_scatter_data_path, partition, val_tf, download=False
    )
    test_ds = test_ds_eval  # Use eval transform for consistency
    
    # Split merged train into train/val for early stopping (before train_size subsetting)
    train_idx, val_idx = dtd_stratified_train_val_split(
        train_merged_ds, val_frac=val_frac, seed=seed
    )
    train_dataset = torch.utils.data.Subset(train_merged_ds, train_idx)
    val_dataset = torch.utils.data.Subset(train_merged_ds, val_idx)
    
    # Apply train_size subsetting to the train portion (after creating Subsets)
    if train_size is not None:
        train_dataset = dtd_stratified_subset(train_dataset, train_size, seed=seed)
    
    # Calculate train_samples_before_split for logging
    train_samples_before_split = len(train_merged_ds)
    
    return train_dataset, val_dataset, test_ds, train_samples_before_split


def _build_minc_legacy_datasets(train_size, val_frac, seed, data_dir='./data'):
    """Build MINC2500 train/val/test datasets matching the legacy framework interface."""
    _ensure_deep_scatter_available("MINC2500 dataset")
    
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for --dataset minc2500. Install with: pip install kagglehub"
        ) from exc
    
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf, val_tf = _imagenet_transforms(mean, std, input_size=224)
    
    os.environ.setdefault("KAGGLEHUB_CACHE", data_dir)
    local_image_root = os.path.join(data_dir, MINC2500_DEEPER_PATH)
    if os.path.isdir(local_image_root):
        image_root = local_image_root
    else:
        kaggle_root = kagglehub.dataset_download(MINC2500_DATASET_ID)
        image_root = os.path.join(kaggle_root, MINC2500_DEEPER_PATH)
    
    if not os.path.isdir(image_root):
        raise FileNotFoundError(
            f"MINC2500 image root not found: {image_root}"
        )
    
    full_train_ds_aug = datasets.ImageFolder(image_root, transform=train_tf)
    full_train_ds_eval = datasets.ImageFolder(image_root, transform=val_tf)

    # Hold out the test set, then split the remaining pool into train/val.
    train_pool_idx, test_idx = stratified_train_val_split(
        full_train_ds_aug, val_frac=MINC2500_TEST_FRAC, seed=seed
    )
    test_ds = torch.utils.data.Subset(full_train_ds_eval, test_idx)
    train_pool_ds_eval = torch.utils.data.Subset(full_train_ds_eval, train_pool_idx)
    # Returns indices into the base dataset.
    train_idx, val_idx = stratified_train_val_split(
        train_pool_ds_eval, val_frac=val_frac, seed=seed
    )
    train_dataset = torch.utils.data.Subset(full_train_ds_aug, train_idx)
    val_dataset = torch.utils.data.Subset(full_train_ds_eval, val_idx)
    
    if train_size is not None:
        train_dataset = stratified_subset(train_dataset, train_size, seed=seed)
    
    train_samples_before_split = len(train_pool_idx)
    
    return train_dataset, val_dataset, test_ds, train_samples_before_split

def _build_eurosat_legacy_datasets(train_size, val_frac, seed, data_dir='./data'):
    """
    Build EuroSAT train/val/test datasets matching the legacy framework interface.
    """
    _ensure_deep_scatter_available("EuroSAT dataset")
    
    cfg = DATASET_CONFIGS.get("eurosat", {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)})
    mean, std = cfg["mean"], cfg["std"]
    
    try:
        from deep_scatter.data import _eurosat_transforms
        train_tf, val_tf = _eurosat_transforms(mean, std)
    except ImportError:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    base_ds_aug = datasets.EuroSAT(root=data_dir, download=True, transform=train_tf)
    base_ds_eval = datasets.EuroSAT(root=data_dir, download=False, transform=val_tf)
    
    # Isolate 20% test set
    test_frac = 0.2
    train_pool_idx, test_idx = stratified_train_val_split(
        base_ds_eval, val_frac=test_frac, seed=seed
    )
    test_dataset = torch.utils.data.Subset(base_ds_eval, test_idx)
    train_pool_ds_eval = torch.utils.data.Subset(base_ds_eval, train_pool_idx)
    # Returns indices into the base dataset.
    train_idx, val_idx = stratified_train_val_split(
        train_pool_ds_eval, val_frac=val_frac, seed=seed
    )
    train_dataset = torch.utils.data.Subset(base_ds_aug, train_idx)
    val_dataset = torch.utils.data.Subset(base_ds_eval, val_idx)
    
    if train_size is not None:
        train_dataset = stratified_subset(train_dataset, train_size, seed=seed)
        
    train_samples_before_split = len(train_pool_idx)
    
    return train_dataset, val_dataset, test_dataset, train_samples_before_split


def _imagefolder_args_shim(val_frac, input_size=None):
    """Minimal ``args`` (val_frac, input_size) for deep_scatter's _build_ImageFolder_dataset."""
    from types import SimpleNamespace
    return SimpleNamespace(val_frac=val_frac if val_frac is not None else 0.2,
                           input_size=input_size)


def _build_fmd_legacy_datasets(train_size, val_frac, seed, data_dir='./data', input_size=None):
    """Build FMD train/val/test datasets in legacy mode."""
    _ensure_deep_scatter_available("FMD dataset")
    return _build_ImageFolder_dataset(dataset = 'fmd', data_dir=data_dir, train_size=train_size, seed=seed,
                                      args=_imagefolder_args_shim(val_frac, input_size))

def _build_alot_legacy_datasets(train_size, val_frac, seed, data_dir='./data', input_size=None):
    """Build ALOT train/val/test datasets in legacy mode."""
    _ensure_deep_scatter_available("ALOT dataset")
    return _build_ImageFolder_dataset(dataset = 'alot', data_dir=data_dir, train_size=train_size, seed=seed,
                                      args=_imagefolder_args_shim(val_frac, input_size))

def _build_surotex_legacy_datasets(train_size, val_frac, seed, data_dir='./data', input_size=None):
    """Build SuroTex train/val/test datasets in legacy mode.

    ImageFolder-style (50 texture classes x 100 images) with no predefined splits,
    so this reuses deep_scatter's shared 20% test / stratified train-val builder.
    ``data_dir`` may point either at the extracted dataset dir or at its parent.
    """
    _ensure_deep_scatter_available("SuroTex dataset")
    return _build_ImageFolder_dataset(dataset='surotex', data_dir=data_dir, train_size=train_size, seed=seed,
                                      args=_imagefolder_args_shim(val_frac, input_size))


def _build_medmnist_legacy_datasets(train_size, val_frac, seed, dataset_name, data_dir='./data',
                                    input_size=None):
    """Build MedMNIST train/val/test datasets in legacy mode.

    input_size optionally selects the MedMNIST+ variant (28/64/128/224); None keeps the
    dataset default.
    """
    _ensure_deep_scatter_available("MedMNIST datasets")
    from types import SimpleNamespace
    args_shim = SimpleNamespace(input_size=input_size) if input_size else None
    return _build_medmnist_dataset(dataset=dataset_name, data_dir=data_dir, train_size=train_size,
                                   seed=seed, args=args_shim)

def _build_so2sat_legacy_datasets(train_size, seed, dataset, data_dir='./data'):
    """Build So2Sat train/val/test datasets in legacy mode."""
    _ensure_deep_scatter_available("So2Sat dataset")
    return _build_so2sat_hf_dataset(data_dir=data_dir, train_size=train_size, seed=seed, dataset=dataset)