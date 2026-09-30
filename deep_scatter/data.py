"""Dataset construction and stratified subset sampling.

Supports CIFAR-10/100, ImageNet, DTD, MINC2500, EuroSAT, FMD, ALOT, SuroTex, MedMNIST
and So2Sat.
"""
import glob
from collections import OrderedDict, defaultdict
import io
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import scipy.io as sio
from utils import is_distributed, get_world_size
import medmnist
from medmnist import INFO as medInfo
import h5py
import gzip
import shutil
from huggingface_hub import snapshot_download
from pathlib import Path



# ---------------------------------------------------------------------------
# Per-dataset constants
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "cifar10": {
        "in_channels": 3,
        "in_spatial": 32,
        "nb_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "cifar100": {
        "in_channels": 3,
        "in_spatial": 32,
        "nb_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
    },
    "imagenet": {
        "in_channels": 3,
        "in_spatial": 224,
        "nb_classes": 1000,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "dtd": {
        "in_channels": 3,
        "in_spatial": 224,
        "nb_classes": 47,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "minc2500": {
        "in_channels": 3,
        "in_spatial": 224,
        "nb_classes": 23,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "eurosat": {
        "in_channels": 3,
        "in_spatial": 64,
        "nb_classes": 10,
        "mean": (0.3444, 0.3803, 0.4078),
        "std": (0.0914, 0.0651, 0.0553),
    },
    "pathmnist": {
        "in_channels": 3,
        "in_spatial": 64,
        "nb_classes": 9,
        "mean": (0.7405, 0.5330, 0.7058),
        "std": (0.0723, 0.1038, 0.0731),
    },
    "octmnist": {
        "in_channels": 1,
        "in_spatial": 28,
        "nb_classes": 4,
        "mean": (0.1889,),
        "std": (0.1694,),
    },
    "fmd": {
        "in_channels": 3,
        "in_spatial": 128,
        "nb_classes": 10,
        "mean": (0.6580, 0.6040, 0.5410),
        "std": (0.2210, 0.2230, 0.2170),
    },
    "alot": {
        "in_channels": 3,
        "in_spatial": 128,
        "nb_classes": 250,
        "mean": (0.6770, 0.5850, 0.5100),
        "std": (0.1380, 0.1350, 0.1310),
    },
    "surotex": {
        # 50 classes x 100 400x400 RGB images; mean/std from get_dataset_stats at 128px.
        "in_channels": 3,
        "in_spatial": 128,
        "nb_classes": 50,
        "mean": (0.5590, 0.5724, 0.5594),
        "std": (0.1069, 0.1103, 0.1095),
    },
    "so2sat_rgb": {
        "in_channels": 3,
        "in_spatial": 32,
        "nb_classes": 17,
        "mean": (0.12375696117681, 0.10927746363683, 0.10108552032678),
        "std": (0.03958795, 0.04777826, 0.06636616),
    },
    "so2sat_s2":
    {
        "in_channels": 10,
        "in_spatial": 32,
        "nb_classes": 17,
        "mean": (0.12375696117681, 0.10927746363683, 0.10108552032678, 0.11423986161140, 0.15926566920230, 0.18147236008771, 
           0.17457403122913, 0.19501607349635, 0.15428468872076, 0.10905050699570),
        "std": (0.03958795, 0.04777826, 0.06636616, 0.06358874, 0.07744387, 0.09101635, 0.09218466, 0.10164581, 0.09991773, 0.08780632),
    },
}

# DataLoader defaults.
DEFAULT_PIN_MEMORY = True
DEFAULT_PREFETCH_FACTOR = 2
IMAGENET_SHARDS_DATALOADER_TIMEOUT = 120

MINC2500_DATASET_ID = "liewyousheng/minc2500"
MINC2500_DEEPER_PATH = "minc-2500/images"
MINC2500_TEST_FRAC = 0.15


# ---------------------------------------------------------------------------
# Stratified subset sampling
# ---------------------------------------------------------------------------

def _get_targets(dataset):
    """Extract integer target labels from a dataset."""
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if isinstance(dataset, torch.utils.data.Subset):
        base_targets = _get_targets(dataset.dataset)
        return base_targets[np.asarray(dataset.indices)]
    raise AttributeError(
        f"{type(dataset).__name__} has no .targets or .labels attribute"
    )

def get_dataset_stats(folder_path, device='cpu'):
    """Per-channel mean/std of an ImageFolder at 128px."""
    dataset = datasets.ImageFolder(folder_path, transform=transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = 0.
    std = 0.
    total_samples = 0.
    for data, _ in loader:
        data = data.to(device)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    mean /= total_samples
    std /= total_samples
    return mean.tolist(), std.tolist()

def stratified_subset(dataset, train_size, seed=42):
    """Return a class-balanced Subset of *dataset*.

    Parameters
    ----------
    dataset : torchvision dataset
        Must expose a ``.targets`` or ``.labels`` attribute.
    train_size : float or int
        If float in (0, 1]: fraction of the dataset to keep.
        If int > 1: absolute number of samples to keep.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.utils.data.Subset
    """
    if isinstance(train_size, float) and train_size >= 1.0:
        return dataset
    if isinstance(train_size, int) and train_size >= len(dataset):
        return dataset

    targets = _get_targets(dataset)
    classes = np.unique(targets)
    rng = np.random.RandomState(seed)

    indices = []
    for c in classes:
        c_idx = np.where(targets == c)[0]
        if isinstance(train_size, float):
            n = max(1, int(len(c_idx) * train_size))
        else:
            n = min(train_size // len(classes), len(c_idx))
            n = max(1, n)
        chosen = rng.choice(c_idx, size=n, replace=False)
        indices.extend(chosen.tolist())

    rng.shuffle(indices)
    return torch.utils.data.Subset(dataset, indices)


def stratified_train_val_split(subset_or_dataset, val_frac=0.2, seed=42):
    """Split indices into train and val, stratified by class.

    Parameters
    ----------
    subset_or_dataset : Subset or Dataset
        Must expose .targets or .labels (directly or via .dataset).
    val_frac : float
        Fraction of samples to use for validation (e.g. 0.2 = 20%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_indices, val_indices : list of int
    """
    if isinstance(subset_or_dataset, torch.utils.data.Subset):
        indices = np.array(subset_or_dataset.indices)
        base = subset_or_dataset.dataset
        targets = np.asarray(_get_targets(base))[indices]
    else:
        indices = np.arange(len(subset_or_dataset))
        targets = _get_targets(subset_or_dataset)

    targets = np.asarray(targets)
    classes = np.unique(targets)
    rng = np.random.RandomState(seed)

    train_indices, val_indices = [], []
    for c in classes:
        c_mask = targets == c
        c_indices = indices[c_mask]
        n_val = max(1, int(len(c_indices) * val_frac))
        n_train = len(c_indices) - n_val
        perm = rng.permutation(len(c_indices))
        val_indices.extend(c_indices[perm[:n_val]].tolist())
        train_indices.extend(c_indices[perm[n_val:]].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _build_train_val_split(train_ds_aug, train_ds_eval, train_size,
                                val_frac, seed):
    """Build train/val subsets from train data."""
    subset = stratified_subset(train_ds_aug, train_size, seed=seed) if train_size is not None else train_ds_aug
    train_idx, val_idx = stratified_train_val_split(subset, val_frac=val_frac, seed=seed)
    train_ds = torch.utils.data.Subset(train_ds_aug, train_idx)
    val_ds = torch.utils.data.Subset(train_ds_eval, val_idx)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def _cifar_transforms(mean, std):
    """Transform pipeline for CIFAR-10 and CIFAR-100."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def _imagenet_transforms(mean, std, input_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


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

def _eurosat_transforms(mean, std):
    """Transform pipeline for EuroSAT (minimal augmentation, similar to ImageNet eval)."""
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((64, 64)),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize((64, 64)),
    ])
    return train_tf, val_tf

def _so2sat_transforms(mean, std):
    """Transform pipeline for So2Sat (ToTensor first because of >3 channels)."""
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf
# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _build_cifar_dataset(args):
    """Build CIFAR-10/100 (downloads if missing; download on a single process first)."""
    cfg = DATASET_CONFIGS[args.dataset]
    mean, std = cfg["mean"], cfg["std"]
    train_tf, val_tf = _cifar_transforms(mean, std)
    ds_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
    train_ds_aug = ds_cls(args.data_dir, train=True, download=True, transform=train_tf)
    train_ds_eval = ds_cls(args.data_dir, train=True, download=True, transform=val_tf)
    test_ds = ds_cls(args.data_dir, train=False, download=True, transform=val_tf)
    train_ds, val_ds = _build_train_val_split(
        train_ds_aug, train_ds_eval,
        args.train_size, args.val_frac, args.seed)
    return train_ds, val_ds, test_ds

def _build_imagenet_dataset(args):
    cfg = DATASET_CONFIGS[args.dataset]
    mean, std = cfg["mean"], cfg["std"]
    train_tf, val_tf = _imagenet_transforms(mean, std, input_size=cfg["in_spatial"])

    if getattr(args, "imagenet_source", "pytorch") == "huggingface":
        return _build_imagenet_hf_dataset(args, train_tf, val_tf)

    imagenet_root = _resolve_imagenet_root(args)
    traindir = os.path.join(imagenet_root, "train")
    valdir = os.path.join(imagenet_root, "val")
    train_ds_aug = datasets.ImageFolder(traindir, transform=train_tf)
    train_ds_eval = datasets.ImageFolder(traindir, transform=val_tf)
    test_ds = datasets.ImageFolder(valdir, transform=val_tf)
    train_ds, val_ds = _build_train_val_split(
        train_ds_aug, train_ds_eval,
        args.train_size, args.val_frac, args.seed)
    return train_ds, val_ds, test_ds

def _build_minc_dataset(args):
    """Build MINC2500 from KaggleHub: 15% test hold-out, then a train/val split."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required for --dataset minc2500. Install with: pip install kagglehub"
        ) from exc

    cfg = DATASET_CONFIGS["minc2500"]
    mean, std = cfg["mean"], cfg["std"]
    train_tf, val_tf = _imagenet_transforms(mean, std, input_size=cfg["in_spatial"])

    os.environ.setdefault("KAGGLEHUB_CACHE", args.data_dir)
    local_image_root = os.path.join(args.data_dir, MINC2500_DEEPER_PATH)
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

    train_pool_idx, test_idx = stratified_train_val_split(
        full_train_ds_aug, val_frac=MINC2500_TEST_FRAC, seed=args.seed
    )
    test_ds = torch.utils.data.Subset(full_train_ds_eval, test_idx)

    # Train/val split of the remaining pool (returns indices into the base dataset).
    train_pool_ds_eval = torch.utils.data.Subset(full_train_ds_eval, train_pool_idx)
    train_idx, val_idx = stratified_train_val_split(
        train_pool_ds_eval, val_frac=args.val_frac, seed=args.seed
    )
    train_ds = torch.utils.data.Subset(full_train_ds_aug, train_idx)
    val_ds = torch.utils.data.Subset(full_train_ds_eval, val_idx)

    if args.train_size is not None:
        train_ds = stratified_subset(train_ds, args.train_size, seed=args.seed)

    return train_ds, val_ds, test_ds


def _build_dtd_dataset(args):
    """Build one DTD partition (fold), optionally subsetting the training set."""
    from dtd_dataset import (
        build_partition_datasets, stratified_subset,
        stratified_train_val_split
    )

    cfg = DATASET_CONFIGS[args.dataset]
    partition = getattr(args, "dtd_partition", 1)
    mean, std = cfg["mean"], cfg["std"]
    train_tf, val_tf = _dtd_transforms(mean, std)

    # Partition train+val (merged) and test split; the test split uses the eval transform.
    train_merged_ds, _ = build_partition_datasets(
        args.data_dir, partition, train_tf, download=False
    )
    _, test_ds = build_partition_datasets(
        args.data_dir, partition, val_tf, download=False
    )

    # Train/val split for early stopping, before train_size subsetting.
    train_idx, val_idx = stratified_train_val_split(
        train_merged_ds, val_frac=args.val_frac, seed=args.seed
    )
    train_ds = torch.utils.data.Subset(train_merged_ds, train_idx)
    val_ds = torch.utils.data.Subset(train_merged_ds, val_idx)

    if args.train_size is not None:
        train_ds = stratified_subset(train_ds, args.train_size, seed=args.seed)

    return train_ds, val_ds, test_ds


def _build_eurosat_dataset(args):
    """Split EuroSAT into stratified train, validation (from train pool) and 20% test subsets."""
    cfg = DATASET_CONFIGS[args.dataset]
    mean, std = cfg["mean"], cfg["std"]
    train_tf, val_tf = _eurosat_transforms(mean, std)

    base_ds_aug = datasets.EuroSAT(root=args.data_dir, download=True, transform=train_tf)
    base_ds_eval = datasets.EuroSAT(root=args.data_dir, download=False, transform=val_tf)

    test_frac = 0.2
    train_pool_idx, test_idx = stratified_train_val_split(
        base_ds_eval, val_frac=test_frac, seed=args.seed
    )
    test_ds = torch.utils.data.Subset(base_ds_eval, test_idx)

    # Train/val split of the remaining pool (returns indices into the base dataset).
    train_pool_ds_eval = torch.utils.data.Subset(base_ds_eval, train_pool_idx)
    train_idx, val_idx = stratified_train_val_split(
        train_pool_ds_eval, val_frac=args.val_frac, seed=args.seed
    )
    train_ds = torch.utils.data.Subset(base_ds_aug, train_idx)
    val_ds = torch.utils.data.Subset(base_ds_eval, val_idx)

    if args.train_size is not None:
        train_ds = stratified_subset(train_ds, args.train_size, seed=args.seed)

    return train_ds, val_ds, test_ds

SUROTEX_DIR_NAME = "SuroTex Surrounding Texture Dataset"
SUROTEX_ARCHIVE_NAME = "SuroTex - Surrounding Texture Dataset.rar"


def _resolve_surotex_root(data_dir):
    """Return the ImageFolder root for SuroTex (the dir holding ``texture000/`` ...).

    Accepts either the dataset directory itself or its parent. If only the RAR
    archive is present, extract it first with::

        cd "<data_dir>/SuroTex Surrounding Texture Dataset"
        bsdtar -xf "SuroTex - Surrounding Texture Dataset.rar"
    """
    candidates = [data_dir, os.path.join(data_dir, SUROTEX_DIR_NAME)]
    for root in candidates:
        if glob.glob(os.path.join(root, "texture*", "*.jpg")):
            return root
    for root in candidates:
        if os.path.exists(os.path.join(root, SUROTEX_ARCHIVE_NAME)):
            raise FileNotFoundError(
                f"SuroTex archive found at {root} but no extracted texture*/ class "
                f"folders. Extract it with: cd '{root}' && "
                f"bsdtar -xf '{SUROTEX_ARCHIVE_NAME}'"
            )
    raise FileNotFoundError(
        f"SuroTex class folders (texture000/ ...) not found under {data_dir} "
        f"or {os.path.join(data_dir, SUROTEX_DIR_NAME)}"
    )


def _build_ImageFolder_dataset(dataset, data_dir, train_size=1.0, seed=42, args=None):
    """Build an ImageFolder dataset that has no predefined splits.

    Holds out a stratified 20% test set, then splits the remaining 80% into
    stratified training and validation sets.
    """
    cfg = DATASET_CONFIGS[dataset]
    mean, std = cfg["mean"], cfg["std"]
    # --input-size overrides the default resolution (also applied in build_dataloaders).
    input_size = (getattr(args, "input_size", None) if args else None) or cfg["in_spatial"]
    train_tf, val_tf = _imagenet_transforms(mean, std, input_size=input_size)

    if dataset == "surotex":
        data_dir = _resolve_surotex_root(data_dir)

    val_frac = args.val_frac if (args and hasattr(args, 'val_frac')) else 0.2

    ds_aug = datasets.ImageFolder(data_dir, transform=train_tf)
    ds_eval = datasets.ImageFolder(data_dir, transform=val_tf)

    test_frac = 0.2
    train_pool_idx, test_idx = stratified_train_val_split(
        ds_eval, val_frac=test_frac, seed=seed
    )
    test_ds = torch.utils.data.Subset(ds_eval, test_idx)

    # Train/val split of the remaining pool (returns indices into the base dataset).
    train_pool_ds_eval = torch.utils.data.Subset(ds_eval, train_pool_idx)
    train_idx, val_idx = stratified_train_val_split(
        train_pool_ds_eval, val_frac=val_frac, seed=seed
    )
    train_ds = torch.utils.data.Subset(ds_aug, train_idx)
    val_ds = torch.utils.data.Subset(ds_eval, val_idx)

    # stratified_subset is a no-op for 1.0 or an absolute count >= len(dataset).
    if train_size is not None:
        train_ds = stratified_subset(train_ds, train_size, seed=seed)

    return train_ds, val_ds, test_ds

def _build_medmnist_dataset(dataset, data_dir, train_size=1.0, seed=42, args=None):
    """Build a MedMNIST dataset using its official train/val/test splits."""
    if dataset not in medInfo:
        raise ValueError(f"Dataset '{dataset}' is not a valid MedMNIST dataset name. "
                            f"Choose from: {list(medInfo.keys())}")
        
    info = medInfo[dataset]
    info.update(DATASET_CONFIGS.get(dataset, {}))
    n_channels = info['n_channels']

    mean = info.get('mean', [0.5] * n_channels)
    std = info.get('std', [0.5] * n_channels)

    # MedMNIST+ supports 28, 64, 128, 224; --input-size overrides the config default.
    input_size = getattr(args, 'input_size', None) or info.get('in_spatial', 28)
    train_tf, val_tf = _imagenet_transforms(mean, std, input_size=input_size)

    DataClass = getattr(medmnist, info['python_class'])

    download_flag = getattr(args, 'download', True)
    kwargs = {'root': data_dir, 'download': download_flag}
    if input_size != 28:
        kwargs['size'] = input_size

    train_ds = DataClass(split='train', transform=train_tf, **kwargs)
    val_ds = DataClass(split='val', transform=val_tf, **kwargs)
    test_ds = DataClass(split='test', transform=val_tf, **kwargs)

    # Expose MedMNIST's `.labels` as a torchvision-style `.targets`.
    for ds in [train_ds, val_ds, test_ds]:
        if hasattr(ds, 'labels') and not hasattr(ds, 'targets'):
            ds.targets = ds.labels.squeeze().tolist()

    if train_size is not None and train_size < 1.0:
        train_ds = stratified_subset(train_ds, train_size, seed=seed)
        
    for ds in [train_ds, val_ds, test_ds]:
        if hasattr(ds, 'labels'):
            ds.labels = ds.labels.squeeze()
        if type(ds) is torch.utils.data.Subset:
            if hasattr(ds.dataset, 'labels'):
                ds.dataset.labels = ds.dataset.labels.squeeze()
    return train_ds, val_ds, test_ds


class H5So2SatDataset(torch.utils.data.Dataset):
    """Reads So2Sat-LCZ42 HDF5 files."""
    def __init__(self, h5_path, transform=None, band_mode='rgb'):
        self.h5_path = h5_path
        self.transform = transform
        self.band_mode = band_mode.lower()
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['label'].shape[0]
            
        # Opened lazily in each worker; an open handle cannot be shared across processes.
        self.dataset = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')
            
        # One-hot (17) -> class index.
        label = np.argmax(self.dataset['label'][idx])

        if self.band_mode == 's1':
            # Sentinel-1 Radar (8 bands)
            img_array = self.dataset['sen1'][idx].astype(np.float32)
            
        elif self.band_mode == 's2':
            # Sentinel-2 Optical (10 bands)
            img_array = self.dataset['sen2'][idx].astype(np.float32)
            
        elif self.band_mode == 'rgb':
            # First three optical bands: B2, B3, B4.
            img_array = self.dataset['sen2'][idx, ..., [0,1,2]].astype(np.float32)
            
        elif self.band_mode == 'all':
            # Multi-modal fusion (18 bands)
            s1 = self.dataset['sen1'][idx].astype(np.float32)
            s2 = self.dataset['sen2'][idx].astype(np.float32)
            img_array = np.concatenate([s1, s2], axis=-1)
            
        else:
            raise ValueError(f"Unknown band_mode: {self.band_mode}")

        if self.transform:
            img_array = self.transform(img_array)
            
        return img_array, label  

def _download_and_extract_so2sat(repo_id, cache_dir):
    """Download So2Sat from the Hugging Face Hub and return {split: path to .h5}."""
    print(f"Syncing {repo_id} from Hugging Face Hub...")
    download_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        repo_type="dataset",
        allow_patterns=["*.h5", "*.h5.gz"]
    )
    
    base_path = Path(download_dir)
    split_paths = {}

    for split in ['training', 'validation', 'testing']:
        h5_files = list(base_path.rglob(f"{split}.h5"))
        gz_files = list(base_path.rglob(f"{split}.h5.gz"))
        
        if h5_files:
            split_paths[split] = str(h5_files[0])
        elif gz_files:
            gz_path = gz_files[0]
            h5_path = gz_path.with_suffix('')

            print(f"Extracting {gz_path.name}... (This may take a minute)")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(h5_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            split_paths[split] = str(h5_path)
        else:
            raise FileNotFoundError(
                f"Could not find {split}.h5 or {split}.h5.gz anywhere in {download_dir}"
            )
    return split_paths

def _build_so2sat_hf_dataset(dataset, data_dir, train_size=1.0, seed=42, args=None, band_mode='rgb'):
    """Build So2Sat datasets from zhu-xlab/So2Sat-LCZ42."""

    if args:
        train_size = getattr(args, 'train_size', train_size)
        seed = getattr(args, 'seed', seed)
        data_dir = getattr(args, 'data_dir', data_dir)

    split_paths = _download_and_extract_so2sat("zhu-xlab/So2Sat-LCZ42", data_dir)

    train_tf, val_tf = _so2sat_transforms(
        mean=DATASET_CONFIGS[dataset]["mean"],
        std=DATASET_CONFIGS[dataset]["std"]
    )

    train_ds = H5So2SatDataset(split_paths["training"], train_tf, band_mode)
    val_ds   = H5So2SatDataset(split_paths["validation"], val_tf, band_mode)
    test_ds  = H5So2SatDataset(split_paths["testing"], val_tf, band_mode)

    if train_size < 1.0:
        subset_len = math.ceil(len(train_ds) * train_size)
        
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_ds), generator=generator)[:subset_len].tolist()
        
        train_ds = torch.utils.data.Subset(train_ds, indices)

    return train_ds, val_ds, test_ds

    
def _find_hf_split_files(root_dir):
    """Return parquet file mapping for HF ImageNet snapshot under root_dir."""
    data_dir = os.path.join(root_dir, "data")
    if not os.path.isdir(data_dir):
        return None

    def _collect(prefix):
        pattern = os.path.join(data_dir, f"{prefix}-*.parquet")
        return sorted(glob.glob(pattern))

    train_files = _collect("train")
    validation_files = _collect("validation") or _collect("val")
    test_files = _collect("test")
    if not train_files:
        return None

    split_files = {"train": train_files}
    if validation_files:
        split_files["validation"] = validation_files
    elif test_files:
        # Most ImageNet HF snapshots provide test shard files for val-set labels.
        split_files["validation"] = test_files
    else:
        raise FileNotFoundError(
            f"No validation/test parquet shards found under {data_dir}"
        )
    return split_files


def _resolve_hf_columns(ds):
    cols = list(ds.column_names)
    image_col = next((c for c in ("image", "img", "jpeg") if c in cols), None)
    label_col = next((c for c in ("label", "labels", "class", "target") if c in cols), None)
    if image_col is None or label_col is None:
        raise KeyError(
            f"Unable to resolve image/label columns from columns={cols}"
        )
    return image_col, label_col


def _to_rgb_pil(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict):
        img_bytes = image_value.get("bytes")
        img_path = image_value.get("path")
        if img_bytes is not None:
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if img_path:
            return Image.open(img_path).convert("RGB")
    if isinstance(image_value, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_value)).convert("RGB")
    if isinstance(image_value, np.ndarray):
        return Image.fromarray(image_value).convert("RGB")
    raise TypeError(f"Unsupported image sample type: {type(image_value)}")


class HFImageNetDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper for Hugging Face parquet-backed ImageNet splits."""

    def __init__(self, hf_ds, transform, image_col, label_col):
        self.hf_ds = hf_ds
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.targets = np.asarray(hf_ds[label_col], dtype=np.int64)

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        row = self.hf_ds[idx]
        img = _to_rgb_pil(row[self.image_col])
        label = int(row[self.label_col])
        return self.transform(img), label


def _build_imagenet_hf_dataset(args, train_tf, val_tf):
    """Build ImageNet datasets from a Hugging Face snapshot (parquet or folders)."""
    imagenet_root = _resolve_imagenet_root(args)

    # If folders are present, keep native torchvision ImageFolder behavior.
    traindir = os.path.join(imagenet_root, "train")
    valdir = os.path.join(imagenet_root, "val")
    if os.path.isdir(traindir) and os.path.isdir(valdir):
        train_ds_aug = datasets.ImageFolder(traindir, transform=train_tf)
        train_ds_eval = datasets.ImageFolder(traindir, transform=val_tf)
        test_ds = datasets.ImageFolder(valdir, transform=val_tf)
        train_ds, val_ds = _build_train_val_split(
            train_ds_aug, train_ds_eval,
            args.train_size, args.val_frac, args.seed)
        return train_ds, val_ds, test_ds

    split_files = _find_hf_split_files(imagenet_root)
    if split_files is None:
        raise FileNotFoundError(
            f"No ImageFolder directories or parquet shards found under {imagenet_root}"
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required to read Hugging Face parquet ImageNet snapshots. "
            "Install with: pip install datasets"
        ) from exc

    ds_dict = load_dataset("parquet", data_files=split_files)
    image_col, label_col = _resolve_hf_columns(ds_dict["train"])

    train_ds_aug = HFImageNetDataset(ds_dict["train"], train_tf, image_col, label_col)
    train_ds_eval = HFImageNetDataset(ds_dict["train"], val_tf, image_col, label_col)
    test_ds = HFImageNetDataset(ds_dict["validation"], val_tf, image_col, label_col)
    train_ds, val_ds = _build_train_val_split(
        train_ds_aug, train_ds_eval,
        args.train_size, args.val_frac, args.seed)
    return train_ds, val_ds, test_ds


def _resolve_imagenet_root(args):
    """Resolve local ImageNet root for torchvision loading.

    For ``imagenet-source=huggingface``, optionally sync the dataset snapshot
    first and return the local root (either ImageFolder layout or parquet shards).
    """
    source = getattr(args, "imagenet_source", "pytorch")
    if source == "huggingface":
        local_dir = getattr(args, "hf_local_dir", None) or args.data_dir
        if getattr(args, "hf_download", False):
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "huggingface_hub is required for --imagenet-source huggingface "
                    "with --hf-download. Install with: pip install huggingface_hub[hf_transfer]"
                ) from exc

            # Opt into hf_transfer acceleration unless user set it explicitly.
            if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

            token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
            download_kw = dict(
                repo_id=getattr(args, "hf_repo_id", "imagenet-1k"),
                repo_type="dataset",
                local_dir=local_dir,
            )
            if token:
                download_kw["token"] = token
            snapshot_download(**download_kw)
        data_root = local_dir
    else:
        data_root = args.data_dir

    if source == "huggingface":
        return data_root

    traindir = os.path.join(data_root, "train")
    valdir = os.path.join(data_root, "val")
    if not os.path.isdir(traindir) or not os.path.isdir(valdir):
        raise FileNotFoundError(
            f"ImageNet directory must contain train/ and val/: {data_root}"
        )
    return data_root
# ---------------------------------------------------------------------------
# ImageNet Tar shards (map-style TarIndex)
# ---------------------------------------------------------------------------


def _load_tar_catalog(data_dir):
    """Load the pre-built tar catalog from _tar_catalog.npz.

    Raises FileNotFoundError if the catalog is missing — run
    build_imagenet_tar_index.py first.
    """
    catalog_path = os.path.join(data_dir, "train", "_tar_catalog.npz")
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(
            f"Tar catalog not found: {catalog_path}. "
            "Run: python build_imagenet_tar_index.py --data-dir <data_dir>"
        )
    data = np.load(catalog_path, allow_pickle=True)
    return (
        data["tar_paths"].tolist(),
        data["path_ids"],
        data["offsets"],
        data["sizes"],
        data["class_ids"],
    )


class ImageNetTarDataset(torch.utils.data.Dataset):
    """Map-style dataset over train/*.tar shards.

    All lookup data (tar paths, byte offsets, sizes, class ids) is stored as
    compact numpy arrays built once in the main process and inherited by forked
    DataLoader workers via copy-on-write — no Python-GC touching, ~20 MB total
    regardless of worker count.  Reads use direct byte-seeks into a small LRU
    of open raw file handles (one open() per tar per worker, never re-scanned).
    """

    def __init__(self, tar_paths, path_ids, offsets, sizes, class_ids,
                 transform, tar_cache_size=8):
        self.tar_paths = tar_paths   # list[str], length = n_tars
        self.path_ids  = path_ids    # int16[N]
        self.offsets   = offsets     # int64[N]
        self.sizes     = sizes       # int32[N]
        self.class_ids = class_ids   # int16[N]
        self.targets   = class_ids   # alias expected by _get_targets()
        self.transform = transform
        self._fh_cache = OrderedDict()
        self._tar_cache_maxsize = tar_cache_size

    def _get_fh(self, tar_path):
        if tar_path in self._fh_cache:
            self._fh_cache.move_to_end(tar_path)
            return self._fh_cache[tar_path]
        if len(self._fh_cache) >= self._tar_cache_maxsize:
            _, old_fh = self._fh_cache.popitem(last=False)
            old_fh.close()
        fh = open(tar_path, "rb")
        self._fh_cache[tar_path] = fh
        return fh

    def __len__(self):
        return len(self.path_ids)

    def __getitem__(self, idx):
        tar_path = self.tar_paths[int(self.path_ids[idx])]
        offset   = int(self.offsets[idx])
        size     = int(self.sizes[idx])
        class_id = int(self.class_ids[idx])
        try:
            fh = self._get_fh(tar_path)
            fh.seek(offset)
            data = fh.read(size)
            img = Image.open(io.BytesIO(data)).convert("RGB")
            return self.transform(img), torch.tensor(class_id, dtype=torch.long)
        except Exception as exc:
            raise RuntimeError(
                f"Failed reading idx={idx} from tar={tar_path}, "
                f"offset={offset}, size={size}"
            ) from exc


def _resolve_subset_indices(dataset):
    """Return (local_indices_array, base_ImageNetTarDataset) for a Dataset or Subset."""
    if isinstance(dataset, torch.utils.data.Subset):
        return np.array(dataset.indices), dataset.dataset
    return np.arange(len(dataset)), dataset


class ShardedSampler(torch.utils.data.Sampler):
    """Shard-aware sampler for ImageNetTarDataset (and Subsets thereof).

    Shuffles at the tar-shard level so consecutive samples come from the same
    tar file, minimising random seeks on network filesystems.  Within each
    shard the sample order is also shuffled each epoch so the model sees full
    diversity.  Handles DDP by assigning a contiguous block of the shard-
    ordered list to each rank (better shard locality than round-robin).
    """

    def __init__(self, dataset, num_replicas=1, rank=0, shuffle=True, seed=0):
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        local_indices, base = _resolve_subset_indices(dataset)
        groups = defaultdict(list)
        for local_idx, base_idx in enumerate(local_indices):
            tar_id = int(base.path_ids[base_idx])
            groups[tar_id].append(local_idx)
        self._groups = list(groups.values())
        self._n_total = len(local_indices)
        self.num_samples = math.ceil(self._n_total / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            group_order = rng.permutation(len(self._groups)).tolist()
            indices = []
            for g in group_order:
                indices.extend(rng.permutation(self._groups[g]).tolist())
        else:
            indices = [i for group in self._groups for i in group]
        # Pad to be evenly divisible across replicas
        while len(indices) < self.total_size:
            indices += indices[: self.total_size - len(indices)]
        indices = indices[: self.total_size]
        # Contiguous block for this rank (better shard locality than round-robin)
        start = self.rank * self.num_samples
        return iter(indices[start: start + self.num_samples])

    def __len__(self):
        return self.num_samples


def _build_imagenet_sharded(args):
    """Build ImageNet train/val/test datasets from train/*.tar and val/*/*.JPEG."""
    cfg = DATASET_CONFIGS["imagenet"]
    mean, std = cfg["mean"], cfg["std"]
    train_tf, val_tf = _imagenet_transforms(mean, std)
    imagenet_root = _resolve_imagenet_root(args)

    val_dir = os.path.join(imagenet_root, "val")
    test_ds = datasets.ImageFolder(val_dir, transform=val_tf)
    if not test_ds.samples:
        raise FileNotFoundError(
            f"No validation images in {val_dir}. "
            "Expected val/<wnid>/*.JPEG."
        )
    class_to_idx = dict(test_ds.class_to_idx)
    nb_classes = len(class_to_idx)
    tar_paths, path_ids, offsets, sizes, class_ids = _load_tar_catalog(imagenet_root)

    tar_cache_size = args.tar_cache_size
    full_ds_aug = ImageNetTarDataset(
        tar_paths, path_ids, offsets, sizes, class_ids,
        train_tf, tar_cache_size=tar_cache_size)
    full_ds_eval = ImageNetTarDataset(
        tar_paths, path_ids, offsets, sizes, class_ids,
        val_tf, tar_cache_size=tar_cache_size)
    train_ds, val_ds = _build_train_val_split(
        full_ds_aug, full_ds_eval,
        args.train_size, args.val_frac, args.seed)

    return train_ds, val_ds, test_ds, nb_classes


# ---------------------------------------------------------------------------
# Shared loader creation and args setup
# ---------------------------------------------------------------------------

def _make_dataloaders(train_ds, val_ds, test_ds, args, *, drop_last_train=False, timeout=0):
    """Create train/val/test DataLoaders with DistributedSampler when distributed."""
    num_replicas = get_world_size()
    rank = dist.get_rank() if is_distributed() else 0

    def _make_sampler(ds, shuffle):
        base = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
        if isinstance(base, ImageNetTarDataset):
            if getattr(args, "imagenet_sampler", "sharded") == "global":
                if is_distributed():
                    return torch.utils.data.distributed.DistributedSampler(
                        ds, shuffle=shuffle)
                return None
            # Shard-aware sampler: groups samples by tar file to minimise seeks
            return ShardedSampler(
                ds, num_replicas=num_replicas, rank=rank,
                shuffle=shuffle, seed=args.seed)
        if is_distributed():
            return torch.utils.data.distributed.DistributedSampler(ds, shuffle=shuffle)
        return None

    train_sampler = _make_sampler(train_ds, True)
    val_sampler = _make_sampler(val_ds, False)
    test_sampler = _make_sampler(test_ds, False)

    loader_kw = dict(
        batch_size=args.batch_size_per_device,
        num_workers=args.workers,
        pin_memory=DEFAULT_PIN_MEMORY,
        persistent_workers=(args.workers > 0 and not args.no_persistent_workers),
    )
    if args.workers > 0:
        loader_kw["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    if args.workers > 0 and timeout > 0:
        loader_kw["timeout"] = timeout

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=drop_last_train,
        **loader_kw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, shuffle=False, sampler=val_sampler, **loader_kw)
    test_loader = torch.utils.data.DataLoader(
        test_ds, shuffle=False, sampler=test_sampler, **loader_kw)
    return train_loader, val_loader, test_loader


def _set_training_args(args, train_loader, val_loader, test_loader, nb_classes, cfg):
    """Set args used by training loop (samples, steps, etc.)."""
    world_size = get_world_size()
    args.actual_train_samples = len(train_loader.dataset)
    args.val_samples = len(val_loader.dataset)
    args.test_samples = len(test_loader.dataset)
    args.in_channels = cfg["in_channels"]
    args.in_spatial = cfg["in_spatial"]
    args.nb_classes = nb_classes
    args.effective_batch_size = args.batch_size_per_device * world_size * args.grad_accum_steps
    args.steps_per_epoch = len(train_loader) // args.grad_accum_steps
    if not args.random_init:
        args.a_total_steps = args.lr_epochs * args.steps_per_epoch
        args.b_total_steps = args.epochs * args.steps_per_epoch
    else:
        args.total_steps = args.epochs * args.steps_per_epoch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    """Build train, validation, and test dataloaders.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain: dataset, data_dir, train_size, batch_size, workers,
        seed, val_frac, distributed.
        For DTD: also dtd_partition (default 1, range 1-10).

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    if args.dataset in DATASET_CONFIGS:
        cfg = DATASET_CONFIGS[args.dataset]
    elif args.dataset in medInfo:
        cfg = medInfo[args.dataset]
        cfg["in_channels"] = cfg.get("n_channels", 3)
        cfg["in_spatial"] = cfg.get("in_spatial", 28)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # --input-size override; copied so DATASET_CONFIGS is not mutated.
    if getattr(args, "input_size", None):
        cfg = dict(cfg)
        cfg["in_spatial"] = args.input_size

    if args.dataset in ("cifar10", "cifar100"):
        train_ds, val_ds, test_ds = _build_cifar_dataset(args)
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args)
    elif args.dataset == "dtd":
        train_ds, val_ds, test_ds = _build_dtd_dataset(args)
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args)
    elif args.dataset == "minc2500":
        train_ds, val_ds, test_ds = _build_minc_dataset(args)
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args)
    elif args.dataset == "imagenet" and getattr(args, "imagenet_shards", False):
        train_ds, val_ds, test_ds, nb_classes = _build_imagenet_sharded(args)
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args,
            drop_last_train=True, timeout=IMAGENET_SHARDS_DATALOADER_TIMEOUT)
    elif args.dataset == "imagenet":
        train_ds, val_ds, test_ds = _build_imagenet_dataset(args)
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args)
    elif args.dataset == "eurosat":
        train_ds, val_ds, test_ds = _build_eurosat_dataset(args)
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args)
    elif args.dataset in ["fmd", "alot", "surotex"]:
        train_ds, val_ds, test_ds =  _build_ImageFolder_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            train_size=args.train_size,
            seed=args.seed,
            args=args,)
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(
            train_ds, val_ds, test_ds, args)
    elif args.dataset in medInfo.keys():
        train_ds, val_ds, test_ds = _build_medmnist_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            train_size=args.train_size,
            seed=args.seed,
            args=args,)
        nb_classes = len(medInfo[args.dataset]['label'])
        train_loader, val_loader, test_loader = _make_dataloaders(train_ds, val_ds, test_ds, args)
    elif args.dataset in ['so2sat_rgb', 'so2sat_s2']:
        train_ds, val_ds, test_ds = _build_so2sat_hf_dataset(
            dataset=args.dataset,
            data_dir=args.data_dir,
            train_size=args.train_size,
            seed=args.seed,
            args=args,
            band_mode='rgb' if args.dataset == 'so2sat_rgb' else 's2'
        )
        nb_classes = cfg["nb_classes"]
        train_loader, val_loader, test_loader = _make_dataloaders(train_ds, val_ds, test_ds, args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    _set_training_args(args, train_loader, val_loader, test_loader, nb_classes, cfg)
    return train_loader, val_loader, test_loader

def reorganize_val_dataset(data_dir):
    """Reorganize ImageNet validation set from flat to class subdirs.

    Moves each image of the flat ILSVRC2012_img_val.tar layout into a subdirectory
    named after its class (e.g., n01440764/).

    Expects:
        data_dir/val/*.JPEG
        data_dir/ILSVRC2012_validation_ground_truth.txt
        data_dir/meta.mat

    After running, the structure will be:
        data_dir/val/n01440764/*.JPEG
        data_dir/val/n01443537/*.JPEG
        ...
    """

    val_dir = os.path.join(data_dir, "val")
    val_groundtruth = os.path.join(data_dir, "ILSVRC2012_validation_ground_truth.txt")
    devkit_meta = os.path.join(data_dir, "meta.mat")

    if not os.path.exists(val_groundtruth):
        raise FileNotFoundError(
            f"Validation ground truth not found: {val_groundtruth}. "
            "Download ILSVRC2012_devkit_t12.tar.gz and extract it."
        )

    # Load validation ground truth labels (1-indexed in file)
    with open(val_groundtruth) as f:
        val_targets = [int(x.strip()) - 1 for x in f.readlines()]
    val_targets = np.array(val_targets)

    # Map synset order to class indices
    meta = sio.loadmat(devkit_meta, squeeze_me=True)["synsets"]
    val_wnids = [meta[i][1] for i in range(1000)]

    # Create subdirectories and move files
    for idx, wnid in enumerate(val_wnids):
        class_dir = os.path.join(val_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)

        target_imgs = np.where(val_targets == idx)

        for j_idx, img_idx in enumerate(target_imgs[0]):
            img_name = f"ILSVRC2012_val_{img_idx+1:08d}.JPEG"
            src_path = os.path.join(val_dir, img_name)
            dst_path = os.path.join(class_dir, img_name)

            if not os.path.exists(src_path):
                if j_idx == 0:
                    print(f"Warning: Expected validation image not found: {src_path}, checking destination")
                if not os.path.exists(dst_path):
                    print(f"Warning: Expected validation image not found at src or destination: {src_path}")
                    continue

            if not os.path.exists(dst_path):
                os.rename(src_path, dst_path)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} / {len(val_wnids)} classes")

