
"""DTD loaders aligned with the VORTEX `datasets.DTD` + `classify_features.py` protocol.

Data layout matches ``torchvision.datasets.DTD`` (``--data-root``/``dtd``/...).

Preprocessing matches VORTEX: ``ToTensor`` → ImageNet ``Normalize`` → ``Resize``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet normalization as in VORTEX `classify_features.py`
VORTEX_MEAN = (0.485, 0.456, 0.406)
VORTEX_STD = (0.229, 0.224, 0.225)

DTD_NUM_CLASSES = 47


def vortex_style_transform(input_size: int) -> transforms.Compose:
    """Same transform chain as VORTEX (order matters for comparability)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(VORTEX_MEAN, VORTEX_STD),
            transforms.Resize((input_size, input_size)),
        ]
    )


def _get_targets(dataset) -> np.ndarray:
    """Extract integer target labels from a dataset.
    
    Handles standard datasets (.targets, .labels), DTD datasets (._labels),
    ConcatDataset by recursively extracting from component datasets,
    and Subset by indexing the base dataset's targets.
    """
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "_labels"):
        return np.asarray(dataset._labels)
    # Handle Subset by getting targets from base dataset and indexing
    if isinstance(dataset, Subset):
        base_targets = _get_targets(dataset.dataset)
        return base_targets[np.array(dataset.indices)]
    # Handle ConcatDataset
    if isinstance(dataset, ConcatDataset):
        all_targets = []
        for ds in dataset.datasets:
            all_targets.append(_get_targets(ds))
        return np.concatenate(all_targets)
    raise AttributeError(
        f"{type(dataset).__name__} has no .targets, .labels, or ._labels attribute"
    )


def stratified_subset(dataset, train_size, seed=42) -> Subset:
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
    return Subset(dataset, indices)


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
    if isinstance(subset_or_dataset, Subset):
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


def _assert_compatible_labels(train_ds: datasets.DTD, val_ds: datasets.DTD) -> None:
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise RuntimeError(
            "Train/val class_to_idx mismatch; use a single joint label space "
            "(check torchvision DTD version / corrupted metadata)."
        )


def build_partition_datasets(
    data_root: str,
    partition: int,
    transform: Callable,
    download: bool = False,
) -> Tuple[ConcatDataset, datasets.DTD]:
    """Train+val (merged) and test sets for one official DTD fold (partition 1..10)."""
    if not (1 <= partition <= 10):
        raise ValueError("partition must be in [1, 10]")

    train_ds = datasets.DTD(
        data_root,
        split="train",
        partition=partition,
        transform=transform,
        download=download,
    )
    val_ds = datasets.DTD(
        data_root,
        split="val",
        partition=partition,
        transform=transform,
        download=download,
    )
    _assert_compatible_labels(train_ds, val_ds)
    test_ds = datasets.DTD(
        data_root,
        split="test",
        partition=partition,
        transform=transform,
        download=download,
    )
    train_merge = ConcatDataset([train_ds, val_ds])
    return train_merge, test_ds


def build_partition_loaders(
    data_root: str,
    partition: int,
    batch_size: int,
    *,
    workers: int = 4,
    input_size: int = 224,
    train_size: Optional[float | int] = None,
    download: bool = False,
    pin_memory: Optional[bool] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """``DataLoader``s for training (train+val merged) and evaluation (test).
    
    Parameters
    ----------
    data_root : str
        Root directory for DTD dataset
    partition : int
        DTD partition (fold) number, 1-10
    batch_size : int
        Batch size per DataLoader
    workers : int, optional
        Number of DataLoader workers
    input_size : int, optional
        Input spatial size for transforms
    train_size : float | int | None, optional
        If provided, subsample training set to this fraction (0, 1] or absolute count
    download : bool, optional
        Whether to download dataset if missing
    pin_memory : bool | None, optional
        Pin memory in DataLoaders
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    train_loader, test_loader : DataLoader
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    transform = vortex_style_transform(input_size)
    train_ds, test_ds = build_partition_datasets(
        data_root, partition, transform, download=download
    )
    
    # Apply train_size subsetting if specified
    if train_size is not None:
        train_ds = stratified_subset(train_ds, train_size, seed=seed)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader
