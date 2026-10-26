import os
from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


_STATS: dict[str, dict] = {
    "cifar10": {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "cifar100": {"mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
}

_DATASET_CLS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}

# Sub-directory each archive extracts to, relative to the --data directory.
_EXTRACTED_DIR = {
    "cifar10": "cifar-10-batches-py",
    "cifar100": "cifar-100-python",
}


def _check_data_dir(dataset_name: str, data_dir: str) -> None:
    """Fail early with a helpful message if the extracted dataset is not present."""
    expected = Path(data_dir) / _EXTRACTED_DIR[dataset_name]
    if not expected.is_dir():
        raise FileNotFoundError(
            f"Could not find '{expected}'.\n"
            f"Download and extract {dataset_name} into '{data_dir}' first — "
            f"see the README for the curl commands."
        )


def get_loaders(
    dataset_name: str,
    batch_size: int,
    data_dir: str,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    _check_data_dir(dataset_name, data_dir)

    cls = _DATASET_CLS[dataset_name]
    stats = _STATS[dataset_name]
    num_workers = min(4, os.cpu_count() or 1)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])

    train_set = cls(root=data_dir, train=True, download=False, transform=train_transform)
    val_set = cls(root=data_dir, train=False, download=False, transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, len(train_set.classes), train_set.classes


def get_inference_transform(dataset_name: str) -> transforms.Compose:
    stats = _STATS[dataset_name]
    return transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(**stats),
    ])
