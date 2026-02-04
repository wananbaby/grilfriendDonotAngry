from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def create_dataloaders(
    data_root: str,
    image_size: int = 112,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    root = Path(data_root)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.ImageFolder(str(root), transform=transform)
    num_classes = len(dataset.classes)
    if num_classes == 0 or len(dataset) == 0:
        raise RuntimeError(f"No images found in {data_root}")
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, num_classes

