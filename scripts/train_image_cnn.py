import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.image_dataset import create_dataloaders


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(name: str, num_classes: int) -> nn.Module:
    if name == "simple":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        m = models.resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"Unsupported model: {name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def save_model(model: nn.Module, num_classes: int, output_path: str, class_names):
    out_dir = Path(os.path.dirname(output_path)) if os.path.dirname(output_path) else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
    }
    torch.save(state, output_path)
    label_map = {i: name for i, name in enumerate(class_names)}
    label_path = out_dir / (Path(output_path).stem + "_labels.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train image emotion CNN on local dataset.")
    parser.add_argument("--data_root", type=str, default="data/valid")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_path", type=str, default="models/image_cnn.pth")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "resnet18"],
    )
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = create_dataloaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=0.2,
        num_workers=0,
    )
    dataset = train_loader.dataset.dataset
    class_names = dataset.classes
    model = create_model(args.model, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, device, optimizer, criterion
        )
        val_loss, val_acc = eval_epoch(model, val_loader, device, criterion)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
    save_model(model, num_classes, args.output_path, class_names)
    print(f"Saved image model to {args.output_path}")


if __name__ == "__main__":
    main()
