import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
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


def main():
    parser = argparse.ArgumentParser(description="Test image model inference on a small batch.")
    parser.add_argument("--data_root", type=str, default="data/valid")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="simple",
        choices=["simple", "resnet18"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
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

    train_loader, val_loader, _ = create_dataloaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_ratio=0.2,
        num_workers=0,
    )
    loader = val_loader if len(val_loader.dataset) > 0 else train_loader
    images, targets = next(iter(loader))

    checkpoint = torch.load(args.model_path, map_location=device)
    num_classes = int(checkpoint.get("num_classes"))
    model = create_model(args.model, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    label_path = Path(os.path.dirname(args.model_path)) / (
        Path(args.model_path).stem + "_labels.json"
    )
    label_map = None
    if label_path.is_file():
        with open(label_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)

    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

    images = images.cpu()
    preds = preds.cpu()
    targets = targets.cpu()

    print("Batch size:", images.size(0))
    print("Num classes:", num_classes)
    for i in range(images.size(0)):
        pred_idx = int(preds[i].item())
        true_idx = int(targets[i].item())
        if label_map is not None:
            pred_label = label_map.get(str(pred_idx), str(pred_idx))
            true_label = label_map.get(str(true_idx), str(true_idx))
        else:
            pred_label = str(pred_idx)
            true_label = str(true_idx)
        prob = float(probs[i, pred_idx].item())
        print(f"Sample {i}: pred={pred_label} (idx={pred_idx}, prob={prob:.4f}), true={true_label} (idx={true_idx})")


if __name__ == "__main__":
    main()

