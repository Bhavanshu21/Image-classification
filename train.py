import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report
from tqdm import tqdm


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, val_tfms


def build_dataloaders_from_folder(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, datasets.ImageFolder]:
    train_tfms, val_tfms = build_transforms(image_size=image_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds


def build_dataloaders_from_cifar10(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, datasets.VisionDataset]:
    train_tfms, val_tfms = build_transforms(image_size=image_size)

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    val_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    # Use a strong pretrained backbone (ResNet-50) for high-end performance
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0.0

    print("\nValidation classification report:")
    print(classification_report(all_labels, all_preds))

    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, class_names, output_dir: str, filename: str = "best_model.pth") -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, save_path)
    print(f"Saved best model to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-end image classification training")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to dataset root. For folder dataset: expects train/ and val/. For CIFAR-10: download location.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "folder"],
        help="Which dataset to use: 'cifar10' (auto-download) or 'folder' (custom train/val folders).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == "cifar10":
        print("Using CIFAR-10 dataset (auto-download).")
        train_loader, val_loader, train_ds = build_dataloaders_from_cifar10(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
    else:
        print("Using custom folder dataset (expects train/ and val/ subfolders).")
        train_loader, val_loader, train_ds = build_dataloaders_from_folder(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
        )
    num_classes = len(train_ds.classes)
    print(f"Found {num_classes} classes: {train_ds.classes}")

    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, train_ds.classes, args.output_dir, filename="best_model.pth")


if __name__ == "__main__":
    main()


