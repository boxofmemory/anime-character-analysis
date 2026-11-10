from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.common.experiment import ExperimentRun
from .arcface_model import ResNet50ArcFace
from .dataset import build_dataset_split


def validate_arcface_config(config: dict) -> None:
    dataset_dir = Path(config["dataset_dir"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"ArcFace dataset not found: {dataset_dir}")


def train_arcface_from_config(config: dict, run: ExperimentRun) -> dict:
    validate_arcface_config(config)
    _set_seed(int(config.get("seed", 42)))

    device = _resolve_device(config.get("device"))
    input_size = int(config.get("input_size", 224))
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 60))
    learning_rate = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 5e-4))
    scale = float(config.get("scale", 30.0))
    margin = float(config.get("margin", 0.5))
    train_ratio = float(config.get("train_ratio", 0.8))
    min_images_per_class = int(config.get("min_images_per_class", 5))
    early_stop_patience = int(config.get("early_stop_patience", 10))
    num_workers = int(config.get("num_workers", 0))
    drop_last = bool(config.get("drop_last", True))

    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    split = build_dataset_split(
        dataset_dir=config["dataset_dir"],
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=train_ratio,
        min_images_per_class=min_images_per_class,
        seed=int(config.get("seed", 42)),
    )
    if split.valid_classes == 0:
        raise RuntimeError("No valid classes found for ArcFace training.")

    train_loader = DataLoader(
        split.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        split.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ResNet50ArcFace(
        num_classes=split.valid_classes,
        embedding_size=int(config.get("embedding_size", 2048)),
        scale=scale,
        margin=margin,
    ).to(device)

    warm_start_path = config.get("pretrained_backbone")
    if warm_start_path and Path(warm_start_path).exists():
        _load_checkpoint_backbone(model, Path(warm_start_path), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_val_targets: list[int] = []
    best_val_predictions: list[int] = []
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
        "epoch_time_sec": [],
    }

    best_path = run.artifacts_dir / "best_arcface.pth"
    last_path = run.artifacts_dir / "last_arcface.pth"

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)

        model.eval()
        val_correct = 0
        val_total = 0
        current_val_targets: list[int] = []
        current_val_predictions: list[int] = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                features = model(images)
                features_norm = F.normalize(features, dim=1)
                weight_norm = F.normalize(model.arcface.weight, dim=1)
                logits = F.linear(features_norm, weight_norm) * model.arcface.scale
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                current_val_targets.extend(labels.cpu().tolist())
                current_val_predictions.extend(predicted.cpu().tolist())

        val_acc = 100.0 * val_correct / max(val_total, 1)
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(round(lr, 8))
        history["epoch_time_sec"].append(round(elapsed, 2))

        checkpoint = _checkpoint_payload(model, split.class_names, epoch, val_acc)
        torch.save(checkpoint, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_val_targets = list(current_val_targets)
            best_val_predictions = list(current_val_predictions)
            torch.save(checkpoint, best_path)
        else:
            patience_counter += 1

        scheduler.step()
        if patience_counter >= early_stop_patience:
            break

    curves_path = run.artifacts_dir / "training_curves.png"
    confusion_matrix_path = run.artifacts_dir / "confusion_matrix.png"
    _save_training_curves(history, curves_path)
    _save_confusion_matrix(best_val_targets, best_val_predictions, split.class_names, confusion_matrix_path)

    return {
        "experiment_id": run.experiment_id,
        "model_name": "resnet50_arcface",
        "dataset_dir": config["dataset_dir"],
        "device": str(device),
        "total_classes": split.total_classes,
        "valid_classes": split.valid_classes,
        "train_samples": split.train_samples,
        "val_samples": split.val_samples,
        "epochs_requested": epochs,
        "epochs_completed": len(history["val_acc"]),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "history": history,
        "artifacts": {
            "best_weights": best_path,
            "last_weights": last_path,
            "training_curves": curves_path if curves_path.exists() else None,
            "confusion_matrix": confusion_matrix_path if confusion_matrix_path.exists() else None,
        },
    }


def _checkpoint_payload(model: ResNet50ArcFace, class_names: list[str], epoch: int, val_acc: float) -> dict:
    return {
        "backbone": model.backbone.state_dict(),
        "bn": model.bn.state_dict(),
        "class_names": class_names,
        "embedding_size": 2048,
        "val_acc": float(val_acc),
        "epoch": epoch,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_value) -> torch.device:
    if device_value is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_value, int):
        return torch.device(f"cuda:{device_value}" if torch.cuda.is_available() else "cpu")
    device_text = str(device_value)
    if device_text.isdigit():
        return torch.device(f"cuda:{device_text}" if torch.cuda.is_available() else "cpu")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_text)


def _load_checkpoint_backbone(model: ResNet50ArcFace, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "backbone" in checkpoint:
        model.backbone.load_state_dict(checkpoint["backbone"], strict=False)
    if "bn" in checkpoint:
        model.bn.load_state_dict(checkpoint["bn"], strict=False)


def _save_training_curves(history: dict, output_path: Path) -> None:
    if not history.get("train_loss"):
        return
    epochs = list(range(1, len(history["train_loss"]) + 1))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#10b981", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Acc", color="#f59e0b", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    if not targets or not predictions or not class_names:
        return

    size = len(class_names)
    matrix = np.zeros((size, size), dtype=int)
    for target, prediction in zip(targets, predictions):
        if 0 <= target < size and 0 <= prediction < size:
            matrix[target, prediction] += 1

    figure, axis = plt.subplots(figsize=(max(6, size * 0.6), max(5, size * 0.55)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_title("Validation Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_xticks(range(size))
    axis.set_yticks(range(size))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(size):
        for column in range(size):
            value = matrix[row, column]
            if value == 0:
                continue
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#111827",
                fontsize=7,
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
