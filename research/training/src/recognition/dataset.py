from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class AnimeFaceDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None, class_to_idx: dict[str, int] | None = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.class_names: list[str] = []
        self.class_to_idx = dict(class_to_idx or {})
        self._scan()

    def _scan(self) -> None:
        if not self.root_dir.exists():
            return
        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue
            class_name = folder.name
            if class_name not in self.class_to_idx:
                class_index = len(self.class_to_idx)
                self.class_to_idx[class_name] = class_index
                self.class_names.append(class_name)
            class_index = self.class_to_idx[class_name]
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() in VALID_IMAGE_SUFFIXES:
                    self.samples.append((str(image_path), class_index))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


@dataclass
class DatasetSplit:
    train_dataset: AnimeFaceDataset
    val_dataset: AnimeFaceDataset
    class_names: list[str]
    class_to_idx: dict[str, int]
    total_classes: int
    valid_classes: int
    train_samples: int
    val_samples: int


def build_dataset_split(
    dataset_dir: str | Path,
    train_transform,
    val_transform,
    train_ratio: float,
    min_images_per_class: int,
    seed: int,
) -> DatasetSplit:
    dataset_dir = Path(dataset_dir)
    full_dataset = AnimeFaceDataset(dataset_dir, transform=None)

    class_counts: dict[str, int] = {}
    idx_to_name = {index: name for name, index in full_dataset.class_to_idx.items()}
    for _, label in full_dataset.samples:
        class_name = idx_to_name[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    valid_classes = {
        name: index
        for name, index in full_dataset.class_to_idx.items()
        if class_counts.get(name, 0) >= min_images_per_class
    }
    new_class_to_idx = {name: index for index, name in enumerate(sorted(valid_classes.keys()))}

    rng = random.Random(seed)
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []

    for class_name in sorted(new_class_to_idx.keys()):
        old_index = full_dataset.class_to_idx[class_name]
        class_images = [(path, label) for path, label in full_dataset.samples if label == old_index]
        rng.shuffle(class_images)
        if len(class_images) <= 1:
            split_index = len(class_images)
        else:
            split_index = int(len(class_images) * train_ratio)
            split_index = min(max(split_index, 1), len(class_images) - 1)
        new_index = new_class_to_idx[class_name]
        for path, _ in class_images[:split_index]:
            train_samples.append((path, new_index))
        for path, _ in class_images[split_index:]:
            val_samples.append((path, new_index))

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    train_dataset = AnimeFaceDataset.__new__(AnimeFaceDataset)
    train_dataset.root_dir = dataset_dir
    train_dataset.transform = train_transform
    train_dataset.samples = train_samples
    train_dataset.class_to_idx = new_class_to_idx
    train_dataset.class_names = sorted(new_class_to_idx.keys())

    val_dataset = AnimeFaceDataset.__new__(AnimeFaceDataset)
    val_dataset.root_dir = dataset_dir
    val_dataset.transform = val_transform
    val_dataset.samples = val_samples
    val_dataset.class_to_idx = new_class_to_idx
    val_dataset.class_names = sorted(new_class_to_idx.keys())

    return DatasetSplit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_names=sorted(new_class_to_idx.keys()),
        class_to_idx=new_class_to_idx,
        total_classes=len(full_dataset.class_to_idx),
        valid_classes=len(new_class_to_idx),
        train_samples=len(train_samples),
        val_samples=len(val_samples),
    )
