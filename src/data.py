from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def image_transform(training: bool = False):
    steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps.extend(
            [
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class FracAtlasDataset(Dataset):
    def __init__(self, project_root: Path, split: str, training: bool = False, include_augmented: bool = True):
        self.project_root = Path(project_root)
        self.csv_path = self.project_root / "outputs" / f"{split}_processed.csv"
        self.rows = pd.read_csv(self.csv_path)
        if not include_augmented and "is_augmented" in self.rows.columns:
            self.rows = self.rows.loc[~self.rows["is_augmented"].astype(bool)].reset_index(drop=True)
        self.transform = image_transform(training=training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.project_root / Path(row["processed_image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        label = float(row["label"])
        return image, label


class ExternalFractureDataset(Dataset):
    """Supplemental labeled images described by a project-relative CSV manifest."""

    def __init__(self, project_root: Path, manifest_path: Path, training: bool = True):
        self.project_root = Path(project_root)
        self.rows = pd.read_csv(manifest_path)
        self.transform = image_transform(training=training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image_path = self.project_root / Path(row["image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        return image, float(row["label"])
