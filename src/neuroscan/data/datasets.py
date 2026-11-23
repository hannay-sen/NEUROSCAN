"""Image loading, preprocessing and PyTorch datasets shared by training and inference."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_gray(src) -> np.ndarray:
    """Path, PIL image or array -> float32 grayscale array in [0, 1]."""
    if isinstance(src, np.ndarray):
        a = src.astype(np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        return a / 255.0 if a.max() > 1.0 else a
    img = src if isinstance(src, Image.Image) else Image.open(src)
    return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def resize(a: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    im = Image.fromarray((a * 255).astype(np.uint8)) if a.dtype != np.uint8 else Image.fromarray(a)
    im = im.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def to_classifier_tensor(gray: np.ndarray, size: int) -> torch.Tensor:
    """HxW [0,1] -> 3xSxS ImageNet-normalized tensor."""
    t = torch.from_numpy(resize(gray, size)).unsqueeze(0).repeat(3, 1, 1)
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def to_seg_tensor(gray: np.ndarray, size: int) -> torch.Tensor:
    """HxW [0,1] -> 1xSxS tensor, per-image z-score."""
    a = resize(gray, size)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return torch.from_numpy(a).unsqueeze(0)


def _augment(img: Image.Image, mask: Image.Image | None = None):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT) if mask is not None else None
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR)
    mask = mask.rotate(angle, resample=Image.NEAREST) if mask is not None else None
    return img, mask


class ClassificationDataset(Dataset):
    def __init__(self, csv_path, classes: list[str], image_size: int, train: bool = False):
        self.df = pd.read_csv(csv_path)
        self.classes = classes
        self.size = image_size
        self.train = train
        self.targets = [classes.index(l) for l in self.df.label]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.image_path.iat[i]).convert("L")
        if self.train:
            img, _ = _augment(img)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.clip(arr * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05), 0, 1)
        else:
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return to_classifier_tensor(arr, self.size), self.targets[i]


class SegmentationDataset(Dataset):
    def __init__(self, csv_path, image_size: int, train: bool = False):
        self.df = pd.read_csv(csv_path)
        self.size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.image_path.iat[i]).convert("L")
        mask = Image.open(self.df.mask_path.iat[i]).convert("L")
        if self.train:
            img, mask = _augment(img, mask)
        x = to_seg_tensor(np.asarray(img, dtype=np.float32) / 255.0, self.size)
        m = resize(np.asarray(mask, dtype=np.float32) / 255.0, self.size, nearest=True)
        y = torch.from_numpy((m > 0.5).astype(np.float32)).unsqueeze(0)
        return x, y
