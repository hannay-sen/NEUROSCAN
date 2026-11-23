"""Load configs/config.yaml once and resolve paths relative to the project root."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(os.environ.get("NEUROSCAN_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_PATH = Path(os.environ.get("NEUROSCAN_CONFIG", PROJECT_ROOT / "configs" / "config.yaml"))


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def path(key: str) -> Path:
    """Absolute path for an entry under `paths:` in the config."""
    p = Path(load_config()["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_device():
    import torch

    want = load_config().get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def seed_everything(seed: int | None = None) -> None:
    import torch

    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
