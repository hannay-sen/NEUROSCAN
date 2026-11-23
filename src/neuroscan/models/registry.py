"""Load models once. Falls back to untrained weights when no checkpoint exists,
so the app and smoke tests run before training (outputs are meaningless until trained)."""
from __future__ import annotations

from functools import lru_cache

import torch

from neuroscan.config import get_device, load_config, path
from neuroscan.models.classifier import TumorClassifier
from neuroscan.models.unet import build_unet


def build_classifier(pretrained: bool = False) -> TumorClassifier:
    cfg = load_config()
    c = cfg["classifier"]
    return TumorClassifier(len(cfg["classes"]), c["model_name"], pretrained, c["drop_rate"])


def build_segmenter():
    s = load_config()["segmentation"]
    return build_unet(s["channels"], s["strides"], s["num_res_units"])


@lru_cache(maxsize=4)
def get_classifier(tag: str = "") -> tuple[TumorClassifier, bool]:
    """Returns (model, is_trained)."""
    model = build_classifier(pretrained=False)
    ckpt = path("checkpoints") / f"classifier{('_' + tag) if tag else ''}.pt"
    trained = ckpt.exists()
    if trained:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model.to(get_device()).eval(), trained


@lru_cache(maxsize=1)
def get_segmenter() -> tuple[torch.nn.Module, bool]:
    model = build_segmenter()
    ckpt = path("checkpoints") / "unet.pt"
    trained = ckpt.exists()
    if trained:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    return model.to(get_device()).eval(), trained
