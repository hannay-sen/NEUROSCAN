"""Helpers shared by the evaluation scripts."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuroscan.config import get_device, load_config, path
from neuroscan.data.datasets import ClassificationDataset


def save_json(name: str, obj: dict) -> None:
    out = path("results")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
    print(f"-> results/{name}.json")


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(e)


@torch.no_grad()
def classifier_outputs(model, split: str, mc_samples: int | None = None) -> dict:
    """Per-image calibrated probs, raw probs, MC entropy and OOD score for a split CSV."""
    import torch.nn.functional as F

    cfg = load_config()
    c = cfg["classifier"]
    ds = ClassificationDataset(path("splits") / f"{split}.csv", cfg["classes"], c["image_size"])
    dl = DataLoader(ds, c["batch_size"], num_workers=c["num_workers"])
    device = get_device()
    mc_samples = mc_samples or c["mc_samples"]
    head = model.backbone.get_classifier()
    p_drop = model._drop_rate()
    raw, cal, ent, ood, labels = [], [], [], [], []
    model.eval()
    for x, y in dl:
        f = model.features(x.to(device))
        logits = head(f)
        raw.append(F.softmax(logits, -1).cpu())
        cal.append(model.calibrated_probs(logits).cpu())
        mc = torch.stack([model.calibrated_probs(head(F.dropout(f, p_drop, training=True)))
                          for _ in range(mc_samples)]).mean(0)
        ent.append((-(mc * (mc + 1e-12).log()).sum(1) / np.log(mc.shape[1])).cpu())
        if model.ood_fitted():
            ood.append(model.mahalanobis(f))
        labels.append(y)
    return {
        "raw": torch.cat(raw).numpy(), "probs": torch.cat(cal).numpy(),
        "entropy": torch.cat(ent).numpy(), "labels": torch.cat(labels).numpy(),
        "ood": torch.cat(ood).numpy() if ood else None,
        "ood_threshold": float(model.ood_threshold) if model.ood_fitted() else None,
        "paths": ds.df.image_path.tolist(),
    }
