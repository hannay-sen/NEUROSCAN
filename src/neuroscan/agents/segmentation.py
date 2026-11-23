"""Segmentation agent: U-Net mask, size (% of brain), location and extent.

Size is in pixels and % of brain area because 2D JPG/PNG slices carry no pixel spacing.
Location is in image coordinates ("image-left"), not patient hemisphere.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from neuroscan.agents.state import NeuroScanState
from neuroscan.config import get_device, load_config
from neuroscan.data.datasets import to_seg_tensor
from neuroscan.models.registry import get_segmenter


def otsu_threshold(a: np.ndarray) -> float:
    hist, edges = np.histogram(a.ravel(), bins=256, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    w = np.cumsum(p)
    mu = np.cumsum(p * edges[:-1])
    between = (mu[-1] * w - mu) ** 2 / (w * (1 - w) + 1e-12)
    return float(edges[np.argmax(between)])


def brain_mask(img: np.ndarray) -> np.ndarray:
    return img > max(otsu_threshold(img) * 0.5, 0.02)


def describe_mask(mask: np.ndarray, img: np.ndarray) -> dict:
    h, w = mask.shape
    area = int(mask.sum())
    brain = int((brain_mask(img) | mask).sum()) or h * w   # tumor counts as brain
    if area == 0:
        return {"area_px": 0, "area_pct_brain": 0.0, "location": "none", "extent": {}}
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / h, xs.mean() / w
    vert = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    horiz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    location = "image-center" if (vert, horiz) == ("middle", "center") else f"image-{vert}-{horiz}"
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return {
        "area_px": area,
        "area_pct_brain": round(100.0 * area / brain, 2),
        "location": location,
        "extent": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                   "width": x1 - x0 + 1, "height": y1 - y0 + 1},
    }


def segmentation_agent(state: NeuroScanState) -> dict:
    cfg = load_config()
    s = cfg["segmentation"]
    try:
        model, trained = get_segmenter()
        img = state["image"]
        x = to_seg_tensor(img, s["image_size"]).unsqueeze(0).to(get_device())
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        prob_full = np.asarray(Image.fromarray((prob * 255).astype(np.uint8))
                               .resize((img.shape[1], img.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
        mask = prob_full >= s["mask_threshold"]
        info = describe_mask(mask, img)
        has_mask = info["area_pct_brain"] >= cfg["safety"]["min_mask_area_pct"]
        return {"segmentation": {"mask": mask, "has_mask": bool(has_mask), "model_trained": trained, **info}}
    except Exception as e:  # noqa: BLE001
        return {"segmentation": {}, "errors": [f"segmentation: {e}"]}
