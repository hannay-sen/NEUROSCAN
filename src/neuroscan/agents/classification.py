"""Classification agent: EfficientNet-B0 + calibration + MC-dropout + OOD + Grad-CAM."""
from __future__ import annotations

import numpy as np

from neuroscan.agents.state import NeuroScanState
from neuroscan.config import get_device, load_config
from neuroscan.data.datasets import to_classifier_tensor
from neuroscan.explain.gradcam import gradcam
from neuroscan.models.registry import get_classifier


def classification_agent(state: NeuroScanState) -> dict:
    cfg = load_config()
    classes = cfg["classes"]
    try:
        model, trained = get_classifier()
        img = state["image"]
        x = to_classifier_tensor(img, cfg["classifier"]["image_size"]).unsqueeze(0).to(get_device())
        out = model.predict(x, mc_samples=cfg["classifier"]["mc_samples"])
        idx = int(np.argmax(out["probs"]))
        cam = gradcam(model, x, idx)
        cam_full = _resize_to(cam, img.shape)
        return {"classification": {
            "label": classes[idx],
            "confidence": float(out["probs"][idx]),
            "probs": {c: float(p) for c, p in zip(classes, out["probs"])},
            "entropy": out["entropy"],
            "ood_score": out["ood_score"],
            "ood_threshold": out["ood_threshold"],
            "cam": cam_full,
            "model_trained": trained,
        }}
    except Exception as e:  # noqa: BLE001
        return {"classification": {}, "errors": [f"classification: {e}"]}


def _resize_to(a: np.ndarray, shape) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0

