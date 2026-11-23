"""Grad-CAM (Selvaraju et al., 2017) for the EfficientNet classifier."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, x: torch.Tensor, class_idx: int, target_layer=None) -> np.ndarray:
    """Returns an HxW heatmap in [0, 1] at the input resolution."""
    layer = target_layer or model.backbone.conv_head
    acts, grads = {}, {}

    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = model(x)
        logits[0, class_idx].backward()
        a, g = acts["a"].detach(), grads["g"].detach()
    finally:
        h1.remove()
        h2.remove()

    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy().astype(np.float32)
