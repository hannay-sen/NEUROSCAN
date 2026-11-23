"""Shared state passed between agents. THIS IS THE CONTRACT BETWEEN AGENTS:
agree on changes before editing it.

Each agent writes only its own key. Arrays (image, mask, cam) stay in memory and are
never written into the JSON report.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Classification(TypedDict, total=False):
    label: str                    # glioma | meningioma | notumor | pituitary
    confidence: float             # calibrated max probability
    probs: dict[str, float]       # calibrated, per class
    entropy: float                # normalized MC-dropout predictive entropy, 0..1
    ood_score: float | None       # Mahalanobis distance (None if OOD stats not fitted)
    ood_threshold: float | None
    cam: Any                      # np.ndarray HxW Grad-CAM in [0,1] at original resolution
    model_trained: bool


class Segmentation(TypedDict, total=False):
    mask: Any                     # np.ndarray HxW bool at original resolution
    area_px: int
    area_pct_brain: float         # % of brain area (2D slices carry no pixel spacing)
    location: str                 # image coordinates, e.g. "image-upper-left"
    extent: dict                  # bbox in px: x0, y0, x1, y1, width, height
    has_mask: bool
    model_trained: bool


class Knowledge(TypedDict, total=False):
    sentences: list[dict]         # {"text", "citation": int, "chunk_id", "grounded": bool|None}
    sources: list[dict]           # retrieved chunks: chunk_id, title, url, text
    mode: str                     # llm | extractive | skipped
    removed: list[dict]           # sentences removed by the grounding review


class Safety(TypedDict, total=False):
    flags: list[str]
    details: dict[str, str]
    needs_review: bool
    cam_in_mask: float | None


class NeuroScanState(TypedDict, total=False):
    image_path: str
    image: Any                    # np.ndarray HxW float32 in [0,1]
    valid: bool
    classification: Classification
    segmentation: Segmentation
    knowledge: Knowledge
    safety: Safety
    report: dict
    errors: Annotated[list[str], operator.add]   # any agent may append
