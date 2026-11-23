"""Smoke tests: run without data, trained weights or API keys."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _fake_slice(size=256, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    brain = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size * 0.4) ** 2
    img = brain * 0.4 + rng.normal(0, 0.03, (size, size))
    blob = ((yy - 90) ** 2 + (xx - 160) ** 2) < 20 ** 2
    return np.clip(img + blob * 0.4, 0, 1).astype(np.float32)


def test_config_loads():
    from neuroscan.config import load_config

    cfg = load_config()
    assert cfg["no_tumor_class"] in cfg["classes"]
    assert 0 < cfg["safety"]["min_confidence"] < 1


def test_classifier_predict_and_calibration():
    from neuroscan.models.registry import build_classifier

    torch.manual_seed(0)
    m = build_classifier(pretrained=False)
    out = m.predict(torch.randn(1, 3, 224, 224), mc_samples=5)
    assert out["probs"].shape == (4,) and abs(out["probs"].sum() - 1) < 1e-4
    assert 0 <= out["entropy"] <= 1 and out["ood_score"] is None

    logits, labels = torch.randn(64, 4) * 5, torch.randint(0, 4, (64,))
    assert 0.05 <= m.fit_temperature(logits, labels) <= 20
    feats = torch.randn(80, 16)
    thr = m.fit_ood(feats, torch.arange(80) % 4, torch.randn(20, 16))
    assert m.ood_fitted() and thr > 0


def test_unet_shapes_and_mask_description():
    from neuroscan.agents.segmentation import describe_mask
    from neuroscan.models.registry import build_segmenter

    y = build_segmenter()(torch.randn(2, 1, 256, 256))
    assert y.shape == (2, 1, 256, 256)
    img = _fake_slice()
    yy, xx = np.mgrid[:256, :256]
    info = describe_mask(((yy - 60) ** 2 + (xx - 200) ** 2) < 15 ** 2, img)
    assert info["location"] == "image-upper-right" and 0 < info["area_pct_brain"] < 100


def test_safety_rules():
    from neuroscan.agents.safety import safety_agent

    cam = np.zeros((100, 100), np.float32); cam[:20, :20] = 1      # attention in a corner
    mask = np.zeros((100, 100), bool); mask[60:80, 60:80] = True     # tumor elsewhere
    state = {
        "valid": True,
        "classification": {"label": "glioma", "confidence": 0.4, "entropy": 0.9,
                           "ood_score": 50.0, "ood_threshold": 10.0, "cam": cam},
        "segmentation": {"has_mask": True, "mask": mask, "area_pct_brain": 4.0},
        "knowledge": {"sources": [{"chunk_id": "a::0", "text": "Gliomas arise from glial cells.",
                                   "title": "t", "url": ""}],
                      "sentences": [{"text": "Gliomas arise from glial cells.", "chunk_id": "a::0", "citation": 1},
                                    {"text": "Gliomas are always benign.", "chunk_id": "a::0", "citation": 1}],
                      "mode": "extractive"},
        "errors": [],
    }
    s = safety_agent(state)
    flags = set(s["safety"]["flags"])
    assert {"low_confidence", "high_uncertainty", "out_of_distribution",
            "attention_outside_tumor", "explanation_poorly_grounded"} <= flags
    assert [x["text"] for x in s["knowledge"]["sentences"]] == ["Gliomas arise from glial cells."]

    s2 = safety_agent({"valid": True, "errors": [],
                       "classification": {"label": "notumor", "confidence": 0.99, "entropy": 0.01,
                                          "ood_score": None, "ood_threshold": None},
                       "segmentation": {"has_mask": True, "mask": mask, "area_pct_brain": 4.0}})
    assert s2["safety"]["flags"] == ["mask_but_no_tumor_predicted"]


def test_graph_end_to_end_untrained(tmp_path):
    from neuroscan.graph import run

    fp = tmp_path / "scan.png"
    Image.fromarray((_fake_slice() * 255).astype(np.uint8)).save(fp)
    state = run(str(fp))
    report = state["report"]
    json.dumps(report)                                   # must be JSON-serializable
    assert report["classification"]["label"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert report["classification"]["model_trained"] is False
    assert state["classification"]["cam"].shape == (256, 256)
    assert state["segmentation"]["mask"].shape == (256, 256)
    assert isinstance(report["flags"], list) and report["disclaimer"]

    bad = run(image=np.zeros((128, 128), np.float32))
    assert bad["report"]["flags"] == ["invalid_image"]
