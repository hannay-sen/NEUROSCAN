"""Report agent: assemble a JSON-serializable report (arrays are left out)."""
from __future__ import annotations

from neuroscan import DISCLAIMER
from neuroscan.agents.state import NeuroScanState


def report_agent(state: NeuroScanState) -> dict:
    cls = state.get("classification") or {}
    seg = state.get("segmentation") or {}
    know = state.get("knowledge") or {}
    safety = state.get("safety") or {}
    sources = {s["chunk_id"]: s for s in know.get("sources", [])}

    report = {
        "image": state.get("image_path"),
        "disclaimer": DISCLAIMER,
        "needs_review": safety.get("needs_review", True),
        "flags": safety.get("flags", []),
        "flag_details": safety.get("details", {}),
        "classification": None if not cls else {
            "label": cls["label"],
            "confidence": round(cls["confidence"], 4),
            "probs": {k: round(v, 4) for k, v in cls["probs"].items()},
            "uncertainty_entropy": round(cls["entropy"], 4),
            "ood_score": None if cls.get("ood_score") is None else round(cls["ood_score"], 3),
            "model_trained": cls.get("model_trained", False),
        },
        "segmentation": None if not seg else {
            "has_mask": seg.get("has_mask", False),
            "area_px": seg.get("area_px", 0),
            "area_pct_brain": seg.get("area_pct_brain", 0.0),
            "location": seg.get("location"),
            "extent": seg.get("extent", {}),
            "model_trained": seg.get("model_trained", False),
            "note": "Size in pixels / % of brain area; location in image coordinates, not patient hemisphere.",
        },
        "explanation": [
            {"text": s["text"], "citation": s.get("citation"),
             "source": sources.get(s.get("chunk_id"), {}).get("title"),
             "url": sources.get(s.get("chunk_id"), {}).get("url")}
            for s in know.get("sentences", [])
        ],
        "explanation_mode": know.get("mode", "skipped"),
        "removed_sentences": [s["text"] for s in know.get("removed", [])],
        "cam_energy_in_mask": safety.get("cam_in_mask"),
        "errors": state.get("errors", []) or [],
    }
    return {"report": report}
