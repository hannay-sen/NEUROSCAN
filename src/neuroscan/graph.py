"""LangGraph workflow.

  intake ─┬─> classification ─┐
          └─> segmentation  ──┴─> knowledge -> safety -> report
  (invalid image: intake -> safety -> report)

Classification and segmentation run in parallel.

Usage:  python -m neuroscan.graph path/to/scan.jpg
"""
from __future__ import annotations

import json
import sys

import numpy as np
from langgraph.graph import END, START, StateGraph

from neuroscan.agents.classification import classification_agent
from neuroscan.agents.knowledge import knowledge_agent
from neuroscan.agents.report import report_agent
from neuroscan.agents.safety import safety_agent
from neuroscan.agents.segmentation import segmentation_agent
from neuroscan.agents.state import NeuroScanState
from neuroscan.data.datasets import load_gray


def intake(state: NeuroScanState) -> dict:
    try:
        img = state["image"] if state.get("image") is not None else load_gray(state["image_path"])
        img = np.asarray(img, dtype=np.float32)
        if img.ndim != 2 or min(img.shape) < 64:
            raise ValueError(f"expected a 2D slice of at least 64x64, got shape {img.shape}")
        if float(img.std()) < 1e-3:
            raise ValueError("image is blank (constant intensity)")
        return {"image": img, "valid": True}
    except Exception as e:  # noqa: BLE001
        return {"image": None, "valid": False, "errors": [f"intake: {e}"]}


def _route(state: NeuroScanState):
    return ["classification", "segmentation"] if state.get("valid") else "safety"


def build_graph():
    g = StateGraph(NeuroScanState)
    g.add_node("intake", intake)
    g.add_node("classification", classification_agent)
    g.add_node("segmentation", segmentation_agent)
    g.add_node("knowledge", knowledge_agent)
    g.add_node("safety", safety_agent)
    g.add_node("report", report_agent)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", _route, ["classification", "segmentation", "safety"])
    g.add_edge(["classification", "segmentation"], "knowledge")   # waits for both
    g.add_edge("knowledge", "safety")
    g.add_edge("safety", "report")
    g.add_edge("report", END)
    return g.compile()


_GRAPH = None


def run(image_path: str | None = None, image=None) -> NeuroScanState:
    """Run the full pipeline. Pass a file path or an array/PIL image."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    init: NeuroScanState = {"image_path": image_path or "<uploaded>", "errors": []}
    if image is not None:
        init["image"] = load_gray(image)
    return _GRAPH.invoke(init)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m neuroscan.graph path/to/scan.jpg")
    print(json.dumps(run(sys.argv[1])["report"], indent=2))
