"""Safety agent: rules + Grad-CAM check + LLM judge (grounding review).

Flags uncertain or inconsistent cases for review and removes explanation sentences
that are not supported by their cited source.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from neuroscan import llm
from neuroscan.agents.state import NeuroScanState
from neuroscan.config import load_config

JUDGE_SYSTEM = "You are a strict fact-checker. Answer with exactly one word: SUPPORTED or UNSUPPORTED."


def judge_sentence(sentence: str, source_text: str) -> bool:
    """True if the LLM judge says the source fully supports the sentence."""
    prompt = (f"Source excerpt:\n{source_text}\n\nSentence:\n{sentence}\n\n"
              "Is every claim in the sentence directly supported by the source excerpt?")
    answer = llm.complete(prompt, system=JUDGE_SYSTEM, max_tokens=5).upper()
    return answer.startswith("SUPPORTED")


def _norm(t: str) -> str:
    return " ".join(t.lower().split())


def grounding_review(knowledge: dict) -> tuple[list[dict], list[dict], bool]:
    """Returns (kept, removed, check_failed)."""
    sources = {s["chunk_id"]: s for s in knowledge.get("sources", [])}
    kept, removed, failed = [], [], False
    for s in knowledge.get("sentences", []):
        src = sources.get(s.get("chunk_id"))
        if src is None:
            ok = False                                         # no valid citation
        elif knowledge.get("mode") == "extractive":
            ok = _norm(s["text"]) in _norm(src["text"])        # verbatim quote check
        else:
            try:
                ok = judge_sentence(s["text"], src["text"])
            except Exception:  # noqa: BLE001
                failed, ok = True, False
        s = {**s, "grounded": ok}
        (kept if ok else removed).append(s)
    return kept, removed, failed


def cam_energy_in_mask(cam: np.ndarray, mask: np.ndarray, dilation_px: int) -> float:
    m = binary_dilation(mask, iterations=dilation_px) if dilation_px > 0 else mask
    total = float(cam.sum())
    return float(cam[m].sum() / total) if total > 0 else 0.0


def safety_agent(state: NeuroScanState) -> dict:
    cfg = load_config()
    t = cfg["safety"]
    no_tumor = cfg["no_tumor_class"]
    flags, details = [], {}

    def flag(name: str, why: str) -> None:
        flags.append(name)
        details[name] = why

    if not state.get("valid", False):
        flag("invalid_image", "the input could not be read as a usable MRI slice")
        return {"safety": {"flags": flags, "details": details, "needs_review": True, "cam_in_mask": None}}

    cls = state.get("classification") or {}
    seg = state.get("segmentation") or {}
    know = state.get("knowledge") or {}
    cam_in_mask = None

    if cls:
        if cls["confidence"] < t["min_confidence"]:
            flag("low_confidence", f"calibrated confidence {cls['confidence']:.2f} < {t['min_confidence']}")
        if cls["entropy"] > t["max_entropy"]:
            flag("high_uncertainty", f"MC-dropout entropy {cls['entropy']:.2f} > {t['max_entropy']}")
        if cls.get("ood_score") is not None and cls["ood_score"] > cls["ood_threshold"]:
            flag("out_of_distribution",
                 f"Mahalanobis {cls['ood_score']:.1f} > {cls['ood_threshold']:.1f} (val p{t['ood_percentile']})")

    tumor_predicted = bool(cls) and cls.get("label") != no_tumor
    if cls and seg:
        if tumor_predicted and not seg.get("has_mask"):
            flag("tumor_predicted_no_mask", f"classifier says {cls['label']} but the segmenter found no tumor")
        if not tumor_predicted and seg.get("has_mask"):
            flag("mask_but_no_tumor_predicted",
                 f"segmenter found {seg['area_pct_brain']}% of brain but classifier says no tumor")
        if tumor_predicted and seg.get("has_mask") and cls.get("cam") is not None:
            cam_in_mask = cam_energy_in_mask(cls["cam"], seg["mask"], t["mask_dilation_px"])
            if cam_in_mask < t["min_cam_in_mask"]:
                flag("attention_outside_tumor",
                     f"only {cam_in_mask:.0%} of Grad-CAM energy falls inside the tumor mask")

    kept, removed = know.get("sentences", []), []
    if tumor_predicted:
        if not know.get("sources"):
            flag("no_knowledge_retrieved", "no knowledge-base passages were found for this tumor type")
        elif know.get("sentences"):
            kept, removed, failed = grounding_review(know)
            if failed:
                flag("grounding_check_failed", "the LLM judge could not be reached for every sentence")
            frac = len(removed) / max(len(know["sentences"]), 1)
            if frac > t["max_removed_fraction"]:
                flag("explanation_poorly_grounded",
                     f"grounding review removed {len(removed)}/{len(know['sentences'])} sentences")

    for err in state.get("errors", []) or []:
        if err.startswith(("classification", "segmentation")) and "pipeline_error" not in flags:
            flag("pipeline_error", err)

    out = {"safety": {"flags": flags, "details": details, "needs_review": bool(flags),
                      "cam_in_mask": cam_in_mask}}
    if know:
        out["knowledge"] = {**know, "sentences": kept, "removed": removed}
    return out
