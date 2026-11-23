"""Reviewer UI.  Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan import DISCLAIMER  # noqa: E402
from neuroscan.config import path  # noqa: E402
from neuroscan.graph import run  # noqa: E402

st.set_page_config(page_title="NeuroScan Agent", layout="wide")
st.title("NeuroScan Agent")
st.warning(DISCLAIMER)


def overlay(img: np.ndarray, layer: np.ndarray, color, alpha: float) -> np.ndarray:
    rgb = np.stack([img] * 3, axis=-1)
    col = np.array(color, dtype=np.float32)
    a = (layer.astype(np.float32) * alpha)[..., None]
    return np.clip(rgb * (1 - a) + col * a, 0, 1)


uploaded = st.file_uploader("Upload one MRI slice (JPG or PNG)", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to run the four agents.")
    st.stop()

with st.spinner("Running classification, segmentation, knowledge and safety agents..."):
    state = run(image_path=uploaded.name, image=Image.open(uploaded))
report = state["report"]

if report["errors"]:
    st.error("Pipeline messages: " + "; ".join(report["errors"]))
if not state.get("valid"):
    st.stop()

img = state["image"]
cls, seg = state.get("classification") or {}, state.get("segmentation") or {}

c1, c2, c3 = st.columns(3)
c1.image(img, caption="Input", clamp=True, use_container_width=True)
if seg.get("mask") is not None:
    c2.image(overlay(img, seg["mask"], (1, 0, 0), 0.45), caption="U-Net tumor mask", use_container_width=True)
if cls.get("cam") is not None:
    c3.image(overlay(img, cls["cam"], (1, 0.8, 0), 0.5), caption="Grad-CAM attention", use_container_width=True)

if not (report["classification"] or {}).get("model_trained") or not (report["segmentation"] or {}).get("model_trained"):
    st.error("Running on UNTRAINED weights. Outputs are meaningless until checkpoints exist.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Findings")
    r = report["classification"]
    if r:
        st.metric("Predicted class", r["label"], f"confidence {r['confidence']:.0%}")
        st.write({k: f"{v:.1%}" for k, v in r["probs"].items()})
        st.caption(f"MC-dropout uncertainty (0-1): {r['uncertainty_entropy']:.2f}   "
                   f"OOD score: {r['ood_score'] if r['ood_score'] is not None else 'not fitted'}")
    s = report["segmentation"]
    if s:
        st.write(f"**Mask:** {'found' if s['has_mask'] else 'none'} · {s['area_pct_brain']}% of brain area · "
                 f"{s['location']}")
        st.caption(s["note"])

with right:
    st.subheader("Safety review")
    if report["flags"]:
        st.error("Needs radiologist review")
        for f in report["flags"]:
            st.write(f"- **{f}**: {report['flag_details'].get(f, '')}")
    else:
        st.success("No flags raised (still requires radiologist confirmation).")

    st.subheader("Background (cited)")
    if report["explanation"]:
        for e in report["explanation"]:
            st.write(f"{e['text']} [{e['citation']}]")
        st.caption("Sources: " + " · ".join(sorted({f"[{e['citation']}] {e['source']}"
                                                   for e in report["explanation"] if e["source"]})))
    else:
        st.caption(f"No explanation ({report['explanation_mode']}).")
    if report["removed_sentences"]:
        with st.expander(f"{len(report['removed_sentences'])} sentence(s) removed by grounding review"):
            for t in report["removed_sentences"]:
                st.write(f"~~{t}~~")

with st.expander("Full JSON report"):
    st.json(report)

st.subheader("Reviewer feedback")
with st.form("feedback"):
    reviewer = st.text_input("Reviewer ID")
    agree = st.radio("Do you agree with the predicted class?", ["yes", "no", "unsure"], horizontal=True)
    true_label = st.selectbox("Your label (if you disagree)", ["", "glioma", "meningioma", "notumor", "pituitary"])
    mask_rating = st.slider("Mask quality", 1, 5, 3)
    expl_rating = st.slider("Explanation helpfulness", 1, 5, 3)
    comment = st.text_area("Comment")
    if st.form_submit_button("Submit feedback"):
        fp = path("feedback")
        fp.parent.mkdir(parents=True, exist_ok=True)
        new = not fp.exists()
        with open(fp, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "reviewer", "image", "predicted_label", "confidence", "needs_review",
                            "flags", "agree_with_label", "reviewer_label", "mask_rating",
                            "explanation_rating", "comment"])
            r = report["classification"] or {}
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), reviewer, report["image"],
                        r.get("label"), r.get("confidence"), report["needs_review"], "|".join(report["flags"]),
                        agree, true_label, mask_rating, expl_rating, comment])
        st.success("Saved. Thank you!")
