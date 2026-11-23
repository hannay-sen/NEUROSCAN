"""Summarize reviewer feedback logged by the Streamlit app (feedback/feedback.csv).

  python -m neuroscan.eval.summarize_feedback

Use the output to retune thresholds, prompts and retrieval. No model weights are
fine-tuned from feedback.
"""
from __future__ import annotations

import pandas as pd

from neuroscan.config import path
from neuroscan.eval.common import save_json


def main() -> None:
    fp = path("feedback")
    if not fp.exists():
        raise SystemExit(f"No feedback yet at {fp}")
    df = pd.read_csv(fp)
    agree = df.agree_with_label.astype(str).str.lower().eq("yes")
    res = {
        "n_reviews": len(df),
        "n_reviewers": int(df.reviewer.nunique()),
        "n_cases": int(df.image.nunique()),
        "label_agreement_rate": float(agree.mean()),
        "agreement_when_flagged": float(agree[df.needs_review].mean()) if df.needs_review.any() else None,
        "agreement_when_unflagged": float(agree[~df.needs_review].mean()) if (~df.needs_review).any() else None,
        "explanation_helpfulness_mean": float(df.explanation_rating.mean()),
        "mask_quality_mean": float(df.mask_rating.mean()),
        "agreement_by_predicted_label": df.assign(agree=agree).groupby("predicted_label").agree.mean().to_dict(),
        "comments": df.comment.dropna().astype(str).tolist()[:50],
    }
    for k, v in res.items():
        if k != "comments":
            print(f"{k:32s} {v}")
    save_json("feedback_summary", res)


if __name__ == "__main__":
    main()
