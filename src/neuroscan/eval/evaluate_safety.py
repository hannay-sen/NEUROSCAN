"""Evaluate the uncertainty-based safety flags (low_confidence, high_uncertainty,
out_of_distribution) as an error detector.

  python -m neuroscan.eval.evaluate_safety --split kaggle_val --sweep   # choose thresholds on val...
  python -m neuroscan.eval.evaluate_safety --split kaggle_test          # ...report on test

A good safety layer flags most misclassifications (high error recall) while leaving most
cases unflagged (high coverage), so the unflagged error rate is low.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from neuroscan.config import load_config
from neuroscan.eval.common import classifier_outputs, save_json
from neuroscan.models.registry import get_classifier


def flag_stats(out: dict, min_conf: float, max_ent: float, use_ood: bool = True) -> dict:
    conf = out["probs"].max(1)
    wrong = out["probs"].argmax(1) != out["labels"]
    flagged = (conf < min_conf) | (out["entropy"] > max_ent)
    if use_ood and out["ood"] is not None:
        flagged |= out["ood"] > out["ood_threshold"]
    n_wrong = max(int(wrong.sum()), 1)
    return {
        "min_confidence": float(min_conf), "max_entropy": float(max_ent),
        "coverage": float(1 - flagged.mean()),
        "flag_rate": float(flagged.mean()),
        "error_recall": float((flagged & wrong).sum() / n_wrong),
        "error_rate_overall": float(wrong.mean()),
        "error_rate_unflagged": float(wrong[~flagged].mean()) if (~flagged).any() else 0.0,
        "error_rate_flagged": float(wrong[flagged].mean()) if flagged.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max-unflagged-error", type=float, default=0.02,
                    help="sweep target: highest coverage with unflagged error at or below this")
    args = ap.parse_args()

    t = load_config()["safety"]
    model, trained = get_classifier()
    if not trained:
        print("WARNING: no trained checkpoint found; evaluating random weights.")
    out = classifier_outputs(model, args.split)

    if not args.sweep:
        res = flag_stats(out, t["min_confidence"], t["max_entropy"])
        res["split"] = args.split
        for k, v in res.items():
            print(f"{k:22s} {v}")
        save_json(f"safety_{args.split}", res)
        return

    grid = [flag_stats(out, c, e)
            for c, e in itertools.product(np.arange(0.40, 0.96, 0.05), np.arange(0.10, 0.91, 0.05))]
    ok = [g for g in grid if g["error_rate_unflagged"] <= args.max_unflagged_error]
    best = max(ok, key=lambda g: g["coverage"]) if ok else max(grid, key=lambda g: g["error_recall"])
    print("Chosen on validation (copy into configs/config.yaml -> safety):")
    print(f"  min_confidence: {best['min_confidence']:.2f}")
    print(f"  max_entropy: {best['max_entropy']:.2f}")
    print(f"  coverage {best['coverage']:.3f}  error recall {best['error_recall']:.3f}  "
          f"unflagged error {best['error_rate_unflagged']:.4f}")
    save_json(f"safety_sweep_{args.split}", {"target_unflagged_error": args.max_unflagged_error,
                                            "chosen": best, "grid": grid})


if __name__ == "__main__":
    main()
