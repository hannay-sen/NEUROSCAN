"""Evaluate the classifier on a split.

  python -m neuroscan.eval.evaluate_classifier --split kaggle_test     # in-distribution
  python -m neuroscan.eval.evaluate_classifier --split figshare_all    # external source
  python -m neuroscan.eval.evaluate_classifier --split kaggle_test --tag nobal   # bias ablation

Figshare has no "notumor" class, so the external test reports metrics over the classes present.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from neuroscan.config import load_config, path
from neuroscan.eval.common import classifier_outputs, ece, save_json
from neuroscan.models.registry import get_classifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    classes = load_config()["classes"]
    model, trained = get_classifier(args.tag)
    if not trained:
        print("WARNING: no trained checkpoint found; evaluating random weights.")
    out = classifier_outputs(model, args.split)
    y, p = out["labels"], out["probs"]
    pred = p.argmax(1)
    present = sorted(set(y.tolist()))

    rep = classification_report(y, pred, labels=list(range(len(classes))), target_names=classes,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    result = {
        "split": args.split, "tag": args.tag, "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_f1_present_classes": f1_score(y, pred, labels=present, average="macro", zero_division=0),
        "ece_uncalibrated": ece(out["raw"], y), "ece_calibrated": ece(p, y),
        "per_class": {c: rep[c] for c in classes},
        "confusion_matrix": cm.tolist(), "classes": classes,
    }
    for k in ("accuracy", "macro_f1_present_classes", "ece_uncalibrated", "ece_calibrated"):
        print(f"{k:28s} {result[k]:.4f}")

    name = f"classifier_{args.split}{('_' + args.tag) if args.tag else ''}"
    save_json(name, result)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path("results") / f"{name}_confusion.png", dpi=150)
    print(f"-> results/{name}_confusion.png")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
