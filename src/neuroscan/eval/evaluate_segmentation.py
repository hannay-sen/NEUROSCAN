"""Evaluate the U-Net: Dice and HD95 (pixels) per slice.

  python -m neuroscan.eval.evaluate_segmentation --split figshare_test
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_hausdorff_distance
from torch.utils.data import DataLoader

from neuroscan.config import get_device, load_config, path
from neuroscan.data.datasets import SegmentationDataset
from neuroscan.eval.common import save_json
from neuroscan.models.registry import get_segmenter


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="figshare_test")
    args = ap.parse_args()

    s = load_config()["segmentation"]
    model, trained = get_segmenter()
    if not trained:
        print("WARNING: no trained checkpoint found; evaluating random weights.")
    ds = SegmentationDataset(path("splits") / f"{args.split}.csv", s["image_size"])
    dl = DataLoader(ds, s["batch_size"], num_workers=s["num_workers"])
    device = get_device()

    dice, hd95 = [], []
    for x, y in dl:
        p = (torch.sigmoid(model(x.to(device))) >= s["mask_threshold"]).float().cpu()
        inter = (p * y).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dice += torch.where(denom > 0, 2 * inter / denom.clamp_min(1e-6), torch.ones_like(denom)).tolist()
        both = (p.sum(dim=(1, 2, 3)) > 0) & (y.sum(dim=(1, 2, 3)) > 0)
        h = torch.full((len(y),), float("nan"))
        if both.any():
            h[both] = compute_hausdorff_distance(p[both], y[both], percentile=95).flatten().float()
        hd95 += h.tolist()

    df = ds.df.assign(dice=dice, hd95_px=hd95)
    out = path("results")
    out.mkdir(parents=True, exist_ok=True)
    df[["id", "label", "patient_id", "dice", "hd95_px"]].to_csv(out / f"segmentation_{args.split}.csv", index=False)
    res = {
        "split": args.split, "n": len(df),
        "dice_mean": float(np.mean(dice)), "dice_median": float(np.median(dice)),
        "hd95_px_mean": float(np.nanmean(hd95)), "hd95_px_median": float(np.nanmedian(hd95)),
        "empty_predictions": int(sum(np.isnan(hd95))),
        "dice_by_class": df.groupby("label").dice.mean().to_dict(),
        "note": "HD95 in pixels at the network resolution; 2D slices carry no pixel spacing.",
    }
    for k, v in res.items():
        print(f"{k:20s} {v}")
    save_json(f"segmentation_{args.split}", res)


if __name__ == "__main__":
    main()
