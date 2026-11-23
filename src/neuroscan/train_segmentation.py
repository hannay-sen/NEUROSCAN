"""Train the 2D U-Net on Figshare tumor masks (patient-level splits).

Usage:  python -m neuroscan.train_segmentation      # -> checkpoints/unet.pt
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader

from neuroscan.config import get_device, load_config, path, seed_everything
from neuroscan.data.datasets import SegmentationDataset
from neuroscan.models.registry import build_segmenter


@torch.no_grad()
def mean_dice(model, loader, device, thr: float) -> float:
    model.eval()
    scores = []
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(device))) >= thr).float().cpu()
        inter = (p * y).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        scores.append(torch.where(denom > 0, 2 * inter / denom.clamp_min(1e-6), torch.ones_like(denom)))
    return float(torch.cat(scores).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    s = cfg["segmentation"]
    epochs = args.epochs or s["epochs"]
    seed_everything()
    device = get_device()
    splits = path("splits")

    train_dl = DataLoader(SegmentationDataset(splits / "figshare_train.csv", s["image_size"], train=True),
                          s["batch_size"], shuffle=True, num_workers=s["num_workers"])
    val_dl = DataLoader(SegmentationDataset(splits / "figshare_val.csv", s["image_size"]),
                        s["batch_size"], num_workers=s["num_workers"])

    model = build_segmenter().to(device)
    loss_fn = DiceCELoss(sigmoid=True)
    opt = torch.optim.AdamW(model.parameters(), lr=s["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        d = mean_dice(model, val_dl, device, s["mask_threshold"])
        history.append({"epoch": ep, "train_loss": total / n, "val_dice": d})
        print(f"epoch {ep:02d}  loss {total/n:.4f}  val_dice {d:.4f}  ({time.time()-t0:.0f}s)")
        if d > best:
            best = d
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ckpt_dir = path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_val_dice": best}, ckpt_dir / "unet.pt")
    res = path("results")
    res.mkdir(parents=True, exist_ok=True)
    (res / "train_unet.json").write_text(json.dumps({"history": history, "best_val_dice": best}, indent=2))
    print(f"saved {ckpt_dir / 'unet.pt'}  (best val Dice {best:.4f})")


if __name__ == "__main__":
    main()
