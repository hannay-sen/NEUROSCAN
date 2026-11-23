"""Train the EfficientNet-B0 classifier on Kaggle train/val, then fit temperature
scaling (on val) and Mahalanobis OOD statistics (train features, threshold from val).

Usage:
  python -m neuroscan.train_classifier                              # -> checkpoints/classifier.pt
  python -m neuroscan.train_classifier --balance none --tag nobal   # ablation for the bias study
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from neuroscan.config import get_device, load_config, path, seed_everything
from neuroscan.data.datasets import ClassificationDataset
from neuroscan.models.registry import build_classifier


@torch.no_grad()
def collect(model, loader, device, want_feats: bool = False):
    model.eval()
    logits, feats, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        f = model.features(x)
        logits.append(model.backbone.get_classifier()(f).cpu())
        if want_feats:
            feats.append(f.cpu())
        labels.append(y)
    return torch.cat(logits), (torch.cat(feats) if want_feats else None), torch.cat(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", choices=["weighted", "none"], default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    c = cfg["classifier"]
    balance = args.balance or c["balance"]
    epochs = args.epochs or c["epochs"]
    seed_everything()
    device = get_device()
    splits = path("splits")
    classes = cfg["classes"]

    train_ds = ClassificationDataset(splits / "kaggle_train.csv", classes, c["image_size"], train=True)
    val_ds = ClassificationDataset(splits / "kaggle_val.csv", classes, c["image_size"])
    train_eval_ds = ClassificationDataset(splits / "kaggle_train.csv", classes, c["image_size"])

    if balance == "weighted":
        counts = np.bincount(train_ds.targets, minlength=len(classes))
        w = 1.0 / counts[train_ds.targets]
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
        train_dl = DataLoader(train_ds, c["batch_size"], sampler=sampler, num_workers=c["num_workers"])
    else:
        train_dl = DataLoader(train_ds, c["batch_size"], shuffle=True, num_workers=c["num_workers"])
    val_dl = DataLoader(val_ds, c["batch_size"], num_workers=c["num_workers"])

    model = build_classifier(pretrained=c["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        logits, _, labels = collect(model, val_dl, device)
        pred = logits.argmax(1)
        acc = float((pred == labels).float().mean())
        f1 = float(f1_score(labels, pred, average="macro"))
        history.append({"epoch": ep, "train_loss": total / n, "val_acc": acc, "val_macro_f1": f1})
        print(f"epoch {ep:02d}  loss {total/n:.4f}  val_acc {acc:.4f}  val_f1 {f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Calibration on val, OOD stats from (unaugmented) train features.
    val_logits, val_feats, val_labels = collect(model, val_dl, device, want_feats=True)
    nll_before = float(F.cross_entropy(val_logits, val_labels))
    t = model.fit_temperature(val_logits, val_labels)
    nll_after = float(F.cross_entropy(val_logits / t, val_labels))
    tr_dl = DataLoader(train_eval_ds, c["batch_size"], num_workers=c["num_workers"])
    _, tr_feats, tr_labels = collect(model, tr_dl, device, want_feats=True)
    thr = model.fit_ood(tr_feats, tr_labels, val_feats, cfg["safety"]["ood_percentile"])
    print(f"temperature {t:.3f}  (val NLL {nll_before:.4f} -> {nll_after:.4f})   OOD threshold {thr:.2f}")

    ckpt_dir = path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = f"classifier{('_' + args.tag) if args.tag else ''}"
    torch.save({"model": model.state_dict(), "classes": classes, "balance": balance,
                "best_val_macro_f1": best_f1, "temperature": t, "ood_threshold": thr},
               ckpt_dir / f"{name}.pt")
    res = path("results")
    res.mkdir(parents=True, exist_ok=True)
    (res / f"train_{name}.json").write_text(json.dumps(
        {"balance": balance, "history": history, "best_val_macro_f1": best_f1, "temperature": t,
         "val_nll_before": nll_before, "val_nll_after": nll_after, "ood_threshold": thr}, indent=2))
    print(f"saved {ckpt_dir / (name + '.pt')}")


if __name__ == "__main__":
    main()
