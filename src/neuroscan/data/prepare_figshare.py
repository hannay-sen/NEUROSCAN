"""Convert Figshare (Cheng et al.) .mat files into PNG image + PNG mask + figshare.csv.

Each .mat file (MATLAB v7.3 / HDF5) holds a struct `cjdata` with:
  label      1 = meningioma, 2 = glioma, 3 = pituitary
  PID        patient ID (stored as uint16 characters)
  image      512x512 int16 slice
  tumorMask  512x512 binary mask

Usage:  python -m neuroscan.data.prepare_figshare
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from neuroscan.config import path

LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def _read_mat(fp: Path) -> dict:
    with h5py.File(fp, "r") as f:
        cj = f["cjdata"]
        label = int(np.array(cj["label"]).squeeze())
        pid_chars = np.array(cj["PID"]).squeeze()
        pid = "".join(chr(int(c)) for c in np.atleast_1d(pid_chars))
        image = np.array(cj["image"]).T.astype(np.float32)      # MATLAB is column-major
        mask = np.array(cj["tumorMask"]).T.astype(np.uint8)
    return {"label": LABELS[label], "pid": pid.strip(), "image": image, "mask": mask}


def _to_uint8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def main() -> None:
    raw = path("figshare_raw")
    out = path("figshare_out")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    files = sorted(raw.rglob("*.mat"), key=lambda p: (len(p.stem), p.stem))
    if not files:
        raise SystemExit(f"No .mat files found under {raw}")

    rows = []
    for fp in files:
        try:
            d = _read_mat(fp)
        except Exception as e:  # noqa: BLE001
            print(f"skip {fp.name}: {e}")
            continue
        sid = f"fs_{fp.stem}"
        img_p = out / "images" / f"{sid}.png"
        msk_p = out / "masks" / f"{sid}.png"
        Image.fromarray(_to_uint8(d["image"])).save(img_p)
        Image.fromarray((d["mask"] > 0).astype(np.uint8) * 255).save(msk_p)
        rows.append({"id": sid, "image_path": str(img_p), "mask_path": str(msk_p),
                     "label": d["label"], "patient_id": d["pid"], "source": "figshare"})

    df = pd.DataFrame(rows)
    df.to_csv(out / "figshare.csv", index=False)
    print(f"Wrote {len(df)} slices from {df.patient_id.nunique()} patients -> {out/'figshare.csv'}")
    print(df.label.value_counts().to_string())


if __name__ == "__main__":
    main()
