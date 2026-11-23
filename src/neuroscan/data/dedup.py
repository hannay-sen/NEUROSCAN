"""Leakage + duplicate check.

The Kaggle Brain Tumor MRI dataset contains the Figshare images. Because Figshare is
the classifier's *external* test set, every Kaggle image that matches a Figshare image
is removed. Internal Kaggle duplicates (including Training/Testing overlap) are removed
too, keeping the first copy (Training before Testing).

Matching uses a difference hash (dHash, hash_size^2 bits); pairs within `max_hamming`
bits are treated as duplicates. Inspect data/splits/dedup_pairs.csv to sanity-check matches.

Usage:  python -m neuroscan.data.dedup
Output: data/splits/kaggle_clean.csv, data/splits/dedup_pairs.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from neuroscan.config import load_config, path

CHUNK = 200
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def dhash(fp: str | Path, hash_size: int = 16) -> np.ndarray:
    """Packed dHash bits as a uint8 vector of length hash_size**2 / 8."""
    img = Image.open(fp).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: N x B, b: M x B packed hashes -> N x M bit distances."""
    return np.bitwise_count(a[:, None, :] ^ b[None, :, :]).sum(axis=-1, dtype=np.int32)


def scan_kaggle(root: Path, classes: list[str]) -> pd.DataFrame:
    rows = []
    for subset in ["Training", "Testing"]:
        for cls in classes:
            d = root / subset / cls
            if not d.exists():
                continue
            for fp in sorted(d.iterdir()):
                if fp.suffix.lower() in IMG_EXT:
                    rows.append({"id": f"kg_{subset[:2].lower()}_{cls}_{fp.stem}",
                                 "image_path": str(fp), "label": cls,
                                 "kaggle_subset": subset.lower(), "source": "kaggle"})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    hs, max_h = cfg["dedup"]["hash_size"], cfg["dedup"]["max_hamming"]
    out_dir = path("splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    kg = scan_kaggle(path("kaggle_raw"), cfg["classes"])
    if kg.empty:
        raise SystemExit(f"No Kaggle images found under {path('kaggle_raw')}")
    print(f"Hashing {len(kg)} Kaggle images...")
    kh = np.stack([dhash(p, hs) for p in kg.image_path])

    pairs = []
    drop = np.zeros(len(kg), dtype=bool)

    fs_csv = path("figshare_out") / "figshare.csv"
    if fs_csv.exists():
        fs = pd.read_csv(fs_csv)
        print(f"Hashing {len(fs)} Figshare images...")
        fh = np.stack([dhash(p, hs) for p in fs.image_path])
        for start in range(0, len(kh), CHUNK):
            d = hamming_matrix(kh[start:start + CHUNK], fh)
            ki, fi = np.nonzero(d <= max_h)
            for i, j in zip(ki + start, fi):
                drop[i] = True
                pairs.append({"kind": "kaggle_vs_figshare", "a": kg.image_path.iat[i],
                              "b": fs.image_path.iat[j], "hamming": int(d[i - start, j])})
    else:
        print(f"WARNING: {fs_csv} not found. Run prepare_figshare first, or leakage is unchecked.")

    # Internal duplicates: keep the earliest row (Training rows come first).
    for start in range(0, len(kh), CHUNK):
        d = hamming_matrix(kh[start:start + CHUNK], kh)
        for r in range(d.shape[0]):
            i = start + r
            if drop[i]:
                continue
            earlier = np.nonzero(d[r, :i] <= max_h)[0]
            earlier = earlier[~drop[earlier]]
            if len(earlier):
                j = int(earlier[0])
                drop[i] = True
                pairs.append({"kind": "kaggle_internal", "a": kg.image_path.iat[i],
                              "b": kg.image_path.iat[j], "hamming": int(d[r, j])})

    pd.DataFrame(pairs, columns=["kind", "a", "b", "hamming"]).to_csv(out_dir / "dedup_pairs.csv", index=False)
    clean = kg.loc[~drop].reset_index(drop=True)
    clean.to_csv(out_dir / "kaggle_clean.csv", index=False)

    n_fs = sum(p["kind"] == "kaggle_vs_figshare" for p in pairs)
    n_in = sum(p["kind"] == "kaggle_internal" for p in pairs)
    print(f"Removed {drop.sum()} of {len(kg)} Kaggle images "
          f"({n_fs} Figshare-overlap pairs, {n_in} internal-duplicate pairs).")
    print(f"Kept {len(clean)} -> {out_dir/'kaggle_clean.csv'}")


if __name__ == "__main__":
    main()
