"""Create train/val/test CSVs.

Kaggle: original Testing folder -> test; Training folder split stratified into train/val.
Figshare: split by patient ID so no patient appears in two splits. figshare_all is the
classifier's external test set (the classifier never trains on Figshare).

Usage:  python -m neuroscan.data.make_splits
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from neuroscan.config import load_config, path


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    out = path("splits")
    out.mkdir(parents=True, exist_ok=True)

    kg = pd.read_csv(out / "kaggle_clean.csv")
    test = kg[kg.kaggle_subset == "testing"]
    trval = kg[kg.kaggle_subset == "training"]
    train, val = train_test_split(trval, test_size=cfg["splits"]["kaggle_val_frac"],
                                  stratify=trval.label, random_state=seed)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_csv(out / f"kaggle_{name}.csv", index=False)
        print(f"kaggle_{name}: {len(df)}  {df.label.value_counts().to_dict()}")

    fs_csv = path("figshare_out") / "figshare.csv"
    if not fs_csv.exists():
        print("figshare.csv not found; skipping Figshare splits.")
        return
    fs = pd.read_csv(fs_csv, dtype={"patient_id": str})
    fs.to_csv(out / "figshare_all.csv", index=False)

    test_frac = cfg["splits"]["figshare_test_frac"]
    val_frac = cfg["splits"]["figshare_val_frac"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trval_idx, test_idx = next(gss.split(fs, groups=fs.patient_id))
    fs_trval, fs_test = fs.iloc[trval_idx], fs.iloc[test_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac / (1 - test_frac), random_state=seed)
    tr_idx, va_idx = next(gss2.split(fs_trval, groups=fs_trval.patient_id))
    parts = {"train": fs_trval.iloc[tr_idx], "val": fs_trval.iloc[va_idx], "test": fs_test}

    ids = [set(p.patient_id) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "patient leakage!"
    for name, df in parts.items():
        df.to_csv(out / f"figshare_{name}.csv", index=False)
        print(f"figshare_{name}: {len(df)} slices, {df.patient_id.nunique()} patients")


if __name__ == "__main__":
    main()
