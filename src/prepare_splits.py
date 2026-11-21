"""
Step 1: build one fixed split file for the whole project.

Run once in a Kaggle notebook with the dataset
  abdallahalidev/plantvillage-dataset
attached. Output: /kaggle/working/splits.csv

Columns: path, label, label_idx, split (train/val/test), hpo_subset (bool)
- 70/20/10 stratified split (same ratios as Bahaa et al., AgriEngineering 2026)
- hpo_subset: stratified 30% of the TRAIN split, used for Phase 1 HPO
"""
import glob
import os

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INPUT_ROOT = "/kaggle/input"
OUT_CSV = "/kaggle/working/splits.csv"

# --- locate the "color" folder (the dataset also has grayscale/segmented versions)
candidates = [d for d in glob.glob(f"{INPUT_ROOT}/**/color", recursive=True) if os.path.isdir(d)]
assert candidates, "Could not find a 'color' folder - check the attached dataset."
COLOR_DIR = candidates[0]
print("Using:", COLOR_DIR)

# --- collect image paths and labels
rows = []
for cls in sorted(os.listdir(COLOR_DIR)):
    cls_dir = os.path.join(COLOR_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue
    for f in os.listdir(cls_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            # store path relative to INPUT_ROOT so it works on any account
            rows.append({"path": os.path.relpath(os.path.join(cls_dir, f), INPUT_ROOT),
                         "label": cls})

df = pd.DataFrame(rows)
classes = sorted(df["label"].unique())
df["label_idx"] = df["label"].map({c: i for i, c in enumerate(classes)})
print(f"{len(df)} images, {len(classes)} classes")  # expect ~54k, 38

# --- 70 / 20 / 10 stratified split
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["label"], random_state=SEED)
train_df = train_df.assign(split="train")
val_df = val_df.assign(split="val")
test_df = test_df.assign(split="test")

# --- 30% stratified HPO subset of the training split
hpo_idx, _ = train_test_split(train_df.index, train_size=0.30,
                              stratify=train_df["label"], random_state=SEED)
train_df["hpo_subset"] = train_df.index.isin(hpo_idx)
val_df["hpo_subset"] = False
test_df["hpo_subset"] = False

out = pd.concat([train_df, val_df, test_df]).sort_values("path").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)

# --- sanity checks
print(out["split"].value_counts())
print("HPO subset size:", out["hpo_subset"].sum())
print("Smallest class in HPO subset:",
      out[out.hpo_subset].label.value_counts().min(), "images")
print("Saved", OUT_CSV)
