# n4/run_pipeline.py
import torch
import glob, os, json, subprocess, pandas as pd, sys
from pathlib import Path

# Directories
PATCH_DIR = "patch"   # folder of patch subfolders (if needed)
FEAT_DIR = "feat"
COMP_DIR = "compress"
BAG_DIR = "bag"
MANIFEST = "manifest_raw.csv"
SLIDE_CSV = "data/slide_labels.csv"

# Device
device = torch.device("cpu")

# Make directories if not exist
os.makedirs(COMP_DIR, exist_ok=True)
os.makedirs(BAG_DIR, exist_ok=True)

# Read slide labels
df = pd.read_csv(SLIDE_CSV)
slide_ids = [str(x) for x in df['slide_id'].tolist()]

for sid in slide_ids:
    pt_path = os.path.join(FEAT_DIR, f"{sid}.pt")
    if not os.path.exists(pt_path):
        print(f"[SKIP] missing feat {pt_path}")
        continue

    # 1. Artifact filter (feature-norm)
    cmd = [
        sys.executable, "n4/artifact_filter.py",
        "--mode", "feat",
        "--pt", pt_path,
        "--z_thresh", "-1.5"
    ]
    print("=>", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # 2. Deduplicate
    out_dedup = os.path.join(COMP_DIR, f"{sid}_dedup.json")
    cmd = [
        sys.executable, "n4/deduplicate.py",
        "--pt", pt_path,
        "--out_json", out_dedup,
        "--sim_thresh", "0.95"
    ]
    subprocess.run(cmd, check=True)

    # 3. Adaptive compress (kavtc)
    out_sel = os.path.join(COMP_DIR, f"{sid}_kavtc.json")
    cmd = [
        sys.executable, "n4/adaptive_compress.py",
        "--pt", pt_path,
        "--mode", "kavtc",
        "--alpha", "0.2",
        "--out_json", out_sel
    ]
    subprocess.run(cmd, check=True)

    # 4. Create bag
    label_row = df[df['slide_id'].astype(str) == sid]
    label = None
    if 'label' in df.columns:
        label = label_row['label'].values[0]
    out_h5 = os.path.join(BAG_DIR, f"{sid}_bag.h5")
    cmd = [
        sys.executable, "n4/create_bag_h5.py",
        "--pt", pt_path,
        "--selected_json", out_sel,
        "--label", str(label),
        "--out", out_h5
    ]
    subprocess.run(cmd, check=True)

    # Log summary
    print(f"[DONE] slide {sid}")
