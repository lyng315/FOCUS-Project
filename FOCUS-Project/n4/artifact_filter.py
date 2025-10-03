# n4/artifact_filter.py
import os, argparse, torch, numpy as np
from PIL import Image
import cv2

def var_laplacian(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return float(cv2.Laplacian(img, cv2.CV_64F).var()) if img is not None else 0.0
    except:
        return 0.0

def filter_by_image(manifest_rows, lap_thresh):
    keep = []
    rem = []
    for r in manifest_rows:
        v = var_laplacian(r['path'])
        if v >= lap_thresh:
            keep.append(r)
        else:
            rem.append(r)
    return keep, rem

def filter_by_feat_norm(pt_path, z_thresh):
    obj = torch.load(pt_path, map_location='cpu')
    feats = obj['features'].cpu().numpy()
    norms = np.linalg.norm(feats, axis=1)
    z = (norms - norms.mean()) / (norms.std() + 1e-9)
    keep_idx = list(np.where(z >= z_thresh)[0])
    rem_idx = list(np.where(z < z_thresh)[0])
    return keep_idx, rem_idx

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image","feat"], required=True)
    p.add_argument("--manifest", default="manifest_raw.csv")
    p.add_argument("--pt", default=None)
    p.add_argument("--slide_id", default=None)
    p.add_argument("--lap_thresh", type=float, default=100.0)
    p.add_argument("--z_thresh", type=float, default=-1.5)
    args = p.parse_args()

    if args.mode == "image":
        import csv
        # manifest expected to contain rows with 'slide_id','patch_id','path'
        df = []
        with open(args.manifest, 'r', encoding='utf-8') as f:
            import pandas as pd
            m = pd.read_csv(args.manifest)
            subset = m[m['slide_id']==int(args.slide_id)]
            rows = subset.to_dict('records')
        keep, rem = filter_by_image(rows, args.lap_thresh)
        print("keep", len(keep), "rem", len(rem))
    else:
        keep_idx, rem_idx = filter_by_feat_norm(args.pt, args.z_thresh)
        print("keep_idx_count", len(keep_idx), "rem_idx_count", len(rem_idx))
