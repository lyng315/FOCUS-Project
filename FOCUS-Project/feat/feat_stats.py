import torch
import pandas as pd
import json

FEATURE_PATH = "TCGA-01.pt"
INDEX_PATH = "feat_index.jsonl"
OUTPUT_STATS = "feat_stats.csv"

# ---- LOAD DATA ----
features = torch.load(FEATURE_PATH)   # (N, 2048)
records = []

with open(INDEX_PATH, "r") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# ---- JOIN FEATURE + LABEL ----
df["feat_index"] = df["index"]
df["label"] = df["label"]

# ---- TÍNH MEAN/STD THEO LABEL ----
stats = []
for label in df["label"].unique():
    idx = df[df["label"] == label]["feat_index"].values
    feats = features[idx]
    mean_feat = feats.mean(dim=0)
    std_feat = feats.std(dim=0)
    
    stats.append({
        "label": label,
        "n_patches": len(idx),
        "mean_first5": mean_feat[:5].tolist(),  # chỉ log 5 giá trị đầu
        "std_first5": std_feat[:5].tolist()
    })

stats_df = pd.DataFrame(stats)
stats_df.to_csv(OUTPUT_STATS, index=False)

print(f"✅ Saved statistics to {OUTPUT_STATS}")
