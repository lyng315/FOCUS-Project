import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm

# ---- CONFIG ----
CSV_PATH = 'mr/manifest_clean.csv'   # input từ người 2
OUTPUT_PT = "TCGA-01.pt"                  # tensor feature
OUTPUT_INDEX = "feat_index.jsonl"         # mapping patch_id -> index

# ---- LOAD DATA ----
df = pd.read_csv(CSV_PATH)

# ---- MODEL (ResNet50 pretrained) ----
model = models.resnet50(pretrained=True)
# bỏ layer cuối cùng -> lấy feature 2048-dim
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

# ---- TRANSFORMS ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

# ---- FEATURE EXTRACTION ----
features = []
index_records = []

with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['path']
        patch_id = row['patch_id']
        slide_id = row['label_id']
        label = row['label']

        try:
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0)  # (1,3,224,224)
            feat = model(img_t).squeeze().numpy()  # (2048,)
            
            features.append(feat)

            record = {
                "index": len(features)-1,
                "slide_id": slide_id,
                "patch_id": patch_id,
                "label": label,
                "path": img_path
            }
            with open(OUTPUT_INDEX, "a") as f:
                f.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"❌ Error at {img_path}: {e}")

# ---- SAVE FEATURE TENSOR ----
features_tensor = torch.tensor(features)
torch.save(features_tensor, OUTPUT_PT)

print(f"✅ Saved features to {OUTPUT_PT}, shape = {features_tensor.shape}")
print(f"✅ Saved index records to {OUTPUT_INDEX}")
