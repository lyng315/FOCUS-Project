import os
import torch
import pandas as pd
import collections
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ===== PATH CONFIG =====
CSV_FILE = "data/slide_labels.csv"   # CSV có slide_id, label
PATCH_DIR = "data/patch"                  # folder chứa thư mục patch/{slide_id}/*.png
OUT_DIR = "feat"                     # nơi lưu feature .pt

# ===== MODEL =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # bỏ layer cuối
model = model.to(device)
model.eval()

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===== ĐỌC SLIDE CSV =====
df = pd.read_csv(CSV_FILE)
os.makedirs(OUT_DIR, exist_ok=True)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    slide_id = str(row["slide_id"])
    label = row["label"]
    slide_path = os.path.join(PATCH_DIR, slide_id)

    if not os.path.exists(slide_path):
        print(f"⚠️ Không tìm thấy thư mục patch cho slide {slide_id}, bỏ qua.")
        continue

    features = []
    patch_ids = []

    for img_name in os.listdir(slide_path):
        if not img_name.lower().endswith((".png",".jpg",".jpeg")):
            continue
        img_path = os.path.join(slide_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x).cpu()  # (1, 2048)
            features.append(feat)
            patch_ids.append(img_name)
        except Exception as e:
            print(f"❌ Lỗi khi đọc {img_path}: {e}")

    if len(features) == 0:
        print(f"⚠️ Slide {slide_id} không có patch hợp lệ.")
        continue

    features = torch.cat(features, dim=0)  # (N_patch, 2048)
    save_path = os.path.join(OUT_DIR, f"{slide_id}.pt")
    torch.save({
        "features": features,
        "patch_ids": patch_ids,
        "label": label
    }, save_path)

    print(f"✅ Saved {save_path}, shape = {features.shape}")
