import os
import cv2
import pandas as pd
from tqdm import tqdm

# === 1. Hàm tính tissue_pct ===
def compute_tissue_pct(img_path, threshold=200, resize=64):
    """
    Tính phần trăm mô trong ảnh patch
    img_path: đường dẫn ảnh
    threshold: ngưỡng để coi pixel là "nền trắng"
    resize: giảm kích thước trước khi tính để chạy nhanh
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (resize, resize))
    total_pixels = img.size
    tissue_pixels = (img < threshold).sum()
    return round(tissue_pixels / total_pixels, 4)

# === 2. Đường dẫn dataset ===
base_dir = "/kaggle/input/ubc-ocean-tiles-w-masks-2048px-scale"
images_dir = os.path.join(base_dir, "train_images")
csv_path = os.path.join(base_dir, "train.csv")

# === 3. Đọc dữ liệu gốc ===
df_train = pd.read_csv(csv_path)

# === 4. Tạo danh sách chứa các bản ghi ===
records = []

for img_name in tqdm(os.listdir(images_dir)):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(images_dir, img_name)

    # slide_id, patch_id
    slide_id = img_name.split("_")[0]
    patch_id = img_name.split(".")[0]

    # Tính tỷ lệ mô
    tissue_pct = compute_tissue_pct(img_path)

    # Lấy label từ train.csv
    row = df_train[df_train["image"] == img_name]
    label = row["label"].values[0] if not row.empty else "Unknown"

    records.append({
        "slide_id": slide_id,
        "patch_id": patch_id,
        "path": img_path,
        "label": label,
        "tissue_pct": tissue_pct
    })

# === 5. Gộp thành DataFrame và lưu ra file CSV ===
df_manifest = pd.DataFrame(records)
df_manifest.to_csv("Mainfest_raw.csv", index=False)
print("✅ File manifest_raw.csv đã được tạo thành công!")
