import os
import cv2
import pandas as pd
from multiprocessing import Pool, cpu_count

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


# === 2. Đường dẫn ===
patches_dir = "/kaggle/input/camelyon17/data/camelyon17_v1.0/patches"
metadata_csv = "/kaggle/input/camelyon17/data/camelyon17_v1.0/metadata.csv"

# === 3. Đọc metadata ===
df_meta = pd.read_csv(metadata_csv)

# Tạo slide_id thủ công (vd: patient_004_node_4)
df_meta["slide_id"] = df_meta["patient"].astype(str) + "_node_" + df_meta["node"].astype(str)

# === 4. Duyệt toàn bộ ảnh patch ===
records = []
for slide_folder in os.listdir(patches_dir):
    slide_path = os.path.join(patches_dir, slide_folder)
    if not os.path.isdir(slide_path):
        continue

    for fname in os.listdir(slide_path):
        if fname.endswith(".png"):
            records.append({
                "slide_id": slide_folder,
                "patch_id": fname,
                "path": os.path.join(slide_path, fname)
            })

df_manifest = pd.DataFrame(records)
print("🧩 Tổng số patch:", len(df_manifest))

# === 5. Gộp nhãn tumor từ metadata ===
df_manifest = df_manifest.merge(df_meta[["slide_id", "tumor"]], on="slide_id", how="left")
df_manifest = df_manifest.rename(columns={"tumor": "label"})

# === 6. Tính tissue_pct song song ===
def process_row(path):
    return compute_tissue_pct(path)

print("⚙️ Đang tính tissue_pct, vui lòng chờ...")

with Pool(cpu_count()) as p:
    df_manifest["tissue_pct"] = p.map(process_row, df_manifest["path"].tolist())

# === 7. Xuất file cuối cùng ===
output_csv = "/kaggle/working/mainfest_rawCML.csv"
df_manifest.to_csv(output_csv, index=False)
print(f"\n✅ Đã tạo xong file: {output_csv}")
print(df_manifest.head())