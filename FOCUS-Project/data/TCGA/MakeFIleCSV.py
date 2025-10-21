import os
import cv2
import pandas as pd
from tqdm import tqdm

# ====== Đường dẫn dataset ======
base_dir = "/kaggle/input/tcga_coad_msi_mss_jpg"
classes = ["MSIMUT_JPEG", "MSS_JPEG"]   # hai thư mục chính

# ====== Hàm tính tỉ lệ mô bệnh (tissue_pct) ======
def calc_tissue_pct(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    tissue_area = (mask == 0).sum()
    total_area = mask.size
    return round(tissue_area / total_area, 4)

# ====== Duyệt file và tạo DataFrame ======
data = []
for label in classes:
    folder = os.path.join(base_dir, label)
    for fname in tqdm(os.listdir(folder), desc=f"Đang xử lý {label}"):
        if fname.endswith(".jpg"):
            # Tách thông tin từ tên file
            parts = fname.replace(".jpg", "").split("-")
            patch_id = parts[1] if len(parts) > 1 else "NA"
            slide_id = "-".join(parts[2:]) if len(parts) > 2 else "NA"
            
            # Đường dẫn tương đối
            rel_path = f"{label}/{fname}"
            full_path = os.path.join(folder, fname)
            
            # Tính tỉ lệ mô bệnh
            tissue_pct = calc_tissue_pct(full_path)
            
            data.append([slide_id, patch_id, rel_path, label.replace("_JPEG", ""), tissue_pct])

# ====== Lưu thành file CSV ======
df = pd.DataFrame(data, columns=["slide_id", "patch_id", "path", "label", "tissue_pct"])
df.to_csv("manifest_raw.csv", index=False)
print("✅ Đã tạo Mainfest_rawTCGA.csv thành công!")
print("📊 Số lượng ảnh mỗi lớp:")
print(df["label"].value_counts())
print(df.head(10))
