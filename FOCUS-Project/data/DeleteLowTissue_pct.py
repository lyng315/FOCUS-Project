import pandas as pd
import os

# ---- Cấu hình ----
manifest_filename = "Mainfest_rawCML.csv" #Thay thế tên file.csv muốn chỉnh sửa

# 1. Đọc file CSV
try:
    df = pd.read_csv(manifest_filename)
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file Manifest: {manifest_filename}.")
    print("Vui lòng đảm bảo file CSV nằm cùng thư mục với script Python này.")
    exit()
except Exception as e:
    print(f"❌ Lỗi khi đọc file CSV: {e}")
    exit()

# Ghi lại số lượng dòng ban đầu để báo cáo
initial_rows = len(df)
print(f"🔎 Tổng số dòng ban đầu trong Manifest: {initial_rows}")

# 2. Chuẩn bị cột tissue_pct
# Đảm bảo cột tissue_pct là kiểu số.
df['tissue_pct'] = pd.to_numeric(df['tissue_pct'], errors='coerce')

# 3. Lọc dữ liệu: Chỉ giữ lại các dòng mà tissue_pct >= 0.1 (tức là XÓA các dòng < 0.1)
# Chúng ta cũng loại bỏ các dòng có tissue_pct là NaN (giá trị lỗi)
df_filtered = df[(df['tissue_pct'] >= 0.1) & (df['tissue_pct'].notna())]

# 4. Tính toán số dòng đã xóa
deleted_rows = initial_rows - len(df_filtered)

# 5. Lưu lại file CSV đã được lọc (ghi đè file cũ)
df_filtered.to_csv(manifest_filename, index=False)

# 6. Hiển thị kết quả
print("=========================================================")
print(f"✅ HOÀN TẤT LỌC DỮ LIỆU")
print(f"Tổng số dòng đã bị XÓA (tissue_pct < 0.1 hoặc lỗi): {deleted_rows}")
print(f"Tổng số dòng còn lại (tissue_pct >= 0.1): {len(df_filtered)}")
print(f"File {manifest_filename} đã được cập nhật.")
print("=========================================================")