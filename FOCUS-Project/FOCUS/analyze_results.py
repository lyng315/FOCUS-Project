import pandas as pd
import numpy as np
import os

# 🔹 Đường dẫn file kết quả
path = r"E:\FOCUS_results\YourSet\YourSet_fold_test_s1\summary_partial_0_8.csv"

# Kiểm tra file tồn tại
if not os.path.exists(path):
    raise FileNotFoundError(f"Không tìm thấy file: {path}")

# Đọc file CSV
df = pd.read_csv(path)
print("📄 5 dòng đầu:")
print(df.head(), "\n")

# 🔹 Tính trung bình ± độ lệch chuẩn
metrics = ['test_acc', 'bal_acc', 'test_auc', 'f1', 'precision', 'recall']
print("📊 Thống kê trung bình ± độ lệch chuẩn:\n")
for m in metrics:
    if m in df.columns:
        mean, std = df[m].mean(), df[m].std()
        print(f"{m.upper():12s}: {mean:.4f} ± {std:.4f}")


