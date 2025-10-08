import os, pandas as pd

S = r"datasets/YourSet_S/h5"
L = r"datasets/YourSet_L/h5"
split_csv = r"splits/YourSet/splits_0.csv"

df = pd.read_csv(split_csv, dtype=str)
miss_s = [sid for sid in df["slide_id"] if not os.path.exists(f"{S}/{sid}.h5")]
miss_l = [sid for sid in df["slide_id"] if not os.path.exists(f"{L}/{sid}.h5")]

print("Thiếu ở S:", len(miss_s), miss_s[:8])
print("Thiếu ở L:", len(miss_l), miss_l[:8])

# Thống kê nhị phân
print("\nTổng số dòng:", len(df))
print("train=1:", int(df["train"].astype(int).sum()))
print("val=1:", int(df["val"].astype(int).sum()))
print("test=1:", int(df["test"].astype(int).sum()))
