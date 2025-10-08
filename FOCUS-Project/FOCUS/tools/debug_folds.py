import os, glob

split_dir = r"splits/YourSet"
print("Thư mục split_dir:", os.path.abspath(split_dir))

candidates = sorted(glob.glob(os.path.join(split_dir, "splits_*.csv")))
print("Tìm thấy files:", [os.path.basename(x) for x in candidates])

print("splits_0.csv tồn tại? ", os.path.exists(os.path.join(split_dir, "splits_0.csv")))
