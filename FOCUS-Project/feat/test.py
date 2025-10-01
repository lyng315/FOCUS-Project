#CODE TEST TCGA-01.pt

import torch

# Đường dẫn tới file .pt
file_path = "TCGA-01.pt"

# Load file
data = torch.load(file_path, map_location="cpu")

print("📂 Kiểu dữ liệu:", type(data))

if isinstance(data, torch.Tensor):
    print("Tensor shape:", data.shape)
    print("5 dòng đầu tiên:\n", data[:5])

elif isinstance(data, dict):
    print("Keys trong file:", list(data.keys())[:10])  # in 10 keys đầu
    for k, v in list(data.items())[:3]:
        print(f"👉 {k}:", type(v))
else:
    print("Nội dung không rõ, in thử:", data)
