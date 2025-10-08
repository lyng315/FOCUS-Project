import os, pandas as pd
from datasets.dataset_generic import Generic_MIL_Dataset

CSV_SPLIT = "splits/YourSet/splits_0.csv"
LABELS    = "datasets/YourSet_S/labels.csv"   # labels phải ở S
DIR_S     = "YourSet_S"   # .h5 nằm TRỰC TIẾP trong thư mục này
DIR_L     = "YourSet_L"

# 0) In vài thông tin cơ bản
print("== CHECK FILES ==")
df = pd.read_csv(CSV_SPLIT, dtype=str)
print("Splits rows:", len(df))
print("Split counts:\n", df["split"].value_counts())

# 1) Kiểm tra tồn tại h5 ở S và L cho MỌI slide_id
miss_s, miss_l = [], []
for sid in df["slide_id"]:
    if not os.path.exists(os.path.join("datasets", DIR_S, f"{sid}.h5")):
        miss_s.append(sid)
    if not os.path.exists(os.path.join("datasets", DIR_L, f"{sid}.h5")):
        miss_l.append(sid)

print(f"\nMissing in S ({DIR_S}):", len(miss_s), "=>", miss_s[:5])
print(f"Missing in L ({DIR_L}):", len(miss_l), "=>", miss_l[:5])

# 2) Kiểm tra labels.csv có đủ nhãn cho các slide_id trong splits
lab = pd.read_csv(LABELS, dtype=str)
need = set(df["slide_id"])
have = set(lab["slide_id"])
missing_labels = sorted(need - have)
print("\nMissing labels for slide_id in labels.csv:", len(missing_labels), "=>", missing_labels[:5])

# 3) Thử khởi tạo dataset như trong main.py và in kích thước từng split
print("\n== INIT GENERIC_MIL_DATASET ==")
dset = Generic_MIL_Dataset(
    csv_path=LABELS, mode="transformer",
    data_dir_s=DIR_S, data_dir_l=DIR_L,
    shuffle=False, print_info=True,
    label_dict={'LUAD':0, 'LUSC':1},
    patient_strat=False, ignore=[]
)

train_set, val_set, test_set = dset.return_splits(
    from_id=False, csv_path=CSV_SPLIT
)
print("\nSizes => train:", len(train_set), "val:", len(val_set), "test:", len(test_set))
