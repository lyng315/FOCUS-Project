# compute_balacc_f1.py  — đọc split_*_results.pkl để:
# (A) gom các metric sẵn có (acc/auc/f1/bal_acc) -> Mean±Std
# (B) nếu không có, in cấu trúc để bạn xem bên trong là gì

import os, glob, pickle, numpy as np, pandas as pd

ROOT = r"E:\FOCUS_results\YourSet\YourSet_fold_test_s1"  # chỉnh nếu khác

def load_obj(fp):
    try:
        return pd.read_pickle(fp)
    except Exception:
        with open(fp, "rb") as f:
            return pickle.load(f)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# tên cột/khóa có thể gặp
name_map = {
    'acc': ['acc','test_acc','accuracy'],
    'auc': ['auc','auroc','test_auc'],
    'f1' : ['f1','test_f1'],
    'bal_acc': ['balanced_acc','bal_acc','balanced_accuracy','test_bal_acc'],
}

def pick_metrics_from_mapping(obj):
    """Cố gắng lấy các số liệu scalar acc/auc/f1/bal_acc từ dict/DataFrame."""
    out = {}
    if isinstance(obj, dict):
        keys_low = {k.lower(): k for k in obj.keys()}
        for std_name, aliases in name_map.items():
            for a in aliases:
                if a in keys_low:
                    v = safe_float(obj[keys_low[a]])
                    if v is not None:
                        out[std_name] = v
                        break
    elif isinstance(obj, pd.DataFrame):
        cols_low = {c.lower(): c for c in obj.columns}
        # ưu tiên 1 dòng duy nhất -> coi như summary
        if len(obj) == 1:
            row = obj.iloc[0]
            for std_name, aliases in name_map.items():
                for a in aliases:
                    if a in cols_low:
                        v = safe_float(row[cols_low[a]])
                        if v is not None:
                            out[std_name] = v
                            break
    return out

def describe_object(obj):
    """In cấu trúc của obj để bạn xem nhanh."""
    if isinstance(obj, dict):
        print("  - type: dict")
        for k, v in obj.items():
            t = type(v).__name__
            shape = ""
            try:
                import numpy as np
                if hasattr(v, 'shape') and v.shape:
                    shape = f", shape={tuple(v.shape)}"
            except Exception:
                pass
            print(f"    • {k}: {t}{shape}")
    elif isinstance(obj, pd.DataFrame):
        print("  - type: DataFrame")
        print(f"    • columns: {list(obj.columns)}")
        print(f"    • head:\n{obj.head().to_string(index=False)}")
    elif isinstance(obj, (list, tuple)):
        print(f"  - type: {type(obj).__name__}, len={len(obj)}")
        print("    • element types:", [type(x).__name__ for x in obj])
    else:
        print(f"  - type: {type(obj).__name__}")
        print(f"    • repr: {repr(obj)[:200]}...")

files = sorted(glob.glob(os.path.join(ROOT, "split_*_results.pkl")))
if not files:
    raise SystemExit(f"Không thấy file split_*_results.pkl trong {ROOT}")

rows = []
missing = []
print(f"🔎 Tìm metric trong {len(files)} file .pkl ...")
for fp in files:
    name = os.path.basename(fp)
    try:
        obj = load_obj(fp)
        met = pick_metrics_from_mapping(obj)
        if met:
            met["file"] = name
            rows.append(met)
            print(f"  [OK] {name} -> {met}")
        else:
            print(f"  [!] {name}: không thấy acc/auc/f1/bal_acc — in cấu trúc để kiểm tra:")
            describe_object(obj)
            missing.append(name)
    except Exception as e:
        print(f"  [ERR] {name}: {e}")

if not rows and missing:
    raise SystemExit("\n❌ Không trích được metric nào từ .pkl (chỉ có cấu trúc). Dựa vào cấu trúc in ở trên để chỉnh parser hoặc bật logging per-sample khi eval.")

if rows:
    df = pd.DataFrame(rows).sort_values("file")
    print("\n📄 Bảng metric đọc được (theo split):")
    print(df.to_string(index=False))

    print("\n== Tổng hợp (Mean ± Std) ==")
    for col in ["bal_acc","acc","auc","f1"]:
        if col in df.columns:
            mean = df[col].mean()
            std  = df[col].std(ddof=1)
            print(f"{col.upper():8s}: {mean:.3f} ± {std:.3f}")

    if missing:
        print("\n(ℹ️ Một số file không có metric sẵn, đã in cấu trúc ở trên để bạn tham khảo):")
        for n in missing:
            print("   -", n)
