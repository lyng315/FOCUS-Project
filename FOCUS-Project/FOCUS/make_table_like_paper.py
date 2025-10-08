# make_table_like_paper.py
import os, math, pandas as pd

PATH = r"E:\FOCUS_results\YourSet\YourSet_fold_test_s1\result_partial_0_8.csv"  # <-- giữ đúng tên file bạn có

if not os.path.exists(PATH):
    raise FileNotFoundError(f"Không thấy file: {PATH}")

df = pd.read_csv(PATH)
df.columns = [c.strip().lower() for c in df.columns]

# Trường hợp 1: file là summary (metric = mean/var)  <-- file bạn đang có
if 'metric' in df.columns and set(df['metric'].str.lower()) >= {'mean','var'}:
    df['metric'] = df['metric'].str.lower()
    mean_row = df[df['metric']=='mean'].iloc[0].to_dict()
    var_row  = df[df['metric']=='var' ].iloc[0].to_dict()
    # lấy các metric có mặt
    keys = [k for k in ['bal_acc','test_auc','f1','test_acc','precision','recall'] if k in df.columns]
    stats = {k: (mean_row[k], math.sqrt(var_row[k])) for k in keys}
else:
    # Trường hợp 2: file là per-fold -> tính mean/std bình thường
    rename = {'auc':'test_auc','auroc':'test_auc','acc':'test_acc','test_f1':'f1'}
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    keys = [k for k in ['bal_acc','test_auc','f1','test_acc','precision','recall'] if k in df.columns]
    stats = {k: (float(df[k].mean()), float(df[k].std(ddof=1))) for k in keys}

def fmt_pair(m):  # 3 chữ số như paper
    mean, std = stats[m]
    return f"{mean:.3f}±{std:.3f}"

dataset_label = "YourSet (2 classes)"
setting = "Your run"  # đổi nếu muốn '4-shot'/'8-shot'/...

print("Tóm tắt Mean ± Std (đúng):")
for k in ['bal_acc','test_auc','f1','test_acc']:
    if k in stats:
        print(f"{k.upper():10s}: {fmt_pair(k)}")
print()

# Markdown row (giống Table 1)
def cell(m): return fmt_pair(m) if m in stats else "—"
md_header = "| Dataset | Methods | Setting | Balanced ACC | AUC | F1 |\n|---|---|---|---:|---:|---:|"
md_row = f"| {dataset_label} | FOCUS (Ours) | {setting} | {cell('bal_acc')} | {cell('test_auc')} | {cell('f1')} |"
print("Markdown:\n" + md_header + "\n" + md_row + "\n")

# LaTeX row
def lcell(m): return cell(m).replace("±", r"$\pm$")
latex_row = f"{dataset_label} & FOCUS (Ours) & {setting} & {lcell('bal_acc')} & {lcell('test_auc')} & {lcell('f1')} \\\\"
print("LaTeX:\n" + latex_row)
