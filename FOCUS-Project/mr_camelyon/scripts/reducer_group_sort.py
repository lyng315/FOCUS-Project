#!/usr/bin/env python3
import sys, json, csv, re
from collections import defaultdict

buckets = defaultdict(list)

for line in sys.stdin:
    line = line.rstrip('\r\n')
    if not line:
        continue
    try:
        label_id, json_str = line.split("\t", 1)
    except ValueError:
        continue
    try:
        obj = json.loads(json_str)
    except:
        continue
    patch_id = obj.get("patch_id","")
    path = obj.get("path","")
    label = obj.get("label","")
    tissue = obj.get("tissue_pct","")

    x = y = -1
    m = re.search(r"_x_(\d+)_y_(\d+)\.png$", patch_id)
    if m:
        x = int(m.group(1))
        y = int(m.group(2))

    buckets[label_id].append({
        "x": x, "y": y, "patch_id": patch_id,
        "path": path, "label": label, "tissue": tissue
    })

writer = csv.writer(sys.stdout)
for slide, arr in buckets.items():
    for rec in sorted(arr, key=lambda r: (r["x"], r["y"])):
        writer.writerow([slide, rec["patch_id"], rec["path"], rec["label"], rec["tissue"]])