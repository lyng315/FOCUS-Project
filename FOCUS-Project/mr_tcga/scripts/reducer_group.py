#!/usr/bin/env python3
import sys, json, csv
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

    patch_id = obj.get("patch_id", "")
    path = obj.get("path", "")
    label = obj.get("label", "")
    tissue = obj.get("tissue_pct", "")

    buckets[label_id].append({
        "patch_id": patch_id,
        "path": path,
        "label": label,
        "tissue": tissue
    })

writer = csv.writer(sys.stdout)
for slide, arr in buckets.items():
    for rec in arr:   # không sort nữa
        writer.writerow([slide, rec["patch_id"], rec["path"], rec["label"], rec["tissue"]])
