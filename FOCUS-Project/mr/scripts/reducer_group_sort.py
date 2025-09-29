#!/usr/bin/env python3
import sys, json, csv
from collections import defaultdict

buckets = defaultdict(list)

for line in sys.stdin:
    line = line.rstrip()
    if not line: 
        continue
    try:
        label_id, json_str = line.split("\t", 1)
    except ValueError:
        # malformed - skip
        continue
    try:
        obj = json.loads(json_str)
    except:
        continue
    patch_id = obj.get("patch_id","")
    path = obj.get("path","")
    label = obj.get("label","")
    tissue = obj.get("tissue_pct", "")
    # try extract coords from patch_id if encoded as ..._x_y
    parts = patch_id.split("_")
    x = -1; y = -1
    if len(parts) >= 3:
        try:
            x = int(parts[-2]); y = int(parts[-1])
        except:
            x = -1; y = -1
    buckets[label_id].append({"x":x,"y":y,"patch_id":patch_id,"path":path,"label":label,"tissue":tissue})

writer = csv.writer(sys.stdout)
# no header here (will be added later by run script)
for slide, arr in buckets.items():
    for rec in sorted(arr, key=lambda r:(r["x"], r["y"])):
        writer.writerow([slide, rec["patch_id"], rec["path"], rec["label"], rec["tissue"]])
