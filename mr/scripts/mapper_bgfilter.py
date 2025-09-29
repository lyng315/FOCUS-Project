#!/usr/bin/env python3
import sys, csv, json

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20

r = csv.reader(sys.stdin)
first = True
for row in r:
    if first:
        first = False
        # skip header if looks like header
        if len(row)>0 and row[0].strip().lower().startswith("label_id"):
            continue
    if len(row) < 5:
        continue
    label_id = row[0].strip()
    patch_id  = row[1].strip()
    path      = row[2].strip()
    label     = row[3].strip()
    try:
        tissue = float(row[4])
    except:
        continue
    if tissue >= THRESH:
        value = {"patch_id": patch_id, "path": path, "label": label, "tissue_pct": tissue}
        # emit key \t json
        print(f"{label_id}\t{json.dumps(value, separators=(',',':'))}")
