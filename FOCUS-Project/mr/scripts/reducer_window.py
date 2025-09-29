#!/usr/bin/env python3
import sys, csv, math

THRESH_PIX = 256  # ngưỡng khoảng cách (có thể chỉnh)

reader = csv.reader(sys.stdin)
writer = csv.writer(sys.stdout)

prev_coord = {}  # label_id -> (x,y)

for row in reader:
    if not row: 
        continue
    # expect: label_id,patch_id,path,label,tissue_pct
    if len(row) < 5:
        continue
    label_id = row[0].strip()
    patch_id = row[1].strip()
    path = row[2].strip()
    label = row[3].strip()
    tissue = row[4].strip()
    # extract coordinate from patch_id ending with _x_y
    parts = patch_id.split("_")
    x = -1; y = -1
    if len(parts) >= 3:
        try:
            x = int(parts[-2]); y = int(parts[-1])
        except:
            x = -1; y = -1

    keep = True
    if label_id in prev_coord:
        px, py = prev_coord[label_id]
        if px >= 0 and py >= 0 and x >=0 and y >= 0:
            dist = math.hypot(x-px, y-py)
            if dist < THRESH_PIX:
                keep = False
    if keep:
        writer.writerow([label_id, patch_id, path, label, tissue])
        prev_coord[label_id] = (x,y)
