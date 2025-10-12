#!/usr/bin/env python3
import sys, csv, math, re

THRESH_PIX = 256  # khoảng cách pixel

reader = csv.reader((line.replace('\r','') for line in sys.stdin))
writer = csv.writer(sys.stdout)

prev_coord = {}  # label_id -> (x, y)

for row in reader:
    if not row or len(row) < 5:
        continue
    label_id, patch_id, path, label, tissue = [c.strip() for c in row]

    x = y = -1
    m = re.search(r"_x_(\d+)_y_(\d+)\.png$", patch_id)
    if m:
        x, y = int(m.group(1)), int(m.group(2))

    keep = True
    if label_id in prev_coord:
        px, py = prev_coord[label_id]
        if px>=0 and py>=0 and x>=0 and y>=0:
            dist = math.hypot(x - px, y - py)
            if dist < THRESH_PIX:
                keep = False
    if keep:
        writer.writerow([label_id, patch_id, path, label, tissue])
        prev_coord[label_id] = (x, y)