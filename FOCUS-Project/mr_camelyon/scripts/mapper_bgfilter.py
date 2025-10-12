#!/usr/bin/env python3
import sys, csv, json, codecs, signal

# Khi pipe đóng (ví dụ head), Python không in BrokenPipeError
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        sys.exit(0)

def main():
    # đọc stdin line-by-line, bỏ BOM và \r
    reader = csv.reader((line.replace('\r','') for line in codecs.iterdecode(sys.stdin.buffer, 'utf-8-sig')))
    first = True
    for row in reader:
        if first:
            first = False
            # bỏ header nếu có
            if len(row) > 0 and row[0].strip().lower().startswith("slide_id"):
                continue
        if len(row) < 5:
            continue
        label_id = row[0].strip()
        patch_id = row[1].strip()
        path = row[2].strip()
        label = row[3].strip()
        try:
            tissue = float(row[4])
        except ValueError:
            continue
        if tissue >= THRESH:
            value = {"patch_id": patch_id, "path": path, "label": label, "tissue_pct": tissue}
            safe_print(f"{label_id}\t{json.dumps(value, separators=(',',':'))}")

if __name__ == "__main__":
    main()
