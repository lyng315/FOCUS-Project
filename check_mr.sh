#!/usr/bin/env bash
set -euo pipefail

# ===========================
# Config
# ===========================
THRESH=0.20
THRESH_PIX=256

# Input CSV
INPUT=data/manifest_raw.csv

# Scripts
MAPPER_BG=mr/scripts/mapper_bgfilter.py
REDUCER_PASS=mr/scripts/reducer_passthrough.py
MAPPER_GROUP=mr/scripts/mapper_group.py
REDUCER_SORT=mr/scripts/reducer_group_sort.py
MAPPER_WINDOW=mr/scripts/mapper_window.py
REDUCER_WINDOW=mr/scripts/reducer_window.py

# ===========================
# Make sure scripts are executable
# ===========================
chmod +x mr/scripts/*.py
echo "[INFO] Set execute permission for Python scripts"

# ===========================
# Test mapper_bgfilter.py
# ===========================
echo "[INFO] Testing mapper_bgfilter.py"
head -n 5 $INPUT | python3 $MAPPER_BG $THRESH

# ===========================
# Test reducer_passthrough.py
# ===========================
echo "[INFO] Testing reducer_passthrough.py"
head -n 5 $INPUT | python3 $MAPPER_BG $THRESH | python3 $REDUCER_PASS

# ===========================
# Test mapper_group.py + reducer_group_sort.py
# ===========================
echo "[INFO] Testing mapper_group.py + reducer_group_sort.py"
head -n 10 $INPUT | python3 $MAPPER_BG $THRESH | python3 $REDUCER_PASS \
  | python3 $MAPPER_GROUP | python3 $REDUCER_SORT

# ===========================
# Test mapper_window.py + reducer_window.py
# ===========================
echo "[INFO] Testing mapper_window.py + reducer_window.py"
head -n 10 $INPUT | python3 $MAPPER_BG $THRESH | python3 $REDUCER_PASS \
  | python3 $MAPPER_GROUP | python3 $REDUCER_SORT \
  | python3 $MAPPER_WINDOW | python3 $REDUCER_WINDOW

echo "[OK] All Python scripts ran successfully on sample input."
