#!/usr/bin/env bash
set -euo pipefail

# ===========================
# Config
# ===========================
THRESH=0.20        # tissue_pct threshold (0..1)
THRESH_PIX=256     # dedup distance in pixels

# HDFS paths
INPUT=/wsi/input/manifest_raw.csv
OUT_BG=/wsi/out_bg
OUT_SORT=/wsi/out_sort
OUT_DEDUP=/wsi/out_dedup

LOCAL_OUT=mr
mkdir -p $LOCAL_OUT

# ===========================
# Hadoop streaming JAR
# ===========================
HADOOP_STREAMING_JAR="/mnt/d/hadoop-3.4.1/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar"

if [ ! -f "$HADOOP_STREAMING_JAR" ]; then
  echo "[ERR] Hadoop streaming jar not found at $HADOOP_STREAMING_JAR"
  exit 1
fi
echo "[INFO] using streaming jar: $HADOOP_STREAMING_JAR"

# ===========================
# Cleanup HDFS outputs
# ===========================
hdfs dfs -rm -r -f $OUT_BG || true
hdfs dfs -rm -r -f $OUT_SORT || true
hdfs dfs -rm -r -f $OUT_DEDUP || true

# ===========================
# JOB 1: Background filter
# ===========================
echo "[INFO] JOB1: background filter (tissue >= $THRESH)"
hadoop jar "$HADOOP_STREAMING_JAR" \
  -files mr/scripts/mapper_bgfilter.py,mr/scripts/reducer_passthrough.py \
  -D mapreduce.job.name="bgfilter" \
  -input "$INPUT" \
  -output "$OUT_BG" \
  -mapper "python3 mapper_bgfilter.py $THRESH" \
  -reducer "python3 reducer_passthrough.py"

# ===========================
# JOB 2: Group & Sort
# ===========================
echo "[INFO] JOB2: group & sort"
hadoop jar "$HADOOP_STREAMING_JAR" \
  -files mr/scripts/mapper_group.py,mr/scripts/reducer_group_sort.py \
  -D mapreduce.job.name="group_sort" \
  -input "$OUT_BG" \
  -output "$OUT_SORT" \
  -mapper "python3 mapper_group.py" \
  -reducer "python3 reducer_group_sort.py"

# ===========================
# JOB 3: Sliding-window dedup
# ===========================
echo "[INFO] JOB3: sliding-window dedup (pixel thresh $THRESH_PIX)"
hadoop jar "$HADOOP_STREAMING_JAR" \
  -files mr/scripts/mapper_window.py,mr/scripts/reducer_window.py \
  -D mapreduce.job.name="window_dedup" \
  -input "$OUT_SORT" \
  -output "$OUT_DEDUP" \
  -mapper "python3 mapper_window.py" \
  -reducer "python3 reducer_window.py"

# ===========================
# Fetch & Merge output
# ===========================
echo "[INFO] fetching results to $LOCAL_OUT/manifest_clean.csv"
hdfs dfs -getmerge $OUT_DEDUP $LOCAL_OUT/manifest_clean.csv || true

HEADER="label_id,patch_id,path,label,tissue_pct"
echo "$HEADER" > $LOCAL_OUT/manifest_clean_with_header.csv
if [ -s $LOCAL_OUT/manifest_clean.csv ]; then
  cat $LOCAL_OUT/manifest_clean.csv >> $LOCAL_OUT/manifest_clean_with_header.csv
fi
mv $LOCAL_OUT/manifest_clean_with_header.csv $LOCAL_OUT/manifest_clean.csv

# Copy to patch list
cp $LOCAL_OUT/manifest_clean.csv $LOCAL_OUT/patch_clean.list

# ===========================
# Stats
# ===========================
TOTAL=$(($(wc -l < data/manifest_raw.csv) - 1))
AFTER_BG=$(hdfs dfs -cat $OUT_BG/part-* 2>/dev/null | wc -l || echo 0)
AFTER_DEDUP=$(wc -l < $LOCAL_OUT/manifest_clean.csv || echo 0)
AFTER_DEDUP=$((AFTER_DEDUP>0 ? AFTER_DEDUP-1 : 0))

REDUCE_PERCENT=$(python3 - <<PY
t=$TOTAL; a=$AFTER_DEDUP
print(round(100*(t-a)/t,2) if t>0 else 0.0)
PY
)

cat > $LOCAL_OUT/mr_stats.json <<EOF
{
  "total_patches": $TOTAL,
  "after_bgfilter": $AFTER_BG,
  "after_dedup": $AFTER_DEDUP,
  "reduction_percent": $REDUCE_PERCENT
}
EOF

# ===========================
# Done
# ===========================
echo "[OK] MR pipeline done. Outputs:"
echo "  - $LOCAL_OUT/manifest_clean.csv"
echo "  - $LOCAL_OUT/patch_clean.list"
echo "  - $LOCAL_OUT/mr_stats.json"
