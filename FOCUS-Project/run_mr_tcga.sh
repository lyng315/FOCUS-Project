#!/usr/bin/env bash
set -euo pipefail

# ===========================
# Config
# ===========================
THRESH=0.20        # tissue_pct threshold (0..1)
#THRESH_PIX=256     # dedup distance in pixels

# HDFS paths
INPUT=/tcga/input/Manifest_rawTCGA.csv
OUT_BG=/tcga/out_bg
OUT_GROUP=/tcga/out_group
#OUT_DEDUP=/tcga/out_dedup

LOCAL_OUT=mr_tcga
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
# Make Python scripts executable
# ===========================
chmod +x mr_tcga/scripts/*.py
echo "[INFO] Set execute permission for all Python scripts"

# ===========================
# Cleanup HDFS outputs
# ===========================
hdfs dfs -rm -r -f $OUT_BG || true
hdfs dfs -rm -r -f $OUT_GROUP || true
#hdfs dfs -rm -r -f $OUT_DEDUP || true

# ===========================
# JOB 1: Background filter
# ===========================
echo "[INFO] JOB1: background filter (tissue >= $THRESH)"
START_JOB1=$(date +%s)
hadoop jar "$HADOOP_STREAMING_JAR" \
  -files mr_tcga/scripts/mapper_bgfilter.py,mr_tcga/scripts/reducer_passthrough.py \
  -D mapreduce.job.name="bgfilter" \
  -input "$INPUT" \
  -output "$OUT_BG" \
  -mapper "python3 mapper_bgfilter.py $THRESH" \
  -reducer "python3 reducer_passthrough.py"
END_JOB1=$(date +%s)
DURATION_JOB1=$((END_JOB1 - START_JOB1))
echo "[DONE] JOB1 completed in ${DURATION_JOB1}s"
echo "-----------------------------------------------------"
# ===========================
# JOB 2: Group
# ===========================
echo "[INFO] JOB2: group"
START_JOB2=$(date +%s)
hadoop jar "$HADOOP_STREAMING_JAR" \
  -files mr_tcga/scripts/mapper_group.py,mr_tcga/scripts/reducer_group.py \
  -D mapreduce.job.name="group" \
  -input "$OUT_BG" \
  -output "$OUT_GROUP" \
  -mapper "python3 mapper_group.py" \
  -reducer "python3 reducer_group.py"
END_JOB2=$(date +%s)
DURATION_JOB2=$((END_JOB2 - START_JOB2))
echo "[DONE] JOB2 completed in ${DURATION_JOB2}s"
echo "-----------------------------------------------------"

# ===========================
# Fetch & Merge output
# ===========================
echo "[INFO] fetching results to $LOCAL_OUT/manifest_clean.csv"
hdfs dfs -getmerge $OUT_GROUP $LOCAL_OUT/manifest_clean.csv || true

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
TOTAL=$(($(wc -l < data/TCGA/Manifest_rawTCGA.csv) - 1))
AFTER_BG=$(hdfs dfs -cat $OUT_BG/part-* 2>/dev/null | wc -l || echo 0)
AFTER_GROUP=$(wc -l < $LOCAL_OUT/manifest_clean.csv || echo 0)
AFTER_GROUP=$((AFTER_GROUP>0 ? AFTER_GROUP-1 : 0))

REDUCE_PERCENT=$(python3 - <<PY
t=$TOTAL; a=$AFTER_GROUP
print(round(100*(t-a)/t,2) if t>0 else 0.0)
PY
)

TOTAL_TIME=$((DURATION_JOB1 + DURATION_JOB2))

cat > $LOCAL_OUT/mr_tcga_stats.json <<EOF
{
  "total_patches": $TOTAL,
  "after_bgfilter": $AFTER_BG,
  "after_group": $AFTER_GROUP,
  "reduction_percent": $REDUCE_PERCENT,
  "job1_time_sec": $DURATION_JOB1,
  "job2_time_sec": $DURATION_JOB2,
  "total_time_sec": $TOTAL_TIME
}
EOF

# ===========================
# Done
# ===========================
echo "[OK] MR pipeline done. Outputs:"
echo "  - $LOCAL_OUT/manifest_clean.csv"
echo "  - $LOCAL_OUT/patch_clean.list"
echo "  - $LOCAL_OUT/mr_tcga_stats.json"
