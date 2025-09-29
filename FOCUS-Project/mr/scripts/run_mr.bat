@echo on
setlocal enabledelayedexpansion

REM =============================
REM Hadoop MapReduce Pipeline (Windows BAT version)
REM =============================

REM Config
set INPUT=/wsi/input/manifest_raw.csv
set OUT_BG=/wsi/out_bg
set OUT_SORT=/wsi/out_sort
set OUT_DEDUP=/wsi/out_dedup
set LOCAL_OUT=mr

REM Threshold tissue_pct (0..1)
set THRESH=0.20

REM Path to Hadoop streaming jar (Hadoop 3.4.1)
set HADOOP_STREAMING_JAR=D:\hadoop-3.4.1\share\hadoop\tools\lib\hadoop-streaming-3.4.1.jar

if not exist %LOCAL_OUT% mkdir %LOCAL_OUT%

echo [INFO] Using Hadoop streaming jar: %HADOOP_STREAMING_JAR%

echo [INFO] Cleaning old HDFS outputs...
hdfs dfs -rm -r -f %OUT_BG%
hdfs dfs -rm -r -f %OUT_SORT%
hdfs dfs -rm -r -f %OUT_DEDUP%

echo [INFO] JOB1: background filter (tissue >= %THRESH%)
hadoop jar "%HADOOP_STREAMING_JAR%" ^
  -D mapreduce.job.name="bgfilter" ^
  -input %INPUT% ^
  -output %OUT_BG% ^
  -mapper "python mr/scripts/mapper_bgfilter.py %THRESH%" ^
  -reducer "python mr/scripts/reducer_passthrough.py" ^
  -file mr/scripts/mapper_bgfilter.py ^
  -file mr/scripts/reducer_passthrough.py

echo [INFO] JOB2: group & sort
hadoop jar "%HADOOP_STREAMING_JAR%" ^
  -D mapreduce.job.name="group_sort" ^
  -input %OUT_BG% ^
  -output %OUT_SORT% ^
  -mapper "python mr/scripts/mapper_group.py" ^
  -reducer "python mr/scripts/reducer_group_sort.py" ^
  -file mr/scripts/mapper_group.py ^
  -file mr/scripts/reducer_group_sort.py

echo [INFO] JOB3: sliding-window dedup
hadoop jar "%HADOOP_STREAMING_JAR%" ^
  -D mapreduce.job.name="window_dedup" ^
  -input %OUT_SORT% ^
  -output %OUT_DEDUP% ^
  -mapper "python mr/scripts/mapper_window.py" ^
  -reducer "python mr/scripts/reducer_window.py" ^
  -file mr/scripts/mapper_window.py ^
  -file mr/scripts/reducer_window.py

echo [INFO] Fetching results...
hdfs dfs -getmerge %OUT_DEDUP% %LOCAL_OUT%\manifest_clean.csv

REM Add header line manually
echo label_id,patch_id,path,label,tissue_pct > %LOCAL_OUT%\manifest_clean_with_header.csv
type %LOCAL_OUT%\manifest_clean.csv >> %LOCAL_OUT%\manifest_clean_with_header.csv
move /Y %LOCAL_OUT%\manifest_clean_with_header.csv %LOCAL_OUT%\manifest_clean.csv

copy /Y %LOCAL_OUT%\manifest_clean.csv %LOCAL_OUT%\patch_clean.list

REM =============================
REM Stats
REM =============================

for /f %%A in ('find /c /v "" ^< data\manifest_raw.csv') do set TOTAL=%%A
set /a TOTAL=%TOTAL%-1

for /f %%A in ('hdfs dfs -cat %OUT_BG%/part-* ^| find /c /v ""') do set AFTER_BG=%%A
for /f %%A in ('find /c /v "" ^< %LOCAL_OUT%\manifest_clean.csv') do set AFTER_DEDUP=%%A
set /a AFTER_DEDUP=%AFTER_DEDUP%-1
if %AFTER_DEDUP% LSS 0 set AFTER_DEDUP=0

set /a REDUCTION=(%TOTAL%-%AFTER_DEDUP%)*100/%TOTAL%

(
echo {
echo   "total_patches": %TOTAL%,
echo   "after_bgfilter": %AFTER_BG%,
echo   "after_dedup": %AFTER_DEDUP%,
echo   "reduction_percent": %REDUCTION%
echo }
) > %LOCAL_OUT%\mr_stats.json

echo [OK] Pipeline completed!
echo Output:
echo   - %LOCAL_OUT%\manifest_clean.csv
echo   - %LOCAL_OUT%\patch_clean.list
echo   - %LOCAL_OUT%\mr_stats.json

endlocal
