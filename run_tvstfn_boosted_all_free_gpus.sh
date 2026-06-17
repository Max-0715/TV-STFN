#!/usr/bin/env bash
set -euo pipefail
ROOT='/data/workplace/jwx/TV-STFN'
PY='/data/workplace/jwx/TV-STFN/.venv/bin/python'
LOG_DIR="$ROOT/cv_logs_20260309_boosted"
mkdir -p "$LOG_DIR"
cd "$ROOT"

run_job() {
  local gpu="$1"
  local folds="$2"
  local tag="$3"
  echo "[$(date '+%F %T')] START gpu=$gpu folds=$folds tag=$tag" | tee -a "$LOG_DIR/dispatcher.log"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u benchmark_tvstfn_fast.py \
    --folds "$folds" \
    --tag "$tag" \
    --epochs 30 \
    --batch-size 24 \
    --lr 1e-4 \
    --early-stop 8 \
    --lambda-focal 1.0 \
    --lambda-rank 0.1 \
    --lambda-mse 1.0 \
    --tune-cls-threshold \
    > "$LOG_DIR/$tag.log" 2>&1
  local code=$?
  echo "[$(date '+%F %T')] END gpu=$gpu folds=$folds tag=$tag exit=$code" | tee -a "$LOG_DIR/dispatcher.log"
  return $code
}

run_job 1 '0-3' 'boost_gpu1_folds0_3' &
PID1=$!
run_job 2 '4-6' 'boost_gpu2_folds4_6' &
PID2=$!
run_job 4 '7-9' 'boost_gpu4_folds7_9' &
PID3=$!

wait $PID1
wait $PID2
wait $PID3

echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG_DIR/dispatcher.log"
