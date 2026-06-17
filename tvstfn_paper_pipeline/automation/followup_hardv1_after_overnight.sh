#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/workplace/jwx/TV-STFN"
cd "$ROOT"

PY="${PY:-$ROOT/.venv/bin/python}"
LOG_DIR="$ROOT/cv_logs_20260311_followup"
mkdir -p "$LOG_DIR"

WAIT_PATTERN='benchmark_tvstfn_fast.py --n-folds 5 --tag tv5overnight_1'
TAG="hard_v1_overnight"

{
  echo "[$(date '+%F %T')] follow-up watcher started"
  echo "[$(date '+%F %T')] waiting for overnight tag tv5overnight_1 to finish"

  while pgrep -af "$WAIT_PATTERN" >/dev/null; do
    echo "[$(date '+%F %T')] overnight still running..."
    sleep 180
  done

  echo "[$(date '+%F %T')] overnight ended, starting hard-sample weighted run"

  CUDA_VISIBLE_DEVICES=4 PYTHONUNBUFFERED=1 "$PY" -u benchmark_tvstfn_fast.py \
    --n-folds 5 \
    --tag "$TAG" \
    --epochs 32 \
    --batch-size 20 \
    --lr 6e-05 \
    --early-stop 10 \
    --val-ratio 0.12 \
    --lambda-focal 1.5 \
    --lambda-rank 0.18 \
    --lambda-mse 0.7 \
    --lambda-consistency 0.18 \
    --focal-alpha 0.22 \
    --focal-gamma-pos 1.0 \
    --focal-gamma-neg 3.0 \
    --seed 73 \
    --model-hidden-dim 640 \
    --model-dropout 0.10 \
    --zero-d-prior 2.4 \
    --cls-skip-weight 0.55 \
    --gate-temperature 0.68 \
    --num-workers 4 \
    --skip-fold-predictions \
    --tune-cls-threshold \
    --calibrate \
    --sample-weights-csv tvstfn_paper_pipeline/outputs/optimization/sample_weights_v1.csv \
    --sample-weight-power 1.2 \
    > "$LOG_DIR/${TAG}.log" 2>&1

  code=$?
  echo "[$(date '+%F %T')] hard_v1 follow-up finished exit=$code"
} > "$LOG_DIR/followup_dispatcher.log" 2>&1
