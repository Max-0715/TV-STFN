#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/workplace/jwx/TV-STFN/non-deep-learning models"
PY="/data/workplace/jwx/TV-STFN/.venv/bin/python"
LOG_DIR="$ROOT/unified_logs_20260308"
mkdir -p "$LOG_DIR"

run_job() {
  local label="$1"
  shift
  echo "[$(date '+%F %T')] START $label" | tee -a "$LOG_DIR/dispatcher.log"
  (
    cd "$ROOT"
    "$@"
  ) > "$LOG_DIR/${label}.log" 2>&1
  local code=$?
  echo "[$(date '+%F %T')] END $label exit=$code" | tee -a "$LOG_DIR/dispatcher.log"
}

run_job CatBoost env CUDA_VISIBLE_DEVICES=0 MODEL_FILTER=CatBoost GPU_ID=0 OUTPUT_TAG=catboost "$PY" benchmark_compare_paper_models.py &
P0=$!
run_job XGBoost env CUDA_VISIBLE_DEVICES=6 MODEL_FILTER=XGBoost GPU_ID=0 OUTPUT_TAG=xgboost "$PY" benchmark_compare_paper_models.py &
P1=$!
run_job LGBM env CUDA_VISIBLE_DEVICES=7 MODEL_FILTER=LGBM GPU_ID=0 OUTPUT_TAG=lgbm "$PY" benchmark_compare_paper_models.py &
P2=$!
run_job CPUClassic env MODEL_FILTER='KNN,RF,SVM (poly),SVM (rbf),DT' OUTPUT_TAG=cpuclassic "$PY" benchmark_compare_paper_models.py &
P3=$!

wait "$P0" "$P1" "$P2" "$P3"
echo "[$(date '+%F %T')] ALL NON-DEEP DONE" | tee -a "$LOG_DIR/dispatcher.log"
