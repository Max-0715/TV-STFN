#!/usr/bin/env bash
set -euo pipefail

# Run from TV-STFN root:
#   bash tvstfn_paper_pipeline/run_all.sh

PYTHON_BIN=${PYTHON_BIN:-python}
MODE=${MODE:-fast}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[pipeline] mode=${MODE}"

# WP0: Freeze baseline assets
$PYTHON_BIN tvstfn_paper_pipeline/wp0_freeze_assets/freeze_assets.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/paper_v1

# WP1: UMAP embeddings + figure
$PYTHON_BIN tvstfn_paper_pipeline/wp1_umap/export_embeddings.py \
  --data-dir tetraview_processed \
  --weights best_tetraview_model.pth \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp1_umap \
  --max-samples 2500

$PYTHON_BIN tvstfn_paper_pipeline/wp1_umap/plot_umap.py \
  --npz tvstfn_paper_pipeline/outputs/wp1_umap/umap_embeddings.npz \
  --out-dir tvstfn_paper_pipeline/outputs/wp1_umap \
  --max-points 2500

# WP2: Stratified robustness
$PYTHON_BIN tvstfn_paper_pipeline/wp2_stratified_robustness/stratified_eval.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp2_stratified

# WP3: Activity cliff + conformer attention
$PYTHON_BIN tvstfn_paper_pipeline/wp3_activity_cliff/find_activity_cliffs.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp3_cliff

$PYTHON_BIN tvstfn_paper_pipeline/wp3_activity_cliff/export_conformer_attention.py \
  --indices-csv tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv \
  --data-dir tetraview_processed \
  --weights best_tetraview_model.pth \
  --out-dir tvstfn_paper_pipeline/outputs/wp3_cliff

$PYTHON_BIN tvstfn_paper_pipeline/wp3_activity_cliff/build_cliff_figure_data.py \
  --cliff-csv tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv \
  --attn-csv tvstfn_paper_pipeline/outputs/wp3_cliff/conformer_attention_top.csv \
  --out-dir tvstfn_paper_pipeline/outputs/wp3_cliff

if [[ "$MODE" == "full" ]]; then
  # WP4: Ablation + figure (expensive)
  $PYTHON_BIN tvstfn_paper_pipeline/wp4_ablation/run_ablation_cv.py \
    --data-dir tetraview_processed \
    --out-dir tvstfn_paper_pipeline/outputs/wp4_ablation \
    --n-folds 5 \
    --epochs 25

  $PYTHON_BIN tvstfn_paper_pipeline/wp4_ablation/plot_ablation.py \
    --summary-csv tvstfn_paper_pipeline/outputs/wp4_ablation/ablation_summary.csv \
    --out-dir tvstfn_paper_pipeline/outputs/wp4_ablation
else
  echo "[pipeline] skip WP4 ablation in fast mode"
fi

# WP5: Stats + calibration
$PYTHON_BIN tvstfn_paper_pipeline/wp5_stats_calibration/stats_and_calibration.py \
  --pred-dir benchmark_results \
  --out-dir tvstfn_paper_pipeline/outputs/wp5_stats \
  --focus-model TVSTFN

echo "All paper pipeline steps finished."