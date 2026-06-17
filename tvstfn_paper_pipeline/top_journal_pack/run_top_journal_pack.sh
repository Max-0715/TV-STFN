#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/workplace/jwx/TV-STFN"
PACK_DIR="$ROOT/tvstfn_paper_pipeline/top_journal_pack"

python "$PACK_DIR/task1_conformer_attention_visualization.py"
python "$PACK_DIR/task2_activity_cliff_analysis.py"
python "$PACK_DIR/task3_dft_dl_closure_figure.py"
python "$PACK_DIR/task4_scaffold_split_validation.py"

echo "All top-journal tasks finished."
