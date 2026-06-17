import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def main():
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    with tempfile.TemporaryDirectory() as td:
        # Build minimal NPZ for UMAP plotting
        n = 128
        d0, df = 64, 128
        y = np.random.randn(n).astype(np.float32)
        np.savez_compressed(
            os.path.join(td, "umap_embeddings.npz"),
            feat_fusion=np.random.randn(n, df).astype(np.float32),
            feat_0d=np.random.randn(n, d0).astype(np.float32),
            y_true=y,
        )

        out_umap = os.path.join(td, "wp1")
        os.makedirs(out_umap, exist_ok=True)
        run(
            [
                sys.executable,
                "tvstfn_paper_pipeline/wp1_umap/plot_umap.py",
                "--npz",
                os.path.join(td, "umap_embeddings.npz"),
                "--out-dir",
                out_umap,
                "--max-points",
                "200",
            ],
            cwd=repo,
        )
        assert os.path.exists(os.path.join(out_umap, "figure_Y_umap_dual.png"))

        # Build minimal fold predictions for stats
        pred_dir = os.path.join(td, "pred")
        os.makedirs(pred_dir, exist_ok=True)
        df_pred = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN", "CCC", "CCCl"],
                "True_LogP": [-5.0, -7.0, -6.5, -5.5],
                "Pred_Score_TVSTFN": [0.9, 0.1, 0.2, 0.8],
            }
        )
        df_pred.to_csv(os.path.join(pred_dir, "fold_0_predictions.csv"), index=False)

        out_stats = os.path.join(td, "wp5")
        os.makedirs(out_stats, exist_ok=True)
        run(
            [
                sys.executable,
                "tvstfn_paper_pipeline/wp5_stats_calibration/stats_and_calibration.py",
                "--pred-dir",
                pred_dir,
                "--out-dir",
                out_stats,
                "--focus-model",
                "TVSTFN",
                "--n-bootstrap",
                "100",
            ],
            cwd=repo,
        )
        assert os.path.exists(os.path.join(out_stats, "stats_summary.csv"))
        assert os.path.exists(os.path.join(out_stats, "reliability_diagram.png"))

    print("smoke_test_fast: PASS")


if __name__ == "__main__":
    main()
