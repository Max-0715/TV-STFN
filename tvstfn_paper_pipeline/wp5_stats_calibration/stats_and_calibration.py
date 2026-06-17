import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tvstfn_paper_pipeline.common.utils import (
    bootstrap_ci,
    ensure_dir,
    find_model_score_columns,
    find_true_labels,
    load_fold_predictions,
    mean_std_ci95,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def main():
    parser = argparse.ArgumentParser(description="Statistical summary + calibration plots")
    parser.add_argument("--pred-dir", type=str, default="benchmark_results")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp5_stats")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--focus-model", type=str, default="TVSTFN")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_fold_predictions(args.pred_dir)
    y_true = find_true_labels(df)

    score_cols = find_model_score_columns(df)
    if args.focus_model not in score_cols:
        raise KeyError(f"No Pred_Score column found for model: {args.focus_model}")

    rows = []
    for model_name, col in score_cols.items():
        y_prob = df[col].to_numpy().astype(float)
        brier = brier_score_loss(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)

        mean_v, std_v, lo, hi = mean_std_ci95(y_prob)
        b_lo, b_hi = bootstrap_ci(y_prob, n_boot=args.n_bootstrap, seed=42)
        rows.append(
            {
                "model": model_name,
                "brier": brier,
                "ece": ece,
                "score_mean": mean_v,
                "score_std": std_v,
                "score_ci95_low": lo,
                "score_ci95_high": hi,
                "score_bootstrap_low": b_lo,
                "score_bootstrap_high": b_hi,
            }
        )

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "stats_summary.csv"), index=False)

    y_prob = df[score_cols[args.focus_model]].to_numpy().astype(float)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    plt.figure(figsize=(5.6, 5.6), dpi=150)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_pred, frac_pos, marker="o", label=args.focus_model)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "reliability_diagram.png"), bbox_inches="tight")

    pd.DataFrame({"mean_pred": mean_pred, "frac_pos": frac_pos}).to_csv(
        os.path.join(args.out_dir, "reliability_points.csv"), index=False
    )

    print(f"Saved stats and calibration outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
