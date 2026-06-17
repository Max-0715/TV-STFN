import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot ablation summary chart")
    parser.add_argument("--summary-csv", type=str, default="tvstfn_paper_pipeline/outputs/wp4_ablation/ablation_summary.csv")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp4_ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.summary_csv)

    metrics = ["ACC", "F1", "AUROC"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, m in zip(axes, metrics):
        ax.bar(df["variant"], df[f"{m}_mean"], yerr=df.get(f"{m}_std", None), capsize=3)
        ax.set_title(m)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "figure_X_ablation.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved ablation figure: {out_path}")


if __name__ == "__main__":
    main()
