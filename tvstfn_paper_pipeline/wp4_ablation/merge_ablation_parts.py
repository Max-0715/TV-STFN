import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge ablation shard outputs")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp4_ablation")
    parser.add_argument("--tag-prefix", type=str, default="", help="Only merge files containing this tag prefix")
    args = parser.parse_args()

    pattern = "ablation_per_fold_*.csv"
    per_fold_files = sorted(glob.glob(os.path.join(args.out_dir, pattern)))
    if args.tag_prefix:
        needle = f"ablation_per_fold_{args.tag_prefix}_"
        per_fold_files = [p for p in per_fold_files if needle in os.path.basename(p)]
    if not per_fold_files:
        raise FileNotFoundError("No ablation_per_fold_*.csv found")

    all_df = pd.concat([pd.read_csv(p) for p in per_fold_files], ignore_index=True)
    all_df = all_df.sort_values(["variant", "fold"]).reset_index(drop=True)
    if args.tag_prefix:
        per_fold_out = os.path.join(args.out_dir, f"ablation_per_fold_{args.tag_prefix}_merged.csv")
        summary_out = os.path.join(args.out_dir, f"ablation_summary_{args.tag_prefix}_merged.csv")
    else:
        per_fold_out = os.path.join(args.out_dir, "ablation_per_fold.csv")
        summary_out = os.path.join(args.out_dir, "ablation_summary.csv")
    all_df.to_csv(per_fold_out, index=False)

    summary = all_df.groupby("variant")[["ACC", "F1", "AUROC"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(summary_out, index=False)

    print("Merged shard outputs:")
    print(f"  per-fold: {per_fold_out}")
    print(f"  summary : {summary_out}")


if __name__ == "__main__":
    main()
