import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build Figure Z panel table from cliff and attention outputs")
    parser.add_argument("--cliff-csv", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv")
    parser.add_argument("--attn-csv", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff/conformer_attention_top.csv")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cliff = pd.read_csv(args.cliff_csv)
    attn = pd.read_csv(args.attn_csv)

    top_by_idx = attn[["dataset_index", "conformer_id", "attention_weight"]].rename(
        columns={"dataset_index": "idx_i", "conformer_id": "top_conf_i", "attention_weight": "top_conf_weight_i"}
    )
    out = cliff.merge(top_by_idx, on="idx_i", how="left")

    top_by_idx_j = attn[["dataset_index", "conformer_id", "attention_weight"]].rename(
        columns={"dataset_index": "idx_j", "conformer_id": "top_conf_j", "attention_weight": "top_conf_weight_j"}
    )
    out = out.merge(top_by_idx_j, on="idx_j", how="left")

    out.to_csv(os.path.join(args.out_dir, "figure_Z_panel_data.csv"), index=False)
    print(f"Saved Figure Z panel data to: {args.out_dir}")


if __name__ == "__main__":
    main()
