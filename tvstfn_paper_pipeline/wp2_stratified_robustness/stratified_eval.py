import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tvstfn_paper_pipeline.common.utils import (
    basic_cls_metrics,
    ensure_dir,
    find_model_reg_columns,
    find_model_score_columns,
    find_true_labels,
    load_fold_predictions,
)


def add_physchem_by_smiles(df: pd.DataFrame) -> pd.DataFrame:
    if "SMILES" not in df.columns:
        raise KeyError("Prediction files need SMILES column for MW/TPSA stratification")

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mw, tpsa = [], []
    for smi in df["SMILES"].astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mw.append(np.nan)
            tpsa.append(np.nan)
        else:
            mw.append(Descriptors.MolWt(mol))
            tpsa.append(Descriptors.TPSA(mol))
    df = df.copy()
    df["MW"] = mw
    df["TPSA"] = tpsa
    return df


def eval_by_bins(df, y_true, model_col, bin_col, bins, labels):
    rows = []
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i + 1]
        mask = (df[bin_col] >= lo) & (df[bin_col] < hi)
        part = df.loc[mask]
        if len(part) == 0:
            rows.append({"bin": label, "n": 0, "ACC": np.nan, "F1": np.nan, "MCC": np.nan, "AUROC": np.nan})
            continue
        y_t = y_true[mask.to_numpy()]
        y_s = part[model_col].to_numpy()
        m = basic_cls_metrics(y_t, y_s, threshold=0.5)
        rows.append({"bin": label, "n": len(part), **m})
    return pd.DataFrame(rows)


def plot_metric(df_plot, metric, title, out_path):
    plt.figure(figsize=(8.4, 4.8), dpi=150)
    for model_name in sorted(df_plot["model"].unique()):
        part = df_plot[df_plot["model"] == model_name]
        plt.plot(part["bin"], part[metric], marker="o", label=model_name)
    plt.title(title)
    plt.xlabel("Bin")
    plt.ylabel(metric)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Compute MW/TPSA stratified robustness metrics")
    parser.add_argument("--pred-dir", type=str, default="benchmark_results")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp2_stratified")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_fold_predictions(args.pred_dir)
    df = add_physchem_by_smiles(df)

    y_true = find_true_labels(df)

    score_cols = find_model_score_columns(df)
    reg_cols = find_model_reg_columns(df)

    if "TVSTFN" not in score_cols and "TVSTFN" in reg_cols:
        s = df[reg_cols["TVSTFN"]].to_numpy()
        s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
        df["Pred_Score_TVSTFN"] = s
        score_cols["TVSTFN"] = "Pred_Score_TVSTFN"

    mw_bins = [0, 800, 1000, np.inf]
    mw_labels = ["<800", "800-1000", ">1000"]
    tpsa_bins = [0, 200, 300, np.inf]
    tpsa_labels = ["<200", "200-300", ">300"]

    rows = []
    for model_name, score_col in score_cols.items():
        mw_df = eval_by_bins(df, y_true, score_col, "MW", mw_bins, mw_labels)
        mw_df["model"] = model_name
        mw_df["stratifier"] = "MW"
        rows.append(mw_df)

        tpsa_df = eval_by_bins(df, y_true, score_col, "TPSA", tpsa_bins, tpsa_labels)
        tpsa_df["model"] = model_name
        tpsa_df["stratifier"] = "TPSA"
        rows.append(tpsa_df)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(args.out_dir, "stratified_metrics.csv"), index=False)

    for metric in ["F1", "AUROC", "MCC"]:
        plot_metric(
            out[out["stratifier"] == "MW"],
            metric,
            f"Figure W1: MW-stratified {metric}",
            os.path.join(args.out_dir, f"figure_W1_MW_{metric}.png"),
        )
        plot_metric(
            out[out["stratifier"] == "TPSA"],
            metric,
            f"Figure W2: TPSA-stratified {metric}",
            os.path.join(args.out_dir, f"figure_W2_TPSA_{metric}.png"),
        )

    print(f"Saved stratified outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
