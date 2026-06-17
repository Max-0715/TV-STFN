import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tvstfn_paper_pipeline.common.utils import ensure_dir, load_fold_predictions


def resolve_dataset_index(df: pd.DataFrame, smiles_csv: str, smiles_col: str) -> pd.Series:
    if "__dataset_index" in df.columns:
        s = pd.to_numeric(df["__dataset_index"], errors="coerce")
        if s.notna().any():
            return s.astype("Int64")

    if not os.path.exists(smiles_csv):
        return pd.Series([-1] * len(df), index=df.index, dtype="int64")

    src_df = pd.read_csv(smiles_csv, low_memory=False)
    if smiles_col not in src_df.columns:
        return pd.Series([-1] * len(df), index=df.index, dtype="int64")

    src = src_df[[smiles_col]].copy().reset_index().rename(columns={"index": "dataset_index"})
    src[smiles_col] = src[smiles_col].astype(str)
    # Use first index for duplicated smiles in source CSV to keep mapping deterministic.
    src = src.drop_duplicates(subset=[smiles_col], keep="first")

    mapped = df["SMILES"].astype(str).map(dict(zip(src[smiles_col], src["dataset_index"])))
    mapped = mapped.fillna(-1).astype("int64")
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Find activity cliff candidate pairs from fold predictions")
    parser.add_argument("--pred-dir", type=str, default="benchmark_results")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff")
    parser.add_argument("--sim-thr", type=float, default=0.8)
    parser.add_argument("--delta-thr", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--smiles-csv", type=str, default="CycPeptMPDB_Peptide_PAMPA.csv")
    parser.add_argument("--smiles-col", type=str, default="SMILES")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = load_fold_predictions(args.pred_dir)
    if "SMILES" not in df.columns or "True_LogP" not in df.columns:
        raise KeyError("Need SMILES and True_LogP in fold prediction files")

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    keep_cols = [
        c
        for c in df.columns
        if c in ["SMILES", "True_LogP", "Pred_LogP_XGB", "Pred_LogP_MSF", "Pred_LogP_TVSTFN", "__dataset_index"]
    ]
    work = df[keep_cols].copy()
    work["__dataset_index"] = resolve_dataset_index(work, args.smiles_csv, args.smiles_col)
    work = work.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    mols = [Chem.MolFromSmiles(s) for s in work["SMILES"].astype(str)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None for m in mols]

    rows = []
    n = len(work)
    for i in range(n):
        if fps[i] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        for j_off, sim in enumerate(sims):
            if sim < args.sim_thr:
                continue
            j = i + 1 + j_off
            delta = abs(float(work.loc[i, "True_LogP"]) - float(work.loc[j, "True_LogP"]))
            if delta < args.delta_thr:
                continue
            row = {
                "i": i,
                "j": j,
                "smiles_i": work.loc[i, "SMILES"],
                "smiles_j": work.loc[j, "SMILES"],
                "sim": float(sim),
                "delta_true": float(delta),
                "score": float(sim * delta),
                "true_i": float(work.loc[i, "True_LogP"]),
                "true_j": float(work.loc[j, "True_LogP"]),
                "idx_i": int(work.loc[i, "__dataset_index"]) if "__dataset_index" in work.columns else -1,
                "idx_j": int(work.loc[j, "__dataset_index"]) if "__dataset_index" in work.columns else -1,
            }
            for c in ["Pred_LogP_XGB", "Pred_LogP_MSF", "Pred_LogP_TVSTFN"]:
                if c in work.columns:
                    row[c + "_i"] = float(work.loc[i, c])
                    row[c + "_j"] = float(work.loc[j, c])
                    row[c + "_delta"] = abs(float(work.loc[i, c]) - float(work.loc[j, c]))
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No cliff pairs found with current thresholds")
        return

    out = out.sort_values(["score", "sim", "delta_true"], ascending=False).head(args.topk)
    out.to_csv(os.path.join(args.out_dir, "cliff_candidates_topk.csv"), index=False)

    shortlist = out.head(10)
    shortlist.to_csv(os.path.join(args.out_dir, "cliff_shortlist_top10.csv"), index=False)
    print(f"Saved cliff candidates to: {args.out_dir}")


if __name__ == "__main__":
    main()
