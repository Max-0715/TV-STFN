import argparse
import os

import numpy as np
import pandas as pd


def build_weights_from_cliff(cliff_csv: str, boost: float = 1.8):
    if not os.path.exists(cliff_csv):
        return {}
    df = pd.read_csv(cliff_csv)
    w = {}
    for c in ["idx_i", "idx_j"]:
        if c in df.columns:
            for idx in df[c].dropna().astype(int).tolist():
                if idx < 0:
                    continue
                w[idx] = max(w.get(idx, 1.0), boost)
    return w


def add_high_tpsa_weights(weights: dict, smiles_csv: str, tpsa_thr: float = 300.0, boost: float = 1.4):
    if not os.path.exists(smiles_csv):
        return weights

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    df = pd.read_csv(smiles_csv, low_memory=False)
    if "SMILES" not in df.columns:
        return weights

    for idx, smi in enumerate(df["SMILES"].astype(str).tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        tpsa = float(Descriptors.TPSA(mol))
        if tpsa > tpsa_thr:
            weights[idx] = max(weights.get(idx, 1.0), boost)
    return weights


def add_high_mw_weights(weights: dict, smiles_csv: str, mw_thr: float = 1000.0, boost: float = 1.3):
    if not os.path.exists(smiles_csv):
        return weights

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    df = pd.read_csv(smiles_csv, low_memory=False)
    if "SMILES" not in df.columns:
        return weights

    for idx, smi in enumerate(df["SMILES"].astype(str).tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mw = float(Descriptors.MolWt(mol))
        if mw > mw_thr:
            weights[idx] = max(weights.get(idx, 1.0), boost)
    return weights


def main():
    parser = argparse.ArgumentParser(description="Build hard-sample weights from WP2/WP3 outputs")
    parser.add_argument("--cliff-csv", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv")
    parser.add_argument("--smiles-csv", type=str, default="CycPeptMPDB_Peptide_PAMPA.csv")
    parser.add_argument("--out-csv", type=str, default="tvstfn_paper_pipeline/outputs/optimization/sample_weights_v1.csv")
    parser.add_argument("--cliff-boost", type=float, default=1.8)
    parser.add_argument("--tpsa-thr", type=float, default=300.0)
    parser.add_argument("--tpsa-boost", type=float, default=1.4)
    parser.add_argument("--mw-thr", type=float, default=1000.0)
    parser.add_argument("--mw-boost", type=float, default=1.3)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    weights = build_weights_from_cliff(args.cliff_csv, boost=args.cliff_boost)
    weights = add_high_tpsa_weights(weights, args.smiles_csv, tpsa_thr=args.tpsa_thr, boost=args.tpsa_boost)
    weights = add_high_mw_weights(weights, args.smiles_csv, mw_thr=args.mw_thr, boost=args.mw_boost)

    if not weights:
        print("No hard-sample weights generated; check inputs.")
        return

    out = pd.DataFrame(
        {
            "dataset_index": list(weights.keys()),
            "weight": list(weights.values()),
        }
    ).sort_values("dataset_index")

    out.to_csv(args.out_csv, index=False)
    print(f"Saved weights: {args.out_csv} | rows={len(out)} | mean={out['weight'].mean():.4f}")


if __name__ == "__main__":
    main()
