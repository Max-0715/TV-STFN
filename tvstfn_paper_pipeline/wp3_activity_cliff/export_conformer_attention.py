import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import TetraViewDataset, tetra_view_collate
from model import TetraViewNet
from tvstfn_paper_pipeline.common.utils import ensure_dir


def move_batch_to_device(batch, device):
    return {
        "view1": {
            "coords": batch["view1"]["coords"].to(device),
            "atom_features": batch["view1"]["atom_features"].to(device),
            "num_atoms": batch["view1"]["num_atoms"].to(device),
        },
        "view2": {
            "input_ids": batch["view2"]["input_ids"].to(device),
            "attention_mask": batch["view2"]["attention_mask"].to(device),
        },
        "view3": batch["view3"].to(device),
        "view4": batch["view4"].to(device),
    }


def parse_indices(path: str):
    df = pd.read_csv(path)
    idx = []
    for c in ["idx_i", "idx_j", "__dataset_index"]:
        if c in df.columns:
            idx.extend(df[c].dropna().astype(int).tolist())
    idx = sorted({x for x in idx if x >= 0})
    return idx


def resolve_indices_from_smiles(indices_csv: str, smiles_csv: str, smiles_col: str = "SMILES"):
    cliff_df = pd.read_csv(indices_csv)
    if not os.path.exists(smiles_csv):
        return []

    src_df = pd.read_csv(smiles_csv, low_memory=False)
    if smiles_col not in src_df.columns:
        return []

    src = src_df[[smiles_col]].copy()
    src[smiles_col] = src[smiles_col].astype(str)
    src = src.reset_index().rename(columns={"index": "dataset_index"})
    smi_to_idx = dict(zip(src[smiles_col].tolist(), src["dataset_index"].astype(int).tolist()))

    out = []
    for c in ["smiles_i", "smiles_j", "SMILES"]:
        if c in cliff_df.columns:
            for smi in cliff_df[c].astype(str).tolist():
                if smi in smi_to_idx:
                    out.append(smi_to_idx[smi])
    out = sorted(set(out))
    return out


def main():
    parser = argparse.ArgumentParser(description="Export conformer attention weights for selected molecules")
    parser.add_argument("--indices-csv", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff/cliff_shortlist_top10.csv")
    parser.add_argument("--data-dir", type=str, default="tetraview_processed")
    parser.add_argument("--weights", type=str, default="best_tetraview_model.pth")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp3_cliff")
    parser.add_argument("--smiles-csv", type=str, default="CycPeptMPDB_Peptide_PAMPA.csv")
    parser.add_argument("--smiles-col", type=str, default="SMILES")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    indices = parse_indices(args.indices_csv)
    if not indices:
        indices = resolve_indices_from_smiles(args.indices_csv, args.smiles_csv, args.smiles_col)
        if indices:
            print(f"[info] fallback resolved {len(indices)} indices from SMILES mapping")
        else:
            raise RuntimeError("No valid dataset index found in indices csv and SMILES fallback failed")

    ds = TetraViewDataset(args.data_dir)
    indices = [i for i in indices if 0 <= i < len(ds)]
    if not indices:
        raise RuntimeError("Resolved indices are empty after dataset bounds check")
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=tetra_view_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TetraViewNet().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            idx = indices[i]
            out = model(move_batch_to_device(batch, device), return_dict=True)
            conf_attn = out.get("conformer_attention")
            if conf_attn is None:
                continue
            conf_attn = conf_attn.detach().cpu().numpy().reshape(-1)
            top_conf = int(np.argmax(conf_attn))
            for conf_id, w in enumerate(conf_attn.tolist()):
                rows.append(
                    {
                        "dataset_index": idx,
                        "conformer_id": conf_id,
                        "attention_weight": float(w),
                        "is_top": int(conf_id == top_conf),
                    }
                )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(args.out_dir, "conformer_attention_weights.csv"), index=False)

    top_df = out_df[out_df["is_top"] == 1].copy()
    top_df.to_csv(os.path.join(args.out_dir, "conformer_attention_top.csv"), index=False)

    print(f"Saved conformer attention exports to: {args.out_dir}")


if __name__ == "__main__":
    main()
