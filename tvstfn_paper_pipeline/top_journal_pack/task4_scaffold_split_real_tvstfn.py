#!/usr/bin/env python3
"""Real scaffold-split test for TV-STFN.

This script performs an actual train/test run of TetraViewNet on an existing
Murcko scaffold split assignment file, then compares scaffold metrics against
random-split metrics from 3.21.csv.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
from torch.utils.data import DataLoader, Subset

ROOT_DIR = "/data/workplace/jwx/TV-STFN"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataset import TetraViewDataset, tetra_view_collate
from loss import CompositeLoss
from model import TetraViewNet


def parse_metric(v: str) -> float:
    s = str(v)
    if "±" in s:
        s = s.split("±")[0].strip()
    return float(s)


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def build_rowwise_scaffold_split(df_raw: pd.DataFrame, seed: int = 42, frac_train=0.8, frac_valid=0.1) -> pd.DataFrame:
    d = df_raw[["SMILES", "Permeability"]].copy()
    d["row_idx"] = d.index.astype(int)
    d["scaffold"] = d["SMILES"].astype(str).map(murcko_scaffold)
    d = d[d["scaffold"].str.len() > 0].reset_index(drop=True)

    scaffold_to_rows: Dict[str, List[int]] = defaultdict(list)
    for _, r in d.iterrows():
        scaffold_to_rows[r["scaffold"]].append(int(r["row_idx"]))

    rng = np.random.default_rng(seed)
    groups = sorted(scaffold_to_rows.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    n = len(d)
    n_train = int(frac_train * n)
    n_valid = int(frac_valid * n)
    train_cnt = 0
    valid_cnt = 0

    split_by_row: Dict[int, str] = {}
    for _, rows in groups:
        rr = rows[:]
        rng.shuffle(rr)
        if train_cnt + len(rr) <= n_train:
            for rid in rr:
                split_by_row[rid] = "train"
            train_cnt += len(rr)
        elif valid_cnt + len(rr) <= n_valid:
            for rid in rr:
                split_by_row[rid] = "valid"
            valid_cnt += len(rr)
        else:
            for rid in rr:
                split_by_row[rid] = "test"

    d["split"] = d["row_idx"].map(split_by_row)
    return d[["row_idx", "SMILES", "Permeability", "scaffold", "split"]]


def move_batch_to_device(batch, device):
    inp = {
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
    return inp


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        targets = batch["targets"].to(device)
        inp = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        preds = model(inp)
        loss_out = criterion(preds, targets)
        loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.item())
    return total / max(len(loader), 1)


def evaluate(model, loader, device, threshold=-6.0) -> Dict[str, float]:
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            inp = move_batch_to_device(batch, device)
            preds = model(inp)
            preds_all.append(preds.cpu().numpy().reshape(-1))
            y_all.append(targets.cpu().numpy().reshape(-1))

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(y_all)

    if not np.isfinite(y_pred).all():
        finite = y_pred[np.isfinite(y_pred)]
        fill = float(np.median(finite)) if finite.size else -7.0
        y_pred = np.nan_to_num(y_pred, nan=fill, posinf=fill, neginf=fill)

    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    score = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10)
    try:
        auroc = roc_auc_score(y_true_bin, score)
    except Exception:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true_bin, score)
    except Exception:
        auprc = float("nan")

    out = {
        "ACC": float(accuracy_score(y_true_bin, y_pred_bin)),
        "F1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "n_test": int(len(y_true)),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
    }
    return out


def optimize_threshold(model, loader, device, grid=None) -> float:
    if grid is None:
        grid = np.linspace(-8.5, -4.5, 81)
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            inp = move_batch_to_device(batch, device)
            preds = model(inp)
            preds_all.append(preds.cpu().numpy().reshape(-1))
            y_all.append(targets.cpu().numpy().reshape(-1))
    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(y_all)

    finite = y_pred[np.isfinite(y_pred)]
    fill = float(np.median(finite)) if finite.size else -7.0
    y_pred = np.nan_to_num(y_pred, nan=fill, posinf=fill, neginf=fill)

    best_t = -6.0
    best_f1 = -1.0
    for t in grid:
        y_true_bin = (y_true >= -6.0).astype(int)
        y_pred_bin = (y_pred >= t).astype(int)
        f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/data/workplace/jwx/TV-STFN/CycPeptMPDB_Peptide_PAMPA.csv")
    parser.add_argument("--assign-csv", default="")
    parser.add_argument("--random-metrics", default="/data/workplace/jwx/结果/3。21.csv")
    parser.add_argument("--data-dir", default="/data/workplace/jwx/TV-STFN/tetraview_processed")
    parser.add_argument(
        "--out-dir",
        default="/data/workplace/jwx/TV-STFN/tvstfn_paper_pipeline/outputs/top_journal_pack/task4_scaffold_split",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "composite"], default="mse")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build row-wise scaffold split mapping.
    df_raw = pd.read_csv(args.csv)
    df_raw = df_raw[df_raw["Permeability"].notna()].copy()
    if args.assign_csv and os.path.exists(args.assign_csv):
        assign_df = pd.read_csv(args.assign_csv)
        if "row_idx" not in assign_df.columns:
            assign_df = build_rowwise_scaffold_split(df_raw, seed=args.seed)
    else:
        assign_df = build_rowwise_scaffold_split(df_raw, seed=args.seed)
    assign_df.to_csv(os.path.join(args.out_dir, "scaffold_split_assignments_real_rowidx.csv"), index=False)
    row_to_split = {int(r["row_idx"]): str(r["split"]) for _, r in assign_df.iterrows()}

    # Dataset order is lexicographic by file name in dataset.py; recover original row ids from filenames.
    dataset = TetraViewDataset(args.data_dir)
    pos_to_rowidx: List[int] = []
    for fn in dataset.file_names:
        # Expected format: data_<rowidx>.pt
        base = os.path.splitext(fn)[0]
        row_idx = int(base.split("_")[-1])
        pos_to_rowidx.append(row_idx)

    train_pos, valid_pos, test_pos = [], [], []
    unmatched = 0
    for pos, row_idx in enumerate(pos_to_rowidx):
        if row_idx not in df_raw.index:
            continue
        r = df_raw.loc[row_idx]
        sp = row_to_split.get(int(row_idx))
        if sp is None:
            unmatched += 1
            continue
        if sp == "train":
            train_pos.append(pos)
        elif sp == "valid":
            valid_pos.append(pos)
        elif sp == "test":
            test_pos.append(pos)

    if len(train_pos) == 0 or len(valid_pos) == 0 or len(test_pos) == 0:
        raise RuntimeError("No valid train/test positions mapped. Check split assignment compatibility.")

    train_loader = DataLoader(
        Subset(dataset, train_pos),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tetra_view_collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(dataset, test_pos),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tetra_view_collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_pos),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tetra_view_collate,
        num_workers=0,
    )

    model = TetraViewNet().to(device)
    encoder_3d_params = list(map(id, model.encoder_3d.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_3d_params, model.parameters())
    optimizer = optim.AdamW(
        [
            {"params": base_params},
            {"params": model.encoder_3d.parameters(), "lr": args.lr * 0.1},
        ],
        lr=args.lr,
        weight_decay=1e-4,
    )
    if args.loss == "composite":
        criterion = CompositeLoss(lambda_focal=1.0, lambda_rank=1.0, lambda_mse=1.0)
    else:
        criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    best_state = None
    best_rmse = float("inf")
    bad_epochs = 0
    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, valid_loader, device, threshold=-6.0)
        val_rmse = float(np.sqrt(np.mean((val_metrics["y_pred_mean"] - val_metrics["y_true_mean"]) ** 2)))
        scheduler.step(train_loss)

        if train_loss < best_rmse:
            best_rmse = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{args.epochs} | train_loss={train_loss:.4f} | valid_acc={val_metrics['ACC']:.4f}")
        if bad_epochs >= args.patience:
            print(f"Early stop at epoch {ep+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    m_fixed = evaluate(model, test_loader, device, threshold=-6.0)
    best_thr = optimize_threshold(model, valid_loader, device)
    m_tuned = evaluate(model, test_loader, device, threshold=best_thr)

    m = {f"fixed_{k}": v for k, v in m_fixed.items()}
    m.update({f"tuned_{k}": v for k, v in m_tuned.items()})
    m["best_threshold_from_valid"] = float(best_thr)
    m["n_train"] = len(train_pos)
    m["n_valid"] = len(valid_pos)
    m["n_test"] = len(test_pos)
    m["unmatched_positions"] = unmatched
    real_path = os.path.join(args.out_dir, "tvstfn_scaffold_real_metrics.csv")
    pd.DataFrame([m]).to_csv(real_path, index=False)

    # Merge with random split metrics from 3.21.csv for TV-STFN only.
    rand_df = pd.read_csv(args.random_metrics)
    tv_rand = rand_df[rand_df["Model"] == "TV-STFN"].iloc[0]
    random_auroc = parse_metric(tv_rand["AUROC"])
    random_f1 = parse_metric(tv_rand["F1"])

    out = pd.DataFrame(
        [
            {
                "Model": "TV-STFN",
                "AUROC_random": random_auroc,
                "AUROC_scaffold": m["fixed_AUROC"],
                "F1_random": random_f1,
                "F1_scaffold": m["fixed_F1"],
            }
        ]
    )
    out["AUROC_drop"] = out["AUROC_random"] - out["AUROC_scaffold"]
    out["F1_drop"] = out["F1_random"] - out["F1_scaffold"]
    out["Mean_drop"] = 0.5 * (out["AUROC_drop"] + out["F1_drop"])
    out_path = os.path.join(args.out_dir, "scaffold_random_drop_metrics_real_tvstfn.csv")
    out.to_csv(out_path, index=False)

    print(f"Saved real metrics: {real_path}")
    print(f"Saved drop compare: {out_path}")


if __name__ == "__main__":
    main()
