import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import TetraViewDataset, tetra_view_collate
from loss import CompositeLoss
from model import TetraViewNet
from tvstfn_paper_pipeline.common.utils import ensure_dir

CLS_THRESHOLD = -6.0


class ZeroEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x, return_attention=False, **kwargs):
        if isinstance(x, dict) and "coords" in x:
            b = x["coords"].shape[0]
            device = x["coords"].device
            num_confs = x["coords"].shape[1] if x["coords"].dim() >= 2 else 1
        elif isinstance(x, dict) and "input_ids" in x:
            b = x["input_ids"].shape[0]
            device = x["input_ids"].device
            num_confs = 1
        elif hasattr(x, "batch"):
            b = int(x.batch.max().item()) + 1 if x.batch.numel() > 0 else 1
            device = x.x.device
            num_confs = 1
        else:
            b = x.shape[0]
            device = x.device
            num_confs = 1

        feat = torch.zeros((b, self.out_dim), device=device)
        if return_attention:
            # Keep interface consistent with DynamicConformerEncoder.forward(return_attention=True)
            attn = torch.zeros((b, num_confs), device=device)
            return feat, attn
        return feat


class ZeroExpert(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        return torch.zeros((x.shape[0], self.out_dim), device=x.device)


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


def build_model(ablation, device):
    model = TetraViewNet().to(device)
    if ablation == "wo_3d":
        model.encoder_3d = ZeroEncoder(256).to(device)
    elif ablation == "wo_1d":
        model.encoder_1d = ZeroEncoder(256).to(device)
    elif ablation == "wo_2d":
        model.encoder_2d = ZeroEncoder(256).to(device)
    elif ablation == "wo_0d":
        model.encoder_0d = ZeroEncoder(256).to(device)
        model.raw_0d_expert = ZeroExpert(512).to(device)
    return model


def build_binary_labels(ds, threshold=CLS_THRESHOLD):
    valid_indices = []
    y_cont = []
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        valid_indices.append(i)
        y_cont.append(float(item["target"]))
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    y_cont = np.asarray(y_cont, dtype=np.float32)
    y_bin = (y_cont >= threshold).astype(np.int64)
    return valid_indices, y_cont, y_bin


def make_weighted_sampler(labels):
    labels = np.asarray(labels, dtype=np.int64)
    cls_cnt = np.bincount(labels, minlength=2).astype(np.float64)
    cls_cnt[cls_cnt == 0] = 1.0
    cls_w = 1.0 / cls_cnt
    sample_w = cls_w[labels]
    sample_w = torch.from_numpy(sample_w).double()
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def find_best_threshold(y_true, y_score):
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.2, 0.8, 61):
        yp = (y_score >= t).astype(int)
        f1 = f1_score(y_true, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def evaluate(model, loader, device, threshold=0.5, return_raw=False):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(move_batch_to_device(batch, device), return_dict=True)
            score = torch.sigmoid(out["classification"]).detach().cpu().numpy().reshape(-1)
            target = batch["targets"].cpu().numpy().reshape(-1)
            y_true.append((target >= CLS_THRESHOLD).astype(int))
            y_score.append(score)
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        # Single-class folds can make AUROC undefined; use neutral baseline.
        auc = 0.5
    metrics = {"ACC": acc, "F1": f1, "AUROC": auc}
    if return_raw:
        return metrics, y_true, y_score
    return metrics


def train_fold(
    model,
    train_loader,
    val_loader,
    device,
    epochs=40,
    lr=2e-4,
    early_stop=10,
    lambda_cls=0.3,
    lambda_entropy=0.02,
):
    criterion = CompositeLoss(lambda_focal=0.3, lambda_rank=0.5, lambda_mse=1.0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best = -1.0
    bad = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            target = batch["targets"].to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out = model(move_batch_to_device(batch, device), return_dict=True)
                reg_loss, _, _, _ = criterion(
                    out["regression"],
                    target,
                    classification_threshold=CLS_THRESHOLD,
                )
                cls_target = (target >= CLS_THRESHOLD).float()
                cls_loss = F.binary_cross_entropy_with_logits(out["classification"], cls_target)
                entropy_term = out.get("modality_entropy", torch.tensor(0.0, device=device))
                # Maximize gate entropy to avoid single-modality collapse.
                loss = reg_loss + lambda_cls * cls_loss - lambda_entropy * entropy_term
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

        m = evaluate(model, val_loader, device)
        key = 0.6 * m["AUROC"] + 0.4 * m["F1"]
        scheduler.step(key)
        if np.isfinite(key) and key > best:
            best = key
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)


def main():
    parser = argparse.ArgumentParser(description="Run TV-STFN ablation CV for wo_0d/1d/2d/3d")
    parser.add_argument("--data-dir", type=str, default="tetraview_processed")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp4_ablation")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cls", type=float, default=0.3)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", type=str, default="full,wo_0d,wo_1d,wo_2d,wo_3d", help="comma-separated variants")
    parser.add_argument("--tag", type=str, default="", help="optional suffix for output files")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = TetraViewDataset(args.data_dir)
    indices, _, y_bin_all = build_binary_labels(ds, threshold=CLS_THRESHOLD)
    kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    valid = {"full", "wo_0d", "wo_1d", "wo_2d", "wo_3d"}
    for v in variants:
        if v not in valid:
            raise ValueError(f"Unknown variant: {v}")

    rows = []
    for variant in variants:
        print(f"Running variant: {variant}")
        for fold, (train_pos, test_pos) in enumerate(kf.split(indices, y_bin_all)):
            train_idx_full = indices[train_pos]
            test_idx = indices[test_pos]
            y_train_full = y_bin_all[train_pos]

            tr_idx, val_idx = train_test_split(
                train_idx_full,
                test_size=0.1,
                random_state=args.seed + fold,
                stratify=y_train_full,
            )

            y_tr = (np.array([float(ds[i]["target"]) for i in tr_idx]) >= CLS_THRESHOLD).astype(np.int64)
            sampler = make_weighted_sampler(y_tr)

            train_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, sampler=sampler, collate_fn=tetra_view_collate)
            val_loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=tetra_view_collate)
            test_loader = DataLoader(Subset(ds, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=tetra_view_collate)

            model = build_model(variant, device)
            train_fold(
                model,
                train_loader,
                val_loader,
                device,
                epochs=args.epochs,
                lr=args.lr,
                lambda_cls=args.lambda_cls,
                lambda_entropy=args.lambda_entropy,
            )

            _, yv_true, yv_score = evaluate(model, val_loader, device, return_raw=True)
            best_t, _ = find_best_threshold(yv_true, yv_score)
            m = evaluate(model, test_loader, device, threshold=best_t)
            rows.append({"variant": variant, "fold": fold, "threshold": best_t, **m})
            print(
                f"  fold {fold}: thr={best_t:.2f}, ACC={m['ACC']:.4f}, F1={m['F1']:.4f}, AUROC={m['AUROC']:.4f}"
            )

    suffix = f"_{args.tag}" if args.tag else ""
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(args.out_dir, f"ablation_per_fold{suffix}.csv"), index=False)

    summary = per_fold.groupby("variant")[["ACC", "F1", "AUROC"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.out_dir, f"ablation_summary{suffix}.csv"), index=False)

    print(f"Saved ablation results to: {args.out_dir}")


if __name__ == "__main__":
    main()
