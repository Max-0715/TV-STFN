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
from tvstfn_paper_pipeline.common.utils import discover_fold_prediction_files, ensure_dir


def sanitize_0d_features(x, clip_q=0.001):
    x = np.asarray(x, dtype=np.float64)
    # Replace non-finite values first.
    finite_mask = np.isfinite(x)
    if not finite_mask.all():
        col_med = np.nanmedian(np.where(finite_mask, x, np.nan), axis=0)
        rows, cols = np.where(~finite_mask)
        x[rows, cols] = col_med[cols]

    # Column-wise robust clipping to suppress extreme outliers.
    low = np.quantile(x, clip_q, axis=0)
    high = np.quantile(x, 1.0 - clip_q, axis=0)
    x = np.clip(x, low, high)

    # Robust scaling (median / IQR) for stable distance-based metrics.
    med = np.median(x, axis=0)
    q1 = np.quantile(x, 0.25, axis=0)
    q3 = np.quantile(x, 0.75, axis=0)
    iqr = q3 - q1
    iqr[iqr < 1e-8] = 1.0
    x = (x - med) / iqr
    # Final guardrail for distance-based visualization stability.
    x = np.clip(x, -20.0, 20.0)
    return x.astype(np.float32)


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


def collect_test_indices(pred_dir):
    idx = []
    for fp in discover_fold_prediction_files(pred_dir):
        df = pd.read_csv(fp)
        if "__dataset_index" in df.columns:
            idx.extend(df["__dataset_index"].astype(int).tolist())
    idx = sorted(set(idx))
    return idx


def main():
    parser = argparse.ArgumentParser(description="Export TV-STFN fusion embeddings and 0D vectors for UMAP")
    parser.add_argument("--data-dir", type=str, default="tetraview_processed")
    parser.add_argument("--weights", type=str, default="best_tetraview_model.pth")
    parser.add_argument("--pred-dir", type=str, default="benchmark_results")
    parser.add_argument("--out-dir", type=str, default="tvstfn_paper_pipeline/outputs/wp1_umap")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=2500, help="cap samples for fast embedding export; 0 means no cap")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    ds = TetraViewDataset(args.data_dir)
    test_indices = collect_test_indices(args.pred_dir)
    if not test_indices:
        # Fallback to the full dataset ordering when fold CSVs do not carry dataset index.
        test_indices = list(range(len(ds)))
        print("[warn] __dataset_index not found; fallback to full dataset index order.")

    if args.max_samples > 0 and len(test_indices) > args.max_samples:
        test_indices = test_indices[: args.max_samples]
        print(f"[info] capped export samples to {len(test_indices)}")

    subset = Subset(ds, test_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, collate_fn=tetra_view_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TetraViewNet().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    all_shared, all_view4, all_attn, all_conf_attn, all_targets, all_preds, all_cls = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].cpu().numpy().reshape(-1)
            batch_in = move_batch_to_device(batch, device)
            out = model(batch_in, return_dict=True)

            shared = out["shared_features"].detach().cpu().numpy()
            attn = out["attention_weights"].detach().cpu().numpy()
            pred = out["regression"].detach().cpu().numpy().reshape(-1)
            cls = torch.sigmoid(out["classification"]).detach().cpu().numpy().reshape(-1)
            conf_attn = out.get("conformer_attention", None)
            if conf_attn is not None:
                conf_attn = conf_attn.detach().cpu().numpy()
            else:
                conf_attn = np.zeros((shared.shape[0], 1), dtype=np.float32)

            all_shared.append(shared)
            all_view4.append(batch["view4"].cpu().numpy())
            all_attn.append(attn)
            all_conf_attn.append(conf_attn)
            all_targets.append(targets)
            all_preds.append(pred)
            all_cls.append(cls)

    shared = np.concatenate(all_shared, axis=0)
    view4 = np.concatenate(all_view4, axis=0)
    view4 = sanitize_0d_features(view4)
    view_attn = np.concatenate(all_attn, axis=0)
    conf_attn = np.concatenate(all_conf_attn, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_score = np.concatenate(all_cls, axis=0)

    np.savez_compressed(
        os.path.join(args.out_dir, "umap_embeddings.npz"),
        dataset_index=np.asarray(test_indices, dtype=np.int64),
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        feat_fusion=shared,
        feat_0d=view4,
        view_attention=view_attn,
        conformer_attention=conf_attn,
    )

    pd.DataFrame(
        {
            "dataset_index": np.asarray(test_indices, dtype=np.int64),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
            "attn_3d": view_attn[:, 0],
            "attn_1d": view_attn[:, 1],
            "attn_2d": view_attn[:, 2],
            "attn_0d": view_attn[:, 3],
        }
    ).to_csv(os.path.join(args.out_dir, "embedding_meta.csv"), index=False)

    print(f"Saved embeddings to: {args.out_dir}")


if __name__ == "__main__":
    main()
