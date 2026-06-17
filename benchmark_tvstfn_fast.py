"""TV-STFN benchmark with configurable CV, hyperparams, and threshold tuning."""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score, cohen_kappa_score
)
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TetraViewDataset, tetra_view_collate
from model import TetraViewNet
from loss import CompositeLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description="TV-STFN fast CV benchmark")
    parser.add_argument('--n-folds', type=int, default=10)
    parser.add_argument('--folds', type=str, default='')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--early-stop', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tetraview_processed'))
    parser.add_argument('--out-dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results'))
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--val-ratio', type=float, default=0.10)
    parser.add_argument('--cls-threshold', type=float, default=-6.0)
    parser.add_argument('--tune-cls-threshold', action='store_true')
    parser.add_argument('--tune-metric', type=str, choices=['acc', 'mcc', 'f1'], default='acc')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--skip-fold-predictions', action='store_true')
    parser.add_argument('--lambda-focal', type=float, default=1.0)
    parser.add_argument('--lambda-rank', type=float, default=1.0)
    parser.add_argument('--lambda-mse', type=float, default=1.0)
    parser.add_argument('--model-hidden-dim', type=int, default=512)
    parser.add_argument('--model-dropout', type=float, default=0.10)
    parser.add_argument('--zero-d-prior', type=float, default=1.2)
    parser.add_argument('--cls-skip-weight', type=float, default=0.30)
    parser.add_argument('--gate-temperature', type=float, default=0.85)
    parser.add_argument('--modality-dropout', type=float, default=0.15)
    return parser.parse_args()


def parse_fold_spec(spec, n_folds):
    if not spec:
        return list(range(n_folds))
    picked = set()
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            left, right = chunk.split('-', 1)
            start = int(left)
            end = int(right)
            for i in range(start, end + 1):
                picked.add(i)
        else:
            picked.add(int(chunk))
    return sorted([i for i in picked if 0 <= i < n_folds])

def move_batch_to_device(batch, device):
    """Move all tensors in batch dict to device."""
    inp = {}
    inp['view1'] = {
        'coords': batch['view1']['coords'].to(device),
        'atom_features': batch['view1']['atom_features'].to(device),
        'num_atoms': batch['view1']['num_atoms'].to(device)
    }
    inp['view2'] = {
        'input_ids': batch['view2']['input_ids'].to(device),
        'attention_mask': batch['view2']['attention_mask'].to(device)
    }
    inp['view3'] = batch['view3'].to(device)
    inp['view4'] = batch['view4'].to(device)
    return inp

def train_one_epoch(model, loader, criterion, optimizer, device, cls_threshold):
    model.train()
    total_loss = 0
    steps = 0
    for batch in loader:
        targets = batch['targets'].to(device)
        inp = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        preds = model(inp)
        if not torch.isfinite(preds).all():
            continue
        loss, _, _, _ = criterion(preds, targets, classification_threshold=cls_threshold)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1), steps

def predict(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            targets = batch['targets'].to(device)
            inp = move_batch_to_device(batch, device)
            preds = model(inp)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    if not np.isfinite(all_preds).all():
        all_preds = np.nan_to_num(all_preds, nan=0.0, posinf=0.0, neginf=0.0)
    return all_preds, all_targets

def compute_metrics(y_true, y_pred, cls_scores, cls_threshold, score_threshold, scores_are_prob):
    """Compute regression + classification metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-10))) * 100

    # Classification
    y_true_bin = (y_true >= cls_threshold).astype(int)
    y_pred_bin = (cls_scores >= score_threshold).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
    kappa = cohen_kappa_score(y_true_bin, y_pred_bin)

    # TNR
    tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC
    try:
        if scores_are_prob:
            preds_prob = cls_scores
        else:
            preds_prob = (cls_scores - cls_scores.min()) / (cls_scores.max() - cls_scores.min() + 1e-10)
        auroc = roc_auc_score(y_true_bin, preds_prob)
        auprc = average_precision_score(y_true_bin, preds_prob)
    except Exception:
        auroc, auprc = 0.0, 0.0

    return {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape,
        'Pearson': pr, 'Spearman': sr,
        'ACC': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
        'MCC': mcc, 'Kappa': kappa, 'TNR': tnr, 'AUROC': auroc, 'AUPRC': auprc
    }


def tune_threshold(y_true_bin, scores, metric):
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return 0.0
    best_thr = thresholds[0]
    best_val = -1.0
    for thr in thresholds:
        pred_bin = (scores >= thr).astype(int)
        if metric == 'mcc':
            val = matthews_corrcoef(y_true_bin, pred_bin)
        elif metric == 'f1':
            val = f1_score(y_true_bin, pred_bin, zero_division=0)
        else:
            val = accuracy_score(y_true_bin, pred_bin)
        if val > best_val:
            best_val = val
            best_thr = thr
    return float(best_thr)


def fit_calibrator(y_true_bin, scores):
    model = LogisticRegression(max_iter=2000)
    model.fit(scores.reshape(-1, 1), y_true_bin)
    return model

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    folds = parse_fold_spec(args.folds, args.n_folds)
    if not folds:
        raise SystemExit('No folds selected. Check --folds setting.')

    print(f"Device: {DEVICE}")
    print(f"Data: {args.data_dir}")
    print(f"Folds: {args.n_folds} (run={folds}) | Epochs: {args.epochs} | BS: {args.batch_size}")
    print("=" * 80)

    dataset = TetraViewDataset(args.data_dir)
    n = len(dataset)
    print(f"Total samples: {n}")

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CycPeptMPDB_Peptide_PAMPA.csv")
    df_orig = pd.read_csv(csv_path, low_memory=False)
    smiles_list = df_orig['SMILES'].values
    target_list = df_orig['Permeability'].values

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    indices = np.arange(n)

    all_fold_metrics = []
    fold_ids_ran = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        if fold_i not in folds:
            continue
        fold_start = time.time()
        print(f"\n--- Fold {fold_i}/{args.n_folds-1} ({len(train_idx)} train, {len(test_idx)} test) ---")

        if args.val_ratio > 0:
            labels = (target_list[train_idx] >= args.cls_threshold).astype(int)
            stratify = labels if len(np.unique(labels)) > 1 else None
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=args.val_ratio,
                random_state=args.seed + fold_i,
                stratify=stratify,
            )
        else:
            val_idx = np.array([], dtype=int)

        train_ds = Subset(dataset, train_idx.tolist())
        test_ds = Subset(dataset, test_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist()) if len(val_idx) else None

        loader_kwargs = {
            'batch_size': args.batch_size,
            'collate_fn': tetra_view_collate,
            'num_workers': args.num_workers,
            'drop_last': False,
            'pin_memory': DEVICE.type == 'cuda',
        }
        if args.num_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 2

        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds else None

        model = TetraViewNet(
            fusion_hidden_dim=args.model_hidden_dim,
            dropout=args.model_dropout,
            modality_prior_0d=args.zero_d_prior,
            cls_skip_weight=args.cls_skip_weight,
            gating_temperature=args.gate_temperature,
            modality_dropout_prob=args.modality_dropout,
        ).to(DEVICE)

        encoder_3d_params = list(map(id, model.encoder_3d.parameters()))
        base_params = filter(lambda p: id(p) not in encoder_3d_params, model.parameters())
        optimizer = optim.AdamW([
            {'params': base_params},
            {'params': model.encoder_3d.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = CompositeLoss(lambda_focal=args.lambda_focal, lambda_rank=args.lambda_rank, lambda_mse=args.lambda_mse)

        best_loss = float('inf')
        patience_cnt = 0
        best_state = None

        for epoch in range(args.epochs):
            loss, steps = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args.cls_threshold)
            if steps == 0:
                print("  Warning: all training batches skipped due to non-finite outputs")
                patience_cnt = args.early_stop
                break
            scheduler.step(loss)

            if loss < best_loss:
                best_loss = loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= args.early_stop:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(DEVICE)

        preds_test, targets_test = predict(model, test_loader, DEVICE)
        if np.std(preds_test) < 1e-8:
            print("  Warning: degenerate predictions detected (std ~ 0)")
        cls_scores_test = preds_test
        score_threshold = args.cls_threshold
        scores_are_prob = False

        if val_loader is not None:
            preds_val, targets_val = predict(model, val_loader, DEVICE)
            y_val_bin = (targets_val >= args.cls_threshold).astype(int)
            cls_scores_val = preds_val

            if args.calibrate:
                try:
                    calibrator = fit_calibrator(y_val_bin, cls_scores_val)
                    cls_scores_val = calibrator.predict_proba(cls_scores_val.reshape(-1, 1))[:, 1]
                    cls_scores_test = calibrator.predict_proba(cls_scores_test.reshape(-1, 1))[:, 1]
                    scores_are_prob = True
                except Exception:
                    scores_are_prob = False

            if args.tune_cls_threshold:
                score_threshold = tune_threshold(y_val_bin, cls_scores_val, args.tune_metric)
            elif scores_are_prob:
                score_threshold = 0.5

        metrics = compute_metrics(
            targets_test,
            preds_test,
            cls_scores_test,
            args.cls_threshold,
            score_threshold,
            scores_are_prob,
        )
        metrics['ScoreThreshold'] = score_threshold
        all_fold_metrics.append(metrics)
        fold_ids_ran.append(fold_i)

        print(
            f"  Fold {fold_i} | RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f} "
            f"Spearman={metrics['Spearman']:.4f} ACC={metrics['ACC']:.4f} "
            f"AUROC={metrics['AUROC']:.4f} | {time.time()-fold_start:.0f}s"
        )

        if not args.skip_fold_predictions:
            existing_fold_csv = os.path.join(args.out_dir, f"fold_{fold_i}_predictions.csv")
            if scores_are_prob:
                pred_scores = cls_scores_test
            else:
                pred_scores = (preds_test - preds_test.min()) / (preds_test.max() - preds_test.min() + 1e-10)

            if os.path.exists(existing_fold_csv):
                df_existing = pd.read_csv(existing_fold_csv)
                df_existing['Pred_LogP_TVSTFN'] = preds_test
                df_existing['Pred_Score_TVSTFN'] = pred_scores
                df_existing.to_csv(existing_fold_csv, index=False)
                print(f"  -> Merged TV-STFN into {existing_fold_csv}")
            else:
                fold_smiles = smiles_list[test_idx] if len(smiles_list) >= n else [f"sample_{i}" for i in test_idx]
                true_labels = (targets_test >= args.cls_threshold).astype(float)
                df_fold = pd.DataFrame({
                    'SMILES': fold_smiles,
                    'True_LogP': targets_test,
                    'True_Label': true_labels,
                    'Pred_LogP_TVSTFN': preds_test,
                    'Pred_Score_TVSTFN': pred_scores,
                })
                df_fold.to_csv(existing_fold_csv, index=False)
                print(f"  -> Saved {existing_fold_csv}")

        del model, optimizer, scheduler, criterion, best_state
        torch.cuda.empty_cache()

    if not all_fold_metrics:
        raise SystemExit('No folds were executed. Nothing to summarize.')

    print("\n" + "=" * 100)
    print("TV-STFN Cross-Validation Benchmark Results")
    print("=" * 100)

    metric_names = list(all_fold_metrics[0].keys())
    summary = {}
    for m in metric_names:
        vals = [f[m] for f in all_fold_metrics]
        mean_v = np.mean(vals)
        std_v = np.std(vals)
        summary[m] = (mean_v, std_v)
        print(f"  {m:<14s}: {mean_v:.4f} ± {std_v:.4f}")

    summary_df = pd.DataFrame({
        'Metric': metric_names,
        'Mean': [summary[m][0] for m in metric_names],
        'Std': [summary[m][1] for m in metric_names]
    })
    suffix = f"_{args.tag}" if args.tag else ""
    summary_path = os.path.join(args.out_dir, f'tvstfn_benchmark_summary{suffix}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    detail_df = pd.DataFrame(all_fold_metrics)
    detail_df.insert(0, 'Fold', fold_ids_ran)
    detail_path = os.path.join(args.out_dir, f'tvstfn_benchmark_per_fold{suffix}.csv')
    detail_df.to_csv(detail_path, index=False)
    print(f"Per-fold detail saved to {detail_path}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
