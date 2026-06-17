import os
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                             accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score, cohen_kappa_score)
from scipy.stats import pearsonr, spearmanr
import numpy as np

# Import our components
from dataset import TetraViewDataset, tetra_view_collate
from model import TetraViewNet
from loss import CompositeLoss

# Configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100  # 增加到100，配合LR Scheduler
EARLY_STOP_PATIENCE = 20  # 增加早停耐心值
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "tetraview_processed"
SAVE_PATH = "best_tetraview_model.pth"
NUM_WORKERS = min(8, os.cpu_count() or 1)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    total_mse = 0
    total_rank = 0
    total_focal = 0
    
    for batch in loader:
        # Move data to device
        targets = batch['targets'].to(device, non_blocking=True)
        
        # Helper to move nested dicts/objects to device
        batch_input = {}
        
        # View 1
        batch_input['view1'] = {
            'coords': batch['view1']['coords'].to(device, non_blocking=True),
            'atom_features': batch['view1']['atom_features'].to(device, non_blocking=True),
            'num_atoms': batch['view1']['num_atoms'].to(device, non_blocking=True)
        }
        # View 2
        batch_input['view2'] = {
            'input_ids': batch['view2']['input_ids'].to(device, non_blocking=True),
            'attention_mask': batch['view2']['attention_mask'].to(device, non_blocking=True)
        }
        # View 3
        batch_input['view3'] = batch['view3'].to(device, non_blocking=True)
        # View 4
        batch_input['view4'] = batch['view4'].to(device, non_blocking=True)
        
        # Forward
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            preds = model(batch_input)
            # Loss
            loss, mse, rank, focal = criterion(preds, targets)
        
        # Backward (AMP)
        scaler.scale(loss).backward()
        
        # 梯度裁剪：防止梯度爆炸 (解决Epoch 34 spike问题)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_mse += mse
        total_rank += rank
        total_focal += focal
        
    avg_loss = total_loss / len(loader)
    avg_mse = total_mse / len(loader)
    avg_rank = total_rank / len(loader)
    avg_focal = total_focal / len(loader)
    return avg_loss, avg_mse, avg_rank, avg_focal

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            targets = batch['targets'].to(device)
            
            batch_input = {}
            batch_input['view1'] = {
                'coords': batch['view1']['coords'].to(device, non_blocking=True),
                'atom_features': batch['view1']['atom_features'].to(device, non_blocking=True),
                'num_atoms': batch['view1']['num_atoms'].to(device, non_blocking=True)
            }
            batch_input['view2'] = {
                'input_ids': batch['view2']['input_ids'].to(device, non_blocking=True),
                'attention_mask': batch['view2']['attention_mask'].to(device, non_blocking=True)
            }
            batch_input['view3'] = batch['view3'].to(device, non_blocking=True)
            batch_input['view4'] = batch['view4'].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                preds = model(batch_input)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    
    # 回归指标 (Regression Metrics)
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    pearson_r, _ = pearsonr(all_targets, all_preds)
    spearman_r, _ = spearmanr(all_targets, all_preds)
    
    # MAPE (Mean Absolute Percentage Error) - 避免除以0
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-10))) * 100
    
    # 分类指标 (Classification Metrics) - 使用中位数作为阈值
    threshold = np.median(all_targets)
    preds_binary = (all_preds >= threshold).astype(int)
    targets_binary = (all_targets >= threshold).astype(int)
    
    acc = accuracy_score(targets_binary, preds_binary)
    precision = precision_score(targets_binary, preds_binary, zero_division=0)
    recall = recall_score(targets_binary, preds_binary, zero_division=0)
    f1 = f1_score(targets_binary, preds_binary, zero_division=0)
    mcc = matthews_corrcoef(targets_binary, preds_binary)
    kappa = cohen_kappa_score(targets_binary, preds_binary)
    
    # AUROC 和 AUPRC (使用预测概率)
    try:
        # 将预测值归一化到[0,1]区间作为概率
        preds_prob = (all_preds - all_preds.min()) / (all_preds.max() - all_preds.min() + 1e-10)
        auroc = roc_auc_score(targets_binary, preds_prob)
        auprc = average_precision_score(targets_binary, preds_prob)
    except:
        auroc = 0.0
        auprc = 0.0
    
    # TNR (True Negative Rate / Specificity)
    tn = np.sum((targets_binary == 0) & (preds_binary == 0))
    fp = np.sum((targets_binary == 0) & (preds_binary == 1))
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        # 回归指标
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2, 'mape': mape,
        'pearson': pearson_r, 'spearman': spearman_r,
        # 分类指标
        'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1,
        'mcc': mcc, 'kappa': kappa, 'tnr': tnr, 'auroc': auroc, 'auprc': auprc
    }

def main():
    set_seed(42)

    if DEVICE.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    full_dataset = TetraViewDataset(DATA_DIR)
    print(f"Total samples: {len(full_dataset)}")
    
    # Split: 80% Train, 10% Val, 10% Test
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    loader_kwargs = {
        'batch_size': BATCH_SIZE,
        'collate_fn': tetra_view_collate,
        'num_workers': NUM_WORKERS,
        'pin_memory': (DEVICE.type == 'cuda')
    }

    if NUM_WORKERS > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    
    # 2. Init Model
    model = TetraViewNet().to(DEVICE)
    
    # 优化：差分学习率 (Differential Learning Rates)
    # 3D SchNet 分支参数较多且敏感，使用较小学习率，防止破坏预训练或初始化
    # 其余部分使用标准学习率
    encoder_3d_params = list(map(id, model.encoder_3d.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_3d_params, model.parameters())

    optimizer = optim.AdamW([
        {'params': base_params},
        {'params': model.encoder_3d.parameters(), 'lr': LEARNING_RATE * 0.1}  # LR for 3D view
    ], lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 优化：引入学习率调度器 (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    criterion = CompositeLoss(lambda_focal=1.0, lambda_rank=1.0, lambda_mse=1.0)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    # 3. Training Loop
    best_rmse = float('inf')
    patience_counter = 0  # 早停计数器
    
    print("\nStarting Training...")
    print(f"Epochs: {EPOCHS} | Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"{'Epoch':<5} | {'Loss':<8} | {'MSE':<7} | {'Rank':<7} | {'Focal':<7} | {'RMSE':<7} | {'R²':<7} | {'Spearman':<8} | {'AUROC':<7}")
    print("-" * 120)
    
    for epoch in range(EPOCHS):
        loss, mse, rank, focal = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        metrics = evaluate(model, val_loader, DEVICE)
        
        print(f"{epoch+1:<5} | {loss:.4f}   | {mse:.4f}  | {rank:.4f}  | {focal:.4f}  | {metrics['rmse']:.4f}  | {metrics['r2']:.4f}  | {metrics['spearman']:.4f}   | {metrics['auroc']:.4f}")
        
        # 更新学习率
        scheduler.step(metrics['rmse'])

        # 早停机制：如果RMSE有提升，重置计数器；否则计数+1
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            torch.save(model.state_dict(), SAVE_PATH)
            patience_counter = 0  # 重置早停计数器
            print(f"      ✓ 新最优模型已保存 (RMSE: {best_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⚠ 早停触发：连续{EARLY_STOP_PATIENCE}轮无提升，停止训练")
                break
            # print("  -> Model saved")
            
    print("\nTraining Complete.")
    print(f"Best Val RMSE: {best_rmse:.4f}")
    
    # 4. Final Test
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=tetra_view_collate,
                             num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
    test_metrics = evaluate(model, test_loader, DEVICE)
    
    print("\n" + "="*80)
    print("Test Set Evaluation (按MSF-CPMP论文标准)")
    print("="*80)
    print("\n【回归指标 Regression Metrics】")
    print(f"  MAE:      {test_metrics['mae']:.4f}")
    print(f"  MSE:      {test_metrics['mse']:.4f}")
    print(f"  RMSE:     {test_metrics['rmse']:.4f}")
    print(f"  MAPE:     {test_metrics['mape']:.2f}%")
    print(f"  R²:       {test_metrics['r2']:.4f}")
    print(f"  Pearson:  {test_metrics['pearson']:.4f}")
    print(f"  Spearman: {test_metrics['spearman']:.4f}")
    
    print("\n【分类指标 Classification Metrics (阈值=中位数)】")
    print(f"  ACC:       {test_metrics['acc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    print(f"  Kappa:     {test_metrics['kappa']:.4f}")
    print(f"  TNR:       {test_metrics['tnr']:.4f}")
    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  AUPRC:     {test_metrics['auprc']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
