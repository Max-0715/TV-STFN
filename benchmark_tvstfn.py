"""
TV-STFN Benchmark — 与 benchmark_repro.py 完全对齐
- 相同 CSV / 相同 dropna / 相同 KFold(10, shuffle=True, random_state=42)
- 相同 30 epochs / 相同梯度裁剪
- 结果直接追加到已有 fold_X_predictions.csv
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch
import torch.nn.utils.rnn as rnn_utils
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TetraViewNet
from loss import CompositeLoss

# ==================== Config ====================
SEED = 42
N_FOLDS = 10
EPOCHS = 30          # 和 benchmark_repro.py 一致
BATCH_SIZE = 32      # TV-STFN 较大, 32 比较安全
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tetraview_processed")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ==================== Dataset ====================
class IndexedTetraViewDataset(Dataset):
    """按 CSV 行号直接加载 data_{idx}.pt, 保证和 benchmark_repro 索引对齐。
    预加载所有数据到内存以加速训练。"""
    def __init__(self, indices, data_dir, cache=None):
        self.indices = list(indices)
        self.data_dir = data_dir
        self.cache = cache  # 共享缓存 dict
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        csv_idx = self.indices[i]
        if self.cache is not None and csv_idx in self.cache:
            return self.cache[csv_idx]
        path = os.path.join(self.data_dir, f"data_{csv_idx}.pt")
        data = torch.load(path, weights_only=False)
        if self.cache is not None:
            self.cache[csv_idx] = data
        return data

def collate_fn(batch):
    """和 dataset.py 的 tetra_view_collate 一致"""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # View 1: 3D
    coords_list = [b['view1_3d']['coords'] for b in batch]
    atom_feat_list = [b['view1_3d']['atom_feat'] for b in batch]
    num_atoms = torch.tensor([c.size(1) for c in coords_list])
    max_atoms = num_atoms.max().item()
    bs = len(batch)
    n_conf = coords_list[0].size(0)
    
    padded_coords = torch.zeros(bs, n_conf, max_atoms, 3)
    padded_feats = torch.zeros(bs, max_atoms, 4)
    for i, (c, f) in enumerate(zip(coords_list, atom_feat_list)):
        n = c.size(1)
        padded_coords[i, :, :n, :] = c
        padded_feats[i, :n, :] = f
    
    view1 = {'coords': padded_coords, 'atom_features': padded_feats, 'num_atoms': num_atoms}
    
    # View 2: 1D
    ids_list = [b['view2_1d']['input_ids'] for b in batch]
    padded_ids = rnn_utils.pad_sequence(ids_list, batch_first=True, padding_value=0)
    attn_mask = (padded_ids != 0).float()
    view2 = {'input_ids': padded_ids, 'attention_mask': attn_mask}
    
    # View 3: 2D (PyG)
    view3 = PyGBatch.from_data_list([b['view3_2d'] for b in batch])
    
    # View 4: 0D
    view4 = torch.stack([b['view4_0d'] for b in batch])
    
    # Target
    targets = torch.tensor([b['target'] for b in batch], dtype=torch.float32).unsqueeze(1)
    
    return {'view1': view1, 'view2': view2, 'view3': view3, 'view4': view4, 'targets': targets}

def to_device(batch, device):
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

# ==================== Train & Predict ====================
def train_and_predict(train_indices, test_indices, task='reg', cache=None):
    """
    和 benchmark_repro.py 的 train_model 对齐:
    - 30 epochs, gradient clipping 1.0
    - task='reg' 输出 LogP, task='cls' 输出 score
    """
    train_ds = IndexedTetraViewDataset(train_indices, DATA_DIR, cache=cache)
    test_ds = IndexedTetraViewDataset(test_indices, DATA_DIR, cache=cache)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    
    model = TetraViewNet().to(DEVICE)
    
    if task == 'reg':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # 和 benchmark_repro 一致: Adam, lr=0.0005
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        steps = 0
        for batch in train_loader:
            if batch is None:
                continue
            targets = batch['targets'].to(DEVICE)
            
            if task == 'cls':
                # 转为二分类标签 (>= -6 为正)
                targets = (targets >= -6.0).float()
            
            inp = to_device(batch, DEVICE)
            optimizer.zero_grad()
            out = model(inp)  # [B, 1]
            
            if task == 'reg':
                loss = criterion(out, targets)
            else:
                loss = criterion(out, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        
        avg = total_loss / max(steps, 1)
        print(f"    [{task.upper()}] Epoch {epoch+1}/{EPOCHS} Loss: {avg:.4f}", flush=True)
    
    # Predict
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            inp = to_device(batch, DEVICE)
            out = model(inp)  # [B, 1]
            if task == 'cls':
                out = torch.sigmoid(out)
            preds.extend(out.cpu().numpy().flatten())
    
    return np.array(preds)

# ==================== Main ====================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1) 加载 CSV — 和 benchmark_repro.py 完全一致
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CycPeptMPDB_Peptide_PAMPA.csv')
    df = pd.read_csv(csv_path, low_memory=False)
    df['Permeability'] = pd.to_numeric(df['Permeability'], errors='coerce')
    df = df.dropna(subset=['Permeability'])
    df = df.dropna(subset=['SMILES'])
    df = df.reset_index(drop=True)
    
    print(f"Device: {DEVICE}", flush=True)
    print(f"Dataset size: {len(df)}", flush=True)
    print(f"Folds: {N_FOLDS}, Epochs: {EPOCHS}, BS: {BATCH_SIZE}", flush=True)
    
    # 预加载所有 .pt 数据到内存（一次加载, 所有 fold 复用）
    print("Pre-loading all tetraview data into memory...", flush=True)
    data_cache = {}
    for idx in range(len(df)):
        path = os.path.join(DATA_DIR, f"data_{idx}.pt")
        if os.path.exists(path):
            data_cache[idx] = torch.load(path, weights_only=False)
        if (idx + 1) % 1000 == 0:
            print(f"  Loaded {idx+1}/{len(df)}...", flush=True)
    print(f"  Done! {len(data_cache)} samples cached in memory.", flush=True)
    print("=" * 70, flush=True)
    
    # 2) KFold — 和 benchmark_repro.py 完全一致
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(df)))):
        fold_t = time.time()
        print(f"\nFold {fold} ({len(train_idx)} train, {len(test_idx)} test)", flush=True)
        
        # 3) 训练回归模型, 预测 LogP
        pred_logp = train_and_predict(train_idx, test_idx, task='reg', cache=data_cache)
        
        # 4) 训练分类模型, 预测 Score
        pred_score = train_and_predict(train_idx, test_idx, task='cls', cache=data_cache)
        
        # 5) 写入结果
        fold_csv = os.path.join(OUT_DIR, f'fold_{fold}_predictions.csv')
        
        if os.path.exists(fold_csv):
            # 追加列到已有 CSV
            res = pd.read_csv(fold_csv)
            # 去掉旧的 TV-STFN 列（如果有）
            res = res.drop(columns=[c for c in res.columns if 'TVSTFN' in c], errors='ignore')
            res['Pred_LogP_TVSTFN'] = pred_logp
            res['Pred_Score_TVSTFN'] = pred_score
            res.to_csv(fold_csv, index=False)
            print(f"  -> Merged into {fold_csv}", flush=True)
        else:
            # 新建 CSV
            test_smiles = df['SMILES'].values[test_idx]
            y_reg = df['Permeability'].values[test_idx]
            y_cls = (y_reg >= -6).astype(float)
            res = pd.DataFrame({
                'SMILES': test_smiles,
                'True_LogP': y_reg,
                'True_Label': y_cls,
                'Pred_LogP_TVSTFN': pred_logp,
                'Pred_Score_TVSTFN': pred_score
            })
            res.to_csv(fold_csv, index=False)
            print(f"  -> Created {fold_csv}", flush=True)
        
        # 6) 打印 fold 指标（和 benchmark_repro 格式一致）
        y_test_reg = df['Permeability'].values[test_idx]
        y_test_cls = (y_test_reg >= -6).astype(float)
        
        acc = accuracy_score(y_test_cls, (pred_score > 0.5).astype(int))
        try:
            auc = roc_auc_score(y_test_cls, pred_score)
        except:
            auc = 0.0
        rho = spearmanr(y_test_reg, pred_logp).correlation
        rmse = np.sqrt(mean_squared_error(y_test_reg, pred_logp))
        
        print(f"  TVSTFN: ACC={acc:.3f}, AUC={auc:.3f}, Spearman={rho:.3f}, RMSE={rmse:.3f} ({time.time()-fold_t:.0f}s)", flush=True)
        
        # 清理 GPU
        torch.cuda.empty_cache()
    
    print(f"\n{'='*70}", flush=True)
    print(f"Done! Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"Results in: {OUT_DIR}/fold_*_predictions.csv", flush=True)

if __name__ == '__main__':
    main()
