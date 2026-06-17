import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from dgl.nn.pytorch import GATConv
import xgboost as xgb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Ensure reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
dgl.seed(SEED)

def get_mol_descriptors(smiles):
    """Calculate 103 RDKit descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(103)  # Handle invalid SMILES
        
    res = []
    # Use a fixed list of descriptors if possible, or just all scalar descriptors available
    # For reproducibility, let's pick 103 specific ones or just first 103 valid ones from rdkit.Chem.Descriptors.descList
    # But usually "103 descriptors" refers to a specific set. 
    # Here we will use all available scalar descriptors in RDKit (approx 200) and take first 103, or pad if fewer.
    # A better approach for "reproducibility" of '103 descriptors' from a paper is tricky without the list.
    # We will use all available float descriptors and PCA or just slice. 
    # LET'S USE ALL AVAILABLE valid float descriptors.
    
    vals = []
    for name, func in Descriptors.descList:
        try:
            v = func(mol)
            if np.isnan(v) or np.isinf(v):
                v = 0.0
            vals.append(v)
        except:
            vals.append(0.0)
    
    # Paper said 103. RDKit has ~208. Let's select the first 103 for consistency with the prompt's constraint.
    if len(vals) > 103:
        vals = vals[:103]
    else:
        vals = vals + [0.0] * (103 - len(vals))
    return np.array(vals, dtype=np.float32)

def smiles_to_graph(smiles):
    """Convert SMILES to DGL graph."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return empty graph with 1 node to avoid crash
        g = dgl.graph(([], []), num_nodes=1)
        g.ndata['h'] = torch.zeros(1, 40) # Assume 40 atom features
        return g
        
    # Add atoms
    num_atoms = mol.GetNumAtoms()
    src = []
    dst = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
    
    g = dgl.graph((src, dst), num_nodes=num_atoms)
    
    # Atom features (Simple one-hot for symbol + degree + etc.)
    atom_feats = []
    for atom in mol.GetAtoms():
        feats = []
        # Symbol (atomic number)
        feats.append(atom.GetAtomicNum()) 
        # Degree
        feats.append(atom.GetTotalDegree())
        # Formal Charge
        feats.append(atom.GetFormalCharge())
        # Hybridization
        feats.append(int(atom.GetHybridization()))
        # Aromatic
        feats.append(int(atom.GetIsAromatic()))
        # Total H
        feats.append(atom.GetTotalNumHs())
        
        # Pad to 40 dims just to be safe/standard
        f = np.zeros(40)
        f[:len(feats)] = feats
        atom_feats.append(f)
        
    g.ndata['h'] = torch.tensor(np.array(atom_feats), dtype=torch.float32)
    return g

class BenchmarkDataset(Dataset):
    def __init__(self, df, cache_path='cached_data.pt'):
        self.smiles = df['SMILES'].values
        self.y_reg = df['Permeability'].values.astype(np.float32)
        self.y_cls = (df['Permeability'].values >= -6).astype(np.float32)
        
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            data = torch.load(cache_path)
            self.descriptors = data['descriptors']
            self.graphs = data['graphs']
        else:
            # Precompute features
            print("Computing descriptors...")
            self.descriptors = np.array([get_mol_descriptors(s) for s in tqdm(self.smiles)])
            print("Computing graphs...")
            self.graphs = [smiles_to_graph(s) for s in tqdm(self.smiles)]
            print(f"Saving data to {cache_path}...")
            torch.save({'descriptors': self.descriptors, 'graphs': self.graphs}, cache_path)
        
        # Normalize descriptors globally
        # Handle constant columns to avoid div by zero in StandardScaler
        # First, remove infinite values just in case
        self.descriptors = np.nan_to_num(self.descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Manually Standardize
        means = np.mean(self.descriptors, axis=0)
        stds = np.std(self.descriptors, axis=0)
        # Avoid division by zero
        stds[stds == 0] = 1.0
        self.descriptors = (self.descriptors - means) / stds
        
        # Double check
        self.descriptors = np.nan_to_num(self.descriptors, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Simple vocab for Seq models
        chars = set()
        for s in self.smiles:
            chars.update(set(s))
        self.char_map = {c: i+1 for i, c in enumerate(sorted(list(chars)))} # 0 is pad
        self.max_len = 100
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        # Tokenize info seq
        seq = [self.char_map[c] for c in smi][:self.max_len]
        seq = seq + [0]*(self.max_len - len(seq))
        
        return {
            'graph': self.graphs[idx],
            'desc': torch.tensor(self.descriptors[idx], dtype=torch.float32),
            'seq': torch.tensor(seq, dtype=torch.long),
            'y_reg': torch.tensor(self.y_reg[idx], dtype=torch.float32),
            'y_cls': torch.tensor(self.y_cls[idx], dtype=torch.float32),
            'smiles': smi
        }

def collate_fn(batch):
    graphs = dgl.batch([b['graph'] for b in batch])
    descs = torch.stack([b['desc'] for b in batch])
    seqs = torch.stack([b['seq'] for b in batch])
    y_regs = torch.stack([b['y_reg'] for b in batch])
    y_clss = torch.stack([b['y_cls'] for b in batch])
    smiles = [b['smiles'] for b in batch]
    return graphs, descs, seqs, y_regs, y_clss, smiles

# --- Models ---

# MSF-CPMP Model
class MSF_CPMP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Branch 1: Graph
        self.gat1 = GATConv(40, 64, num_heads=2)
        self.gat2 = GATConv(128, 64, num_heads=1)
        
        # Branch 2: Seq (BiLSTM)
        self.embed = nn.Embedding(vocab_size + 1, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        
        # Branch 3: Tabular
        self.mlp_desc = nn.Sequential(
            nn.Linear(103, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 64, 128), # Graph(64) + Seq(128) + Tab(64)
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.head = nn.Linear(64, 1) # Shared head for simplicity, output logit for cls, val for reg
        
    def forward(self, g, desc, seq):
        # Graph
        h = g.ndata['h']
        h = self.gat1(g, h).flatten(1)
        h = F.relu(h)
        h = self.gat2(g, h).flatten(1) # shape: (num_nodes, 64)
        g.ndata['h_out'] = h
        hg = dgl.mean_nodes(g, 'h_out')
        
        # Seq
        x_seq = self.embed(seq)
        _, (hn, _) = self.lstm(x_seq)
        h_seq = torch.cat([hn[-2], hn[-1]], dim=1) # (B, 128)
        
        # Tabular
        h_tab = self.mlp_desc(desc)
        
        # Fusion
        cat = torch.cat([hg, h_seq, h_tab], dim=1)
        feat = self.fusion(cat)
        out = self.head(feat)
        return out.squeeze()

# CycPeptMP Model (Simplified)
class CycPeptMP(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Atom Level (GCN)
        from dgl.nn import GraphConv
        self.gcn1 = GraphConv(40, 64)
        self.gcn2 = GraphConv(64, 64)
        
        # Seq Level (CNN)
        self.embed = nn.Embedding(vocab_size + 1, 64)
        self.cnn = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        
        # Fusion
        self.fusion = nn.Linear(64 + 64, 64)
        self.head = nn.Linear(64, 1)
        
    def forward(self, g, desc, seq):
        # Atom
        h = self.gcn1(g, g.ndata['h'])
        h = F.relu(h)
        h = self.gcn2(g, h)
        g.ndata['h_out'] = h
        h_atom = dgl.mean_nodes(g, 'h_out')
        
        # Seq
        x = self.embed(seq).permute(0, 2, 1) # (B, C, L)
        x = F.relu(self.cnn(x))
        h_seq = F.max_pool1d(x, x.size(2)).squeeze(2)
        
        # Fusion
        cat = torch.cat([h_atom, h_seq], dim=1)
        feat = F.relu(self.fusion(cat))
        out = self.head(feat)
        return out.squeeze()


def train_model(model_cls, vocab_size, train_loader, test_loader, device, task='reg'):
    model = model_cls(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss() if task == 'reg' else nn.BCEWithLogitsLoss()
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        steps = 0
        for g, d, s, yr, yc, _ in train_loader:
            g, d, s = g.to(device), d.to(device), s.to(device)
            target = yr.to(device) if task == 'reg' else yc.to(device)
            
            # Check for nans in input
            if torch.isnan(d).any() or torch.isinf(d).any():
                # Zero out bad descriptors in batch if any (should actully be handled in preprocessing)
                d = torch.nan_to_num(d)
            
            optimizer.zero_grad()
            out = model(g, d, s)
            loss = criterion(out, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                # Skip bad batch
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient Clipping
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        
        if steps > 0:
            print(f"    [{task.upper()}] Epoch {epoch+1}/30 Loss: {total_loss/steps:.4f}")
        else:
             print(f"    [{task.upper()}] Epoch {epoch+1}/30 Loss: NAN (Skipped All)")
            
    model.eval()
    preds = []
    with torch.no_grad():
        for g, d, s, _, _, _ in test_loader:
            g, d, s = g.to(device), d.to(device), s.to(device)
            out = model(g, d, s)
            if task == 'cls':
                out = torch.sigmoid(out)
            preds.extend(out.cpu().numpy())
    return np.array(preds)

def main():
    # Setup paths
    input_path = '/data/workplace/jwx/MSF-CPMP/datasets_process/CycPeptMPDB_Peptide_PAMPA.csv'
    if not os.path.exists(input_path):
        input_path = 'CycPeptMPDB_Peptide_PAMPA.csv'
    
    out_dir = '/data/workplace/jwx/TV-STFN/benchmark_results'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Data
    df = pd.read_csv(input_path, low_memory=False)
    # Ensure Permeability is numeric
    df['Permeability'] = pd.to_numeric(df['Permeability'], errors='coerce') 
    df = df.dropna(subset=['Permeability']) # Clean data
    
    # Also drop any rows where SMILES is missing/invalid
    df = df.dropna(subset=['SMILES'])
    
    dataset = BenchmarkDataset(df)
    vocab_size = len(dataset.char_map)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold}...")
        
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        test_sub = torch.utils.data.Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, collate_fn=collate_fn, num_workers=4)
        test_loader = DataLoader(test_sub, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=4)
        
        # Gold Labels
        y_test_reg = dataset.y_reg[test_idx]
        y_test_cls = dataset.y_cls[test_idx]
        test_smiles = dataset.smiles[test_idx]
        
        # --- Model A: XGBoost ---
        # Prepare tabular data
        X_train_desc = dataset.descriptors[train_idx]
        X_test_desc = dataset.descriptors[test_idx]
        
        # XGB Reg
        xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, n_jobs=4)
        xgb_reg.fit(X_train_desc, dataset.y_reg[train_idx])
        pred_logp_xgb = xgb_reg.predict(X_test_desc)
        
        # XGB Cls
        xgb_cls = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=4)
        xgb_cls.fit(X_train_desc, dataset.y_cls[train_idx])
        pred_score_xgb = xgb_cls.predict_proba(X_test_desc)[:, 1]
        
        # --- Model B: MSF-CPMP ---
        # Regression
        pred_logp_msf = train_model(MSF_CPMP, vocab_size, train_loader, test_loader, device, 'reg')
        # Classification
        pred_score_msf = train_model(MSF_CPMP, vocab_size, train_loader, test_loader, device, 'cls')
        
        # --- Model C: CycPeptMP ---
        # Regression
        pred_logp_cyc = train_model(CycPeptMP, vocab_size, train_loader, test_loader, device, 'reg')
        # Classification
        pred_score_cyc = train_model(CycPeptMP, vocab_size, train_loader, test_loader, device, 'cls')
        
        # Save results
        res_df = pd.DataFrame({
            'SMILES': test_smiles,
            'True_LogP': y_test_reg,
            'True_Label': y_test_cls,
            'Pred_LogP_XGB': pred_logp_xgb,
            'Pred_Score_XGB': pred_score_xgb,
            'Pred_LogP_MSF': pred_logp_msf,
            'Pred_Score_MSF': pred_score_msf,
            'Pred_LogP_Cyc': pred_logp_cyc,
            'Pred_Score_Cyc': pred_score_cyc
        })
        res_df.to_csv(os.path.join(out_dir, f'fold_{fold}_predictions.csv'), index=False)
        
        # Metrics Print
        acc_x = accuracy_score(y_test_cls, (pred_score_xgb > 0.5).astype(int))
        auc_x = roc_auc_score(y_test_cls, pred_score_xgb)
        rho_x = spearmanr(y_test_reg, pred_logp_xgb).correlation
        
        acc_m = accuracy_score(y_test_cls, (pred_score_msf > 0.5).astype(int))
        auc_m = roc_auc_score(y_test_cls, pred_score_msf)
        rho_m = spearmanr(y_test_reg, pred_logp_msf).correlation
        
        acc_c = accuracy_score(y_test_cls, (pred_score_cyc > 0.5).astype(int))
        auc_c = roc_auc_score(y_test_cls, pred_score_cyc)
        rho_c = spearmanr(y_test_reg, pred_logp_cyc).correlation
        
        print(f"Fold {fold} Results:")
        print(f"  XGB: ACC={acc_x:.3f}, AUC={auc_x:.3f}, Spearman={rho_x:.3f}")
        print(f"  MSF: ACC={acc_m:.3f}, AUC={auc_m:.3f}, Spearman={rho_m:.3f}")
        print(f"  Cyc: ACC={acc_c:.3f}, AUC={auc_c:.3f}, Spearman={rho_c:.3f}")

if __name__ == '__main__':
    main()
