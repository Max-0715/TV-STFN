import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import dgl
    from dgl.nn.pytorch import GATConv
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    class GATConv(nn.Module): 
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args, **kwargs): return None

import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
# from dgl.nn.pytorch import GATConv
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Import TV-STFN
sys.path.append('/data/workplace/jwx/TV-STFN')
# We need to install torch-geometric if not available, or adapt the encoders.
# Assuming encoders.py and model.py are independent or we have deps.
# However, encoders.py likely needs specific inputs.
try:
    from model import TetraViewNet
    from encoders import DynamicConformerEncoder
except ImportError:
    print("Could not import TetraViewNet. Check requirements.")
    sys.exit(1)

# Ensure reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if HAS_DGL:
    dgl.seed(SEED)

def get_mol_descriptors(smiles):
    """Calculate 729 RDKit descriptors for TV-STFN (0D View)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(729)
        
    # Morgan FPs (512 bit)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    fp_arr = np.zeros((1,), dtype=np.float32)
    b = np.array(fp) 
    
    # RDKit Descriptors (approx 200 -> 217 in paper, we use all valid)
    vals = []
    for name, func in Descriptors.descList:
        try:
            v = func(mol)
            if np.isnan(v) or np.isinf(v): v = 0.0
            vals.append(v)
        except:
            vals.append(0.0)
            
    # Combine
    combined = np.concatenate([b, np.array(vals)])
    # Pad or truncate to 729
    if len(combined) > 729:
        combined = combined[:729]
    elif len(combined) < 729:
        combined = np.concatenate([combined, np.zeros(729 - len(combined))])
        
    return combined.astype(np.float32)

def smiles_to_pyg(smiles):
    """Convert SMILES to PyG Data object for TV-STFN (View 3)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return PyGData(x=torch.zeros(1, 98), edge_index=torch.empty((2, 0), dtype=torch.long))

    # Features: [atom(64) + hybrid(32) + aromatic(1) + ring(1)] = 98?
    # Encoders.py GraphTopologyEncoder expects:
    # node_feat_dim=4 ?? Wait, code says:
    # "input_feat_dim = 64 + 32 + 2 = 98" in __init__
    # But docstring says: "x: [total_nodes, 4] - node features"
    # Let's check encoders.py source again. It uses Embedding layers for first 2, but expects 4 inputs?
    # self.atom_emb(x[:, 0]), self.hybrid_emb(x[:, 1])
    # x[:, 2] is arom, x[:, 3] is ring.
    # So we need to feed INDICES for first two, and float/int for others.
    
    atom_feats = []
    
    ATOM_LIST = [1, 6, 7, 8, 9, 16, 17, 35]
    HYBRID_LIST = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]
    
    for atom in mol.GetAtoms():
        # 1. Atomic Num Index
        an = atom.GetAtomicNum()
        try:
            an_idx = ATOM_LIST.index(an)
        except:
            an_idx = len(ATOM_LIST) # Other
            
        # 2. Hybrid Index
        hyb = str(atom.GetHybridization())
        try:
            hyb_idx = HYBRID_LIST.index(hyb)
        except:
            hyb_idx = len(HYBRID_LIST) - 1 # Other
            
        # 3. Aromatic
        arom = 1 if atom.GetIsAromatic() else 0
        
        # 4. Ring
        ring = 1 if atom.IsInRing() else 0
        
        atom_feats.append([an_idx, hyb_idx, arom, ring])
        
    x = torch.tensor(atom_feats, dtype=torch.long)
    
    # Edges
    src = []
    dst = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])
        
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # Edge attr if needed? Encoder uses edge_attr? 
    # forward: graph_2d_batch.edge_index
    # It doesn't seem to use edge_attr in GCN/GAT branch logic provided in my snippet.
    
    return PyGData(x=x, edge_index=edge_index, num_nodes=len(atom_feats))

def smiles_to_graph(smiles):
    """Convert SMILES to DGL graph."""
    if not HAS_DGL:
        return None
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
    def __init__(self, df, cache_path='cached_data_tv.pt'):
        self.smiles = df['SMILES'].values
        self.y_reg = df['Permeability'].values.astype(np.float32)
        self.y_cls = (df['Permeability'].values >= -6).astype(np.float32)
        
        if os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}...")
            try:
                data = torch.load(cache_path)
                self.descriptors = data['descriptors']
                self.graphs = data['graphs'] 
                self.pyg_graphs = data.get('pyg_graphs', None) # Add PyG Support
                
                if self.pyg_graphs is None:
                    raise Exception("PyG graphs missing")
            except:
                print("Cache load failed or outdated. Recomputing...")
                self._compute_data(cache_path)
        else:
            self._compute_data(cache_path)
            
        # Normalize descriptors 
        self.descriptors = np.nan_to_num(self.descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        means = np.mean(self.descriptors, axis=0)
        stds = np.std(self.descriptors, axis=0)
        stds[stds == 0] = 1.0
        self.descriptors = (self.descriptors - means) / stds
        
        # Simple vocab for Seq models (View 2)
        chars = set()
        for s in self.smiles:
            chars.update(set(s))
        self.char_map = {c: i+1 for i, c in enumerate(sorted(list(chars)))} # 0 is pad
        self.max_len = 100
        
    def _compute_data(self, cache_path):
        print("Computing descriptors (729 dim)...")
        self.descriptors = np.array([get_mol_descriptors(s) for s in tqdm(self.smiles)])
        print("Computing DGL graphs...")
        self.graphs = [smiles_to_graph(s) for s in tqdm(self.smiles)]
        print("Computing PyG graphs (for TV-STFN)...")
        self.pyg_graphs = [smiles_to_pyg(s) for s in tqdm(self.smiles)]
        
        print(f"Saving data to {cache_path}...")
        torch.save({'descriptors': self.descriptors, 'graphs': self.graphs, 'pyg_graphs': self.pyg_graphs}, cache_path)

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        seq = [self.char_map[c] for c in smi][:self.max_len]
        seq = seq + [0]*(self.max_len - len(seq))
        
        # View 4 (0D): Descriptors
        feat_0d = torch.tensor(self.descriptors[idx], dtype=torch.float32)
        
        return {
            'graph': self.graphs[idx],
            'pyg': self.pyg_graphs[idx], # PyG Graph
            'desc': feat_0d,
            'seq': torch.tensor(seq, dtype=torch.long),
            'y_reg': torch.tensor(self.y_reg[idx], dtype=torch.float32),
            'y_cls': torch.tensor(self.y_cls[idx], dtype=torch.float32),
            'smiles': smi,
            'idx': idx
        }

def collate_fn(batch):
    if HAS_DGL:
        try:
            graphs = dgl.batch([b['graph'] for b in batch])
        except:
            graphs = None
    else:
        graphs = None

    descs = torch.stack([b['desc'] for b in batch])
    seqs = torch.stack([b['seq'] for b in batch])
    y_regs = torch.stack([b['y_reg'] for b in batch])
    y_clss = torch.stack([b['y_cls'] for b in batch])
    smiles = [b['smiles'] for b in batch]
    
    # TV-STFN Inputs
    # View 1: 3D (Simulate Dictionary Input)
    # Keys: 'coords', 'atom_features', 'num_atoms'
    # We fake 'coords' with random [1, N, 3] per mol
    # We use 'pyg.x' (atomic indices) for 'atom_features'
    
    view_1_coords = []
    view_1_feats = []
    view_1_num = []
    max_atoms = 0
    
    for b in batch:
        pyg = b['pyg']
        n = pyg.num_nodes
        if n > max_atoms: max_atoms = n
        view_1_num.append(n)
        
    # Pad and stack
    # Max atoms in this batch
    coords_batch = torch.zeros(len(batch), 1, max_atoms, 3) 
    feats_batch = torch.zeros(len(batch), max_atoms, 4, dtype=torch.long)
    
    for i, b in enumerate(batch):
        n = view_1_num[i]
        # Random coords (range -2 to 2)
        coords_batch[i, 0, :n, :] = torch.randn(n, 3) 
        # Features from PyG
        feats_batch[i, :n, :] = b['pyg'].x
        
    view1 = {
        'coords': coords_batch,
        'atom_features': feats_batch,
        'num_atoms': torch.tensor(view_1_num)
    }
    
    # View 2: 1D (B, L)
    # The Encoder expects a dict with 'input_ids' and 'attention_mask'
    # Our 'seqs' are already Token IDs.
    view2 = {
        'input_ids': seqs,
        'attention_mask': (seqs != 0).long() # Assuming 0 is pad
    }
    
    # View 3: 2D (PyG Batch)
    view3 = PyGBatch.from_data_list([b['pyg'] for b in batch])
    
    # View 4: 0D (B, 729)
    view4 = descs
    
    return {
        'view1': view1,
        'view2': view2,
        'view3': view3,
        'view4': view4,
        'y_reg': y_regs,
        'y_cls': y_clss,
        'smiles': smiles,
        # Legacy returns for other models
        'legacy': (graphs, descs, seqs)
    }
# --- Models ---

# MSF-CPMP Model
class MSF_CPMP(nn.Module):
    def __init__(self, vocab_size, input_dim=729):
        super().__init__()
        # Branch 1: Graph
        self.gat1 = GATConv(40, 64, num_heads=2)
        self.gat2 = GATConv(128, 64, num_heads=1)
        
        # Branch 2: Seq (BiLSTM)
        self.embed = nn.Embedding(vocab_size + 1, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        
        # Branch 3: Tabular
        self.mlp_desc = nn.Sequential(
            nn.Linear(input_dim, 128),
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
        if HAS_DGL:
            from dgl.nn import GraphConv
            self.gcn1 = GraphConv(40, 64)
            self.gcn2 = GraphConv(64, 64)
        else:
            self.gcn1 = None
            self.gcn2 = None
        
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
    # Adapt simple training loop for TV-STFN
    if model_cls == TetraViewNet:
        # TV-STFN init might vary
        model = model_cls().to(device)
    else:
        model = model_cls(vocab_size).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss() if task == 'reg' else nn.BCEWithLogitsLoss()
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        steps = 0
        
        # Loader yields dict now from collate_fn
        for batch in train_loader:
            if isinstance(batch, tuple): # Handle legacy if collate changed
                g, d, s, yr, yc, _ = batch
            else:
                # New collate dict
                g, d, s = batch['legacy']
                yr, yc = batch['y_reg'], batch['y_cls']
            
            # g, d, s = g.to(device), d.to(device), s.to(device)
            if g is not None:
                g = g.to(device)
            d = d.to(device)
            s = s.to(device)
            
            target = yr.to(device) if task == 'reg' else yc.to(device)
            
            # Check for nans
            if torch.isnan(d).any() or torch.isinf(d).any():
                d = torch.nan_to_num(d)
                
            optimizer.zero_grad()
            
            if model_cls == TetraViewNet:
                # TV-STFN requires dict input
                # Move view1 dict tensors to device manually
                v1 = batch['view1']
                v1_device = {
                    'coords': v1['coords'].to(device),
                    'atom_features': v1['atom_features'].to(device),
                    'num_atoms': v1['num_atoms'].to(device)
                }
                
                v2 = batch['view2']
                v2_device = {
                    'input_ids': v2['input_ids'].to(device),
                    'attention_mask': v2['attention_mask'].to(device)
                }
                
                inputs = {
                   'view1': v1_device,
                   'view2': v2_device,
                   'view3': batch['view3'].to(device), # PyG batch supports .to()
                   'view4': batch['view4'].to(device)
                }
                out = model(inputs).squeeze()
            else:
                out = model(g, d, s)
                
            loss = criterion(out, target)
            
            if torch.isnan(loss) or torch.isinf(loss): continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        for batch in test_loader:
            if isinstance(batch, tuple):
                 g, d, s, _, _, _ = batch
            else:
                 g, d, s = batch['legacy']
                 
            g, d, s = g.to(device), d.to(device), s.to(device)
            
            if model_cls == TetraViewNet:
                v1 = batch['view1']
                v1_device = {
                    'coords': v1['coords'].to(device),
                    'atom_features': v1['atom_features'].to(device),
                    'num_atoms': v1['num_atoms'].to(device)
                }
                
                v2 = batch['view2']
                v2_device = {
                    'input_ids': v2['input_ids'].to(device),
                    'attention_mask': v2['attention_mask'].to(device)
                }
                
                inputs = {
                   'view1': v1_device,
                   'view2': v2_device,
                   'view3': batch['view3'].to(device),
                   'view4': batch['view4'].to(device)
                }
                out = model(inputs).squeeze()
            else:
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
        
        # --- SKIP OTHER MODELS FOR SPEED ---
        # Initialize placeholders for results dataframe
        pred_logp_xgb = np.zeros_like(y_test_reg)
        pred_score_xgb = np.zeros_like(y_test_cls)
        pred_logp_msf = np.zeros_like(y_test_reg) 
        pred_score_msf = np.zeros_like(y_test_cls)
        pred_logp_cyc = np.zeros_like(y_test_reg)
        pred_score_cyc = np.zeros_like(y_test_cls)

        # Dataset for legacy models (needed for placeholders? no, we skip)
        
        # --- Model D: TV-STFN (Your Model) ---
        print("Training TV-STFN...")
        # Regression
        pred_logp_tv = train_model(TetraViewNet, vocab_size, train_loader, test_loader, device, 'reg')
        # Classification
        pred_score_tv = train_model(TetraViewNet, vocab_size, train_loader, test_loader, device, 'cls')
        
        # Save initial results
        res_df = pd.DataFrame({
            'SMILES': test_smiles,
            'True_LogP': y_test_reg,
            'True_Label': y_test_cls,
            # Placeholders
            'Pred_LogP_XGB': pred_logp_xgb, 'Pred_Score_XGB': pred_score_xgb,
            'Pred_LogP_MSF': pred_logp_msf, 'Pred_Score_MSF': pred_score_msf,
            'Pred_LogP_Cyc': pred_logp_cyc, 'Pred_Score_Cyc': pred_score_cyc,
            # TV-STFN
            'Pred_LogP_TV': pred_logp_tv,
            'Pred_Score_TV': pred_score_tv
        })
        
        # Calculate TV metrics locally first needed for print
        acc_t = accuracy_score(y_test_cls, (pred_score_tv > 0.5).astype(int))
        auc_t = roc_auc_score(y_test_cls, pred_score_tv)
        rho_t = spearmanr(y_test_reg, pred_logp_tv).correlation
        
        # Load other results if available to merge
        results_merged = False
        prev_res_path = os.path.join(out_dir, f'fold_{fold}_predictions.csv')
        if os.path.exists(prev_res_path):
            try:
                prev_df = pd.read_csv(prev_res_path)
                # Ensure alignment (SMILES match)
                if (prev_df['SMILES'].iloc[0] == res_df['SMILES'].iloc[0]):
                    # Merge columns
                    res_df['Pred_LogP_XGB'] = prev_df['Pred_LogP_XGB']
                    res_df['Pred_Score_XGB'] = prev_df['Pred_Score_XGB']
                    res_df['Pred_LogP_MSF'] = prev_df['Pred_LogP_MSF']
                    res_df['Pred_Score_MSF'] = prev_df['Pred_Score_MSF']
                    res_df['Pred_LogP_Cyc'] = prev_df['Pred_LogP_Cyc']
                    res_df['Pred_Score_Cyc'] = prev_df['Pred_Score_Cyc']
                    results_merged = True
                    print(f"Fold {fold}: Merged with previous benchmark results.")
            except Exception as e:
                print(f"Fold {fold}: Merge failed ({e}). Saving TV results only.")

        res_df.to_csv(os.path.join(out_dir, f'fold_{fold}_predictions_merged.csv'), index=False)
        
        # Metrics Print
        if results_merged:
             acc_x = accuracy_score(y_test_cls, (res_df['Pred_Score_XGB'] > 0.5).astype(int))
             auc_x = roc_auc_score(y_test_cls, res_df['Pred_Score_XGB'])
             rho_x = spearmanr(y_test_reg, res_df['Pred_LogP_XGB']).correlation
             
             acc_m = accuracy_score(y_test_cls, (res_df['Pred_Score_MSF'] > 0.5).astype(int))
             auc_m = roc_auc_score(y_test_cls, res_df['Pred_Score_MSF'])
             rho_m = spearmanr(y_test_reg, res_df['Pred_LogP_MSF']).correlation
             
             acc_c = accuracy_score(y_test_cls, (res_df['Pred_Score_Cyc'] > 0.5).astype(int))
             auc_c = roc_auc_score(y_test_cls, res_df['Pred_Score_Cyc'])
             rho_c = spearmanr(y_test_reg, res_df['Pred_LogP_Cyc']).correlation
             
             print(f"Fold {fold} Results:")
             print(f"  XGB: ACC={acc_x:.3f}, AUC={auc_x:.3f}, Spearman={rho_x:.3f}")
             print(f"  MSF: ACC={acc_m:.3f}, AUC={auc_m:.3f}, Spearman={rho_m:.3f}")
             print(f"  Cyc: ACC={acc_c:.3f}, AUC={auc_c:.3f}, Spearman={rho_c:.3f}")
             print(f"  TV-STFN: ACC={acc_t:.3f}, AUC={auc_t:.3f}, Spearman={rho_t:.3f}")
        else:
             print(f"Fold {fold} Results (TV-STFN Only):")
             print(f"  TV-STFN: ACC={acc_t:.3f}, AUC={auc_t:.3f}, Spearman={rho_t:.3f}")

if __name__ == '__main__':
    main()
