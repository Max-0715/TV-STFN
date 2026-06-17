from joblib import Parallel, delayed
"""
Preprocess CycPeptMPDB data into Tetra-View format
Generates 4 views: 3D conformers, 1D tokens, 2D graph, 0D descriptors
"""

import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers, make it optional
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("⚠ Warning: transformers not found. Will use simple character-level tokenization.")
    HAS_TRANSFORMERS = False

# Configuration
CSV_FILE = "CycPeptMPDB_Peptide_PAMPA.csv"
OUTPUT_DIR = "tetraview_processed"
NUM_CONFORMERS = 10
MAX_SEQ_LEN = 200
TOKENIZER_NAME = 'seyonec/ChemBERTa-zinc-base-v1'

# Atomic constants
ATOM_LIST = [1, 6, 7, 8, 9, 16, 17, 35]  # H, C, N, O, F, S, Cl, Br
HYBRIDIZATION_LIST = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]

def compute_physchem_features(mol):
    """
    Computes 8 key physicochemical features using RDKit.
    Matches the input expected by PhysicochemicalEncoder in model.py.
    """
    if mol is None:
        return None
        
    from rdkit.Chem import Descriptors, rdMolDescriptors
    
    try:
        features = [
            Descriptors.MolLogP(mol),          # LogP (Lipophilicity)
            Descriptors.TPSA(mol),             # TPSA (Polar surface area)
            Descriptors.MolWt(mol),            # Molecular Weight
            Descriptors.NumHDonors(mol),       # H-Bond Donors
            Descriptors.NumHAcceptors(mol),    # H-Bond Acceptors
            rdMolDescriptors.CalcNumRings(mol),# Ring Count
            Descriptors.NumRotatableBonds(mol),# Rotatable Bonds
            mol.GetNumHeavyAtoms()             # Heavy Atom Count
        ]
        return torch.tensor(features, dtype=torch.float)
    except Exception as e:
        print(f"  ⚠ PhysChem feature extraction failed: {e}")
        return None


def extract_atom_features(mol):
    """Extract atomic features for a molecule"""
    features = []
    for atom in mol.GetAtoms():
        # Atomic number
        atomic_num = atom.GetAtomicNum()
        atomic_idx = ATOM_LIST.index(atomic_num) if atomic_num in ATOM_LIST else len(ATOM_LIST)
        
        # Hybridization
        hybrid = str(atom.GetHybridization())
        hybrid_idx = HYBRIDIZATION_LIST.index(hybrid) if hybrid in HYBRIDIZATION_LIST else HYBRIDIZATION_LIST.index("OTHER")
        
        # Aromaticity and ring membership
        is_aromatic = int(atom.GetIsAromatic())
        is_in_ring = int(atom.IsInRing())
        
        features.append([atomic_idx, hybrid_idx, is_aromatic, is_in_ring])
    
    return torch.tensor(features, dtype=torch.long)


def generate_3d_conformers(smiles, num_confs=10):
    """
    Generate 3D conformers with multiple fallback strategies.
    Strategy 1: ETKDGv3 (best for macrocycles)
    Strategy 2: ETKDG (general purpose)
    Strategy 3: Random coordinates + optimization
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        num_generated = 0
        
        # Strategy 1: ETKDGv3 (best for macrocycles like cyclic peptides)
        try:
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxIterations = 2000  # Increased iterations
            params.randomSeed = 42
            params.numThreads = 1  # Use all cores
            params.useSmallRingTorsions = True  # Better for ring systems
            params.useMacrocycleTorsions = True  # Critical for cyclic peptides
            
            num_generated = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        except Exception:
            pass
        
        # Strategy 2: Standard ETKDG if ETKDGv3 fails
        if num_generated == 0:
            try:
                params = AllChem.ETKDG()
                params.useRandomCoords = True
                params.maxIterations = 2000
                params.randomSeed = 42
                num_generated = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            except Exception:
                pass
        
        # Strategy 3: Random coordinates with distance geometry
        if num_generated == 0:
            try:
                num_generated = AllChem.EmbedMultipleConfs(
                    mol, 
                    numConfs=num_confs,
                    useRandomCoords=True,
                    maxAttempts=100,
                    randomSeed=42
                )
            except Exception:
                pass
        
        # Strategy 4: Single conformer with random coords as last resort
        if num_generated == 0:
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
                num_generated = 1
            except Exception:
                return None
        
        if num_generated == 0:
            return None
        
        # MMFF optimization with fallback to UFF
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)
        except Exception:
            try:
                # Fallback to UFF force field
                for conf_id in range(mol.GetNumConformers()):
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
            except Exception:
                pass  # Use unoptimized coordinates
        
        # Extract coordinates
        coords_list = []
        for conf_id in range(mol.GetNumConformers()):
            conf = mol.GetConformer(conf_id)
            coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(mol.GetNumAtoms())
            ])
            coords_list.append(coords)
        
        # Pad if necessary (duplicate first conformer)
        while len(coords_list) < num_confs:
            coords_list.append(coords_list[0])
        
        coords_tensor = torch.tensor(np.array(coords_list[:num_confs]), dtype=torch.float)
        atom_features = extract_atom_features(mol)
        
        return {
            'coords': coords_tensor,  # [num_confs, num_atoms, 3]
            'atom_feat': atom_features  # [num_atoms, 4]
        }
        
    except Exception as e:
        print(f"  ⚠ 3D generation failed for SMILES: {smiles[:50]}... Error: {e}")
        return None


def tokenize_smiles(smiles, tokenizer, max_len=200):
    """Tokenize SMILES for molecular language model"""
    try:
        if HAS_TRANSFORMERS and tokenizer is not None:
            encoded = tokenizer(
                smiles,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        else:
            # Simple character-level tokenization fallback
            char_to_idx = {c: i+1 for i, c in enumerate(set(smiles))}
            char_to_idx['<PAD>'] = 0
            
            tokens = [char_to_idx.get(c, 0) for c in smiles[:max_len]]
            tokens += [0] * (max_len - len(tokens))  # Padding
            
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.tensor([1 if t > 0 else 0 for t in tokens], dtype=torch.long)
            }
        
    except Exception as e:
        print(f"  ⚠ Tokenization failed: {e}")
        return None


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # Node features
        x = extract_atom_features(mol)
        
        # Edge features
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            
            edge_index.extend([[i, j], [j, i]])
            edge_attr.extend([[bond_type], [bond_type]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    except Exception as e:
        print(f"  ⚠ Graph conversion failed: {e}")
        return None




def process_molecule(row, idx, tokenizer):
    """Process single molecule into 4-view format"""
    smiles = row['SMILES']
    permeability = row['Permeability']
    
    # Skip invalid entries
    if pd.isna(smiles) or pd.isna(permeability):
        return None
    
    try:
        permeability = float(permeability)
    except:
        return None
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # View 1: 3D conformers
    view1 = generate_3d_conformers(smiles, NUM_CONFORMERS)
    if view1 is None:
        return None
    
    # View 2: 1D tokens
    view2 = tokenize_smiles(smiles, tokenizer, MAX_SEQ_LEN)
    if view2 is None:
        return None
    
    # View 3: 2D graph
    view3 = smiles_to_graph(smiles)
    if view3 is None:
        return None
    
    # View 4: 0D descriptors (8 PhysChem features)
    view4 = compute_physchem_features(mol)
    if view4 is None:
        return None
    
    return {
        'view1_3d': view1,
        'view2_1d': view2,
        'view3_2d': view3,
        'view4_0d': view4,
        'target': permeability
    }



def process_wrapper(row, idx, tokenizer, output_dir):
    # 检查文件是否存在，存在则跳过
    out_file = os.path.join(output_dir, f"data_{idx}.pt")
    if os.path.exists(out_file):
        return 1
    try:
        # 不存在则计算
        res = process_molecule(row, idx, tokenizer)
        if res is not None:
            torch.save(res, out_file)
            return 1
        return 0
    except:
        return 0


def main():
    print("=" * 80)
    print("Tetra-View Data Preprocessing")
    print("=" * 80)
    
    # Load CSV
    print(f"\nLoading {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    print(f"Total entries: {len(df)}")
    
    # Filter valid permeability values
    df = df[df['Permeability'].notna()]
    print(f"Valid permeability entries: {len(df)}")
    
    # Load tokenizer
    tokenizer = None
    if HAS_TRANSFORMERS:
        print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        except:
            print("  ⚠ Failed to load tokenizer, using character-level fallback")
    else:
        print("\nUsing simple character-level tokenization")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process molecules
    print(f"\nProcessing molecules...")
    successful=0
    failed=0
    # === Smart Mode (32 cores) ===
    n_jobs = 32
    print(f"Using {n_jobs} cores (Resume enabled)...")
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(process_wrapper)(row, idx, tokenizer, OUTPUT_DIR) 
        for idx, row in tqdm(df.iterrows(), total=len(df), mininterval=1.0)
    )
    successful = sum(results)
    failed = len(results) - successful
    # ============================
    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print("=" * 80)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Success rate: {successful/(successful+failed)*100:.2f}%")
    print(f"\nOutput directory: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
