"""
Preprocess CycPeptMPDB data into Tetra-View format
Generates 4 views: 3D conformers, 1D tokens, 2D graph, 0D descriptors
Supports Parallel Processing & Resume Capability
"""

import os
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
import multiprocessing
import time

warnings.filterwarnings('ignore')

# Try to import transformers, make it optional
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    print("⚠ Warning: transformers not found. Will use simple character-level tokenization.")
    HAS_TRANSFORMERS = False

# Configuration
# Robust path handling: always relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "CycPeptMPDB_Peptide_PAMPA.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "tetraview_processed")

NUM_CONFORMERS = 10
MAX_SEQ_LEN = 200
TOKENIZER_NAME = 'seyonec/ChemBERTa-zinc-base-v1'

# Atomic constants
ATOM_LIST = [1, 6, 7, 8, 9, 16, 17, 35]  # H, C, N, O, F, S, Cl, Br
HYBRIDIZATION_LIST = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]

# Global tokenizer variable for workers
_global_tokenizer = None

def compute_physchem_features(mol):
    """
    Computes comprehensive physicochemical features (Morgan FP + RDKit Descriptors).
    Input dimension: 512 (Morgan) + ~217 (RDKit) = ~729.
    Matches the improved MSF-CPMP-like feature set.
    """
    if mol is None:
        return torch.zeros(729, dtype=torch.float)
        
    try:
        # Part 1: Morgan Fingerprints (512 bits) - Substructure features
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        fp_arr = np.zeros((512,), dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
        
        # Part 2: RDKit Descriptors (~200 dims) - Macro properties
        desc_list = []
        for _, func in Descriptors.descList:
            try:
                val = func(mol)
                if not np.isfinite(val):
                    val = 0.0
                desc_list.append(val)
            except:
                desc_list.append(0.0)
        
        # Concatenate: [512] + [217] -> [729]
        features = np.concatenate([fp_arr, np.array(desc_list, dtype=np.float32)])
        return torch.tensor(features, dtype=torch.float)
        
    except Exception as e:
        # print(f"  ⚠ PhysChem feature extraction failed: {e}")
        return torch.zeros(729, dtype=torch.float)


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
    Thread-safe version (numThreads=1 for parallel execution).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        num_generated = 0
        
        # Strategy 1: ETKDGv3
        try:
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxIterations = 2000
            params.randomSeed = 42
            params.numThreads = 1  # IMPORTANT: Set to 1 for parallel processing
            params.useSmallRingTorsions = True 
            params.useMacrocycleTorsions = True
            
            num_generated = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
        except Exception:
            pass
        
        # Strategy 2: Standard ETKDG
        if num_generated == 0:
            try:
                params = AllChem.ETKDG()
                params.useRandomCoords = True
                params.maxIterations = 2000
                params.randomSeed = 42
                params.numThreads = 1 # Set to 1
                num_generated = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
            except Exception:
                pass
        
        # Strategy 3: Random coordinates
        if num_generated == 0:
            try:
                num_generated = AllChem.EmbedMultipleConfs(
                    mol, 
                    numConfs=num_confs,
                    useRandomCoords=True,
                    maxAttempts=100,
                    randomSeed=42,
                    numThreads=1
                )
            except Exception:
                pass
        
        # Strategy 4: Fallback
        if num_generated == 0:
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttempts=100)
                num_generated = 1
            except Exception:
                return None
        
        if num_generated == 0:
            return None
        
        # MMFF optimization
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)
        except Exception:
            try:
                for conf_id in range(mol.GetNumConformers()):
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=500)
            except Exception:
                pass
        
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
        
        # Pad
        while len(coords_list) < num_confs:
            coords_list.append(coords_list[0])
        
        coords_tensor = torch.tensor(np.array(coords_list[:num_confs]), dtype=torch.float)
        atom_features = extract_atom_features(mol)
        
        return {
            'coords': coords_tensor,
            'atom_feat': atom_features
        }
        
    except Exception:
        return None


def tokenize_smiles(smiles, tokenizer, max_len=200):
    """Tokenize SMILES"""
    try:
        if tokenizer is not None:
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
            # Fallback
            char_to_idx = {c: i+1 for i, c in enumerate(set(smiles))}
            char_to_idx['<PAD>'] = 0
            
            tokens = [char_to_idx.get(c, 0) for c in smiles[:max_len]]
            tokens += [0] * (max_len - len(tokens))
            
            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.tensor([1 if t > 0 else 0 for t in tokens], dtype=torch.long)
            }
        
    except Exception as e:
        # print(f"  ⚠ Tokenization failed: {e}")
        return None


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        x = extract_atom_features(mol)
        
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
        
    except Exception:
        return None


def process_wrapper(idx, smiles, permeability, output_dir, tokenizer_name, has_transformers):
    """
    Wrapper for parallel execution.
    Target function for joblib workers.
    """
    output_path = os.path.join(output_dir, f'data_{idx}.pt')
    
    # 1. Resume Capability: Skip if exists
    if os.path.exists(output_path):
        return 'skipped'
            
    # Load tokenizer lazily in worker
    global _global_tokenizer
    if _global_tokenizer is None and has_transformers:
        try:
            from transformers import AutoTokenizer
            # Suppress tokenizer warning in workers
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)
            _global_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            pass # Fallback to char level

    # Skip invalid data
    if pd.isna(smiles) or pd.isna(permeability):
        return 'failed_nan'
        
    try:
        target = float(permeability)
    except:
        return 'failed_target'
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'failed_mol'

    # View 1: 3D
    view1 = generate_3d_conformers(smiles, NUM_CONFORMERS)
    if view1 is None:
        return 'failed_3d'
    
    # View 2: 1D
    view2 = tokenize_smiles(smiles, _global_tokenizer, MAX_SEQ_LEN)
    if view2 is None:
        return 'failed_1d'
    
    # View 3: 2D
    view3 = smiles_to_graph(smiles)
    if view3 is None:
        return 'failed_2d'
    
    # View 4: 0D
    view4 = compute_physchem_features(mol)
    if view4 is None:
        return 'failed_0d'
    
    result = {
        'view1_3d': view1,
        'view2_1d': view2,
        'view3_2d': view3,
        'view4_0d': view4,
        'target': target
    }
    
    try:
        torch.save(result, output_path)
        return 'success'
    except Exception as e:
        return f'failed_save_{str(e)}'


def main():
    print("=" * 80)
    print("Tetra-View Data Preprocessing (Parallel Accelerated)")
    print("=" * 80)
    
    # 1. Load Data
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found!")
        return

    df = df[df['Permeability'].notna()]
    total_files = len(df)
    print(f"Total valid entries: {total_files}")
    
    # 2. Setup Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3. Pre-load Tokenizer (Caching)
    if HAS_TRANSFORMERS:
        print(f"Pre-loading tokenizer cache: {TOKENIZER_NAME}")
        try:
            AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        except Exception as e:
            print(f"Warning: Could not pre-load tokenizer: {e}")
        
    # 4. CPU Configuration
    num_cores = multiprocessing.cpu_count()
    # Aggressive parallelization for preprocessing
    n_jobs = max(1, num_cores - 1) 
    print(f"Using {n_jobs} / {num_cores} CPU cores for parallel processing.")
    
    # 5. Parallel Execution
    print(f"Starting parallel processing...")
    print(f"Resume capability enabled: Skipping existing files in '{OUTPUT_DIR}'")
    
    # Prepare arguments list
    tasks = [
        (idx, row['SMILES'], row['Permeability'], OUTPUT_DIR, TOKENIZER_NAME, HAS_TRANSFORMERS) 
        for idx, row in df.iterrows()
    ]
    
    # Run
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_wrapper)(*task) 
        for task in tqdm(tasks, desc="Processing Molecules")
    )
    
    # 6. Summary
    from collections import Counter
    summary = Counter(results)
    
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Total Tasks: {total_files}")
    print(f"Successful (New): {summary['success']}")
    print(f"Skipped (Already Done): {summary['skipped']}")
    
    failures = 0
    print("Failures breakdown:")
    for k, v in summary.items():
        if k not in ['success', 'skipped']:
            print(f"  - {k}: {v}")
            failures += v
            
    success_rate = (summary['success'] + summary['skipped']) / total_files * 100
    print(f"Effective Success Rate: {success_rate:.2f}%")


if __name__ == '__main__':
    # Fix for multiprocessing context
    multiprocessing.freeze_support()
    main()
