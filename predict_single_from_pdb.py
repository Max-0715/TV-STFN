import argparse
import os
import torch
from rdkit import Chem
from preprocess_tetraview import generate_3d_conformers, tokenize_smiles, smiles_to_graph, compute_physchem_features
from dataset import tetra_view_collate
from model import TetraViewNet


CHECKPOINT_PATH = "archive_unneeded/root_artifacts/best_tetraview_model.pth"


def mol_from_pdb_select_largest(pdb_path):
    # Try to load as RDKit PDB; select largest fragment
    try:
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
        if mol is None:
            with open(pdb_path, 'r') as f:
                block = f.read()
            mol = Chem.MolFromPDBBlock(block, removeHs=False)
        if mol is None:
            return None

        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) == 0:
            return None
        # pick fragment with most atoms
        frag = max(frags, key=lambda m: m.GetNumAtoms())
        smiles = Chem.MolToSmiles(frag, isomericSmiles=True)
        return smiles
    except Exception as e:
        return None


def build_sample_from_smiles(smiles):
    # View1: 3D conformers
    v1 = generate_3d_conformers(smiles)
    if v1 is None:
        raise RuntimeError('3D conformer generation failed')

    # View2: 1D tokens (use fallback tokenizer inside function)
    v2 = tokenize_smiles(smiles, tokenizer=None)
    if v2 is None:
        raise RuntimeError('1D tokenization failed')

    # View3: 2D graph
    v3 = smiles_to_graph(smiles)
    if v3 is None:
        raise RuntimeError('2D graph conversion failed')

    # View4: 0D descriptors
    # compute_physchem_features expects an RDKit Mol
    try:
        mol = Chem.MolFromSmiles(smiles)
        v4 = compute_physchem_features(mol)
    except Exception:
        v4 = compute_physchem_features(None)

    sample = {
        'view1_3d': {
            'coords': v1['coords'],
            'atom_feat': v1['atom_feat']
        },
        'view2_1d': {
            'input_ids': v2['input_ids'],
            'attention_mask': v2['attention_mask']
        },
        'view3_2d': v3,
        'view4_0d': v4,
        'target': 0.0
    }
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, required=True)
    args = parser.parse_args()

    pdb = args.pdb
    if not os.path.exists(pdb):
        print('PDB not found:', pdb)
        return

    print('Reading PDB and extracting largest fragment as SMILES...')
    smiles = mol_from_pdb_select_largest(pdb)
    if smiles is None:
        print('Failed to extract SMILES from PDB')
        return
    print('SMILES:', smiles)

    print('Building 4-view sample...')
    sample = build_sample_from_smiles(smiles)

    print('Collating batch...')
    batch = tetra_view_collate([sample])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    print('Loading model and checkpoint (strict=False)...')
    model = TetraViewNet().to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), CHECKPOINT_PATH)
    if not os.path.exists(ckpt_path):
        print('Checkpoint not found at', ckpt_path)
        return
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # move batch to device
    batch_in = {
        'view1': {
            'coords': batch['view1']['coords'].to(device),
            'atom_features': batch['view1']['atom_features'].to(device),
            'num_atoms': batch['view1']['num_atoms'].to(device),
        },
        'view2': {
            'input_ids': batch['view2']['input_ids'].to(device),
            'attention_mask': batch['view2']['attention_mask'].to(device),
        },
        'view3': batch['view3'].to(device),
        'view4': batch['view4'].to(device)
    }

    with torch.no_grad():
        out = model(batch_in, return_dict=True)

    reg = out['regression'].cpu().numpy().reshape(-1)[0]
    cls_logit = out['classification'].cpu().numpy().reshape(-1)[0]
    cls_prob = float(1.0 / (1.0 + torch.exp(-torch.tensor(cls_logit)).item()))

    print('\n=== Prediction ===')
    print('Regression (permeability):', reg)
    print('Classification logit:', cls_logit)
    print('Classification sigmoid:', cls_prob)
    print('Attention weights (3d,1d,2d,0d):', out['attention_weights'].cpu().numpy().reshape(-1).tolist())


if __name__ == '__main__':
    main()
