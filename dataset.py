import os
import pickle
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

class TetraViewDataset(Dataset):
    """
    Dataset loader for pre-processed Tetra-View data (.pt files).
    """
    def __init__(self, processed_dir):
        """
        Args:
            processed_dir (str): Path to the directory containing .pt files
        """
        self.processed_dir = processed_dir
        
        # Filter out empty, corrupted, or non-finite samples
        all_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
        self.file_names = []
        invalid_files = []

        for f in all_files:
            file_path = os.path.join(processed_dir, f)
            if os.path.getsize(file_path) == 0:
                invalid_files.append(f)
                continue
            try:
                data = torch.load(file_path, weights_only=False)
            except (EOFError, RuntimeError, pickle.UnpicklingError):
                invalid_files.append(f)
                continue
            if not self._is_valid_sample(data):
                invalid_files.append(f)
                continue
            self.file_names.append(f)

        def _sort_key(name):
            stem = name.rsplit('.', 1)[0]
            tail = stem.split('_')[-1]
            return int(tail) if tail.isdigit() else stem

        self.file_names.sort(key=_sort_key)  # Ensure deterministic order
        self.invalid_files = invalid_files
        filtered = len(all_files) - len(self.file_names)
        print(f"Loaded {len(self.file_names)} valid samples (filtered {filtered} invalid/corrupted files)")
        if invalid_files:
            preview = ", ".join(invalid_files[:5])
            print(f"Invalid samples skipped: {len(invalid_files)} (first: {preview})")

    @staticmethod
    def _is_finite_tensor(tensor):
        if not torch.is_tensor(tensor):
            return False
        return torch.isfinite(tensor).all().item()

    def _is_valid_sample(self, data):
        if not isinstance(data, dict):
            return False
        required_keys = ['view1_3d', 'view2_1d', 'view3_2d', 'view4_0d', 'target']
        if not all(k in data for k in required_keys):
            return False

        view1 = data['view1_3d']
        if not isinstance(view1, dict):
            return False
        coords = view1.get('coords')
        atom_feat = view1.get('atom_feat')
        if coords is None or atom_feat is None:
            return False
        if coords.ndim != 3 or coords.numel() == 0:
            return False
        if atom_feat.ndim != 2 or atom_feat.numel() == 0:
            return False
        if not self._is_finite_tensor(coords):
            return False
        if coords.abs().sum().item() == 0.0:
            return False
        if not self._is_finite_tensor(atom_feat.float()):
            return False

        view2 = data['view2_1d']
        if not isinstance(view2, dict):
            return False
        input_ids = view2.get('input_ids')
        attention_mask = view2.get('attention_mask')
        if input_ids is None:
            return False
        if not self._is_finite_tensor(input_ids.float()):
            return False
        if attention_mask is not None and not self._is_finite_tensor(attention_mask.float()):
            return False

        view3 = data['view3_2d']
        if not isinstance(view3, Data):
            return False
        for attr in ['x', 'edge_attr', 'pos']:
            tensor = getattr(view3, attr, None)
            if tensor is not None and not self._is_finite_tensor(tensor.float()):
                return False

        view4 = data['view4_0d']
        if view4 is None or view4.numel() == 0:
            return False
        if not self._is_finite_tensor(view4.float()):
            return False
        if view4.abs().sum().item() == 0.0:
            return False

        target = data['target']
        if isinstance(target, torch.Tensor):
            if not self._is_finite_tensor(target.float()):
                return False
        else:
            try:
                if not torch.isfinite(torch.tensor(float(target))):
                    return False
            except (TypeError, ValueError):
                return False

        return True

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.processed_dir, self.file_names[idx])
        try:
            data = torch.load(file_path, weights_only=False)
            
            # Support new TetraView format (dict from preprocess_tetraview.py)
            if isinstance(data, dict):
                # Ensure all required views are present
                required_keys = ['view1_3d', 'view2_1d', 'view3_2d', 'view4_0d', 'target']
                if all(k in data for k in required_keys):
                    return data
            
            # Fallback for legacy format (List of Data objects)
            if isinstance(data, list) and len(data) > 0:
                print(f"Warning: Legacy data format detected in {self.file_names[idx]}")
                # Legacy conversion logic removed for clarity - assume new format
                return None
                
            return None
            
        except (EOFError, RuntimeError, pickle.UnpicklingError) as e:
            print(f"⚠ Warning: Failed to load {self.file_names[idx]}: {e}")
            return None

def tetra_view_collate(batch):
    """
    Custom collate function to batch data for all 4 views.
    
    Expected batch item structure (dict):
        'view1_3d': {'coords': [n_conf, n_atom, 3], 'atom_feat': [n_atom, 4]}
        'view2_1d': {'input_ids': [seq_len]}
        'view3_2d': PyG Data object
        'view4_0d': Tensor [712]
        'target': float
    """
    
    # Filter out None values from corrupted files
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        raise ValueError("All samples in batch are corrupted!")
    
    # === View I: 3D Dynamic ===
    # Pad atoms to max atoms in batch
    coords_list = [item['view1_3d']['coords'] for item in batch]
    atom_feat_list = [item['view1_3d']['atom_feat'] for item in batch]
    num_atoms = torch.tensor([c.size(1) for c in coords_list])
    max_atoms = num_atoms.max().item()
    
    # Pad coordinates [N_conf, N_atoms, 3]
    # We use a simple padding strategy: pad with 0. 
    # The encoder uses 'num_atoms' to mask, so 0 padding is safe.
    batch_size = len(batch)
    num_confs = coords_list[0].size(0)
    
    padded_coords = torch.zeros(batch_size, num_confs, max_atoms, 3)
    padded_atom_feats = torch.zeros(batch_size, max_atoms, 4)
    
    for i, (coords, feats) in enumerate(zip(coords_list, atom_feat_list)):
        n_at = coords.size(1)
        padded_coords[i, :, :n_at, :] = coords
        padded_atom_feats[i, :n_at, :] = feats

    batch_3d = {
        'coords': padded_coords,
        'atom_features': padded_atom_feats,
        'num_atoms': num_atoms
    }

    # === View II: 1D Semantic ===
    # Pad sequences
    input_ids_list = [item['view2_1d']['input_ids'] for item in batch]
    padded_input_ids = rnn_utils.pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask = (padded_input_ids != 0).float()
    
    batch_1d = {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask
    }

    # === View III: 2D Topology ===
    # Use PyTorch Geometric Batch
    graph_list = [item['view3_2d'] for item in batch]
    batch_2d = Batch.from_data_list(graph_list)

    # === View IV: 0D Physicochemical ===
    # Simple stacking
    batch_0d = torch.stack([item['view4_0d'] for item in batch])

    # === Targets ===
    targets = torch.tensor([item['target'] for item in batch], dtype=torch.float32).unsqueeze(1)

    return {
        'view1': batch_3d,
        'view2': batch_1d,
        'view3': batch_2d,
        'view4': batch_0d,
        'targets': targets
    }
