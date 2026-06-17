"""
Tetra-View Feature Encoders for TV-STFN
Four parallel branches for multi-modal cyclic peptide representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, radius_graph


# Constants for atomic features (defined here to avoid circular import)
ATOM_LIST = [1, 6, 7, 8, 9, 16, 17, 35]  # H, C, N, O, F, S, Cl, Br
HYBRIDIZATION_LIST = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"]


# ============================================================================
# Shared Components
# ============================================================================

class GaussianSmearing(nn.Module):
    """Convert distances to RBF features (for SchNet)"""
    def __init__(self, start=0.0, end=10.0, num_gaussians=50):
        super().__init__()
        self.start = start
        self.end = end
        self.num_gaussians = num_gaussians
        offset = torch.linspace(start, end, num_gaussians)
        self.register_buffer('offset', offset)
        coeff = -0.5 / ((end - start) / num_gaussians) ** 2
        self.register_buffer('coeff', torch.tensor(coeff))

    def forward(self, dist):
        """
        Args:
            dist: [num_edges] distances
        Returns:
            rbf: [num_edges, num_gaussians] RBF features
        """
        dist = dist.unsqueeze(-1) - self.offset.unsqueeze(0)
        return torch.exp(self.coeff * dist ** 2)


class InteractionBlock(nn.Module):
    """SchNet-like interaction block for 3D molecular modeling"""
    def __init__(self, hidden_channels, num_gaussians=50, cutoff=10.0):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        
        self.rbf = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        # Continuous filter network
        self.filter_net = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Node update network
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, x, pos, edge_index):
        """
        Args:
            x: [num_nodes, hidden_channels] node features
            pos: [num_nodes, 3] node positions
            edge_index: [2, num_edges] edge indices
        Returns:
            x: [num_nodes, hidden_channels] updated node features
        """
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1)
        dist = torch.clamp(dist, max=self.cutoff)
        
        # RBF expansion
        rbf_feat = self.rbf(dist)
        
        # Filter network
        filter_out = self.filter_net(rbf_feat)
        
        # Aggregate neighborhood information
        out = torch.zeros_like(x)
        out.index_add_(0, row, filter_out)
        out.index_add_(0, col, filter_out)
        
        # Node update with residual connection
        out = self.update_net(out)
        return x + out


class GatedAttention(nn.Module):
    """Gated attention for Multiple Instance Learning (MIL)"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: [num_instances, hidden_dim] instance features
        Returns:
            aggregated: [hidden_dim] aggregated feature
            attn_weights: [num_instances] attention weights
        """
        attn_scores = self.attention(x).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=0)
        aggregated = torch.sum(attn_weights.unsqueeze(-1) * x, dim=0)
        return aggregated, attn_weights

def get_physchem_features(mol):
    """
    Computes 8 key physicochemical features using RDKit.
    Normalization should be done at the dataset level (StandardScaler).
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object
        
    Returns:
        list: [LogP, TPSA, MW, H-Donors, H-Acceptors, RingCount, RotatableBonds, HeavyAtomCount]
    """
    if mol is None:
        return [0.0] * 8
        
    from rdkit.Chem import Descriptors, rdMolDescriptors
    
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
    return features


# ============================================================================
# Encoder 1: DynamicConformerEncoder (View I - 3D Dynamic)
# ============================================================================

class DynamicConformerEncoder(nn.Module):
    """
    SchNet + MIL for conformational ensemble encoding.
    Modified from DCEA-NET: removes regression head, outputs 256-dim features.
    
    Architecture:
        Input: Multiple 3D conformers with atom coordinates
        -> Atomic Embeddings
        -> SchNet Interaction Blocks (3D geometry learning)
        -> Conformer-level pooling
        -> MIL Gated Attention (conformer aggregation)
        -> Output: 256-dim feature vector
    """
    
    def __init__(self, 
                 hidden_dim=128, 
                 output_dim=256,
                 num_interaction_layers=3, 
                 cutoff=10.0,
                 dropout=0.1,
                 reuse_knn_graph=True):
        super().__init__()
        self.reuse_knn_graph = reuse_knn_graph
        
        # Atomic feature embeddings (same as DCEA-NET)
        self.atom_emb = nn.Embedding(
            num_embeddings=len(ATOM_LIST) + 1,
            embedding_dim=64
        )
        self.hybrid_emb = nn.Embedding(
            num_embeddings=len(HYBRIDIZATION_LIST),
            embedding_dim=32
        )
        
        # Input feature dimension: atom(64) + hybrid(32) + aromatic(1) + ring(1) + pos(3) = 101
        self.node_feat_dim = 64 + 32 + 2 + 3
        self.lin_input = nn.Linear(self.node_feat_dim, hidden_dim)
        
        # SchNet-like 3D interaction blocks
        self.interaction_blocks = nn.ModuleList([
            InteractionBlock(hidden_channels=hidden_dim, num_gaussians=50, cutoff=cutoff)
            for _ in range(num_interaction_layers)
        ])
        
        # MIL gated attention for conformer aggregation
        self.attention = GatedAttention(hidden_dim, hidden_dim, dropout)
        
        # Output projection (removed regression head from DCEA-NET)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, conformers_3d_batch, return_attention=False):
        """
        Args:
            conformers_3d_batch: dict with keys:
                'coords': [batch, num_confs, max_atoms, 3]
                'atom_features': [batch, max_atoms, 4] - (atomic_idx, hybrid_idx, aromatic, ring)
                'num_atoms': [batch] - actual number of atoms per molecule
        
        Returns:
            features: [batch, 256] - aggregated conformer features
        """
        batch_size = conformers_3d_batch['coords'].size(0)
        num_confs = conformers_3d_batch['coords'].size(1)
        max_atoms = conformers_3d_batch['coords'].size(2)
        
        batch_features = []
        batch_attn = []
        
        for mol_idx in range(batch_size):
            # Get data for current molecule
            coords = conformers_3d_batch['coords'][mol_idx]  # [num_confs, max_atoms, 3]
            atom_feat = conformers_3d_batch['atom_features'][mol_idx]  # [max_atoms, 4]
            num_atoms = conformers_3d_batch['num_atoms'][mol_idx].item()
            
            # Trim padding
            coords = coords[:, :num_atoms, :]  # [num_confs, num_atoms, 3]
            atom_feat = atom_feat[:num_atoms, :]  # [num_atoms, 4]
            
            conf_features = []
            edge_index_cached = None
            
            # Process each conformer
            for conf_idx in range(num_confs):
                pos = coords[conf_idx]  # [num_atoms, 3]
                
                # Extract and embed atomic features
                atomic_idx = atom_feat[:, 0].long()
                hybrid_idx = atom_feat[:, 1].long()
                bool_feat = atom_feat[:, 2:].float()  # [num_atoms, 2]
                
                atom_embed = self.atom_emb(atomic_idx)  # [num_atoms, 64]
                hybrid_embed = self.hybrid_emb(hybrid_idx)  # [num_atoms, 32]
                
                # Concatenate all features
                x = torch.cat([atom_embed, hybrid_embed, bool_feat, pos], dim=1)  # [num_atoms, 101]
                x = self.lin_input(x)  # [num_atoms, hidden_dim]
                
                if self.reuse_knn_graph and edge_index_cached is not None:
                    edge_index = edge_index_cached
                else:
                    # Build k-NN graph for 3D interactions (more memory efficient than radius)
                    # Use simple k=8 nearest neighbors
                    k = min(8, num_atoms - 1)
                    # Calculate pairwise distances
                    dist_matrix = torch.cdist(pos, pos)  # [num_atoms, num_atoms]
                    # Get k nearest neighbors for each atom (excluding self)
                    _, knn_idx = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
                    knn_idx = knn_idx[:, 1:]  # Remove self (距离=0)
                    
                    # Build edge_index
                    src = torch.arange(num_atoms, device=pos.device).unsqueeze(1).expand(-1, k).flatten()
                    dst = knn_idx.flatten()
                    edge_index = torch.stack([src, dst], dim=0)
                    if self.reuse_knn_graph:
                        edge_index_cached = edge_index
                
                # Apply SchNet interaction blocks
                for block in self.interaction_blocks:
                    x = block(x, pos, edge_index)
                
                # Conformer-level pooling
                conf_feat = x.mean(dim=0)  # [hidden_dim] - global mean pooling
                conf_features.append(conf_feat)
            
            # Stack conformer features
            conf_features = torch.stack(conf_features)  # [num_confs, hidden_dim]
            
            # MIL attention aggregation across conformers
            mol_feat, conf_attn = self.attention(conf_features)  # [hidden_dim]
            
            # Project to output dimension
            mol_feat = self.output_projection(mol_feat)  # [output_dim]
            batch_features.append(mol_feat)
            batch_attn.append(conf_attn)

        features = torch.stack(batch_features)  # [batch, 256]
        if return_attention:
            return features, torch.stack(batch_attn)  # [batch, num_confs]
        return features


# ============================================================================
# Encoder 2: SemanticEncoder (View II - 1D Semantic)
# ============================================================================

class EDCN_Block(nn.Module):
    """
    Encoder-Decoder Convolutional Network (EDCN) Block
    Captures local and long-range dependencies in sequential data
    """
    def __init__(self, hidden_dim, kernel_sizes=[3, 5, 7], dropout=0.1):
        super().__init__()
        
        # Multi-scale 1D convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * len(kernel_sizes), hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            out: [batch, seq_len, hidden_dim]
        """
        residual = x
        
        # Transpose for 1D convolution: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Multi-scale convolutions
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        
        # Concatenate and fuse
        x = torch.cat(conv_outs, dim=1)  # [batch, hidden_dim * len(kernel_sizes), seq_len]
        x = self.fusion(x)  # [batch, hidden_dim, seq_len]
        
        # Transpose back
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        # Residual connection + LayerNorm
        x = self.layer_norm(x + residual)
        
        return x


class SemanticEncoder(nn.Module):
    """
    Molecular Language Model + EDCN for sequence semantic encoding.
    
    Architecture:
        Input: Tokenized SMILES (token IDs)
        -> Embedding Layer (or pre-trained MolLM)
        -> EDCN Blocks (multi-scale 1D convolutions)
        -> **Global Average Pooling** (CRITICAL: not max pooling!)
        -> Linear projection to 256-dim
    
    Constraint: MUST use Global Average Pooling (not Max Pooling)
    """
    
    def __init__(self,
                 vocab_size=2000,
                 embedding_dim=768,
                 hidden_dim=512,
                 output_dim=256,
                 num_edcn_blocks=2,
                 dropout=0.1,
                 pretrained_embeddings=None):
        super().__init__()
        
        # Token embeddings (can be replaced with pre-trained MolLM)
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Projection to hidden dimension
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # EDCN blocks for local/long-range dependency modeling
        self.edcn_blocks = nn.ModuleList([
            EDCN_Block(hidden_dim, kernel_sizes=[3, 5, 7], dropout=dropout)
            for _ in range(num_edcn_blocks)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, tokens_1d_batch):
        """
        Args:
            tokens_1d_batch: dict with keys:
                'input_ids': [batch, max_seq_len]
                'attention_mask': [batch, max_seq_len]
        
        Returns:
            features: [batch, 256] - sequence semantic features
        """
        input_ids = tokens_1d_batch['input_ids']  # [batch, max_seq_len]
        attention_mask = tokens_1d_batch['attention_mask']  # [batch, max_seq_len]
        
        # Embedding
        x = self.embedding(input_ids)  # [batch, max_seq_len, embedding_dim]
        x = self.input_projection(x)  # [batch, max_seq_len, hidden_dim]
        
        # EDCN blocks
        for block in self.edcn_blocks:
            x = block(x)  # [batch, max_seq_len, hidden_dim]
        
        # ========== CRITICAL: Global Average Pooling (NOT Max Pooling!) ==========
        # Apply attention mask to ignore padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, max_seq_len, 1]
        x_masked = x * mask_expanded  # Zero out padding positions
        
        # Sum over sequence length and divide by number of valid tokens
        seq_sum = x_masked.sum(dim=1)  # [batch, hidden_dim]
        seq_len = mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1] - avoid division by zero
        x_pooled = seq_sum / seq_len  # [batch, hidden_dim] - Average pooling
        # =========================================================================
        
        # Output projection
        features = self.output_projection(x_pooled)  # [batch, 256]
        
        return features


# ============================================================================
# Encoder 3: GraphTopologyEncoder (View III - 2D Topology)
# ============================================================================

class GraphTopologyEncoder(nn.Module):
    """
    Graph Neural Network for 2D molecular topology encoding.
    
    Architecture:
        Input: 2D molecular graph (nodes: atoms, edges: bonds)
        -> 3-layer GCN or GAT
        -> Global Mean Pooling
        -> Linear projection to 256-dim
    """
    
    def __init__(self,
                 node_feat_dim=4,  # (atomic_idx, hybrid_idx, aromatic, ring)
                 edge_feat_dim=1,  # bond type
                 hidden_dim=128,
                 output_dim=256,
                 num_layers=3,
                 gnn_type='gcn',  # 'gcn' or 'gat'
                 dropout=0.1):
        super().__init__()
        
        # Atomic feature embeddings
        self.atom_emb = nn.Embedding(
            num_embeddings=len(ATOM_LIST) + 1,
            embedding_dim=64
        )
        self.hybrid_emb = nn.Embedding(
            num_embeddings=len(HYBRIDIZATION_LIST),
            embedding_dim=32
        )
        
        # Input projection: (64 + 32 + 2) = 98
        input_feat_dim = 64 + 32 + 2
        self.input_projection = nn.Linear(input_feat_dim, hidden_dim)
        
        # GNN layers
        self.gnn_type = gnn_type
        if gnn_type == 'gcn':
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        elif gnn_type == 'gat':
            self.convs = nn.ModuleList([
                GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        self.batch_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, graph_2d_batch):
        """
        Args:
            graph_2d_batch: PyTorch Geometric Batch object with:
                x: [total_nodes, 4] - node features
                edge_index: [2, total_edges] - edge indices
                edge_attr: [total_edges, 1] - edge features
                batch: [total_nodes] - batch assignment
        
        Returns:
            features: [batch_size, 256] - graph topology features
        """
        x = graph_2d_batch.x  # [total_nodes, 4]
        edge_index = graph_2d_batch.edge_index  # [2, total_edges]
        batch = graph_2d_batch.batch  # [total_nodes]
        
        # Extract and embed atomic features
        atomic_idx = x[:, 0].long()
        hybrid_idx = x[:, 1].long()
        bool_feat = x[:, 2:].float()  # [total_nodes, 2]
        
        atom_embed = self.atom_emb(atomic_idx)  # [total_nodes, 64]
        hybrid_embed = self.hybrid_emb(hybrid_idx)  # [total_nodes, 32]
        
        # Concatenate features
        x = torch.cat([atom_embed, hybrid_embed, bool_feat], dim=1)  # [total_nodes, 98]
        x = self.input_projection(x)  # [total_nodes, hidden_dim]
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x_residual = x
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection (if dimensions match)
            if x.size(-1) == x_residual.size(-1):
                x = x + x_residual
        
        # Global mean pooling
        x_pooled = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Output projection
        features = self.output_projection(x_pooled)  # [batch_size, 256]
        
        return features


# ============================================================================
# Encoder 4: PhysicochemicalEncoder (View IV - 0D Physicochemical)
# ============================================================================

class PhysicochemicalEncoder(nn.Module):
    """
    Multi-layer Perceptron for physicochemical property encoding.
    
    Architecture:
        Input: Morgan Fingerprints (512-bit) + RDKit Descriptors (200-dim) = 712-dim
        -> Linear(712, 512) -> ReLU -> Dropout
        -> Linear(512, 256)
    """
    
    def __init__(self,
                 input_dim=712,  # 512 (Morgan) + 200 (RDKit)
                 hidden_dim=512,
                 output_dim=256,
                 dropout=0.1):
        super().__init__()
        
        # Add Input Normalization (LayerNorm) to handle raw descriptors magnitude differences
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),  # Online normalization for raw features
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, features_0d_batch):
        """
        Args:
            features_0d_batch: [batch, 712] - physicochemical features
        
        Returns:
            features: [batch, 256] - encoded features
        """
        return self.network(features_0d_batch)


# ============================================================================
# Test Code
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("Testing Tetra-View Encoders")
    print("=" * 80)
    
    batch_size = 4
    num_confs = 10
    max_atoms = 50
    max_seq_len = 100
    
    # ========== Test View I: DynamicConformerEncoder ==========
    print("\n[View I] DynamicConformerEncoder (3D)")
    encoder_3d = DynamicConformerEncoder(hidden_dim=128, output_dim=256)
    
    dummy_3d = {
        'coords': torch.randn(batch_size, num_confs, max_atoms, 3),
        'atom_features': torch.randint(0, 4, (batch_size, max_atoms, 4)),
        'num_atoms': torch.tensor([max_atoms] * batch_size)
    }
    
    out_3d = encoder_3d(dummy_3d)
    print(f"  Output shape: {out_3d.shape}")  # [batch, 256]
    assert out_3d.shape == (batch_size, 256)
    print("  ✓ Passed")
    
    # ========== Test View II: SemanticEncoder ==========
    print("\n[View II] SemanticEncoder (1D)")
    encoder_1d = SemanticEncoder(
        vocab_size=2000,
        embedding_dim=768,
        hidden_dim=512,
        output_dim=256
    )
    
    dummy_1d = {
        'input_ids': torch.randint(0, 2000, (batch_size, max_seq_len)),
        'attention_mask': torch.ones(batch_size, max_seq_len)
    }
    
    out_1d = encoder_1d(dummy_1d)
    print(f"  Output shape: {out_1d.shape}")  # [batch, 256]
    assert out_1d.shape == (batch_size, 256)
    print("  ✓ Passed (Using Average Pooling)")
    
    # ========== Test View III: GraphTopologyEncoder ==========
    print("\n[View III] GraphTopologyEncoder (2D)")
    encoder_2d = GraphTopologyEncoder(
        hidden_dim=128,
        output_dim=256,
        num_layers=3,
        gnn_type='gcn'
    )
    
    from torch_geometric.data import Data, Batch
    
    # Create dummy graph batch
    graphs = []
    for _ in range(batch_size):
        num_nodes = torch.randint(10, 30, (1,)).item()
        x = torch.randint(0, 4, (num_nodes, 4))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_attr = torch.randn(num_nodes * 2, 1)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    
    graph_batch = Batch.from_data_list(graphs)
    out_2d = encoder_2d(graph_batch)
    print(f"  Output shape: {out_2d.shape}")  # [batch, 256]
    assert out_2d.shape == (batch_size, 256)
    print("  ✓ Passed")
    
    # ========== Test View IV: PhysicochemicalEncoder ==========
    print("\n[View IV] PhysicochemicalEncoder (0D)")
    encoder_0d = PhysicochemicalEncoder(input_dim=712, output_dim=256)
    
    dummy_0d = torch.randn(batch_size, 712)
    out_0d = encoder_0d(dummy_0d)
    print(f"  Output shape: {out_0d.shape}")  # [batch, 256]
    assert out_0d.shape == (batch_size, 256)
    print("  ✓ Passed")
    
    print("\n" + "=" * 80)
    print("All encoders passed! Output dimensions: 256")
    print("=" * 80)
