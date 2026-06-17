import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders import (
    DynamicConformerEncoder,
    SemanticEncoder,
    GraphTopologyEncoder,
    PhysicochemicalEncoder
)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x

class TetraViewNet(nn.Module):
    """
    TV-STFN: Tetra-View Spatio-Temporal Fusion Network
    Integrates 3D, 1D, 2D, and 0D representations for peptide property prediction.
    """
    def __init__(self,
                 dim_3d=256,
                 dim_1d=256,
                 dim_2d=256,
                 dim_0d=256,
                 raw_0d_dim=729,
                 fusion_hidden_dim=512,
                 dropout=0.1,
                 modality_prior_0d=1.2,
                 cls_skip_weight=0.30,
                 gating_temperature=0.85,
                 modality_dropout_prob=0.15):
        super().__init__()
        self.cls_skip_weight = cls_skip_weight
        self.gating_temperature = gating_temperature
        self.modality_dropout_prob = modality_dropout_prob
        
        # --- Encoders ---
        # View I: 3D Dynamic
        self.encoder_3d = DynamicConformerEncoder(output_dim=dim_3d)
        
        # View II: 1D Semantic
        self.encoder_1d = SemanticEncoder(output_dim=dim_1d)
        
        # View III: 2D Topology
        self.encoder_2d = GraphTopologyEncoder(output_dim=dim_2d)
        
        # View IV: 0D Physicochemical
        self.encoder_0d = PhysicochemicalEncoder(input_dim=raw_0d_dim, output_dim=dim_0d)
        self.raw_0d_expert = nn.Sequential(
            nn.LayerNorm(raw_0d_dim),
            nn.Linear(raw_0d_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            ResidualMLPBlock(fusion_hidden_dim, dropout=dropout)
        )
        
        # --- Fusion & Prediction ---
        # 优化：引入投影层和注意力机制，实现特征的自适应加权融合
        hidden_dim = fusion_hidden_dim
        
        # Projections to align dimensions
        self.proj_3d = nn.Sequential(nn.Linear(dim_3d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.proj_1d = nn.Sequential(nn.Linear(dim_1d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.proj_2d = nn.Sequential(nn.Linear(dim_2d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.proj_0d = nn.Sequential(nn.Linear(dim_0d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.proj_0d_raw = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.gate_0d = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.refine_3d = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.refine_1d = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.refine_2d = ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.refine_0d = ResidualMLPBlock(hidden_dim, dropout=dropout)

        self.view_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 4)
        )
        self.view_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.modality_prior = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, modality_prior_0d], dtype=torch.float32))

        self.shared_input_proj = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cross_modal_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.shared_head = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.LayerNorm(hidden_dim)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.classification_input_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classification_head = nn.Sequential(
            ResidualMLPBlock(hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, batch_data, return_dict=False):
        """
        Args:
            batch_data (dict): Output from TetraViewDataset.collate_fn
        """
        # Parallel Encoding
        conformer_attention = None
        if return_dict:
            feat_3d, conformer_attention = self.encoder_3d(batch_data['view1'], return_attention=True)
        else:
            feat_3d = self.encoder_3d(batch_data['view1'])
        feat_1d = self.encoder_1d(batch_data['view2'])
        feat_2d = self.encoder_2d(batch_data['view3'])
        feat_0d = self.encoder_0d(batch_data['view4'])
        feat_0d_raw = self.raw_0d_expert(batch_data['view4'])
        
        # 1. Projection with LayerNorm & GELU
        p_3d = self.proj_3d(feat_3d)
        p_1d = self.proj_1d(feat_1d)
        p_2d = self.proj_2d(feat_2d)
        p_0d = self.proj_0d(feat_0d)
        p_0d_raw = self.proj_0d_raw(feat_0d_raw)
        gate_0d = self.gate_0d(torch.cat([p_0d, p_0d_raw], dim=-1))
        p_0d_fused = gate_0d * p_0d + (1.0 - gate_0d) * p_0d_raw

        p_3d = self.refine_3d(p_3d)
        p_1d = self.refine_1d(p_1d)
        p_2d = self.refine_2d(p_2d)
        p_0d_fused = self.refine_0d(p_0d_fused)

        stacked_features = torch.stack([p_3d, p_1d, p_2d, p_0d_fused], dim=1)

        # Training-time view dropout encourages true cross-modal collaboration.
        if self.training and self.modality_dropout_prob > 0.0:
            bsz = stacked_features.size(0)
            keep = torch.rand((bsz, 4), device=stacked_features.device) > self.modality_dropout_prob
            # Ensure at least one modality remains for every sample.
            none_kept = keep.sum(dim=1) == 0
            if none_kept.any():
                keep[none_kept, torch.randint(0, 4, (none_kept.sum().item(),), device=keep.device)] = True
            stacked_features = stacked_features * keep.unsqueeze(-1).float()

        flat_features = torch.cat([p_3d, p_1d, p_2d, p_0d_fused], dim=-1)
        gate_logits = self.view_gate(flat_features)
        conf_logits = self.view_confidence(stacked_features.reshape(-1, stacked_features.size(-1))).view(stacked_features.size(0), 4)
        moe_logits = (gate_logits + conf_logits + self.modality_prior.unsqueeze(0)) / max(self.gating_temperature, 1e-4)
        attn_weights = F.softmax(moe_logits, dim=1).unsqueeze(-1)
        modality_entropy = -(attn_weights.squeeze(-1) * torch.log(attn_weights.squeeze(-1).clamp_min(1e-8))).sum(dim=1).mean()

        fused_features = torch.sum(stacked_features * attn_weights, dim=1)
        pooled_mean = stacked_features.mean(dim=1)
        pooled_max = stacked_features.max(dim=1).values
        pair_3d_1d = p_3d * p_1d
        pair_3d_2d = p_3d * p_2d
        pair_3d_0d = p_3d * p_0d_fused
        cross_modal = self.cross_modal_proj(torch.cat([pair_3d_1d, pair_3d_2d, pair_3d_0d], dim=-1))

        shared_input = torch.cat([fused_features, pooled_mean, pooled_max, p_0d_fused, cross_modal], dim=-1)
        shared_features = self.shared_head(self.shared_input_proj(shared_input))

        regression = self.regression_head(shared_features + 0.10 * fused_features)
        cls_context = torch.cat([
            shared_features,
            p_0d_fused,
            shared_features * p_0d_fused
        ], dim=-1)
        cls_features = self.classification_input_proj(cls_context)
        classification = self.classification_head(cls_features + self.cls_skip_weight * p_0d_fused)
        
        if return_dict:
            out = {
                'regression': regression,
                'classification': classification,
                'attention_weights': attn_weights.squeeze(-1),
                'modality_entropy': modality_entropy,
                'fused_features': fused_features,
                'shared_features': shared_features,
                'view_features': {
                    'view3d': p_3d,
                    'view1d': p_1d,
                    'view2d': p_2d,
                    'view0d': p_0d_fused
                }
            }
            if conformer_attention is not None:
                out['conformer_attention'] = conformer_attention
            return out
        return regression
