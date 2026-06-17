import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    """
    Composite Loss Function for TV-STFN.
    Combines Focal Loss (for class imbalance/Recall), Margin Ranking Loss (for ordering),
    and MSE Loss (for regression accuracy).
    
    Total_Loss = λ1 * Focal + λ2 * Rank + λ3 * MSE
    """
    def __init__(self, 
                 lambda_focal=1.0, 
                 lambda_rank=1.0, 
                 lambda_mse=1.0,
                 focal_alpha=0.25, 
                 focal_gamma=2.0, 
                 margin=0.1):
        super().__init__()
        self.lambda_focal = lambda_focal
        self.lambda_rank = lambda_rank
        self.lambda_mse = lambda_mse
        
        self.focal_edge_alpha = focal_alpha  # Class weight for positive (minority) samples
        self.focal_gamma = focal_gamma
        
        self.mse_criterion = nn.MSELoss() # Or SmoothL1Loss if preferred
        self.rank_criterion = nn.MarginRankingLoss(margin=margin)

    def binary_focal_loss(self, preds, targets):
        """
        Custom Binary Focal Loss implementation
        preds: Raw logits (before sigmoid)
        targets: Binary labels (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss) # prob of correct class
        focal_loss = (1 - pt) ** self.focal_gamma * bce_loss
        
        # Apply weighting
        if self.focal_edge_alpha is not None:
            alpha_t = self.focal_edge_alpha * targets + (1 - self.focal_edge_alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

    def forward(self, preds, targets, classification_threshold=None):
        """
        Args:
            preds: [batch_size, 1] predicted values (regression scale)
            targets: [batch_size, 1] ground truth values (logPe)
            classification_threshold (float, optional): Threshold to binarize regression targets for focal loss.
                                                        If None, median of batch or global constant is used.
        """
        # 1. MSE Loss (Regression)
        mse_loss = self.mse_criterion(preds, targets)
        
        # 2. Ranking Loss (Structure-Activity Relationship)
        batch_size = preds.size(0)
        rank_loss = torch.tensor(0.0, device=preds.device)
        if batch_size > 1:
            pred_diff = preds - preds.t()
            target_diff = targets - targets.t()
            target_sign = torch.sign(target_diff)
            mask = target_sign != 0
            
            if mask.sum() > 0:
                preds_i = preds.expand(batch_size, batch_size)[mask]
                preds_j = preds.t().expand(batch_size, batch_size)[mask]
                targets_rel = target_sign[mask]
                rank_loss = self.rank_criterion(preds_i, preds_j, targets_rel)

        # 3. Focal Loss (Classification, Auxiliary)
        # We need to construct binary labels from continuous targets
        if classification_threshold is None:
             # Default to -6.0 or global median if not provided; here using a dynamic batch median for stability or fixed value
             classification_threshold = -6.0 

        binary_targets = (targets >= classification_threshold).float()
        
        # Since preds are regression values (e.g. -5.0, -8.0), we need to map them to logits for BCE
        # A simple way is to treat the regression value itself as a logit? 
        # Or more robustly, learn a separate classifier head. 
        # Here, we assume the regression output strongly correlates with probability "logit" 
        # BUT regression outputs are ~ -10 to -4. This is not a standard logit centered at 0.
        # Strategy: Use a shifted version of preds as logits, e.g., (preds - threshold)
        classify_logits = preds - classification_threshold
        # Prevent extreme logits causing overflow/NaN in BCE/focal computations
        classify_logits = torch.clamp(classify_logits, min=-10.0, max=10.0)
        focal_loss = self.binary_focal_loss(classify_logits, binary_targets)

        # Combined Loss
        total_loss = (self.lambda_mse * mse_loss + 
                      self.lambda_rank * rank_loss + 
                      self.lambda_focal * focal_loss)
        
        return total_loss, mse_loss.item(), rank_loss.item(), focal_loss.item()
