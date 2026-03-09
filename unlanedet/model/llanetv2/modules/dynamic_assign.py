
import torch
import torch.nn.functional as F
from .line_iou import line_iou

def dynamic_assign(
    predictions,
    targets,
    img_w,
    img_h,
    distance_cost_weight=3.0,
    cls_cost_weight=1.0,
):
    """
    Computes dynamic matching based on cost, fully vectorized.
    Args:
        predictions (Tensor): predictions, shape: (num_priors, 78)
        targets (Tensor): lane targets, shape: (num_targets, 78)
    return:
        matched_row_inds (Tensor): matched predictions index
        matched_col_inds (Tensor): matched targets index
    """
    # predictions: [N, 78]
    # targets: [M, 78]
    
    # 1. Distance Cost
    # Calculate pairwise distance between all priors and all targets
    # predictions: [N, 72] (coordinates)
    # targets: [M, 72]
    
    pred_lines = predictions[:, 6:] * (img_w - 1)
    tgt_lines = targets[:, 6:]
    
    # [N, 1, 72] - [1, M, 72] -> [N, M, 72]
    diff = torch.abs(pred_lines.unsqueeze(1) - tgt_lines.unsqueeze(0))
    
    # Mask invalid points in target
    # valid_mask: [M, 72] -> [1, M, 72] -> [N, M, 72] (broadcast)
    valid_mask = (tgt_lines >= 0) & (tgt_lines < img_w)
    valid_mask = valid_mask.unsqueeze(0)
    
    # Apply mask
    diff = diff * valid_mask.float()
    
    # Sum over points and normalize by valid length
    # [N, M]
    lengths = valid_mask.sum(dim=-1).float()
    distances_score = diff.sum(dim=-1) / (lengths + 1e-9)
    
    # Normalize distance score
    # min-max normalization per matrix? Or global?
    # Original code: 1 - (distances_score / torch.max(distances_score)) + 1e-2
    # This normalizes by the maximum distance in the entire matrix.
    max_dist = torch.max(distances_score)
    if max_dist > 0:
        distances_score = 1 - (distances_score / max_dist) + 1e-2
    else:
        distances_score = torch.ones_like(distances_score) # If max is 0, all are 0 distance, score 1?
        
    # 2. Classification Cost
    # predictions[:, :2] -> logits
    # targets[:, 1] -> label (always 1 for targets passed here?)
    # Original code: targets[:, 1].long()
    # Focal cost
    cls_pred = predictions[:, :2] # [N, 2]
    gt_labels = targets[:, 1].long() # [M] (should be all 1s if filtering was done)
    
    # focal_cost function from original code
    # We can inline it for speed/clarity or reuse
    # neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    # pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    # cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    
    # But wait, we need pairwise cost?
    # No, classification cost is usually "how well does prior i predict class of target j?"
    # Actually, for CLRNet, it seems we want the cost of "assigning prior i to target j".
    # Prior i has a predicted class score. Target j has class 1.
    # So we want the loss if we assign i to j (which means i should be 1).
    # AND the loss if we DON'T assign i to j (which means i should be 0)?
    # Usually in DETR matching: cost_class = -prob[class_id]
    # Here `focal_cost` returns [N, M]?
    # Let's look at `focal_cost` implementation in `dynamic_assign.py`
    # `cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]`
    # `pos_cost`: [N, 2], `neg_cost`: [N, 2].
    # `gt_labels`: [M].
    # `pos_cost[:, gt_labels]`: Selects columns based on gt_labels.
    # Result: [N, M].
    # Yes, this returns pairwise cost.
    
    alpha = 0.25
    gamma = 2.0
    eps = 1e-12
    cls_pred_sigmoid = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred_sigmoid + eps).log() * (1 - alpha) * cls_pred_sigmoid.pow(gamma)
    pos_cost = -(cls_pred_sigmoid + eps).log() * alpha * (1 - cls_pred_sigmoid).pow(gamma)
    
    # [N, M]
    # We broadcast selection: for each target j (label L_j), we pick column L_j from cost matrices.
    # Since all targets are lane (label 1), we effectively pick column 1.
    # If gt_labels varies, we use advanced indexing.
    cls_score = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    
    # 3. Start XY Cost
    # predictions[:, 2:4], targets[:, 2:4] (start_y, start_x)
    # Scaled by img_h
    pred_start = predictions[:, 2:4].clone()
    pred_start[:, 0] *= (img_h - 1)
    tgt_start = targets[:, 2:4].clone()
    tgt_start[:, 0] *= (img_h - 1)
    
    # Pairwise distance [N, M]
    start_xys_score = torch.cdist(pred_start, tgt_start, p=2)
    max_start = torch.max(start_xys_score)
    if max_start > 0:
        start_xys_score = (1 - start_xys_score / max_start) + 1e-2
    else:
        start_xys_score = torch.ones_like(start_xys_score)

    # 4. Theta Cost
    pred_theta = predictions[:, 4].unsqueeze(1) # [N, 1]
    tgt_theta = targets[:, 4].unsqueeze(1) # [M, 1]
    # Pairwise L1 dist
    theta_score = torch.abs(pred_theta - tgt_theta.T) * 180
    max_theta = torch.max(theta_score)
    if max_theta > 0:
        theta_score = (1 - theta_score / max_theta) + 1e-2
    else:
        theta_score = torch.ones_like(theta_score)

    # Total Cost
    # Using similarity scores (higher is better) to compute cost (lower is better for assignment usually, but here used for dynamic k)
    # The formula: -((dist * start * theta)^2) * w + cls * w
    # Wait, `dynamic_k_assign` takes `cost`.
    # And `dynamic_k_assign` selects `largest=False` (lowest cost).
    # `distances_score` is 1 - norm_dist (so higher is better match).
    # `cls_score` is cost (lower is better match? No, Focal Loss: -log(p). So lower is better).
    # BUT `focal_cost` returns `pos - neg`.
    # If p is high (good match), pos is low, neg is high. pos - neg is low (negative).
    # So lower `cls_score` means better match.
    # `distances_score`: high means close distance.
    # Formula: `-(sim_scores ** 2)`. High sim -> Low (negative) value.
    # So we want to MINIMIZE this cost.
    
    cost = (
        -((distances_score * start_xys_score * theta_score) ** 2) * distance_cost_weight 
        + cls_score * cls_cost_weight
    )
    
    # IoU for Dynamic K
    # [N, M]
    iou = line_iou(pred_lines, tgt_lines, img_w, aligned=False)
    
    # Dynamic K Assignment
    # iou matrix [N, M]
    # Top K IoUs determine K for each GT
    
    # Clamp IoU
    iou = torch.clamp(iou, min=0.0)
    
    n_candidate_k = 4
    # TopK along dim 0 (priors)
    topk_ious, _ = torch.topk(iou, min(n_candidate_k, iou.shape[0]), dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    
    # Assign
    num_gt = targets.shape[0]
    num_priors = predictions.shape[0]
    
    # Create matching matrix [N, M]
    # For each GT, pick top K priors with lowest cost
    
    # Vectorized TopK selection for each column
    # We can't fully vectorize if K varies per column easily without a loop or masks.
    # But K is small (1-4).
    # We can iterate over GTs (M is small, usually 4-10 lanes).
    # M is much smaller than N (192).
    # So loop over M is fine.
    
    matching_matrix = torch.zeros_like(cost)
    for gt_idx in range(num_gt):
        k = dynamic_ks[gt_idx].item()
        _, pos_idx = torch.topk(cost[:, gt_idx], k=k, largest=False)
        matching_matrix[pos_idx, gt_idx] = 1.0
        
    # Handle duplicates (one prior matched to multiple GTs)
    # Sum over GTs
    matched_gt = matching_matrix.sum(1) # [N]
    duplicate_mask = matched_gt > 1
    if duplicate_mask.any():
        # Priors matched to >1 GTs
        # For these priors, find which GT has min cost
        # cost[duplicate_mask, :] -> [D, M]
        # We only care about columns where matching_matrix is 1
        # Mask cost with matching_matrix (set unassigned to inf)
        masked_cost = cost.clone()
        masked_cost[matching_matrix == 0] = float('inf')
        
        # Argmin over GTs
        best_gt_idx = masked_cost[duplicate_mask].argmin(dim=1)
        
        # Zero out rows for duplicates
        matching_matrix[duplicate_mask] = 0.0
        # Set best match to 1
        matching_matrix[duplicate_mask, best_gt_idx] = 1.0

    # Get indices
    # nonzero returns [n_matches, 2] (row, col) -> (prior_idx, gt_idx)
    matches = matching_matrix.nonzero()
    prior_idx = matches[:, 0]
    gt_idx = matches[:, 1]
    
    return prior_idx, gt_idx
