"""
Vectorized Dynamic Assignment for fCLRNet.

Key optimisation over CLRNet/dynamic_assign.py:
──────────────────────────────────────────────
``dynamic_k_assign`` (original):
    for gt_idx in range(num_gt):           # ← Python loop, O(num_gt) round-trips
        _, pos_idx = torch.topk(...)
        matching_matrix[pos_idx, gt_idx] = 1.0

``dynamic_k_assign`` (this file):
    Replaced entirely with argsort + scatter_, running entirely on GPU.
    Zero Python iterations over num_gt.
    Supports BATCHED execution.

``distance_cost`` (original):
    Uses repeat_interleave + torch.cat which allocates two large copies.
    Replaced with direct broadcasting.

Public API ``assign()`` supports both single-image (backward-compatible) and batched inputs.
"""

import torch

from .f_lane_iou import LaneIoUCost, LaneIoULoss

# ─────────────────────────────────────────────────────────────────────────────
# Cost components
# ─────────────────────────────────────────────────────────────────────────────


def distance_cost(predictions: torch.Tensor, targets: torch.Tensor, img_w: float) -> torch.Tensor:
    """
    Mean absolute x-coordinate distance between every (prior, gt) pair.
    Supports both (Np, D) vs (Nt, D) and (B, Np, D) vs (B, Nt, D).
    """
    pred_xs = predictions[..., 6:]  # (..., Np, Nr)
    target_xs = targets[..., 6:]  # (..., Nt, Nr)

    # valid_mask: (..., 1, Nt, Nr)
    target_broad = target_xs.unsqueeze(-3)
    valid_mask = ((target_broad >= 0) & (target_broad < img_w)).float()

    lengths = valid_mask.sum(dim=-1)  # (..., 1, Nt)
    distances = (pred_xs.unsqueeze(-2) - target_xs.unsqueeze(-3)).abs()  # (..., Np, Nt, Nr)

    # Mask invalid positions via multiplication
    distances = (distances * valid_mask).sum(dim=-1) / (lengths + 1e-5)  # (..., Np, Nt)
    return distances


def focal_cost(
    cls_pred: torch.Tensor, gt_labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-12
) -> torch.Tensor:
    """
    Focal classification cost.
    Supports both (Np, C) vs (Nt,) and (B, Np, C) vs (B, Nt).
    """
    p = cls_pred.sigmoid()
    neg_cost = -(1 - p + eps).log() * (1 - alpha) * p.pow(gamma)
    pos_cost = -(p + eps).log() * alpha * (1 - p).pow(gamma)

    target_shape = gt_labels.shape
    Np = cls_pred.shape[-2]

    if gt_labels.ndim == 1:
        # Single image case: (Nt,)
        return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    else:
        # Batched case: (B, Nt)
        gt_labels_expand = gt_labels.unsqueeze(-2).expand(*target_shape[:-1], Np, target_shape[-1])
        indices = gt_labels_expand.unsqueeze(-1)  # (B, Np, Nt, 1)

        pos_cost_exp = pos_cost.unsqueeze(2).expand(*pos_cost.shape[:-1], target_shape[-1], pos_cost.shape[-1])
        neg_cost_exp = neg_cost.unsqueeze(2).expand(*neg_cost.shape[:-1], target_shape[-1], neg_cost.shape[-1])

        indices = indices.long()
        pos_gather = pos_cost_exp.gather(-1, indices).squeeze(-1)
        neg_gather = neg_cost_exp.gather(-1, indices).squeeze(-1)

        return pos_gather - neg_gather


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised dynamic-k assignment
# ─────────────────────────────────────────────────────────────────────────────


def dynamic_k_assign_single(cost: torch.Tensor, pair_wise_ious: torch.Tensor):
    """Original single-image implementation (kept for compatibility)."""
    num_priors, num_gt = cost.shape
    device = cost.device

    ious = pair_wise_ious.clone()
    ious[ious < 0] = 0.0
    n_candidate_k = min(4, num_priors)
    topk_ious, _ = torch.topk(ious, n_candidate_k, dim=0)
    dynamic_ks = topk_ious.sum(0).int().clamp(min=1)

    sorted_idx = cost.argsort(dim=0)
    ranks = torch.arange(num_priors, device=device).unsqueeze(1).expand(num_priors, num_gt)
    topk_mask = (ranks < dynamic_ks.unsqueeze(0)).float()

    matching_matrix = torch.zeros_like(cost)
    matching_matrix.scatter_(0, sorted_idx, topk_mask)

    matched_gt_count = matching_matrix.sum(dim=1)
    conflict_mask = matched_gt_count > 1

    if conflict_mask.any():
        _, cost_argmin = cost[conflict_mask].min(dim=1)
        matching_matrix[conflict_mask] = 0.0
        matching_matrix[conflict_mask, cost_argmin] = 1.0

    prior_idx = matching_matrix.sum(dim=1).nonzero(as_tuple=False)
    gt_idx = matching_matrix[prior_idx].argmax(dim=-1)
    return prior_idx.flatten(), gt_idx.flatten()


def dynamic_k_assign_batched(cost: torch.Tensor, pair_wise_ious: torch.Tensor, valid_mask: torch.Tensor = None):
    """Batched implementation."""
    B, Np, Nt = cost.shape
    device = cost.device

    # 1. Dynamic K
    ious = pair_wise_ious.clone()
    ious[ious < 0] = 0.0
    n_candidate_k = min(4, Np)
    topk_ious, _ = torch.topk(ious, n_candidate_k, dim=1)  # (B, k, Nt)
    dynamic_ks = topk_ious.sum(1).int().clamp(min=1)  # (B, Nt)

    # 2. Sort
    sorted_idx = cost.argsort(dim=1)  # (B, Np, Nt)

    # 3. TopK Mask
    ranks = torch.arange(Np, device=device).view(1, Np, 1)
    topk_mask = (ranks < dynamic_ks.unsqueeze(1)).float()  # (B, Np, Nt)

    matching_matrix = torch.zeros_like(cost)
    matching_matrix.scatter_(1, sorted_idx, topk_mask)

    # 4. Mask Invalid Targets
    if valid_mask is not None:
        invalid_mask = ~valid_mask  # (B, Nt)
        if invalid_mask.any():
            invalid_mask_exp = invalid_mask.unsqueeze(1).expand_as(matching_matrix)
            matching_matrix[invalid_mask_exp] = 0.0

    # 5. Conflict Resolution
    matched_gt_count = matching_matrix.sum(dim=2)  # (B, Np)
    conflict_mask = matched_gt_count > 1  # (B, Np)

    if conflict_mask.any():
        cost_masked = cost.clone()
        cost_masked[matching_matrix == 0] = float('inf')
        cost_argmin = cost_masked.argmin(dim=2)  # (B, Np)
        resolved = torch.zeros_like(matching_matrix)
        resolved.scatter_(2, cost_argmin.unsqueeze(2), 1.0)
        c_mask_exp = conflict_mask.unsqueeze(2).expand_as(matching_matrix)
        matching_matrix = torch.where(c_mask_exp, resolved, matching_matrix)

    return matching_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Public assign
# ─────────────────────────────────────────────────────────────────────────────


def assign(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    img_w: float,
    img_h: float,
    distance_cost_weight: float = 3.0,
    cls_cost_weight: float = 1.0,
    valid_mask: torch.Tensor = None,
):
    """
    Args:
        predictions : (num_priors, 78) or (B, num_priors, 78)
        targets     : (num_targets, 78) or (B, num_targets, 78)
        valid_mask  : (B, num_targets) boolean mask, True for valid targets.
    """
    is_batch = predictions.ndim == 3

    # Predictions are already relative scale for coords [6:] in CLRerNet's input!
    # Let's clone them
    predictions = predictions.detach().clone().float()
    targets = targets.detach().clone().float()

    if is_batch:
        valid_mask = (targets[..., 1] == 1)  # (B, Nt_actual)

    if is_batch and valid_mask is not None:
        invalid = ~valid_mask  
        if invalid.any():
            inv_exp = invalid.unsqueeze(-1)  
            safe = torch.zeros_like(targets) 
            safe[..., 6:] = -1.0             
            targets = torch.where(inv_exp.expand_as(targets), safe, targets)

    # ── Calculate dynamic k IoU ──────────────────────────────────────────────
    lane_iou_dynamic = LaneIoUCost(use_pred_start_end=False, use_giou=True)
    lane_iou_cost = LaneIoUCost(lane_width=30 / 800, use_pred_start_end=True, use_giou=True)

    pred_xs = predictions[..., 6:]
    target_xs = targets[..., 6:] / (img_w - 1)  # abs -> relative

    iou_dynamick = lane_iou_dynamic(pred_xs, target_xs)

    # ── Calculate clrernet_cost ──────────────────────────────────────────────
    # length and y0 are at indices 5 and 2 respectively
    length = predictions[..., 5]
    y0 = predictions[..., 2]
    start = y0.clamp(min=0, max=1)
    end = (start + length).clamp(min=0, max=1)

    iou_cost = lane_iou_cost(pred_xs, target_xs, start, end)
    
    # Normalise iou_score exactly like original clrernet_cost
    if is_batch:
        if valid_mask is not None:
            vexp = valid_mask.unsqueeze(1).expand_as(iou_cost)
            max_iou_cost = (1 - iou_cost).masked_fill(~vexp, -1e9).flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5).clamp(min=1e-5)
            iou_score = 1 - (1 - iou_cost) / max_iou_cost + 1e-2
        else:
            max_iou_cost = (1 - iou_cost).flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5).clamp(min=1e-5)
            iou_score = 1 - (1 - iou_cost) / max_iou_cost + 1e-2
    else:
        iou_score = 1 - (1 - iou_cost) / torch.max(1 - iou_cost) + 1e-2

    cls_score = focal_cost(predictions[..., :2], targets[..., 1].long())

    # Note: original uses reg_weight=3 (distance_cost_weight is 3.0 by default)
    cost = -iou_score * distance_cost_weight + cls_score

    if is_batch and valid_mask is not None:
        mask_expanded = valid_mask.unsqueeze(1).expand_as(cost)
        cost[~mask_expanded] = float('inf')

    if is_batch:
        return dynamic_k_assign_batched(cost, iou_dynamick, valid_mask)
    else:
        return dynamic_k_assign_single(cost, iou_dynamick)
