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

from .f_line_iou import line_iou

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
    distances = (distances * valid_mask).sum(dim=-1) / (lengths + 1e-9)  # (..., Np, Nt)
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

    # cls_pred: (..., Np, C)
    # gt_labels: (..., Nt)
    # We want cost[..., p, t] using gt_labels[..., t]

    # Expand gt_labels to (..., Np, Nt, 1) to gather from pos_cost (..., Np, C)
    # gt_labels has shape (..., Nt).
    # Unsqueeze(-2) -> (..., 1, Nt). Expand -> (..., Np, Nt). Unsqueeze(-1) -> (..., Np, Nt, 1)

    target_shape = gt_labels.shape
    Np = cls_pred.shape[-2]

    # Careful with broadcasting.
    # gt_labels: (B, Nt) or (Nt,)
    if gt_labels.ndim == 1:
        # Single image case: (Nt,)
        # pos_cost: (Np, C)
        return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    else:
        # Batched case: (B, Nt)
        # pos_cost: (B, Np, C)
        gt_labels_expand = gt_labels.unsqueeze(-2).expand(*target_shape[:-1], Np, target_shape[-1])
        indices = gt_labels_expand.unsqueeze(-1)  # (B, Np, Nt, 1)

        # pos_cost needs to be expanded to (B, Np, Nt, C) before gather?
        # No, gather on dim -1 does not require expanding other dims if they match.
        # But here other dims don't match: pos_cost has (B, Np, C), indices has (B, Np, Nt, 1).
        # We need pos_cost to be (B, Np, Nt, C) to gather along C.
        # Wait, gather requires input and index to have same number of dimensions.
        # pos_cost: (B, Np, C). Index: (B, Np, Nt, 1) -> 4 dims vs 3 dims.

        # Correct approach for Batched:
        # We want out[b, i, j] = pos_cost[b, i, gt_labels[b, j]]
        # We can expand pos_cost to (B, Np, Nt, C) then gather.
        pos_cost_exp = pos_cost.unsqueeze(2).expand(*pos_cost.shape[:-1], target_shape[-1], pos_cost.shape[-1])
        neg_cost_exp = neg_cost.unsqueeze(2).expand(*neg_cost.shape[:-1], target_shape[-1], neg_cost.shape[-1])
        # pos_cost_exp: (B, Np, Nt, C)

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

    # 4. Mask Invalid Targets (CRITICAL FIX)
    if valid_mask is not None:
        # valid_mask: (B, Nt)
        # matching_matrix: (B, Np, Nt)
        # Set invalid target columns to 0.0
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            invalid_mask_exp = invalid_mask.unsqueeze(1).expand_as(matching_matrix)
            matching_matrix[invalid_mask_exp] = 0.0

    # 5. Conflict Resolution
    matched_gt_count = matching_matrix.sum(dim=2)  # (B, Np)
    conflict_mask = matched_gt_count > 1  # (B, Np)

    if conflict_mask.any():
        # Mask cost where not matched to avoid selecting non-matched columns
        # (Though unmatched columns are usually high cost, but safe to be sure)
        # Actually we want min cost among the *currently matched* ones.
        # But simplest is to take min cost among all columns for that prior,
        # assuming the 'true' match has lowest cost.
        # A safer way:
        cost_masked = cost.clone()
        # Set non-matched entries to infinity so they are not picked
        cost_masked[matching_matrix == 0] = float('inf')

        cost_argmin = cost_masked.argmin(dim=2)  # (B, Np)

        # Create resolved matches
        resolved = torch.zeros_like(matching_matrix)
        resolved.scatter_(2, cost_argmin.unsqueeze(2), 1.0)

        # Apply only to conflicts
        # We need to expand conflict_mask to (B, Np, Nt) to index matching_matrix?
        # No, boolean indexing matching_matrix[conflict_mask] selects (N_conflicts, Nt) rows.
        # But conflict_mask is (B, Np).

        # Let's use where
        # mask needs to be (B, Np, Nt)
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
                      Only used in batched mode.
    Returns:
        If single: (matched_row_inds, matched_col_inds)
        If batched: matching_matrix (B, num_priors, num_targets)
    """
    is_batch = predictions.ndim == 3

    predictions = predictions.detach().clone()
    predictions[..., 3] *= img_w - 1
    predictions[..., 6:] *= img_w - 1
    targets = targets.detach().clone()

    # ── Distance cost ────────────────────────────────────────────────────────
    dist_scores = distance_cost(predictions, targets, img_w)
    if is_batch:
        max_dist = dist_scores.flatten(1).max(dim=1).values.view(-1, 1, 1)
        dist_scores = 1.0 - (dist_scores / (max_dist + 1e-9)) + 1e-2
    else:
        dist_scores = 1.0 - (dist_scores / (dist_scores.max() + 1e-9)) + 1e-2

    # ── Focal classification cost ────────────────────────────────────────────
    cls_scores = focal_cost(predictions[..., :2], targets[..., 1].long())

    # ── Start-point (y, x) cost ──────────────────────────────────────────────
    pred_start = predictions[..., 2:4].clone()
    target_start = targets[..., 2:4].clone()
    pred_start[..., 0] *= img_h - 1
    target_start[..., 0] *= img_h - 1

    # cdist supports batch
    start_scores = torch.cdist(pred_start, target_start, p=2)
    if is_batch:
        max_start = start_scores.flatten(1).max(dim=1).values.view(-1, 1, 1)
        start_scores = 1.0 - (start_scores / (max_start + 1e-9)) + 1e-2
    else:
        start_scores = 1.0 - (start_scores / (start_scores.max() + 1e-9)) + 1e-2

    # ── Theta cost ───────────────────────────────────────────────────────────
    # (..., Np, 1) vs (..., Nt, 1)
    theta_scores = torch.cdist(predictions[..., 4:5], targets[..., 4:5], p=1) * 180.0
    if is_batch:
        max_theta = theta_scores.flatten(1).max(dim=1).values.view(-1, 1, 1)
        theta_scores = 1.0 - (theta_scores / (max_theta + 1e-9)) + 1e-2
    else:
        theta_scores = 1.0 - (theta_scores / (theta_scores.max() + 1e-9)) + 1e-2

    # ── Combined cost ────────────────────────────────────────────────────────
    cost = -((dist_scores * start_scores * theta_scores) ** 2) * distance_cost_weight + cls_scores * cls_cost_weight

    # ── Masking (Batched only) ───────────────────────────────────────────────
    if is_batch and valid_mask is not None:
        # valid_mask: (B, Nt) -> (B, 1, Nt)
        # cost: (B, Np, Nt)
        # Set cost of invalid targets to infinity
        mask_expanded = valid_mask.unsqueeze(1).expand_as(cost)
        cost[~mask_expanded] = float('inf')

    # ── Pairwise Line-IoU ────────────────────────────────────────────────────
    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)

    if is_batch:
        return dynamic_k_assign_batched(cost, iou, valid_mask)
    else:
        return dynamic_k_assign_single(cost, iou)
