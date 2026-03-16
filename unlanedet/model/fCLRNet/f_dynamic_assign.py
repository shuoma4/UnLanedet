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
                      Only used in batched mode.  If its Nt dimension differs
                      from targets.shape[1] (e.g. due to different dataset
                      padding strategies), it is recomputed from targets.
    Returns:
        If single: (matched_row_inds, matched_col_inds)
        If batched: matching_matrix (B, num_priors, num_targets)
    """
    is_batch = predictions.ndim == 3

    predictions = predictions.detach().clone()
    predictions[..., 3] *= img_w - 1
    predictions[..., 6:] *= img_w - 1
    targets = targets.detach().clone()

    # ── Reconcile valid_mask with actual targets shape ───────────────────────
    # valid_mask passed from fclr_head is computed from batch['lane_line'],
    # which may have a different Nt than targets here if the dataset class
    # pads to a different number than param_config.max_lanes.  To be safe,
    # always derive valid_mask from the targets tensor that we actually have.
    if is_batch:
        valid_mask = (targets[..., 1] == 1)  # (B, Nt_actual)

    # ── Safe-guard: neutralise invalid target slots with torch.where ────────
    # Invalid (padding) targets must not contaminate cost normalisation.
    # Rules:
    #   xs  (idx 6:)  → -1.0  so distance_cost's validity check (xs >= 0)
    #                          correctly masks them out → contribution = 0.
    #   yxtl (idx 2:6) →  0.0  so torch.cdist produces moderate distances
    #                          instead of exploding with original padding values.
    # Using torch.where instead of boolean-index assignment avoids the
    # IndexError that arises when valid_mask.shape[1] != targets.shape[1],
    # and is correct across all PyTorch versions.
    if is_batch and valid_mask is not None:
        invalid = ~valid_mask  # (B, Nt)
        if invalid.any():
            inv_exp = invalid.unsqueeze(-1)  # (B, Nt, 1) – broadcasts over D
            # Build a "safe" replacement for invalid rows
            safe = torch.zeros_like(targets)         # cls + yxtl = 0
            safe[..., 6:] = -1.0                     # xs = -1 (will be masked)
            targets = torch.where(inv_exp.expand_as(targets), safe, targets)

    # ── Distance cost ─────────────────────────────────────────────────────
    dist_scores = distance_cost(predictions, targets, img_w)
    # Normalise using max over VALID target columns only to prevent padding
    # rows from inflating the denominator and squashing valid signal.
    if is_batch:
        if valid_mask is not None:
            vexp = valid_mask.unsqueeze(1).expand_as(dist_scores)  # (B, Np, Nt)
            max_dist = dist_scores.masked_fill(~vexp, 0.0).flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5)
        else:
            max_dist = dist_scores.flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5)
        dist_scores = 1.0 - (dist_scores / max_dist) + 1e-2
    else:
        dist_scores = 1.0 - (dist_scores / (dist_scores.max() + 1e-5)) + 1e-2

    # ── Focal classification cost ────────────────────────────────────────────
    cls_scores = focal_cost(predictions[..., :2], targets[..., 1].long())

    # ── Start-point (y, x) cost ──────────────────────────────────────────────
    pred_start = predictions[..., 2:4].clone()
    target_start = targets[..., 2:4].clone()
    pred_start[..., 0] *= img_h - 1
    target_start[..., 0] *= img_h - 1

    start_scores = torch.cdist(pred_start, target_start, p=2)
    if is_batch:
        if valid_mask is not None:
            vexp = valid_mask.unsqueeze(1).expand_as(start_scores)
            max_start = start_scores.masked_fill(~vexp, 0.0).flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5)
        else:
            max_start = start_scores.flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5)
        start_scores = 1.0 - (start_scores / max_start) + 1e-2
    else:
        start_scores = 1.0 - (start_scores / (start_scores.max() + 1e-5)) + 1e-2

    # ── Theta cost ───────────────────────────────────────────────────────────
    theta_scores = torch.cdist(predictions[..., 4:5], targets[..., 4:5], p=1) * 180.0
    if is_batch:
        if valid_mask is not None:
            vexp = valid_mask.unsqueeze(1).expand_as(theta_scores)
            max_theta = theta_scores.masked_fill(~vexp, 0.0).flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5)
        else:
            max_theta = theta_scores.flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-5)
        theta_scores = 1.0 - (theta_scores / max_theta) + 1e-2
    else:
        theta_scores = 1.0 - (theta_scores / (theta_scores.max() + 1e-5)) + 1e-2

    # ── Combined cost ────────────────────────────────────────────────────────
    cost = -((dist_scores * start_scores * theta_scores) ** 2) * distance_cost_weight + cls_scores * cls_cost_weight

    # ── Masking: set invalid target columns cost to +inf ─────────────────────
    if is_batch and valid_mask is not None:
        mask_expanded = valid_mask.unsqueeze(1).expand_as(cost)
        cost[~mask_expanded] = float('inf')

    # ── Pairwise Line-IoU ────────────────────────────────────────────────────
    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)

    if is_batch:
        return dynamic_k_assign_batched(cost, iou, valid_mask)
    else:
        return dynamic_k_assign_single(cost, iou)
