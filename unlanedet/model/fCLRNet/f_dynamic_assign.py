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

``distance_cost`` (original):
    Uses repeat_interleave + torch.cat which allocates two large copies.
    Replaced with direct broadcasting: (Np,1,Nr) vs (1,Nt,Nr).

Public API ``assign()`` is fully backward-compatible with CLRNet.
"""
import torch

from .f_line_iou import line_iou


# ─────────────────────────────────────────────────────────────────────────────
# Cost components
# ─────────────────────────────────────────────────────────────────────────────

def distance_cost(predictions: torch.Tensor,
                  targets:     torch.Tensor,
                  img_w:       float) -> torch.Tensor:
    """
    Mean absolute x-coordinate distance between every (prior, gt) pair.

    Original implementation used ``repeat_interleave`` + ``torch.cat`` which
    created two (Np*Nt, Nr) copies.  Here we use broadcasting to avoid those
    intermediate allocations.

    Args:
        predictions : (num_priors,  D)  full prediction tensor
        targets     : (num_targets, D)  full target tensor
    Returns:
        distances   : (num_priors, num_targets)
    """
    pred_xs   = predictions[..., 6:]   # (Np, Nr)
    target_xs = targets[..., 6:]       # (Nt, Nr)

    # Validity is determined by target coordinates only (same as original)
    # broadcast to (Np, Nt, Nr)
    invalid_masks = (target_xs[None] < 0) | (target_xs[None] >= img_w)
    lengths   = (~invalid_masks).sum(dim=-1).float()                     # (Np, Nt)
    distances = (pred_xs[:, None, :] - target_xs[None, :, :]).abs()      # (Np, Nt, Nr)
    distances[invalid_masks] = 0.0
    distances = distances.sum(dim=-1) / (lengths + 1e-9)                 # (Np, Nt)
    return distances


def focal_cost(cls_pred:  torch.Tensor,
               gt_labels: torch.Tensor,
               alpha:     float = 0.25,
               gamma:     float = 2.0,
               eps:       float = 1e-12) -> torch.Tensor:
    """
    Focal classification cost — unchanged from CLRNet.

    Args:
        cls_pred  : (num_priors, num_classes)  raw logits
        gt_labels : (num_targets,)             integer labels
    Returns:
        cls_cost  : (num_priors, num_targets)
    """
    p        = cls_pred.sigmoid()
    neg_cost = -(1 - p + eps).log() * (1 - alpha) * p.pow(gamma)
    pos_cost = -(    p + eps).log() *      alpha   * (1 - p).pow(gamma)
    return pos_cost[:, gt_labels] - neg_cost[:, gt_labels]


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised dynamic-k assignment  (core optimisation)
# ─────────────────────────────────────────────────────────────────────────────

def dynamic_k_assign(cost:           torch.Tensor,
                     pair_wise_ious: torch.Tensor):
    """
    Assign ground-truths to priors dynamically — **fully vectorised**.

    The original CLRNet code iterated over GT indices with a Python for-loop:

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], ...)
            matching_matrix[pos_idx, gt_idx] = 1.0

    We replace this with a single ``argsort`` followed by a ``scatter_``:

        sorted_idx = cost.argsort(dim=0)          # rank each prior per GT
        topk_mask  = (rank < dynamic_ks).float()  # 1 iff within top-k budget
        matching_matrix.scatter_(0, sorted_idx, topk_mask)

    This is O(1) on the Python side (no for-loop) and maps to a single
    GPU kernel, giving significant speed-up when num_gt or batch size is large.

    Additionally, conflict resolution (prior matched to >1 GT) clears ALL
    columns of a conflicting prior before re-assigning to the lowest-cost GT,
    which is a more correct implementation than the original (which only
    cleared column 0).

    Args:
        cost           : (num_priors, num_gt)
        pair_wise_ious : (num_priors, num_gt)
    Returns:
        prior_idx : (M,) matched prior indices
        gt_idx    : (M,) corresponding GT indices
    """
    num_priors, num_gt = cost.shape
    device = cost.device

    # ── 1. Dynamic k per GT ──────────────────────────────────────────────────
    ious = pair_wise_ious.clone()
    ious[ious < 0] = 0.0
    n_candidate_k = min(4, num_priors)
    topk_ious, _ = torch.topk(ious, n_candidate_k, dim=0)   # (k, num_gt)
    dynamic_ks   = topk_ious.sum(0).int().clamp(min=1)       # (num_gt,)

    # ── 2. Vectorised top-k selection ────────────────────────────────────────
    # sorted_idx[rank, gt] = prior index with rank-th lowest cost for that GT
    sorted_idx = cost.argsort(dim=0)                          # (num_priors, num_gt)

    # topk_mask[rank, gt] = 1  iff  rank < dynamic_ks[gt]
    ranks     = torch.arange(num_priors, device=device) \
                     .unsqueeze(1).expand(num_priors, num_gt)
    topk_mask = (ranks < dynamic_ks.unsqueeze(0)).float()    # (num_priors, num_gt)

    # Scatter: matching_matrix[sorted_idx[r,g], g] = topk_mask[r,g]
    matching_matrix = torch.zeros_like(cost)
    matching_matrix.scatter_(0, sorted_idx, topk_mask)

    # ── 3. Resolve conflicts (prior matched to >1 GT) ────────────────────────
    matched_gt_count = matching_matrix.sum(dim=1)            # (num_priors,)
    conflict_mask    = matched_gt_count > 1

    if conflict_mask.any():
        # Keep only the minimum-cost GT for each conflicting prior.
        _, cost_argmin = cost[conflict_mask].min(dim=1)      # (num_conflict,)
        matching_matrix[conflict_mask]            = 0.0      # clear all
        matching_matrix[conflict_mask, cost_argmin] = 1.0

    # ── 4. Extract matched indices ───────────────────────────────────────────
    prior_idx = matching_matrix.sum(dim=1).nonzero(as_tuple=False)  # (M, 1)
    gt_idx    = matching_matrix[prior_idx].argmax(dim=-1)            # (M,)
    return prior_idx.flatten(), gt_idx.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Public assign — drop-in replacement for CLRNet.dynamic_assign.assign
# ─────────────────────────────────────────────────────────────────────────────

def assign(predictions:          torch.Tensor,
           targets:              torch.Tensor,
           img_w:                float,
           img_h:                float,
           distance_cost_weight: float = 3.0,
           cls_cost_weight:      float = 1.0):
    """
    Vectorised drop-in replacement for ``CLRNet.dynamic_assign.assign``.

    Args:
        predictions : (num_priors, 78)  per-stage model output
        targets     : (num_targets, 78) ground-truth lane annotations
    Returns:
        matched_row_inds : (M,) matched prior indices
        matched_col_inds : (M,) matched GT indices
    """
    predictions = predictions.detach().clone()
    predictions[:, 3]  *= img_w - 1
    predictions[:, 6:] *= img_w - 1
    targets = targets.detach().clone()

    num_priors  = predictions.shape[0]
    num_targets = targets.shape[0]

    # ── Distance cost (vectorised broadcasting) ──────────────────────────────
    dist_scores = distance_cost(predictions, targets, img_w)
    dist_scores = 1.0 - (dist_scores / (dist_scores.max() + 1e-9)) + 1e-2

    # ── Focal classification cost ────────────────────────────────────────────
    cls_scores = focal_cost(predictions[:, :2], targets[:, 1].long())

    # ── Start-point (y, x) cost ──────────────────────────────────────────────
    pred_start   = predictions[:, 2:4].clone()
    target_start = targets[:, 2:4].clone()
    pred_start[:, 0]   *= img_h - 1
    target_start[:, 0] *= img_h - 1
    start_scores = torch.cdist(pred_start, target_start, p=2)   # (Np, Nt)
    start_scores = 1.0 - (start_scores / (start_scores.max() + 1e-9)) + 1e-2

    # ── Theta cost ───────────────────────────────────────────────────────────
    theta_scores = torch.cdist(
        predictions[:, 4:5], targets[:, 4:5], p=1
    ).reshape(num_priors, num_targets) * 180.0
    theta_scores = 1.0 - (theta_scores / (theta_scores.max() + 1e-9)) + 1e-2

    # ── Combined cost ────────────────────────────────────────────────────────
    cost = (
        -((dist_scores * start_scores * theta_scores) ** 2) * distance_cost_weight
        + cls_scores * cls_cost_weight
    )

    # ── Pairwise Line-IoU ────────────────────────────────────────────────────
    iou = line_iou(predictions[..., 6:], targets[..., 6:], img_w, aligned=False)

    return dynamic_k_assign(cost, iou)
