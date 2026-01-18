import torch
from .line_iou import line_iou


def distance_cost(predictions, targets, img_w):
    """Broadcasting optimized distance cost"""
    # preds: (N, 72) -> (N, 1, 72)
    # targs: (M, 72) -> (1, M, 72)
    preds_pts = predictions[..., 6:]
    targs_pts = targets[..., 6:]

    # Broadcasting abs diff: (N, M, 72)
    dists = torch.abs(preds_pts.unsqueeze(1) - targs_pts.unsqueeze(0))

    # Invalid masks: (1, M, 72)
    invalid_masks = (targs_pts < 0) | (targs_pts >= img_w)
    invalid_masks = invalid_masks.unsqueeze(0)

    # Lengths: (1, M)
    lengths = (~invalid_masks).sum(dim=-1)

    # Zero out invalid distances
    dists = torch.where(invalid_masks, torch.zeros_like(dists), dists)

    # Sum over points: (N, M)
    cost = dists.sum(dim=-1) / (lengths.float() + 1e-9)
    return cost


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)
    cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


def dynamic_k_assign(cost, pair_wise_ious):
    ious_matrix = pair_wise_ious.clamp(min=0.0)
    topk_ious, _ = torch.topk(ious_matrix, k=min(4, ious_matrix.size(0)), dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

    num_gt = cost.shape[1]
    matching_matrix = torch.zeros_like(cost)

    for gt_idx in range(num_gt):
        _, pos_idx = torch.topk(
            cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
        )
        matching_matrix[pos_idx, gt_idx] = 1.0

    matched_gt = matching_matrix.sum(1)
    if (matched_gt > 1).sum() > 0:
        conflict_mask = matched_gt > 1
        cost_conflict = cost[conflict_mask]
        cost_argmin = torch.min(cost_conflict, dim=1)[1]
        matching_matrix[conflict_mask] = 0.0
        matching_matrix[conflict_mask, cost_argmin] = 1.0

    prior_idx, gt_idx = matching_matrix.nonzero(as_tuple=True)
    return prior_idx, gt_idx


def assign(
    predictions, targets, img_w, img_h, distance_cost_weight=3.0, cls_cost_weight=1.0
):
    # Clone only when necessary
    preds_scaled = predictions.clone()
    preds_scaled[..., 6:] *= img_w - 1
    preds_scaled[..., 3] *= img_w - 1

    targs_scaled = targets.clone()
    targs_scaled[..., 6:] *= img_w - 1

    # 1. Dist Cost
    dist_score = distance_cost(preds_scaled, targs_scaled, img_w)
    max_dist = torch.max(dist_score)
    if max_dist > 0:
        dist_score = 1 - (dist_score / max_dist) + 1e-2
    else:
        dist_score = torch.ones_like(dist_score)

    # 2. Cls Cost
    cls_score = focal_cost(predictions[:, :2], targets[:, 1].long())

    # 3. Start XY Cost
    pred_start = predictions[:, 2:4].clone()
    pred_start[:, 0] *= img_h - 1
    pred_start[:, 1] *= img_w - 1
    targ_start = targets[:, 2:4].clone()
    targ_start[:, 0] *= img_h - 1
    targ_start[:, 1] *= img_w - 1

    start_score = torch.cdist(pred_start, targ_start, p=2)
    max_start = torch.max(start_score)
    if max_start > 0:
        start_score = (1 - start_score / max_start) + 1e-2

    # 4. Theta Cost
    theta_score = (
        torch.cdist(predictions[:, 4].unsqueeze(-1), targets[:, 4].unsqueeze(-1), p=1)
        * 180
    )
    max_theta = torch.max(theta_score)
    if max_theta > 0:
        theta_score = (1 - theta_score / max_theta) + 1e-2

    cost = (
        -((dist_score * start_score * theta_score) ** 2) * distance_cost_weight
        + cls_score * cls_cost_weight
    )

    # 5. IoU
    iou = line_iou(preds_scaled[..., 6:], targs_scaled[..., 6:], img_w, aligned=False)

    return dynamic_k_assign(cost, iou)
