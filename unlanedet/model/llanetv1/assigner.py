import torch
import torch.nn.functional as F

from unlanedet.model.LLANet.line_iou import pairwise_line_iou
from unlanedet.model.fCLRNet.f_dynamic_assign import assign as clrnet_assign


def _to_absolute(lanes, img_w, img_h):
    lanes_abs = lanes.detach().clone()
    lanes_abs[..., 2] *= lanes.shape[-1] - 7
    lanes_abs[..., 3] *= img_w - 1
    lanes_abs[..., 4] *= 180.0
    lanes_abs[..., 5] *= lanes.shape[-1] - 7
    lanes_abs[..., 6:] *= img_w - 1
    return lanes_abs


def _prepare_sample_ys(sample_ys, n_offsets, img_h, device):
    if sample_ys is None:
        sample_ys = torch.linspace(1, 0, steps=n_offsets, device=device)
    else:
        sample_ys = sample_ys.to(device)
    return sample_ys.view(1, 1, -1)


def geometry_aware_assign(predictions, targets, img_w, img_h, valid_mask=None, cfg=None, sample_ys=None):
    squeeze = False
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
        targets = targets.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0) if valid_mask is not None else None
        squeeze = True

    device = predictions.device
    B, N, _ = predictions.shape
    _, M, _ = targets.shape
    if valid_mask is None:
        valid_mask = torch.ones((B, M), dtype=torch.bool, device=device)

    if M == 0:
        matching_matrix = predictions.new_zeros((B, N, M))
        return matching_matrix[0] if squeeze else matching_matrix

    predictions_abs = _to_absolute(predictions, img_w, img_h)
    targets_abs = _to_absolute(targets, img_w, img_h)
    n_offsets = predictions_abs.shape[-1] - 6
    n_strips = n_offsets - 1
    sample_ys = _prepare_sample_ys(sample_ys, n_offsets, img_h, device)

    pred_start_y = predictions_abs[..., 2]
    pred_start_x = predictions_abs[..., 3]
    pred_theta = predictions_abs[..., 4]
    pred_length = predictions_abs[..., 5]
    pred_xs_abs = predictions_abs[..., 6:]

    gt_start_y = targets_abs[..., 2]
    gt_start_x = targets_abs[..., 3]
    gt_theta = targets_abs[..., 4]
    gt_length = targets_abs[..., 5]
    gt_xs_abs = targets_abs[..., 6:]

    pred_start_idx = pred_start_y.round().long().clamp(0, n_strips)
    pred_end_idx = (pred_start_idx + pred_length.round().long() - 1).clamp(0, n_offsets - 1)
    pred_valid = (
        (torch.arange(n_offsets, device=device).view(1, 1, -1) >= pred_start_idx.unsqueeze(-1))
        & (torch.arange(n_offsets, device=device).view(1, 1, -1) <= pred_end_idx.unsqueeze(-1))
    )
    pred_xs_abs = pred_xs_abs.masked_fill(~pred_valid, -1e5)

    gt_start_idx = gt_start_y.round().long().clamp(0, n_strips)
    gt_end_idx = (gt_start_idx + gt_length.round().long() - 1).clamp(0, n_offsets - 1)
    gt_valid = (
        (torch.arange(n_offsets, device=device).view(1, 1, -1) >= gt_start_idx.unsqueeze(-1))
        & (torch.arange(n_offsets, device=device).view(1, 1, -1) <= gt_end_idx.unsqueeze(-1))
    )

    gt_invalid = (~gt_valid) | (gt_xs_abs < 0) | (gt_xs_abs >= img_w)
    gt_xs_abs = gt_xs_abs.masked_fill(gt_invalid, -1e5)

    pair_wise_ious = pairwise_line_iou(
        pred_xs_abs,
        gt_xs_abs,
        img_w,
        length=float(getattr(cfg, 'assign_iou_width', 15.0)),
        invalid_value=-1e5,
    )
    pair_wise_ious = torch.nan_to_num(pair_wise_ious, nan=0.0)
    pair_wise_ious = pair_wise_ious.masked_fill(~valid_mask.unsqueeze(1), 0.0)

    delta_x = (pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1)) / max(img_w - 1, 1)
    delta_y = (pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)) / max(n_strips, 1)
    dist_cost = torch.sqrt(delta_x.pow(2) + delta_y.pow(2) + 1e-8)
    theta_cost = torch.abs(pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1)) / 180.0
    length_cost = torch.abs(pred_length.unsqueeze(2) - gt_length.unsqueeze(1)) / max(n_strips, 1)
    cls_cost = -torch.log(F.softmax(predictions[..., :2], dim=-1)[..., 1].unsqueeze(2).clamp(min=1e-8))
    iou_cost = 1.0 - pair_wise_ious

    w_cls = float(getattr(cfg, 'w_cls', 2.0))
    w_geom = float(getattr(cfg, 'w_geom', 4.0))
    w_iou = float(getattr(cfg, 'w_iou', 2.0))
    w_dist = float(getattr(cfg, 'w_dist', 1.0))
    w_theta = float(getattr(cfg, 'w_theta', 2.0))
    w_length = float(getattr(cfg, 'w_length', 1.0))
    geom_cost = w_dist * dist_cost + w_theta * theta_cost + w_length * length_cost
    total_cost = w_cls * cls_cost + w_geom * geom_cost + w_iou * iou_cost
    total_cost = total_cost.masked_fill(~valid_mask.unsqueeze(1), float('inf'))

    n_candidate_k = min(int(getattr(cfg, 'simota_q', 10)), N)
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1, max=N)

    max_k = int(dynamic_ks.max().item()) if dynamic_ks.numel() > 0 else 1
    _, topk_indices = torch.topk(total_cost, max_k, dim=1, largest=False)
    rank_mask = torch.arange(max_k, device=device).view(1, -1, 1) < dynamic_ks.unsqueeze(1)

    matching_matrix = torch.zeros_like(total_cost)
    matching_matrix.scatter_(1, topk_indices, rank_mask.float())
    matching_matrix.masked_fill_(~valid_mask.unsqueeze(1), 0.0)

    conflict_mask = matching_matrix.sum(dim=2) > 1
    if conflict_mask.any():
        cost_with_inf = total_cost.masked_fill(matching_matrix == 0, float('inf'))
        min_cost_idx = cost_with_inf.argmin(dim=2)
        matching_matrix[conflict_mask] = 0.0
        b_idx, n_idx = torch.where(conflict_mask)
        matching_matrix[b_idx, n_idx, min_cost_idx[b_idx, n_idx]] = 1.0

    return matching_matrix[0] if squeeze else matching_matrix


def assign(predictions, targets, masks, img_w, img_h, cfg=None, current_iter=0, sample_ys=None):
    method_name = getattr(cfg, 'assign_method', 'CLRNet') if cfg is not None else 'CLRNet'
    if method_name == 'GeometryAware':
        return geometry_aware_assign(
            predictions,
            targets,
            img_w,
            img_h,
            valid_mask=masks,
            cfg=cfg,
            sample_ys=sample_ys,
        )
    return clrnet_assign(predictions, targets, img_w, img_h, valid_mask=masks)
