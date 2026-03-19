import torch
import torch.nn.functional as F

from unlanedet.model.fCLRNet.f_line_iou import line_iou
from unlanedet.model.fCLRNet.f_dynamic_assign import focal_cost
from unlanedet.model.fCLRNet.f_dynamic_assign import assign as clrnet_assign
from unlanedet.model.fCLRNet.f_dynamic_assign import distance_cost


def _to_absolute(lanes, img_w, img_h):
    lanes_abs = lanes.detach().clone().float()
    lanes_abs[..., 2] *= lanes.shape[-1] - 7
    lanes_abs[..., 3] *= img_w - 1
    lanes_abs[..., 4] *= 180.0
    lanes_abs[..., 5] *= lanes.shape[-1] - 7
    lanes_abs[..., 6:] *= img_w - 1
    return lanes_abs


def _target_to_absolute(lanes, img_w, img_h):
    lanes_abs = lanes.detach().clone().float()
    lanes_abs[..., 2] *= lanes.shape[-1] - 7
    # lanes_abs[..., 3] is already absolute in dataloader
    lanes_abs[..., 4] *= 180.0
    # lanes_abs[..., 5] is already absolute in dataloader
    # lanes_abs[..., 6:] is already absolute in dataloader
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
    targets_abs = _target_to_absolute(targets, img_w, img_h)
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

    # Let CLRNet line_iou handle invalid masks dynamically
    pair_wise_ious = line_iou(pred_xs_abs, gt_xs_abs, img_w, aligned=False)
    
    n_strips = float(n_offsets - 1)

    # 1. Start point absolute distances [normalized 0-1]
    delta_x = torch.abs(pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1)) / img_w
    delta_y = torch.abs(pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)) / max(1.0, n_strips)
    
    # 2. Line distance (Mean absolute x-coord error) [normalized 0-1]
    # Pass absolute coordinates to calculate proper absolute distance, then normalize by image width
    line_dist_c = distance_cost(predictions_abs, targets_abs, img_w) / img_w
    
    # 3. Theta cost [normalized 0-1]
    theta_c = torch.abs(pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1)) / 180.0
    
    # 4. Length cost [normalized 0-1]
    length_c = torch.abs(pred_length.unsqueeze(2) - gt_length.unsqueeze(1)) / max(1.0, n_strips)
    
    # 5. Cls cost
    cls_cost = focal_cost(predictions[..., :2], targets[..., 1].long())
    
    w_cls = float(getattr(cfg, 'w_cls', 1.0))
    w_geom = float(getattr(cfg, 'w_geom', 3.0))

    # NEW TOTAL COST (additive like standard matchers)
    # The scale is now properly constrained since distance is [0, 1] instead of [0, 800+].
    total_cost = (
        cls_cost * w_cls 
        + line_dist_c * w_geom 
        + delta_x * w_geom 
        + delta_y * 1.5 
        + theta_c * 0.5 
        + length_c * 0.5
    )
    
    total_cost = total_cost.masked_fill(~valid_mask.unsqueeze(1), float('inf'))

    n_candidate_k = min(4, N)
    topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
    dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)

    max_k = int(dynamic_ks.max().item()) if dynamic_ks.numel() > 0 else 1
    _, topk_indices = torch.topk(total_cost, max_k, dim=1, largest=False)
    rank_mask = torch.arange(max_k, device=device).view(1, -1, 1) < dynamic_ks.unsqueeze(1)

    matching_matrix = torch.zeros_like(total_cost)
    matching_matrix.scatter_(1, topk_indices, rank_mask.float())
    matching_matrix.masked_fill_(~valid_mask.unsqueeze(1), 0.0)

    conflict_mask = matching_matrix.sum(dim=2) > 1
    if conflict_mask.any():
        c_mapped = total_cost.clone()
        c_mapped[matching_matrix == 0] = float('inf')
        cost_argmin = c_mapped.argmin(dim=2)
        resolved = torch.zeros_like(matching_matrix)
        resolved.scatter_(2, cost_argmin.unsqueeze(2), 1.0)
        c_mask_exp = conflict_mask.unsqueeze(2).expand_as(matching_matrix)
        matching_matrix = torch.where(c_mask_exp, resolved, matching_matrix)

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
