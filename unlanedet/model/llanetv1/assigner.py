import torch
import torch.nn.functional as F

from unlanedet.model.fCLRNet.f_line_iou import line_iou
from unlanedet.model.fCLRNet.f_dynamic_assign import focal_cost
from unlanedet.model.fCLRNet.f_dynamic_assign import assign as clrnet_assign

def _to_absolute(lanes, img_w, img_h):
    lanes_abs = lanes.detach().clone().float()
    lanes_abs[..., 2] *= lanes.shape[-1] - 7  # mapping start_y to strip index
    lanes_abs[..., 3] *= img_w - 1
    lanes_abs[..., 4] *= 180.0
    lanes_abs[..., 5] *= lanes.shape[-1] - 7  # mapping length to strip count
    lanes_abs[..., 6:] *= img_w - 1
    return lanes_abs

def _target_to_absolute(lanes, img_w, img_h):
    lanes_abs = lanes.detach().clone().float()
    lanes_abs[..., 2] *= lanes.shape[-1] - 7  # mapping start_y to strip index
    lanes_abs[..., 4] *= 180.0
    return lanes_abs


def curve_distance(pred_xs, target_xs, img_w):
    target_broad = target_xs.unsqueeze(-3)
    valid_mask = ((target_broad >= 0) & (target_broad < img_w)).float()
    lengths = valid_mask.sum(dim=-1)
    distances = (pred_xs.unsqueeze(-2) - target_xs.unsqueeze(-3)).abs()
    distances = (distances * valid_mask).sum(dim=-1) / (lengths + 1e-5)
    return distances / img_w  # Normalize by img_w to keep scale 0~1 like others

def gaussian_penalty(err, sigma):
    return 1.0 - torch.exp(-(err ** 2) / (2 * sigma ** 2))

def geometry_aware_assign(predictions, targets, img_w, img_h, valid_mask=None, cfg=None, sample_ys=None):
    """
    创新路径一：Gaussian-Decoupled Geometry Cost (Fully Vectorized Batched)
    """
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

    matching_matrix = torch.zeros((B, N, M), device=device)

    if M == 0:
        return matching_matrix[0] if squeeze else matching_matrix

    predictions_abs = _to_absolute(predictions, img_w, img_h)
    targets_abs = _target_to_absolute(targets, img_w, img_h)
    n_offsets = predictions_abs.shape[-1] - 6
    n_strips = float(n_offsets - 1)

    w_cls = float(getattr(cfg, 'w_cls', 1.0)) if getattr(cfg, 'w_cls', None) is not None else 1.0
    w_geom = float(getattr(cfg, 'w_geom', 3.0)) if getattr(cfg, 'w_geom', None) is not None else 3.0

    # Batched extractions
    pred_start_y = predictions_abs[..., 2]
    pred_start_x = predictions_abs[..., 3]
    pred_theta = predictions_abs[..., 4]
    pred_length = predictions_abs[..., 5]

    gt_start_y = targets_abs[..., 2]
    gt_start_x = targets_abs[..., 3]
    gt_theta = targets_abs[..., 4]
    gt_length = targets_abs[..., 5]

    w_geom_curve = 3.0
    w_geom_start = 1.0
    w_geom_theta = 1.0

    cost_x = torch.abs(pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1)) / img_w
    cost_y = torch.abs(pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)) / max(1.0, float(n_strips))
    cost_theta = torch.abs(pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1)) / 180.0
    cost_curve = curve_distance(predictions_abs[..., 6:], targets_abs[..., 6:], img_w)

    cls_cost = focal_cost(predictions[..., :2], targets[..., 1].long())

    geom_cost = (cost_curve * w_geom_curve) + \
                (cost_x * w_geom_start) + \
                (cost_y * w_geom_start) + \
                (cost_theta * w_geom_theta)

    cost = geom_cost + cls_cost * w_cls
    cost = cost.masked_fill(~valid_mask.unsqueeze(1), float('inf'))

    iou_matrix = line_iou(predictions_abs[..., 6:], targets_abs[..., 6:], img_w, aligned=False)
    
    # 修正Dynamic K机制，使用严格的IoU，确保与CLRNet保持同级别的选点数量
    sim_proxy = iou_matrix.clone()
    sim_proxy[sim_proxy < 0] = 0.0
    sim_proxy = sim_proxy.masked_fill(~valid_mask.unsqueeze(1), 0.0)
    
    n_candidate_k = getattr(cfg, 'n_candidate_k', 4) # CLRNet default is 4
    n_candidate_k = min(n_candidate_k, N)
    topk_ious, _ = torch.topk(sim_proxy, n_candidate_k, dim=1)

    dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)

    max_k = int(dynamic_ks.max().item()) if dynamic_ks.numel() > 0 else 1
    _, topk_indices = torch.topk(cost, max_k, dim=1, largest=False)
    rank_mask = torch.arange(max_k, device=device).view(1, -1, 1) < dynamic_ks.unsqueeze(1)

    matching_matrix.scatter_(1, topk_indices, rank_mask.float())
    matching_matrix.masked_fill_(~valid_mask.unsqueeze(1), 0.0)

    conflict_mask = matching_matrix.sum(dim=2) > 1
    if conflict_mask.any():
        c_mapped = cost.clone()
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
