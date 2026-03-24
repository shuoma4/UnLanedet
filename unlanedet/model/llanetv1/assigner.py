import torch
import torch.nn.functional as F

from unlanedet.model.fCLRNet.f_line_iou import line_iou
from unlanedet.model.fCLRNet.f_dynamic_assign import focal_cost
from unlanedet.model.fCLRNet.f_dynamic_assign import assign as clrnet_assign
from unlanedet.model.fCLRerNet.f_dynamic_assign import assign as clrernet_assign


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
    return 1.0 - torch.exp(-(err**2) / (2 * sigma**2))


def geometry_aware_assign(
    predictions, targets, img_w, img_h, valid_mask=None, cfg=None, sample_ys=None
):
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

    w_cls = (
        float(getattr(cfg, "w_cls", 1.0))
        if getattr(cfg, "w_cls", None) is not None
        else 1.0
    )
    w_geom = (
        float(getattr(cfg, "w_geom", 3.0))
        if getattr(cfg, "w_geom", None) is not None
        else 3.0
    )

    # Batched extractions
    pred_start_y = predictions_abs[..., 2]
    pred_start_x = predictions_abs[..., 3]
    pred_theta = predictions_abs[..., 4]
    pred_length = predictions_abs[..., 5]

    gt_start_y = targets_abs[..., 2]
    gt_start_x = targets_abs[..., 3]
    gt_theta = targets_abs[..., 4]
    gt_length = targets_abs[..., 5]

    # 1. 计算相对误差 (限定在 0 到 1 的主响应区间) Tensor Shape: [B, N, M]
    delta_x = torch.abs(pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1)) / img_w
    delta_y = torch.abs(pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)) / max(
        1.0, float(n_strips)
    )
    theta_c = torch.abs(pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1)) / 180.0
    length_c = torch.abs(pred_length.unsqueeze(2) - gt_length.unsqueeze(1)) / max(
        1.0, float(n_strips)
    )
    delta_curve_x = curve_distance(
        predictions_abs[..., 6:], targets_abs[..., 6:], img_w
    )

    iou_matrix = line_iou(
        predictions_abs[..., 6:], targets_abs[..., 6:], img_w, aligned=False
    )
    iou_score = iou_matrix.clone()
    iou_score[iou_score < 0] = 0.0

    # 乘以多维高斯惩罚（Multi-dimensional Gaussian），以严格控制匹配条件（起到“逻辑与”的作用）
    # 加入全曲线平均相对距离约束(delta_curve_x)，大幅缓解由于仅评估起点造成的形状错误匹配导致的 Recall 下降问题
    score_curve = torch.exp(-(delta_curve_x**2) / (2 * 0.15**2))
    score_x = torch.exp(-(delta_x**2) / (2 * 0.15**2))
    score_y = torch.exp(-(delta_y**2) / (2 * 0.20**2))
    score_theta = torch.exp(-(theta_c**2) / (2 * 0.20**2))
    score_length = torch.exp(-(length_c**2) / (2 * 0.30**2))

    geom_score = score_curve * score_x * score_y * score_theta * score_length

    cls_score = focal_cost(predictions[..., :2], targets[..., 1].long())

    # cls_score is focal cost (loss value, positive), so larger is worse, adding it is correct
    cost = -(geom_score**2) * w_geom + cls_score * w_cls
    cost = cost.masked_fill(~valid_mask.unsqueeze(1), float("inf"))

    # 修正Dynamic K机制，使用严格的IoU，确保与CLRNet保持同级别的选点数量
    sim_proxy = iou_matrix.clone()
    sim_proxy[sim_proxy < 0] = 0.0
    sim_proxy = sim_proxy.masked_fill(~valid_mask.unsqueeze(1), 0.0)

    n_candidate_k = getattr(cfg, "n_candidate_k", 4)  # CLRNet default is 4
    n_candidate_k = min(n_candidate_k, N)
    topk_ious, _ = torch.topk(sim_proxy, n_candidate_k, dim=1)

    dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1)

    max_k = int(dynamic_ks.max().item()) if dynamic_ks.numel() > 0 else 1
    _, topk_indices = torch.topk(cost, max_k, dim=1, largest=False)
    rank_mask = torch.arange(max_k, device=device).view(
        1, -1, 1
    ) < dynamic_ks.unsqueeze(1)

    matching_matrix.scatter_(1, topk_indices, rank_mask.float())
    matching_matrix.masked_fill_(~valid_mask.unsqueeze(1), 0.0)

    conflict_mask = matching_matrix.sum(dim=2) > 1
    if conflict_mask.any():
        c_mapped = cost.clone()
        c_mapped[matching_matrix == 0] = float("inf")
        cost_argmin = c_mapped.argmin(dim=2)
        resolved = torch.zeros_like(matching_matrix)
        resolved.scatter_(2, cost_argmin.unsqueeze(2), 1.0)
        c_mask_exp = conflict_mask.unsqueeze(2).expand_as(matching_matrix)
        matching_matrix = torch.where(c_mask_exp, resolved, matching_matrix)

    return matching_matrix[0] if squeeze else matching_matrix


def assign(
    predictions, targets, masks, img_w, img_h, cfg=None, current_iter=0, sample_ys=None
):
    method_name = (
        getattr(cfg, "assign_method", "CLRNet") if cfg is not None else "CLRNet"
    )
    if method_name == "GeometryAware":
        return geometry_aware_assign(
            predictions,
            targets,
            img_w,
            img_h,
            valid_mask=masks,
            cfg=cfg,
            sample_ys=sample_ys,
        )
    if method_name == "CLRerAssign":
        return clrernet_assign(predictions, targets, img_w, img_h, valid_mask=masks)
    return clrnet_assign(predictions, targets, img_w, img_h, valid_mask=masks)
