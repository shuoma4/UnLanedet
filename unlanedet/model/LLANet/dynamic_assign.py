import torch
import torch.nn as nn
import torch.nn.functional as F
from .line_iou import line_iou


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    Args:
        cls_pred (Tensor): Predicted classification logits, shape
            [num_query, num_class] or [batch, num_query, num_class].
        gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,) or [batch, num_gt].

    Returns:
        torch.Tensor: cls_cost value
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)

    if cls_pred.dim() == 3:
        # Batch mode: [B, N, C] vs [B, M]
        gt_labels_expanded = gt_labels.unsqueeze(1).expand(
            -1, cls_pred.shape[1], -1
        )  # [B, N, M]
        cls_cost = pos_cost.gather(2, gt_labels_expanded) - neg_cost.gather(
            2, gt_labels_expanded
        )
    else:
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


class GeometryAwareAssign(nn.Module):
    """
    [方案 A] 几何感知动态分配器 (Geometry-Enhanced SimOTA) - 加法 Cost 模型
    特点：Cost = w_cls * Cls + w_geom * Geom + w_iou * IoU
    """

    def __init__(self, cfg=None):
        super(GeometryAwareAssign, self).__init__()
        self.simota_q = 10

        self.w_cls = getattr(cfg, "w_cls", 4.0)
        self.w_geom = getattr(cfg, "w_geom", 5.0)
        self.w_iou = getattr(cfg, "w_iou", 2.0)

        # 动态权重配置
        self.start_w_cls = getattr(cfg, "start_w_cls", 0.5)
        self.warmup_epochs = getattr(cfg, "warmup_epochs", 5)
        self.epoch_per_iter = getattr(cfg, "epoch_per_iter", 1)

        self.w_dist = 1.0
        self.w_theta = 2.0

    def forward(
        self, preds, targets, masks, img_w, img_h, current_iter=0, prior_ys=None
    ):
        # 动态权重计算
        # warmup_iters = self.warmup_epochs * self.epoch_per_iter
        # if current_iter < warmup_iters:
        #     alpha = current_iter / warmup_iters
        #     current_w_cls = self.start_w_cls + alpha * (
        #         self.target_w_cls - self.start_w_cls
        #     )
        # else:
        #     current_w_cls = self.target_w_cls

        device = preds.device
        batch_size, num_priors, _ = preds.shape
        _, max_targets, _ = targets.shape

        if max_targets == 0:
            return torch.zeros(
                (batch_size, num_priors), dtype=torch.bool, device=device
            ), torch.full((batch_size, num_priors), -1, dtype=torch.long, device=device)

        # 1. 提取预测值
        pred_logits = preds[..., :2]
        pred_scores = F.softmax(pred_logits, dim=-1)[..., 1]

        pred_start_y = preds[..., 2]
        pred_start_x = preds[..., 3]
        pred_theta = preds[..., 4]

        # 2. 提取目标值
        gt_start_y = targets[..., 2]
        gt_start_x = targets[..., 3]
        gt_theta = targets[..., 4]

        # ========================================
        # A. 计算综合几何 Cost (Geometry Cost)
        # ========================================

        # A.1 起始点欧氏距离
        # pred_start_x is normalized, gt_start_x is normalized
        # Both pred_start_x and gt_start_x are normalized to 0-1
        pred_start_x_pixel = pred_start_x * (img_w - 1)
        gt_start_x_pixel = gt_start_x * (img_w - 1)

        delta_x = pred_start_x_pixel.unsqueeze(2) - gt_start_x_pixel.unsqueeze(1)

        # Scale Y to pixel space (gt_start_y is index/n_strips, normalized)
        # pred_start_y is normalized (0-1), gt_start_y is normalized (0-1)
        delta_y = (pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)) * (img_h - 1)

        dist_cost = torch.sqrt(delta_x.pow(2) + delta_y.pow(2) + 1e-8)
        dist_cost = dist_cost / (img_w - 1)  # Normalize to 0-1 scale

        # A.2 角度差异
        theta_cost = torch.abs(pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1))

        # A.3 组合几何 Cost
        geom_cost = self.w_dist * dist_cost + self.w_theta * theta_cost

        # ========================================
        # B. 其他 Cost
        # ========================================
        cls_cost = -torch.log(pred_scores.unsqueeze(2).clamp(min=1e-8))

        # IoU 计算 (需要绝对坐标)
        pred_lines = preds[..., 6:] * (img_w - 1)
        gt_lines = targets[..., 6:] * (img_w - 1)

        n_offsets = preds.shape[-1] - 6
        offset_index = torch.arange(n_offsets, device=device).view(1, 1, -1)
        pred_start_idx = (1.0 - pred_start_y) * (n_offsets - 1)
        pred_len_idx = preds[..., 5] * (n_offsets - 1)
        pred_mask = (offset_index >= pred_start_idx.unsqueeze(-1)) & (
            offset_index <= (pred_start_idx + pred_len_idx).unsqueeze(-1)
        )

        ious_list = []
        for i in range(batch_size):
            valid_gt_mask = masks[i].bool()
            num_valid = int(valid_gt_mask.sum())

            if num_valid == 0:
                ious_list.append(torch.zeros(num_priors, max_targets, device=device))
                continue

            # 【修复】：不要 unsqueeze，保持 2D Tensor，交给 line_iou 处理
            cur_pred = pred_lines[i]
            cur_gt = gt_lines[i, :num_valid]

            l_iou = line_iou(
                cur_pred,
                cur_gt,
                img_w,
                length=15,
                aligned=False,
                pred_mask=pred_mask[i],
            )
            l_iou = torch.nan_to_num(l_iou, nan=0.0)

            # 填充到 max_targets
            padded_iou = torch.zeros(num_priors, max_targets, device=device)
            padded_iou[:, :num_valid] = l_iou
            ious_list.append(padded_iou)

        pair_wise_ious = torch.stack(ious_list)
        iou_cost = 1.0 - pair_wise_ious

        # ========================================
        # C. 总 Cost
        # ========================================
        mask_expanded = masks.unsqueeze(1).expand(-1, num_priors, -1)

        total_cost = (
            self.w_cls * cls_cost
            + self.w_geom * geom_cost
            + self.w_iou * iou_cost
            + 100000.0 * (~mask_expanded.bool()).float()
        )

        # ========================================
        # D. SimOTA
        # ========================================
        n_candidate_k = min(self.simota_q, num_priors)
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1, max=num_priors)

        matched_targets_matrix = torch.full(
            (batch_size, num_priors), -1, dtype=torch.long, device=device
        )
        matched_min_costs = torch.full(
            (batch_size, num_priors), 1e8, dtype=torch.float32, device=device
        )

        for b in range(batch_size):
            num_gt = int(masks[b].sum())
            if num_gt == 0:
                continue

            cost_b = total_cost[b, :, :num_gt]
            k_b = dynamic_ks[b, :num_gt]

            for gt_idx in range(num_gt):
                k = k_b[gt_idx].item()
                k = max(1, min(k, num_priors))

                current_costs = cost_b[:, gt_idx]
                topk_costs, pos_idx = torch.topk(current_costs, k, largest=False)

                current_min_costs = matched_min_costs[b, pos_idx]
                update_mask = topk_costs < current_min_costs
                valid_indices = pos_idx[update_mask]
                valid_costs = topk_costs[update_mask]

                if len(valid_indices) > 0:
                    matched_targets_matrix[b, valid_indices] = gt_idx
                    matched_min_costs[b, valid_indices] = valid_costs

        assigned_mask = matched_targets_matrix >= 0
        return assigned_mask, matched_targets_matrix


class CLRNetAssign(nn.Module):
    """
    [方案 B] CLRNet 风格的动态分配器 (Similarity-based SimOTA) - 乘法 Cost 模型
    特点：Cost = -(Dist_Score * XY_Score * Theta_Score)^2 * w_reg + w_cls * Cls
    优化点：适配 Batch 处理，优化内存，增加动态权重
    """

    def __init__(self, cfg=None):
        super(CLRNetAssign, self).__init__()
        self.simota_q = 10

        # CLRNet 默认参数
        self.w_reg = 3.0  # 距离 Cost 权重

        # 动态权重配置
        self.target_w_cls = 1.0
        self.start_w_cls = getattr(cfg, "start_w_cls", 0.1)
        self.warmup_epochs = getattr(cfg, "warmup_epochs", 5)
        self.epoch_per_iter = getattr(cfg, "epoch_per_iter", 1)

    def forward(
        self,
        preds,
        targets,
        masks,
        img_w,
        img_h,
        current_iter=0,
        prior_ys=None,
    ):
        device = preds.device
        batch_size = preds.shape[0]
        num_priors = preds.shape[1]
        num_targets = targets.shape[1]  # M

        if num_targets == 0:
            return (
                torch.zeros((batch_size, num_priors), device=device, dtype=torch.bool),
                torch.zeros((batch_size, num_priors), device=device, dtype=torch.long),
            )

        matched_targets_matrix = torch.full(
            (batch_size, num_priors), -1, dtype=torch.long, device=device
        )
        preds_abs = preds.clone()
        preds_abs[..., 3] *= img_w - 1
        preds_abs[..., 6:] *= img_w - 1
        targets_abs = targets.clone()
        targets_abs[..., 3] *= img_w - 1
        targets_abs[..., 6:] *= img_w - 1
        target_valid_mask = (targets_abs[..., 6:] >= 0) & (targets_abs[..., 6:] < img_w)
        combined_mask = target_valid_mask.unsqueeze(1)
        combined_len = combined_mask.sum(dim=3).float().clamp(min=1.0)  # [B, 1, M]
        diff = torch.abs(
            preds_abs[..., 6:].unsqueeze(2) - targets_abs[..., 6:].unsqueeze(1)
        )
        diff = diff * combined_mask
        distances = diff.sum(dim=3) / combined_len  # [B, N, M]
        dist_max = distances.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        distances_score = 1 - (distances / dist_max) + 1e-2
        cur_pred_start = preds_abs[..., 2:4].clone()
        cur_pred_start[..., 0] *= img_h - 1
        cur_target_start = targets_abs[..., 2:4].clone()
        cur_target_start[..., 0] *= img_h - 1
        start_dists = torch.cdist(cur_pred_start, cur_target_start, p=2)
        start_max = start_dists.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        start_xys_score = 1 - (start_dists / start_max) + 1e-2
        cur_pred_theta = preds_abs[..., 4].unsqueeze(2)  # [B, N, 1]
        cur_target_theta = targets_abs[..., 4].unsqueeze(1)  # [B, 1, M]
        theta_dists = torch.abs(cur_pred_theta - cur_target_theta) * 180
        theta_max = theta_dists.amax(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        theta_score = 1 - (theta_dists / theta_max) + 1e-2
        gt_labels = targets_abs[..., 1].long()  # [B, M]
        cls_cost = focal_cost(preds_abs[..., :2], gt_labels)  # [B, N, M]
        reg_score = (
            distances_score.clamp(min=1e-3)
            * start_xys_score.clamp(min=1e-3)
            * theta_score.clamp(min=1e-3)
        )
        total_cost = -(reg_score**2) * self.w_reg + cls_cost * 1.0
        valid_targets_mask = masks.unsqueeze(1).expand(-1, num_priors, -1).bool()
        total_cost[~valid_targets_mask] = 1e8  # Assign high cost to invalid targets

        l_iou = line_iou(
            preds_abs[..., 6:],
            targets_abs[..., 6:],
            img_w,
            length=15,
            aligned=False,
        )
        l_iou = torch.nan_to_num(l_iou, nan=0.0)
        l_iou[~valid_targets_mask] = 0.0

        n_candidate_k = min(4, num_priors)
        topk_ious, _ = torch.topk(l_iou, n_candidate_k, dim=1)  # [B, K, M]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # [B, M]
        dynamic_ks[~masks.bool()] = 0
        max_k = n_candidate_k

        if max_k > 0:
            _, topk_indices = torch.topk(total_cost, k=max_k, dim=1, largest=False)
            grid = torch.arange(max_k, device=device).view(1, -1, 1)
            selection_mask = grid < dynamic_ks.unsqueeze(1)
            matching_matrix = torch.zeros(
                (batch_size, num_priors, num_targets), device=device
            )
            matching_matrix.scatter_(1, topk_indices, selection_mask.float())
            matched_counts = matching_matrix.sum(dim=2)  # [B, N]
            conflict_mask = matched_counts > 1  # [B, N]
            if conflict_mask.any():
                masked_cost = total_cost.clone()
                masked_cost[matching_matrix == 0] = float("inf")
                best_target_idx = masked_cost.argmin(dim=2)  # [B, N]
                new_matches = F.one_hot(
                    best_target_idx, num_classes=num_targets
                ).float()
                matching_matrix[conflict_mask] = new_matches[conflict_mask]
            has_match = matching_matrix.sum(dim=2) > 0.5
            target_indices = matching_matrix.argmax(dim=2)
            matched_targets_matrix[has_match] = target_indices[has_match]
        assigned_mask = matched_targets_matrix >= 0
        return assigned_mask, matched_targets_matrix


def assign(
    preds, targets, masks, img_w, img_h, cfg=None, current_iter=0, prior_ys=None
):
    """
    Unified assignment interface.
    """
    # Default to CLRNetAssign if not specified
    assign_method = CLRNetAssign(cfg)
    return assign_method(preds, targets, masks, img_w, img_h, current_iter, prior_ys)
