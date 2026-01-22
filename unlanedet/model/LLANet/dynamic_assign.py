import torch
import torch.nn as nn
import torch.nn.functional as F
from .line_iou import line_iou


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

    def forward(self, preds, targets, masks, img_w, img_h, current_iter=0):
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
        delta_x = pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1)
        delta_y = pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)
        dist_cost = torch.sqrt(delta_x.pow(2) + delta_y.pow(2) + 1e-8)

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

            l_iou = line_iou(cur_pred, cur_gt, img_w, length=15, aligned=False)
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

    def forward(self, preds, targets, masks, img_w, img_h, current_iter=0):
        device = preds.device
        batch_size, num_priors, _ = preds.shape
        _, max_targets, _ = targets.shape

        if max_targets == 0:
            return torch.zeros(
                (batch_size, num_priors), dtype=torch.bool, device=device
            ), torch.full((batch_size, num_priors), -1, dtype=torch.long, device=device)

        # 准备输出容器
        matched_targets_matrix = torch.full(
            (batch_size, num_priors), -1, dtype=torch.long, device=device
        )

        # 预处理：转回绝对坐标
        preds_abs = preds.clone()
        preds_abs[..., 3] *= img_w - 1  # Start X
        preds_abs[..., 6:] *= img_w - 1  # Lane Points

        targets_abs = targets.clone()
        targets_abs[..., 3] *= img_w - 1
        targets_abs[..., 6:] *= img_w - 1

        for b in range(batch_size):
            num_gt = int(masks[b].sum())
            if num_gt == 0:
                continue

            cur_pred = preds_abs[b]
            cur_target = targets_abs[b, :num_gt]

            # --- 1. Distance Cost (Lane Points) ---
            valid_mask = (cur_target[:, 6:] >= 0) & (cur_target[:, 6:] < img_w)
            valid_len = valid_mask.sum(dim=1).float().clamp(min=1.0)

            diff = torch.abs(
                cur_pred[:, 6:].unsqueeze(1) - cur_target[:, 6:].unsqueeze(0)
            )
            diff = diff * valid_mask.unsqueeze(0)
            distances = diff.sum(dim=2) / valid_len.unsqueeze(0)

            distances_score = 1 - (distances / (torch.max(distances) + 1e-8)) + 1e-2

            # --- 2. Start XY Cost ---
            cur_pred_start = cur_pred[:, 2:4].clone()
            cur_pred_start[:, 0] *= img_h - 1
            cur_target_start = cur_target[:, 2:4].clone()
            cur_target_start[:, 0] *= img_h - 1

            start_dists = torch.cdist(cur_pred_start, cur_target_start, p=2)
            start_xys_score = 1 - (start_dists / (torch.max(start_dists) + 1e-8)) + 1e-2

            # --- 3. Theta Cost ---
            cur_pred_theta = cur_pred[:, 4].unsqueeze(1) * 180
            cur_target_theta = cur_target[:, 4].unsqueeze(0) * 180
            theta_dists = torch.abs(cur_pred_theta - cur_target_theta)
            theta_score = 1 - (theta_dists / (torch.max(theta_dists) + 1e-8)) + 1e-2

            # --- 4. Classification Cost ---
            pred_logits = cur_pred[:, :2]
            pred_probs = pred_logits.softmax(dim=1)[:, 1]
            cls_cost = (
                -torch.log(pred_probs.clamp(min=1e-8)).unsqueeze(1).expand(-1, num_gt)
            )

            # --- 5. Total Cost (乘法模型) ---
            reg_score = distances_score * start_xys_score * theta_score
            total_cost = -(reg_score**2) * self.w_reg + cls_cost * 1.0

            # --- 6. SimOTA (Dynamic K) ---
            # 【重要】这里同样要修复：传入 2D Tensor，不要 unsqueeze
            l_iou = line_iou(
                cur_pred[:, 6:], cur_target[:, 6:], img_w, length=15, aligned=False
            )
            l_iou = torch.nan_to_num(l_iou, nan=0.0)

            n_candidate_k = min(4, num_priors)
            topk_ious, _ = torch.topk(l_iou, n_candidate_k, dim=0)
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

            matching_matrix = torch.zeros_like(total_cost)
            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    total_cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[pos_idx, gt_idx] = 1.0

            matched_gt_counts = matching_matrix.sum(1)
            if (matched_gt_counts > 1).sum() > 0:
                conflict_mask = matched_gt_counts > 1
                conflict_costs = total_cost[conflict_mask]
                _, cost_argmin = torch.min(conflict_costs, dim=1)
                matching_matrix[conflict_mask] = 0.0
                matching_matrix[conflict_mask, cost_argmin] = 1.0

            prior_idx, gt_idx = matching_matrix.nonzero(as_tuple=True)
            matched_targets_matrix[b, prior_idx] = gt_idx

        assigned_mask = matched_targets_matrix >= 0
        return assigned_mask, matched_targets_matrix


def assign(preds, targets, masks, img_w, img_h, cfg=None, current_iter=0):
    """
    统一分配入口。
    通过 config 中的 'assign_method' 参数选择分配策略。
    默认使用 GeometryAwareAssign (加法模型)。
    """
    method = getattr(cfg, "assign_method", "GeometryAware")  # GeometryAware or CLRNet

    if method == "CLRNet":
        assigner = CLRNetAssign(cfg)
    else:
        assigner = GeometryAwareAssign(cfg)

    return assigner(preds, targets, masks, img_w, img_h, current_iter)
