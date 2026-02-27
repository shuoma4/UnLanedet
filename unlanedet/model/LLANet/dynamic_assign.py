import torch
import torch.nn as nn
import torch.nn.functional as F

from .line_iou import pairwise_line_iou


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    计算 Focal Loss 风格的分类匹配代价 (Classification Cost)，用于 SimOTA 动态分配阶段。
    Args:
        cls_pred (Tensor):
            预测分类 logits。
            - 非 Batch: [num_priors, num_classes]
            - Batch:     [batch_size, num_priors, num_classes]
        gt_labels (Tensor):
            GT 类别标签索引。
            - 非 Batch: [num_targets]
            - Batch:     [batch_size, num_targets]

        alpha (float): Focal Loss 中正样本权重系数。
        gamma (float): Focal Loss 中难样本调制指数。
        eps (float):   数值稳定项，防止 log(0)。

    Returns:
        cls_cost (Tensor):
            分类代价矩阵。
            - 非 Batch: [num_priors, num_targets]
            - Batch:     [batch_size, num_priors, num_targets]
            数值越小，表示预测 prior 与 GT 在分类层面越匹配。
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)

    if cls_pred.dim() == 3:
        # Batch mode: [B, N, C] vs [B, M]
        gt_labels_expanded = gt_labels.unsqueeze(1).expand(-1, cls_pred.shape[1], -1)  # [B, N, M]
        cls_cost = pos_cost.gather(2, gt_labels_expanded) - neg_cost.gather(2, gt_labels_expanded)
    else:
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


class GeometryAwareAssign(nn.Module):
    """
    [方案 A] 几何感知动态分配器 (Geometry-Enhanced SimOTA, Additive Cost Model)

    Cost 构成：
        Cost = w_cls * C_cls + w_geom * C_geom + w_iou * C_iou

    其中：
        - C_cls  ：分类代价
        - C_geom ：几何代价（起点距离 + 角度差异）
        - C_iou  ：曲线 IoU 代价

    该分配器强调：
        - 车道线起点几何一致性
        - 车道线方向一致性
        - 车道线整体曲线重合程度
    """

    def __init__(self, cfg=None):
        super(GeometryAwareAssign, self).__init__()
        self.simota_q = 10

        self.w_cls = getattr(cfg, 'w_cls', 4.0)
        self.w_geom = getattr(cfg, 'w_geom', 5.0)
        self.w_iou = getattr(cfg, 'w_iou', 2.0)

        # 动态权重配置
        self.start_w_cls = getattr(cfg, 'start_w_cls', 0.5)
        self.warmup_epochs = getattr(cfg, 'warmup_epochs', 5)
        self.epoch_per_iter = getattr(cfg, 'epoch_per_iter', 1)

        self.w_dist = 1.0
        self.w_theta = 2.0

    def forward(self, preds, targets, masks, img_w, img_h, current_iter=0, sample_ys=None):
        device = preds.device
        batch_size, num_priors, _ = preds.shape
        _, max_targets, _ = targets.shape

        if max_targets == 0:
            return torch.zeros((batch_size, num_priors), dtype=torch.bool, device=device), torch.full(
                (batch_size, num_priors), -1, dtype=torch.long, device=device
            )

        pred_start_y = preds[..., 2]
        pred_start_x = preds[..., 3]
        pred_theta = preds[..., 4]
        pred_delta_x = preds[..., 6:]

        if sample_ys is None:
            sample_ys = torch.linspace(0, 1, steps=pred_delta_x.shape[-1], device=device)
            sample_ys = sample_ys.view(1, 1, -1) * (img_h - 1)
        else:
            sample_ys = sample_ys.to(device).view(1, 1, -1)

        pred_tan_theta = torch.tan(torch.deg2rad(pred_theta))
        pred_tan_theta = torch.clamp(pred_tan_theta, -1e3, 1e3)
        pred_x_prior = pred_start_x.unsqueeze(-1) + (pred_start_y.unsqueeze(-1) - sample_ys) * pred_tan_theta.unsqueeze(
            -1
        )
        pred_xs_abs = pred_x_prior + pred_delta_x

        gt_start_y = targets[..., 2]
        gt_start_x = targets[..., 3]
        gt_theta = targets[..., 4]
        gt_delta_x = targets[..., 6:]

        gt_tan_theta = torch.tan(torch.deg2rad(gt_theta))
        gt_tan_theta = torch.clamp(gt_tan_theta, -1e3, 1e3)
        gt_x_prior = gt_start_x.unsqueeze(-1) + (gt_start_y.unsqueeze(-1) - sample_ys) * gt_tan_theta.unsqueeze(-1)
        gt_xs_abs = gt_x_prior + gt_delta_x  # Absolute Pixel Coordinates

        invalid_mask = gt_delta_x < -1e4
        gt_xs_abs[invalid_mask] = -1e5

        pred_logits = preds[..., :2]
        pred_scores = F.softmax(pred_logits, dim=-1)[..., 1]

        gt_start_y_norm = gt_start_y / (img_h - 1)
        gt_start_x_norm = gt_start_x / (img_w - 1)

        pred_start_y_norm = preds[..., 2] / (img_h - 1)
        pred_start_x_norm = preds[..., 3] / (img_w - 1)

        delta_x = pred_start_x_norm.unsqueeze(2) - gt_start_x_norm.unsqueeze(1)
        delta_y = pred_start_y_norm.unsqueeze(2) - gt_start_y_norm.unsqueeze(1)
        dist_cost = torch.sqrt(delta_x.pow(2) + delta_y.pow(2) + 1e-8)

        gt_theta_norm = gt_theta / 90.0
        pred_theta_norm = preds[..., 4] / 90.0
        theta_cost = torch.abs(pred_theta_norm.unsqueeze(2) - gt_theta_norm.unsqueeze(1))

        geom_cost = self.w_dist * dist_cost + self.w_theta * theta_cost
        cls_cost = -torch.log(pred_scores.unsqueeze(2).clamp(min=1e-8))

        # 4. IoU Cost (Pixel Space)
        n_offsets = preds.shape[-1] - 6
        offset_index = torch.arange(n_offsets, device=device).view(1, 1, -1)
        # approximate index from physical start_y
        pred_start_idx = (1.0 - (preds[..., 2] / (img_h - 1))) * (n_offsets - 1)
        # approximate length in indices from physical length
        pred_len_idx = (preds[..., 5] / img_h) * (n_offsets - 1)
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

            cur_pred = pred_xs_abs[i].clone()
            cur_pred[~pred_mask[i]] = -1e5

            # Fix: Select valid targets using mask, not slicing (handles non-contiguous valid targets)
            cur_gt = gt_xs_abs[i][valid_gt_mask]

            l_iou = pairwise_line_iou(cur_pred, cur_gt, length=15, invalid_value=-1e5)
            l_iou = torch.nan_to_num(l_iou, nan=0.0)
            padded_iou = torch.zeros(num_priors, max_targets, device=device)
            # Fix: Place IoUs back to correct positions
            padded_iou[:, valid_gt_mask] = l_iou
            ious_list.append(padded_iou)
        pair_wise_ious = torch.stack(ious_list)
        iou_cost = 1.0 - pair_wise_ious
        mask_expanded = masks.unsqueeze(1).expand(-1, num_priors, -1)
        total_cost = (
            self.w_cls * cls_cost
            + self.w_geom * geom_cost
            + self.w_iou * iou_cost
            + 100000.0 * (~mask_expanded.bool()).float()
        )
        n_candidate_k = min(self.simota_q, num_priors)
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1, max=num_priors)
        matched_targets_matrix = torch.full((batch_size, num_priors), -1, dtype=torch.long, device=device)
        matched_min_costs = torch.full((batch_size, num_priors), 1e8, dtype=torch.float32, device=device)
        for b in range(batch_size):
            valid_gt_idxs = torch.where(masks[b])[0]
            num_gt = len(valid_gt_idxs)
            if num_gt == 0:
                continue

            # Fix: Use valid indices to slice cost and k
            cost_b = total_cost[b][:, valid_gt_idxs]
            k_b = dynamic_ks[b][valid_gt_idxs]

            for i, gt_idx in enumerate(valid_gt_idxs):
                gt_idx = gt_idx.item()
                k = k_b[i].item()
                k = max(1, min(k, num_priors))
                current_costs = cost_b[:, i]
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
    def __init__(self, cfg=None):
        super().__init__()
        self.w_reg = 3.0
        self.w_cls = 1.0
        self.simota_q = 4

    def forward(
        self,
        preds,  # [B, N, C] 真实坐标
        targets,  # [B, M, C] 真实坐标
        masks,  # [B, M]
        img_w,
        img_h,
        current_iter=0,
        sample_ys=None,
    ):
        device = preds.device
        B, N, _ = preds.shape

        assigned_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        matched_targets_matrix = torch.full((B, N), -1, dtype=torch.long, device=device)

        for b in range(B):
            valid_mask = masks[b].bool()
            if valid_mask.sum() == 0:
                continue

            pred = preds[b].detach()  # [N, C]
            tgt = targets[b][valid_mask].detach()  # [M, C]
            num_priors = pred.shape[0]
            num_targets = tgt.shape[0]

            # ===== Distance cost (offset 序列距离)=====
            distances = torch.abs(pred[:, None, 6:] - tgt[None, :, 6:])  # [N, M, L]
            invalid_mask = (tgt[None, :, 6:] < 0) | (tgt[None, :, 6:] >= img_w)  # [1, M, L]
            valid_len = (~invalid_mask).sum(dim=-1).clamp(min=1)
            distances = distances.masked_fill(invalid_mask, 0.0)
            distances = distances.sum(dim=-1) / (valid_len.float() + 1e-6)

            distances_score = 1 - distances / distances.max().clamp(min=1e-6) + 1e-2

            # ===== 分类 cost =====
            cls_cost = focal_cost(pred[:, :2], tgt[:, 1].long())  # [N, M]

            # ===== 起点坐标 cost =====
            start_dists = torch.cdist(pred[:, 2:4], tgt[:, 2:4], p=2)
            start_xys_score = 1 - start_dists / start_dists.max().clamp(min=1e-6) + 1e-2

            # ===== 角度 cost =====
            theta_dists = torch.cdist(pred[:, 4].unsqueeze(-1), tgt[:, 4].unsqueeze(-1), p=1)
            theta_score = 1 - theta_dists / theta_dists.max().clamp(min=1e-6) + 1e-2

            # ===== 总 cost =====
            cost = -((distances_score * start_xys_score * theta_score) ** 2) * self.w_reg + cls_cost * self.w_cls

            # ===== IoU =====
            iou = pairwise_line_iou(pred[:, 6:], tgt[:, 6:], img_w, length=15)  # [N, M]

            # ===== dynamic-k（向量化）=====
            iou_matrix = iou.clamp(min=0.0)
            topk_ious, _ = torch.topk(iou_matrix, k=min(self.simota_q, num_priors), dim=0)
            dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

            max_k = dynamic_ks.max().item()
            cost_t = cost.t()  # [M, N]
            _, topk_indices = torch.topk(cost_t, k=max_k, dim=1, largest=False)

            grid = torch.arange(max_k, device=device).view(1, -1)
            select_mask = grid < dynamic_ks.view(-1, 1)

            matching_matrix = torch.zeros_like(cost)
            matching_matrix.scatter_(
                0,
                topk_indices.t(),
                select_mask.t().float(),
            )

            matched_gt = matching_matrix.sum(1)
            conflict = matched_gt > 1
            if conflict.any():
                min_cost_idx = cost[conflict].argmin(dim=1)
                matching_matrix[conflict] *= 0.0
                matching_matrix[conflict, min_cost_idx] = 1.0

            prior_idx = matching_matrix.sum(1).nonzero().flatten()
            gt_idx_relative = matching_matrix[prior_idx].argmax(dim=1)

            # Map back to original indices
            valid_gt_idxs = torch.where(valid_mask)[0]
            gt_idx_original = valid_gt_idxs[gt_idx_relative]

            assigned_mask[b, prior_idx] = True
            matched_targets_matrix[b, prior_idx] = gt_idx_original

        return assigned_mask, matched_targets_matrix


def assign(preds, targets, masks, img_w, img_h, cfg=None, current_iter=0, sample_ys=None):
    """
    Args:
        preds (Tensor): 网络输出预测张量
            [B, N, C]，其中：
            - C 前 2 维：分类 logits(0-背景logits, 1-车道线logits)
            - C 第 2 维：起点 y, pts
            - C 第 3 维：起点 x, pts
            - C 第 4 维：车道线角度 theta(-90~90°)
            - C 第 5 维：车道线长度, 归一化到, pts
            - C 第 6:  : 各条 strip 上的横向坐标偏移, pts

        targets (Tensor): GT 车道线参数
            [B, M, C]，其中：
            - C 前 2 维：分类 logits(0-背景logits, 1-车道线logits)
            - C 第 2 维：起点 y, pts
            - C 第 3 维：起点 x, pts
            - C 第 4 维：车道线角度 theta(-90~90°)
            - C 第 5 维：车道线长度, pts
            - C 第 6:  : 各条 strip 上的横向坐标偏移, pts

        masks (Tensor): GT 有效性掩码
            [B, M]，True 表示该 GT 有效，False 表示 padding 的无效 GT。

        img_w (int): 输入图像宽度（像素）
        img_h (int): 输入图像高度（像素）

    Returns:
        assigned_mask (Tensor):
            [B, N]，bool 类型，表示每个 prior 是否被分配为正样本。

        matched_targets_matrix (Tensor):
            [B, N]，long 类型，表示每个 prior 匹配到的 GT 索引，
            若为 -1 表示该 prior 为负样本。
    """
    # Default to CLRNetAssign if not specified
    assign_method = CLRNetAssign(cfg)
    return assign_method(preds, targets, masks, img_w, img_h, current_iter, sample_ys)
