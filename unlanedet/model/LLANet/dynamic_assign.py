import torch
import torch.nn as nn
import torch.nn.functional as F

from .line_iou import pairwise_line_iou


def focal_cost(cls_pred, gt_labels, alpha=0.25, gamma=2, eps=1e-12):
    """
    计算 Focal Loss 风格的分类匹配代价
    完全支持 [B, N, M] 向量化计算
    """
    cls_pred = cls_pred.sigmoid()
    neg_cost = -(1 - cls_pred + eps).log() * (1 - alpha) * cls_pred.pow(gamma)
    pos_cost = -(cls_pred + eps).log() * alpha * (1 - cls_pred).pow(gamma)

    if cls_pred.dim() == 3:
        # [B, N, C] vs[B, M] -> [B, N, M]
        gt_labels_expanded = gt_labels.unsqueeze(1).expand(-1, cls_pred.shape[1], -1)
        cls_cost = pos_cost.gather(2, gt_labels_expanded) - neg_cost.gather(2, gt_labels_expanded)
    else:
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
    return cls_cost


class GeometryAwareAssign(nn.Module):
    """
    [极速版] 几何感知动态分配器 (Geometry-Enhanced SimOTA)
    """

    def __init__(self, cfg=None):
        super(GeometryAwareAssign, self).__init__()
        self.simota_q = getattr(cfg, 'simota_q', 10)

        self.w_cls = getattr(cfg, 'w_cls', 4.0)
        self.w_geom = getattr(cfg, 'w_geom', 5.0)
        self.w_iou = getattr(cfg, 'w_iou', 2.0)

        self.w_dist = 1.0
        self.w_theta = 2.0

    def forward(self, preds, targets, masks, img_w, img_h, current_iter=0, sample_ys=None):
        device = preds.device
        B, N, _ = preds.shape
        _, M, _ = targets.shape

        if M == 0:
            return torch.zeros((B, N), dtype=torch.bool, device=device), torch.full(
                (B, N), -1, dtype=torch.long, device=device
            )

        # 1. 提取真实物理坐标
        pred_start_y = preds[..., 2]
        pred_start_x = preds[..., 3]
        pred_theta = preds[..., 4]
        pred_length = preds[..., 5]
        pred_delta_x = preds[..., 6:]

        gt_start_y = targets[..., 2]
        gt_start_x = targets[..., 3]
        gt_theta = targets[..., 4]
        gt_delta_x = targets[..., 6:]

        if sample_ys is None:
            n_offsets = pred_delta_x.shape[-1]
            sample_ys = torch.linspace(img_h - 1, 0, steps=n_offsets, device=device).view(1, 1, -1)
        else:
            sample_ys = sample_ys.to(device).view(1, 1, -1)

        # 2. 并行构建绝对坐标系[B, N, L] 和 [B, M, L]
        pred_tan_theta = torch.clamp(torch.tan(torch.deg2rad(pred_theta)), -1e3, 1e3)
        pred_x_prior = pred_start_x.unsqueeze(-1) + (pred_start_y.unsqueeze(-1) - sample_ys) * pred_tan_theta.unsqueeze(
            -1
        )
        pred_xs_abs = pred_x_prior + pred_delta_x

        gt_tan_theta = torch.clamp(torch.tan(torch.deg2rad(gt_theta)), -1e3, 1e3)
        gt_x_prior = gt_start_x.unsqueeze(-1) + (gt_start_y.unsqueeze(-1) - sample_ys) * gt_tan_theta.unsqueeze(-1)
        gt_xs_abs = gt_x_prior + gt_delta_x

        # 3. 截断无效与越界点 (Masked Fill)
        y_min = pred_start_y - pred_length
        y_max = pred_start_y
        pred_mask = (sample_ys >= y_min.unsqueeze(-1)) & (sample_ys <= y_max.unsqueeze(-1))  # [B, N, L]
        pred_xs_abs = pred_xs_abs.masked_fill(~pred_mask, -1e5)

        invalid_mask = (gt_delta_x < -1e4) | (gt_xs_abs < 0) | (gt_xs_abs >= img_w)  # [B, M, L]
        gt_xs_abs = gt_xs_abs.masked_fill(invalid_mask, -1e5)

        # 4. IoU 代价 [B, N, M] (直接传递 3D 张量到 pairwise_line_iou 计算矩阵)
        pair_wise_ious = pairwise_line_iou(pred_xs_abs, gt_xs_abs, img_w, length=15, invalid_value=-1e5)
        pair_wise_ious = torch.nan_to_num(pair_wise_ious, nan=0.0)
        pair_wise_ious = pair_wise_ious.masked_fill(~masks.unsqueeze(1), 0.0)  # 屏蔽无效 GT
        iou_cost = 1.0 - pair_wise_ious

        # 5. 几何代价 [B, N, M]
        delta_x = (pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1)) / (img_w - 1)
        delta_y = (pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1)) / (img_h - 1)
        dist_cost = torch.sqrt(delta_x.pow(2) + delta_y.pow(2) + 1e-8)

        theta_cost = torch.abs((pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1)) / 90.0)
        geom_cost = self.w_dist * dist_cost + self.w_theta * theta_cost

        # 6. 分类代价[B, N, M]
        pred_logits = preds[..., :2]
        pred_scores = F.softmax(pred_logits, dim=-1)[..., 1]
        cls_cost = -torch.log(pred_scores.unsqueeze(2).clamp(min=1e-8))

        # 7. 总 Cost 矩阵 [B, N, M]
        total_cost = self.w_cls * cls_cost + self.w_geom * geom_cost + self.w_iou * iou_cost
        total_cost = total_cost.masked_fill(~masks.unsqueeze(1), 100000.0)  # 将 Padding GT 的 cost 设为极大值

        # ----------------------------------------------------
        # 8. 极速向量化 SimOTA 分配 (Dynamic-K)
        # ----------------------------------------------------
        n_candidate_k = min(self.simota_q, N)
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)  # [B, K, M]
        dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1, max=N)  # [B, M]

        max_k = dynamic_ks.max().item()  # 找出当前 Batch 内的最大 K，统一申请张量
        topk_costs, topk_indices = torch.topk(total_cost, max_k, dim=1, largest=False)  # [B, max_k, M]

        # 构建动态 K 的索引掩码
        k_mask = torch.arange(max_k, device=device).view(1, -1, 1) < dynamic_ks.unsqueeze(1)  # [B, max_k, M]

        # 投射分配矩阵
        matching_matrix = torch.zeros_like(total_cost)  # [B, N, M]
        matching_matrix.scatter_(1, topk_indices, k_mask.float())
        matching_matrix.masked_fill_(~masks.unsqueeze(1), 0.0)  # 二次保护，不分配给无效 GT

        # 9. 冲突解决 (多对一转一对一)
        matched_gt = matching_matrix.sum(dim=2)  # [B, N]
        conflict_mask = matched_gt > 1  # [B, N]

        if conflict_mask.any():
            # 取出只在分配区域内的 cost，非分配区置为 inf
            cost_with_inf = total_cost.masked_fill(matching_matrix == 0, float('inf'))
            min_cost_idx = cost_with_inf.argmin(dim=2)  # 找到冲突 prior 对应 cost 最小的 GT 索引

            # 清零冲突行
            matching_matrix[conflict_mask] = 0.0

            # 重新打上最小 Cost 对应 GT 的标签
            b_idx, n_idx = torch.where(conflict_mask)
            matching_matrix[b_idx, n_idx, min_cost_idx[b_idx, n_idx]] = 1.0

        # 10. 解析输出
        assigned_mask = matching_matrix.sum(dim=2) > 0  # [B, N]
        matched_targets_matrix = matching_matrix.argmax(dim=2)  # [B, N]
        matched_targets_matrix[~assigned_mask] = -1

        return assigned_mask, matched_targets_matrix


class CLRNetAssign(nn.Module):
    """
    [极速版] 兼容第二版老旧算法逻辑的分配器
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.w_reg = 3.0
        self.w_cls = 1.0
        self.simota_q = getattr(cfg, 'simota_q', 4)

    def forward(self, preds, targets, masks, img_w, img_h, current_iter=0, sample_ys=None):
        device = preds.device
        B, N, _ = preds.shape
        _, M, _ = targets.shape

        if M == 0:
            return torch.zeros((B, N), dtype=torch.bool, device=device), torch.full(
                (B, N), -1, dtype=torch.long, device=device
            )

        pred_delta_x = preds[..., 6:]  # [B, N, L]
        tgt_delta_x = targets[..., 6:]  # [B, M, L]

        # ===== 1. Distance cost [B, N, M] =====
        invalid_masks = (tgt_delta_x < 0) | (tgt_delta_x >= img_w) | (tgt_delta_x < -1e4)  # [B, M, L]
        lengths = (~invalid_masks).sum(dim=-1).clamp(min=1)  # [B, M]

        distances = torch.abs(tgt_delta_x.unsqueeze(1) - pred_delta_x.unsqueeze(2))  # [B, N, M, L]
        distances = distances.masked_fill(invalid_masks.unsqueeze(1), 0.0)
        distances = distances.sum(dim=-1) / lengths.unsqueeze(1).float()  # [B, N, M]

        max_dist = distances.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        distances_score = 1 - (distances / max_dist) + 1e-2

        # ===== 2. 起点 cost [B, N, M] =====
        start_xys_score = torch.cdist(preds[..., 2:4], targets[..., 2:4], p=2)
        max_xy = start_xys_score.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        start_xys_score = 1 - (start_xys_score / max_xy) + 1e-2

        # ===== 3. 角度 cost [B, N, M] =====
        theta_score = torch.cdist(preds[..., 4:5], targets[..., 4:5], p=1)
        max_theta = theta_score.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        theta_score = 1 - (theta_score / max_theta) + 1e-2

        # ===== 4. 分类 cost [B, N, M] =====
        cls_cost = focal_cost(preds[..., :2], targets[..., 1].long())

        # ===== 5. 总组合代价 [B, N, M] =====
        cost = -((distances_score * start_xys_score * theta_score) ** 2) * self.w_reg + cls_cost * self.w_cls
        cost = cost.masked_fill(~masks.unsqueeze(1), 100000.0)

        # ===== 6. 退化版 IoU =====
        iou = pairwise_line_iou(pred_delta_x, tgt_delta_x, img_w, length=15, invalid_value=-1e5)
        iou = iou.masked_fill(~masks.unsqueeze(1), 0.0)

        # ===== 7. 极速向量化 SimOTA =====
        ious_matrix = iou.clamp(min=0.0)
        n_candidate_k = min(self.simota_q, N)
        topk_ious, _ = torch.topk(ious_matrix, n_candidate_k, dim=1)  # [B, K, M]
        dynamic_ks = torch.clamp(topk_ious.sum(dim=1).int(), min=1, max=N)  # [B, M]

        max_k = dynamic_ks.max().item()
        topk_costs, topk_indices = torch.topk(cost, max_k, dim=1, largest=False)  # [B, max_k, M]

        k_mask = torch.arange(max_k, device=device).view(1, -1, 1) < dynamic_ks.unsqueeze(1)  # [B, max_k, M]

        matching_matrix = torch.zeros_like(cost)
        matching_matrix.scatter_(1, topk_indices, k_mask.float())
        matching_matrix.masked_fill_(~masks.unsqueeze(1), 0.0)

        matched_gt = matching_matrix.sum(dim=2)  # [B, N]
        conflict_mask = matched_gt > 1

        if conflict_mask.any():
            cost_with_inf = cost.masked_fill(matching_matrix == 0, float('inf'))
            min_cost_idx = cost_with_inf.argmin(dim=2)
            matching_matrix[conflict_mask] = 0.0

            b_idx, n_idx = torch.where(conflict_mask)
            matching_matrix[b_idx, n_idx, min_cost_idx[b_idx, n_idx]] = 1.0

        assigned_mask = matching_matrix.sum(dim=2) > 0  # [B, N]
        matched_targets_matrix = matching_matrix.argmax(dim=2)  # [B, N]
        matched_targets_matrix[~assigned_mask] = -1

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
    method_name = getattr(cfg, 'assign_method', 'CLRNet')
    if method_name == 'GeometryAware':
        assign_method = GeometryAwareAssign(cfg)
    else:
        assign_method = CLRNetAssign(cfg)
    return assign_method(preds, targets, masks, img_w, img_h, current_iter, sample_ys)
