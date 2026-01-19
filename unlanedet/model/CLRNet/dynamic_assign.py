import torch
import torch.nn as nn
import torch.nn.functional as F
from .line_iou import line_iou


class DynamicAssign(nn.Module):
    def __init__(self, cfg=None):
        super(DynamicAssign, self).__init__()
        self.simota_q = 10
        self.w_cls = getattr(cfg, "w_cls", 3.0)
        self.w_iou = getattr(cfg, "w_iou", 3.0)
        self.w_reg = getattr(cfg, "w_reg", 3.0)

    def forward(self, preds, targets, masks, img_w, img_h):
        """
        Args:
            preds: (B, Num_Priors, Pred_Dim)
            targets: (B, Max_Targets, Target_Dim)
            masks: (B, Max_Targets)
        """
        device = preds.device
        batch_size, num_priors, _ = preds.shape
        _, max_targets, _ = targets.shape

        if max_targets == 0:
            return torch.zeros(
                (batch_size, num_priors), dtype=torch.bool, device=device
            ), torch.full((batch_size, num_priors), -1, dtype=torch.long, device=device)

        # 1. 提取并转换预测值 (Logits -> Probs)
        pred_logits = preds[..., :2]
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_scores = pred_probs[..., 1]  # (B, Num_Priors)

        pred_start_y = preds[..., 2]
        pred_start_x = preds[..., 3]
        pred_theta = preds[..., 4]

        # 2. 提取目标值
        gt_start_y = targets[..., 2]
        gt_start_x = targets[..., 3]
        gt_theta = targets[..., 4]

        # ========================================
        # A. 计算 Cost Matrix
        # ========================================

        # A.1 Classification Cost
        pair_wise_cls_cost = -torch.log(pred_scores.unsqueeze(2).clamp(min=1e-8))

        # A.2 Regression Cost
        cost_x = torch.abs(pred_start_x.unsqueeze(2) - gt_start_x.unsqueeze(1))
        cost_y = torch.abs(pred_start_y.unsqueeze(2) - gt_start_y.unsqueeze(1))
        cost_theta = torch.abs(pred_theta.unsqueeze(2) - gt_theta.unsqueeze(1))
        pair_wise_reg_cost = cost_x + cost_y + cost_theta

        # A.3 IoU Cost (修复维度广播)
        pred_lines = preds[..., 6:] * (img_w - 1)
        gt_lines = targets[..., 6:] * (img_w - 1)

        iou_cost_list = []
        ious_list = []

        for i in range(batch_size):
            valid_gt_mask = masks[i].bool()
            num_valid = int(valid_gt_mask.sum())

            if num_valid == 0:
                iou_cost_list.append(
                    torch.zeros(num_priors, max_targets, device=device)
                )
                ious_list.append(torch.zeros(num_priors, max_targets, device=device))
                continue

            # (Num_Priors, 1, 72) vs (1, Num_Valid, 72) -> 自动广播
            cur_pred_lines = pred_lines[i].unsqueeze(1)
            cur_gt_lines = gt_lines[i, :num_valid].unsqueeze(0)

            # (Num_Priors, Num_Valid)
            l_iou = line_iou(cur_pred_lines, cur_gt_lines, img_w, length=15)
            l_iou = torch.nan_to_num(l_iou, nan=0.0)  # 安全防护

            # Fill Padding
            padded_iou = torch.zeros(num_priors, max_targets, device=device)
            padded_iou[:, :num_valid] = l_iou

            padded_iou_cost = -torch.log(padded_iou[:, :num_valid].clamp(min=1e-8))
            full_iou_cost = torch.zeros(num_priors, max_targets, device=device)
            full_iou_cost[:, :num_valid] = padded_iou_cost

            iou_cost_list.append(full_iou_cost)
            ious_list.append(padded_iou)

        pair_wise_iou_cost = torch.stack(iou_cost_list)
        pair_wise_ious = torch.stack(ious_list)

        # A.4 Total Cost
        mask_expanded = masks.unsqueeze(1).expand(-1, num_priors, -1)
        total_cost = (
            self.w_cls * pair_wise_cls_cost
            + self.w_reg * pair_wise_reg_cost
            + self.w_iou * pair_wise_iou_cost
            + 100000.0 * (~mask_expanded.bool()).float()
        )

        # ========================================
        # B. SimOTA (简化且安全的版本)
        # ========================================
        n_candidate_k = min(self.simota_q, num_priors)
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1, max=num_priors)

        # 初始化分配矩阵
        matched_targets_matrix = torch.full(
            (batch_size, num_priors), -1, dtype=torch.long, device=device
        )
        # 记录当前每个 Prior 被分配时的最小 Cost，用于解决冲突
        matched_min_costs = torch.full(
            (batch_size, num_priors), 1e8, dtype=torch.float32, device=device
        )

        for b in range(batch_size):
            num_gt = int(masks[b].sum())
            if num_gt == 0:
                continue

            cost_b = total_cost[b, :, :num_gt]  # (Num_Priors, Num_GT)
            k_b = dynamic_ks[b, :num_gt]  # (Num_GT)

            # 遍历每个 GT 进行分配
            for gt_idx in range(num_gt):
                k = k_b[gt_idx].item()
                k = max(1, min(k, num_priors))  # 确保 k 有效

                # 获取该 GT cost 最小的 topk 个 prior
                current_costs = cost_b[:, gt_idx]
                topk_costs, pos_idx = torch.topk(current_costs, k, largest=False)

                # 安全更新逻辑：
                # 只有当 (Prior 未被分配) OR (Prior 已被分配但新 Cost 更小) 时，才更新

                # 1. 获取选中的 Prior 当前记录的最小 Cost
                current_min_costs = matched_min_costs[b, pos_idx]

                # 2. 判断是否需要更新 (New Cost < Current Min Cost)
                update_mask = topk_costs < current_min_costs

                # 3. 筛选出需要更新的 Prior 索引
                valid_indices = pos_idx[update_mask]
                valid_costs = topk_costs[update_mask]

                # 4. 执行更新
                if len(valid_indices) > 0:
                    matched_targets_matrix[b, valid_indices] = gt_idx
                    matched_min_costs[b, valid_indices] = valid_costs

        assigned_mask = matched_targets_matrix >= 0
        return assigned_mask, matched_targets_matrix


def assign(preds, targets, masks, img_w, img_h):
    assigner = DynamicAssign()
    return assigner(preds, targets, masks, img_w, img_h)
