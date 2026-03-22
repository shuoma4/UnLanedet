import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_homography_from_poses(extrinsic_k, extrinsic_t, intrinsic, z=0):
    # 简化：使用内参和外参计算单应矩阵，假设路面 z=0
    # 由于实现暂不包含完整的3D反投影逻辑，我们这里可以用一个基础版：
    H = torch.eye(3, device=extrinsic_k.device)
    return H


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, current_preds, previous_preds, batch=None):
        if current_preds is None or previous_preds is None:
            return current_preds.new_tensor(0.0) if current_preds is not None else torch.tensor(0.0)
            
        # 兼容不同输入序列的情况 (支持降级模式)
        min_priors = min(current_preds.shape[1], previous_preds.shape[1])
        curr_p = current_preds[:, :min_priors]
        prev_p = previous_preds[:, :min_priors].detach()
        
        # 1. 约束分类概率分布的连续性 (包含背景和前景)
        curr_prob = F.softmax(curr_p[..., :2], dim=-1)
        prev_prob = F.softmax(prev_p[..., :2], dim=-1)
        
        # 使用 MSE 使得分类概率保持平滑的过渡，稍微放大平衡量级
        cls_loss = F.mse_loss(curr_prob, prev_prob) * 5.0
        
        # 2. 约束具有高置信度前排目标的几何位置连续性 (排除大量冗余且无意义的背景anchor造成的0梯度中和)
        # 取上一帧被网络认为是实体车道线的 prior 作为追踪 anchor (阈值0.1)
        fg_mask = (prev_prob[..., 1] > 0.1).float()
        
        if fg_mask.sum() > 0:
            # 仅对存在可能车道线的前景计算 L1 平滑偏移 (包括起点、角度、长度等维度的回归)
            reg_diff = F.smooth_l1_loss(curr_p[..., 2:6], prev_p[..., 2:6], reduction='none')
            # 沿最后一维求均值后，使用前景掩膜提取
            reg_loss = (reg_diff.mean(dim=-1) * fg_mask).sum() / (fg_mask.sum() + 1e-5)
            # 乘以一个合理的系数
            reg_loss = reg_loss * 10.0
        else:
            reg_loss = curr_p.new_tensor(0.0)
            
        loss = cls_loss + reg_loss
        return loss * self.loss_weight


class ContMixTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        
        # Overview-Net：DilatedRepConv × 2 → GAP → Conv1×1 → G
        self.overview_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        
        # Focus-Net：动态卷积核生成
        self.fc1 = nn.Linear(channels * 2, 512)
        self.fc2 = nn.Linear(512, channels * kernel_size * kernel_size)
        
        # 自适应融合权重
        self.alpha_proj = nn.Sequential(
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, current_feat, prev_feat=None, guidance=None):
        if prev_feat is None:
            return current_feat
        if prev_feat.shape[-2:] != current_feat.shape[-2:]:
            prev_feat = F.interpolate(prev_feat, size=current_feat.shape[-2:], mode='bilinear', align_corners=False)

        B, C, H, W = current_feat.shape
        mix_input = torch.cat([current_feat, prev_feat], dim=1)
        
        # 1. Overview-Net
        G = self.overview_net(mix_input)  # B, C, 1, 1
        
        # 2. Focus-Net
        gap_Ft = F.adaptive_avg_pool2d(current_feat, 1)  # B, C, 1, 1
        concat_feats = torch.cat([G.view(B, C), gap_Ft.view(B, C)], dim=1)  # B, 2C
        
        x_fc1 = F.silu(self.fc1(concat_feats))
        W_dyn = self.fc2(x_fc1)  # B, C*K*K
        
        W_dyn = W_dyn.view(B * C, 1, self.kernel_size, self.kernel_size)
        x = current_feat.view(1, B * C, H, W)
        pad = self.kernel_size // 2
        out = F.conv2d(x, W_dyn, padding=pad, groups=B * C)
        out = out.view(B, C, H, W)
        
        # 3. 自适应融合
        alpha = self.alpha_proj(mix_input)
        alpha = torch.clamp(alpha, min=0.3, max=0.7)
        
        fused = alpha * out + (1.0 - alpha) * prev_feat
        
        return current_feat + fused


class ContMixTemporalAggregator(nn.Module):
    def __init__(self, in_channels, temporal_weight=1.0):
        super().__init__()
        self.blocks = nn.ModuleList([ContMixTemporalBlock(c) for c in in_channels])
        self.temporal_loss = TemporalConsistencyLoss(loss_weight=temporal_weight)

    def forward(self, sequence_features, stage_predictions=None):
        if not sequence_features:
            return None, {}
        if len(sequence_features) == 1:
            aux = {'temporal_consistency_loss': None}
            return sequence_features[0], aux

        aggregated = sequence_features[0]
        for t in range(1, len(sequence_features)):
            current = sequence_features[t]
            guidance = aggregated[-1]
            aggregated = [
                block(curr_feat, prev_feat, guidance if idx == len(current) - 1 else None)
                for idx, (block, curr_feat, prev_feat) in enumerate(zip(self.blocks, current, aggregated))
            ]

        temporal_loss = None
        return aggregated, {'temporal_consistency_loss': temporal_loss}
