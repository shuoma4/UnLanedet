import torch
import torch.nn as nn
import torch.nn.functional as F


class ContMixT(nn.Module):
    """ContMix-T: Temporal Feature Aggregation Module (Section 4.2.2)

    Three-stage architecture:
      (1) Overview-Net (GCM) — Eq 4-1
      (2) Focus-Net dynamic convolution — Eq 4-2, 4-3
      (3) Dynamic fusion — Eq 4-4, 4-5, 4-6
    """

    def __init__(self, in_channels, hidden_channels=None):
        super(ContMixT, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.in_channels = in_channels
        self.k = 3

        # (1) Overview-Net: DilatedRepConv ×2 → GAP → Conv1×1 → G  (Eq 4-1)
        self.overview_net = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        # (2) Focus-Net: FC1 + SiLU → FC2 → dynamic weights W  (Eq 4-2, Fig 4-3)
        self.fc1 = nn.Linear(in_channels * 2, 512)
        self.fc2 = nn.Linear(512, in_channels * self.k * self.k)

        # (3) Adaptive fusion weight α — 两处融合各自独立，避免梯度干扰
        # alpha_prev: 融合 f_{t-1} 和 f_{t-2} 构建历史特征
        self.alpha_conv_prev = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        # alpha_fusion: 融合当前帧与历史特征（偏置初始化为正，使当前帧默认主导）
        self.alpha_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)
        nn.init.constant_(self.alpha_conv.bias, 1.0)   # sigmoid(1)≈0.73 → alpha_init≈0.82

    def _focus_net_pair(self, G, f_a, f_b, B, C, H, W):
        """对同一 G 下的两帧特征各做一次 Focus-Net，合并为单次 FC + 单次 depthwise conv。"""
        F_stacked = torch.cat([f_a, f_b], dim=0)
        g_flat = G.reshape(B, C).repeat_interleave(2, dim=0)
        gap = self.global_pool(F_stacked).reshape(2 * B, C)
        fc_in = torch.cat([g_flat, gap], dim=1)
        w = self.fc2(F.silu(self.fc1(fc_in)))
        w = w.reshape(2 * B * C, 1, self.k, self.k)
        x = F_stacked.reshape(1, 2 * B * C, H, W)
        pad = self.k // 2
        out = F.conv2d(x, w, padding=pad, groups=2 * B * C)
        out = out.reshape(2 * B, C, H, W)
        return out[:B], out[B:]

    def forward(self, f_t_minus_2, f_t_minus_1, f_t):
        B, C, H, W = f_t.shape

        # (1) Overview-Net: Global context from all 3 frames → G ∈ R^{C×1×1}  (Eq 4-1)
        f_concat = torch.cat([f_t_minus_2, f_t_minus_1, f_t], dim=1)
        g_feat = self.overview_net(f_concat)
        g_pooled = self.global_pool(g_feat)
        G = self.g_conv(g_pooled)  # [B, C, 1, 1]

        # (2)(3) Focus-Net：F_t 与 F_{t-1} 共享一次 batched 前向
        f_t_mod, f_t1_mod = self._focus_net_pair(G, f_t, f_t_minus_1, B, C, H, W)

        # 构建历史特征 f_prev：用独立的 alpha_conv_prev，平衡 t-1 与 t-2
        f_prev_cat = torch.cat([f_t1_mod, f_t_minus_2], dim=1)
        alpha_prev = 0.3 + 0.4 * torch.sigmoid(self.alpha_conv_prev(f_prev_cat))
        f_prev = alpha_prev * f_t1_mod + (1 - alpha_prev) * f_t_minus_2

        # (4) Dynamic Fusion: F_t^enhanced  (Eq 4-5, 4-6)
        # alpha_conv 偏置初始化为 1.0 → sigmoid≈0.73 → alpha_init≈0.82
        # 确保训练初期当前帧贡献 ≥ 60%，避免历史噪声污染检测特征
        f_cat_fusion = torch.cat([f_t_mod, f_prev], dim=1)
        alpha = 0.6 + 0.3 * torch.sigmoid(self.alpha_conv(f_cat_fusion))  # α ∈ [0.6, 0.9]

        f_enhanced = alpha * f_t_mod + (1 - alpha) * f_prev
        return f_enhanced


class TemporalFusionWrapper(nn.Module):
    def __init__(self, in_channels=64, num_levels=3, cfg=None):
        super(TemporalFusionWrapper, self).__init__()
        self.num_levels = num_levels
        self.mixers = nn.ModuleList([
            ContMixT(in_channels=in_channels, hidden_channels=in_channels)
            for _ in range(num_levels)
        ])

        from .temporal import TemporalConsistencyLoss
        self.temporal_loss = TemporalConsistencyLoss(loss_weight=1.0, cfg=cfg)

    def forward(self, sequence_features):
        T = len(sequence_features)
        enhanced_levels = []

        for level in range(self.num_levels):
            if T >= 3:
                f_t_minus_2 = sequence_features[-3][level]
            else:
                f_t_minus_2 = sequence_features[0][level]
            f_t_minus_1 = sequence_features[-2][level] if T >= 2 else sequence_features[0][level]
            f_t = sequence_features[-1][level]

            f_enhanced = self.mixers[level](f_t_minus_2, f_t_minus_1, f_t)
            enhanced_levels.append(f_enhanced)

        return enhanced_levels, {'temporal_consistency_loss': None}
