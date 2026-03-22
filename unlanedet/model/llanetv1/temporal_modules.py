import torch
import torch.nn as nn
import torch.nn.functional as F

class ContMixT(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super(ContMixT, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels

        self.k = 3
        self.overview_net = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.g_conv = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        self.fc1 = nn.Linear(in_channels * 2, 512)
        self.fc2 = nn.Linear(512, in_channels * self.k * self.k)
        
        self.alpha_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=1)

    def forward(self, f_t_minus_2, f_t_minus_1, f_t):
        B, C, H, W = f_t.shape
        
        f_concat = torch.cat([f_t_minus_2, f_t_minus_1, f_t], dim=1)
        
        # Overview-Net: Global Context
        g_feat = self.overview_net(f_concat)
        g_pooled = self.global_pool(g_feat)
        g = self.g_conv(g_pooled) # [B, C, 1, 1]
        
        # Focus-Net: Dynamic Conv
        local_pooled = self.global_pool(f_t).reshape(B, C)
        g_flat = g.reshape(B, C)
        fc_in = torch.cat([g_flat, local_pooled], dim=1)
        w = F.silu(self.fc2(self.fc1(fc_in))) # [B, C*K*K]
        w = w.reshape(B*C, 1, self.k, self.k) # Depthwise dynamic conv weights
        
        f_t_flat = f_t.reshape(1, B*C, H, W)
        f_t_mod_flat = F.conv2d(f_t_flat, w, padding=1, groups=B*C)
        f_t_mod = f_t_mod_flat.reshape(B, C, H, W)
        
        # Adaptive Fusion
        f_prev = (f_t_minus_2 + f_t_minus_1) / 2.0  # Optional: recursive prior representation
        f_cat_fusion = torch.cat([f_t_mod, f_prev], dim=1)
        alpha_raw = torch.sigmoid(self.alpha_conv(f_cat_fusion))
        alpha = 0.3 + 0.4 * alpha_raw  # Alpha in [0.3, 0.7]
        
        f_enhanced = alpha * f_t_mod + (1 - alpha) * f_prev
        return f_enhanced


class TemporalFusionWrapper(nn.Module):
    def __init__(self, in_channels=64, num_levels=3):
        super(TemporalFusionWrapper, self).__init__()
        self.num_levels = num_levels
        # Since standard FPN outputs the same number of channels (e.g. 64) across all levels
        self.mixers = nn.ModuleList([
            ContMixT(in_channels=in_channels, hidden_channels=in_channels)
            for _ in range(num_levels)
        ])
        
        from .temporal import TemporalConsistencyLoss
        self.temporal_loss = TemporalConsistencyLoss(loss_weight=1.0)
        
    def forward(self, sequence_features):
        # sequence_features: List of length T=3.
        # Each element is a list/tuple of length num_levels containing feature maps.
        T = len(sequence_features)
        enhanced_levels = []
        temporal_loss = None
        
        for level in range(self.num_levels):
            f_t_minus_2 = sequence_features[-3][level]
            f_t_minus_1 = sequence_features[-2][level]
            f_t = sequence_features[-1][level]
            
            f_enhanced = self.mixers[level](f_t_minus_2, f_t_minus_1, f_t)
            enhanced_levels.append(f_enhanced)
            
        return enhanced_levels, {'temporal_consistency_loss': temporal_loss}
