import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, current_preds, previous_preds):
        if current_preds is None or previous_preds is None:
            return current_preds.new_tensor(0.0) if current_preds is not None else torch.tensor(0.0)
        min_priors = min(current_preds.shape[1], previous_preds.shape[1])
        if min_priors == 0:
            return current_preds.new_tensor(0.0)
        loss = F.smooth_l1_loss(current_preds[:, :min_priors, 2:6], previous_preds[:, :min_priors, 2:6].detach())
        return loss * self.loss_weight


class ContMixTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        padding = kernel_size // 2
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.local_mixer = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=max(1, channels // 16), bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, current_feat, prev_feat=None, guidance=None):
        if prev_feat is None:
            return current_feat
        if prev_feat.shape[-2:] != current_feat.shape[-2:]:
            prev_feat = F.interpolate(prev_feat, size=current_feat.shape[-2:], mode='bilinear', align_corners=False)
        if guidance is not None and guidance.shape[-2:] != current_feat.shape[-2:]:
            guidance = F.interpolate(guidance, size=current_feat.shape[-2:], mode='bilinear', align_corners=False)

        mix_input = torch.cat([current_feat, prev_feat], dim=1)
        gate = self.global_context(mix_input)
        mixed = self.local_mixer(mix_input)
        if guidance is not None:
            mixed = mixed + guidance
        fused = gate * current_feat + (1.0 - gate) * (prev_feat + mixed)
        return current_feat + self.out_proj(fused)


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
        if stage_predictions is not None and len(stage_predictions) >= 2:
            temporal_loss = self.temporal_loss(stage_predictions[-1], stage_predictions[-2])
        return aggregated, {'temporal_consistency_loss': temporal_loss}
