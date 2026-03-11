import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    def __init__(self, student_channels, teacher_channels):
        super().__init__()
        self.proj = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


class LaneDistillationLoss(nn.Module):
    def __init__(self, feature_pairs=None, feature_weight=1.0, logits_weight=1.0, temperature=4.0):
        super().__init__()
        self.feature_pairs = feature_pairs or []
        self.feature_weight = feature_weight
        self.logits_weight = logits_weight
        self.temperature = temperature
        self.adapters = nn.ModuleList(
            [FeatureAdapter(pair['student_channels'], pair['teacher_channels']) for pair in self.feature_pairs]
        )

    def _feature_loss(self, student_features, teacher_features):
        if not self.feature_pairs:
            return student_features[0].new_tensor(0.0)
        total = student_features[0].new_tensor(0.0)
        for adapter, pair in zip(self.adapters, self.feature_pairs):
            s_feat = student_features[pair['student_idx']]
            t_feat = teacher_features[pair['teacher_idx']].detach()
            s_feat = adapter(s_feat)
            if s_feat.shape[-2:] != t_feat.shape[-2:]:
                s_feat = F.interpolate(s_feat, size=t_feat.shape[-2:], mode='bilinear', align_corners=False)
            total = total + F.mse_loss(s_feat, t_feat)
        return total * self.feature_weight

    def _kl_loss(self, student_logits, teacher_logits):
        if student_logits is None or teacher_logits is None:
            if student_logits is not None:
                return student_logits.new_tensor(0.0)
            if teacher_logits is not None:
                return teacher_logits.new_tensor(0.0)
            return torch.tensor(0.0)
        student_logits = student_logits.mean(dim=1)
        teacher_logits = teacher_logits.detach().mean(dim=1)
        log_p = F.log_softmax(student_logits / self.temperature, dim=-1)
        q = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(log_p, q, reduction='batchmean') * (self.temperature**2) * self.logits_weight

    def forward(self, student_features, teacher_features, student_outputs, teacher_outputs):
        losses = {}
        losses['distill_feature_loss'] = self._feature_loss(student_features, teacher_features)
        losses['distill_lane_logits_loss'] = self._kl_loss(
            student_outputs.get('distill_cls_logits'),
            teacher_outputs.get('distill_cls_logits'),
        )
        losses['distill_category_logits_loss'] = self._kl_loss(
            student_outputs.get('category'),
            teacher_outputs.get('category'),
        )
        return losses
