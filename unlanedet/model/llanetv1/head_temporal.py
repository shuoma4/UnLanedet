import torch
import torch.nn.functional as F

from .head import LLANetV1Head
from .temporal_modules import ContMixT
import torch.nn as nn

class LLANetV1TemporalHead(LLANetV1Head):
    def __init__(self, **kwargs):
        super(LLANetV1TemporalHead, self).__init__(**kwargs)
        # For LLANet, all FPN feature maps passed to the head have 64 channels
        self.contmix_t_list = nn.ModuleList([
            ContMixT(in_channels=64, hidden_channels=64),
            ContMixT(in_channels=64, hidden_channels=64),
            ContMixT(in_channels=64, hidden_channels=64)
        ])
        self.lambda_t = float(getattr(self.cfg, 'lambda_t', 1.0))

    def forward(self, x, **kwargs):
        batch_features = list(x[len(x) - self.refine_layers :])
        batch_features.reverse()
        total_batch_size = batch_features[0].shape[0]
        
        T = 3
        # Handle inference or cases where batch is not perfectly divisible by 3
        if total_batch_size % T != 0:
            return super(LLANetV1TemporalHead, self).forward(x, **kwargs)

        B = total_batch_size // T
        
        enhanced_features = []
        for stage, feat in enumerate(batch_features):
            C_feat, H_feat, W_feat = feat.shape[1], feat.shape[2], feat.shape[3]
            feat_T = feat.reshape(B, T, C_feat, H_feat, W_feat)
            
            f_t_minus_2 = feat_T[:, 0, :, :, :]
            f_t_minus_1 = feat_T[:, 1, :, :, :]
            f_t = feat_T[:, 2, :, :, :]
            
            f_enhanced = self.contmix_t_list[stage](f_t_minus_2, f_t_minus_1, f_t)
            enhanced_features.append(f_enhanced.repeat_interleave(T, dim=0))
            
        enhanced_features.reverse()
        # combine the unchanged part of x with the enhanced features
        new_x = list(x[:len(x) - self.refine_layers]) + enhanced_features
        
        return super(LLANetV1TemporalHead, self).forward(new_x, **kwargs)

    def loss(self, outputs, batch):
        loss_dict = super(LLANetV1TemporalHead, self).loss(
            outputs, batch
        )
        
        # Temporal consistency loss pseudo-code
        # In a real scenario, you can compare outputs from t-1 and t, 
        # or use ground-truth lanes if available across frames.
        loss_temporal = torch.tensor(0.0, device=outputs[0]['cls_logits'].device)
        # loss_dict['loss_temporal'] = loss_temporal * self.lambda_t
        
        return loss_dict

