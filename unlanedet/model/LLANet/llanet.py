"""
LLANet Model Definition

Combines MobileNetV4-Small backbone, GSA-FPN neck, and LLANetHead into a single model.
"""

import torch.nn as nn

from .mobilenetv4_small import MobileNetV4Small
from .gsa_fpn import GSAFPN
from .llanet_head import LLANetHead


class LLANet(nn.Module):
    """LLANet model combining MobileNetV4-Small backbone, GSA-FPN neck, and LLANetHead."""

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, data):
        img = data['img']  # Extract image tensor from data dict
        features = self.backbone(img)
        neck_features = self.neck(features)

        if self.training:
            # During training, forward returns outputs and losses
            outputs = self.head(neck_features)
            losses = self.head.loss(outputs, data)
            return losses
        else:
            # During inference, forward returns predictions
            return self.head(neck_features)

