"""
LLANet Model Definition

Combines MobileNetV4-Small backbone, GSA-FPN neck, and LLANetHead into a single model.
"""

import torch
import torch.nn as nn


class LLANet(nn.Module):
    """LLANet model combining MobileNetV4-Small backbone, GSA-FPN neck, and LLANetHead."""

    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, data):
        img = data["img"]  # Extract image tensor from data dict
        device = next(self.backbone.parameters()).device
        if img.device != device:
            img = img.to(device)
        if torch.isnan(img).any():
            print("NaN detected in input image!", flush=True)
        features = self.backbone(img)
        neck_features = self.neck(features)

        if self.training:
            outputs = self.head(neck_features)
            current_iter = data.get("iter", 0)
            losses = self.head.loss(outputs, data, current_iter=current_iter)
            return losses
        else:
            return self.head(neck_features)

    def get_lanes(self, output):
        return self.head.get_lanes(output)
