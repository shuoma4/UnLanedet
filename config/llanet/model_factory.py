"""
简单模型工厂 - 专门用于生成独立的LLANet模型实例
避免多个训练实例间的冲突
"""

import os
import random
import numpy as np
import torch
from unlanedet.config import LazyCall as L
from unlanedet.model.LLANet.mobilenetv4_small import MobileNetV4Small
from unlanedet.model.LLANet.llanet_head import LLANetHead
from unlanedet.model.LLANet.gsa_fpn import GSAFPN
from unlanedet.model.LLANet.llanet import LLANet


def create_llanet_model(cfg):
    """
    创建独立的LLANet模型实例

    Args:
        cfg: 配置参数

    Returns:
        模型配置
    """
    # 生成唯一随机种子确保独立性
    seed = random.randint(1000, 9999)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 创建模型
    model = L(LLANet)(
        backbone=L(MobileNetV4Small)(width_mult=1.0),
        neck=L(GSAFPN)(
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            scm_kernel_size=3,
            enable_global_semantic=True,
        ),
        head=L(LLANetHead)(
            num_priors=cfg.num_priors,
            refine_layers=3,
            fc_hidden_dim=64,
            sample_points=36,
            cfg=cfg,
            enable_category=True,
            enable_attribute=True,
        ),
    )

    return model
