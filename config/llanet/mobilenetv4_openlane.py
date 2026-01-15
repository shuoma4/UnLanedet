from ..modelzoo import get_config

import os
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

# Import LLANet components
from unlanedet.model.LLANet.mobilenetv4_small import MobileNetV4Small
from unlanedet.model.LLANet.llanet_head import LLANetHead
from unlanedet.model.LLANet.gsa_fpn import GSAFPN
from unlanedet.model.LLANet.llanet import LLANet

# Import dataset and transform
from unlanedet.data.transform import *
from unlanedet.data.transform.generate_lane_line_openlane import (
    GenerateLaneLineOpenLane,
)

from fvcore.common.param_scheduler import CosineParamScheduler

from ..modelzoo import get_config

from unlanedet.evaluation import LaneAttributeEvaluator

# OpenLane specific parameters
iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
category_loss_weight = 1.0
attribute_loss_weight = 0.5

num_points = 72
max_lanes = 24
sample_y = range(589, 230, -20)

# Lane attribute parameters
num_lane_categories = (
    15  # OpenLane has 15 lane categories (13 regular lanes + 2 curbsides)
)
num_lr_attributes = 4  # 4 left-right attributes

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

# OpenLane image dimensions
ori_img_w = 1920  # Waymo images
ori_img_h = 1280
img_w = 800
img_h = 320
cut_height = 270

# MobileNetV4 期望 RGB 输入，范围 0-1 归一化 (ImageNet 标准)
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 64  # Match FPN output channels
num_classes = 4 + 1  # 4 lanes + background

# OpenLane dataset path
data_root = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw"

param_config = OmegaConf.create()
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.category_loss_weight = category_loss_weight
param_config.attribute_loss_weight = attribute_loss_weight
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = [i for i in range(589, 230, -20)]
param_config.test_parameters = test_parameters
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.img_w = img_w
param_config.img_h = img_h
param_config.cut_height = cut_height
param_config.img_norm = img_norm
param_config.data_root = data_root
param_config.ignore_label = ignore_label
param_config.bg_weight = bg_weight
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_classes = num_classes
param_config.num_lane_categories = num_lane_categories
param_config.num_lr_attributes = num_lr_attributes

# Model configuration with attribute prediction
model = L(LLANet)(
    backbone=L(MobileNetV4Small)(
        width_mult=1.0,  # Standard width multiplier
    ),
    neck=L(GSAFPN)(
        in_channels=[
            128,  # Stage 2 output (stride 8)
            256,  # Stage 3 output (stride 16)
            512,  # Stage 5 output (stride 32)
        ],  # MobileNetV4-Small output channels for stages 2,3,5 (corrected)
        out_channels=64,  # FPN output channels
        num_outs=3,  # Number of output feature maps
        scm_kernel_size=3,  # SCM module kernel size
        enable_global_semantic=True,  # Enable global semantic injection
    ),
    head=L(LLANetHead)(
        num_priors=192,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config,
        enable_category=True,
        enable_attribute=True,
    ),
)

# Training configuration
train = get_config("config/common/train.py").train
epochs = 15
batch_size = 4 * 30  # Reduce for single GPU training with limited memory
epoch_per_iter = 200000 // batch_size + 1  # Adjust based on OpenLane size
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = 100
train.eval_period = 500
train.output_dir = "./output/llanet/mobilenetv4_small_gsafpn_openlane"  # 训练输出目录

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.6e-3
optimizer.weight_decay = 0.01

lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

# Data augmentation
train_process = [
    L(GenerateLaneLineOpenLane)(
        transforms=[
            dict(
                name="Resize",
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            ),
            dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
            dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
            dict(
                name="MultiplyAndAddToBrightness",
                parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                p=0.6,
            ),
            dict(name="AddToHueAndSaturation", parameters=dict(value=(-10, 10)), p=0.7),
            dict(
                name="OneOf",
                transforms=[
                    dict(name="MotionBlur", parameters=dict(k=(3, 5))),
                    dict(name="MedianBlur", parameters=dict(k=(3, 5))),
                ],
                p=0.2,
            ),
            dict(
                name="Affine",
                parameters=dict(
                    translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
                    rotate=(-10, 10),
                    scale=(0.8, 1.2),
                ),
                p=0.7,
            ),
            dict(
                name="Resize",
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            ),
        ],
        cfg=param_config,
    ),
    L(ToTensor)(
        keys=["img", "lane_line", "seg"],
        collect_keys=["lane_categories", "lane_attributes"],
    ),
]

val_process = [
    L(GenerateLaneLine)(
        transforms=[
            dict(
                name="Resize",
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            ),
        ],
        training=False,
        cfg=param_config,
    ),
    L(ToTensor)(keys=["img"]),
]

# Dataset configuration
dataloader = get_config("config/common/openlane.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config  # Pass param_config, not the full cfg
dataloader.train.total_batch_size = batch_size
dataloader.train.num_workers = 2  # 从4减少到2,降低进程数
dataloader.train.pin_memory = True
dataloader.train.prefetch_factor = 1  # 减少预取,降低内存占用
dataloader.train.persistent_workers = False  # Disable to free memory
dataloader.train.timeout = 120  # 增加超时时间,避免长时间卡顿

dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = batch_size
dataloader.test.num_workers = 2  # Reduce for validation
dataloader.test.pin_memory = True

# Evaluation config
dataloader.evaluator = L(LaneAttributeEvaluator)(
    iou_threshold=0.5, width=30, metric="detection/f1"
)

# DDP config to handle unused parameters from attribute prediction
train.ddp = dict(
    broadcast_buffers=False,
    find_unused_parameters=True,  # Enable to handle unused parameters from attribute branches
    fp16_compression=False,
)
