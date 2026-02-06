from ..modelzoo import get_config

import os
import sys
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L

from unlanedet.model.CLRNet import CLRNet, CLRHead
from unlanedet.model import ResNetWrapper, FPN

# import dataset and transform
from unlanedet.data.transform import *
from unlanedet.data.transform.openlane_generate import (
    OpenLaneGenerate,
)
from unlanedet.evaluation.openlane_evaluator import OpenLaneEvaluator

from fvcore.common.param_scheduler import CosineParamScheduler

# Resource Calc
MAX_TOTAL_WORKERS = 12
TARGET_BATCH_PER_GPU = 12  # ResNet34 might need smaller batch size than MobileNet
runtime_num_gpus = 2  # Default to 2
if "--num-gpus" in sys.argv:
    try:
        idx = sys.argv.index("--num-gpus")
        if idx + 1 < len(sys.argv):
            runtime_num_gpus = int(sys.argv[idx + 1])
    except (ValueError, IndexError):
        pass

safe_workers_per_gpu = max(1, MAX_TOTAL_WORKERS // runtime_num_gpus)
dynamic_total_batch_size = TARGET_BATCH_PER_GPU * runtime_num_gpus

iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0

num_points = 72
max_lanes = 24
sample_y = range(589, 230, -20)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

ori_img_w = 1920
ori_img_h = 1280
img_w = 800
img_h = 320
cut_height = 270
img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 64  # Match LLANet? No, CLRNet usually uses 64 or 192. culane config used 192. Let's use 64 for ResNet34 + FPN typically.
# Wait, culane config says: out_channels=64 in FPN. But featuremap_out_channel = 192.
# Let's check CLRHead init. prior_feat_channels=64 (default).
# In culane config: head = L(CLRHead)(..., fc_hidden_dim=64).
# But featuremap_out_channel is passed to param_config? CLRHead doesn't seem to use param_config.featuremap_out_channel directly for channel dimension.
# However, FPN output channels is 64.
# Let's stick to 64 for FPN out.

num_classes = 4 + 1  # OpenLane has more classes? LLANet uses 4+1?
# OpenLane has 14 classes + BG?
# LLANet config says num_lane_categories = 15.
# But num_classes = 4 + 1 in LLANet config too.
# CLRNet usually treats binary classification (lane/bg) for detection, and maybe category separately?
# CLRNet original code supports only 1 class for detection?
# Let's stick to 4+1 if that's what CULane uses, but OpenLane is different.
# LLANet config: num_classes = 4 + 1.
# But LLANet has separate category head.
# CLRNet might not have category head enabled by default?
# Let's keep num_classes = 4 + 1 for now, as strict CLRNet might not support multiclass lane types in the same way.

data_root = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/"
lane_anno_dir = "lane3d_300/"

param_config = OmegaConf.create()
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
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
param_config.featuremap_out_channel = 64
param_config.num_classes = num_classes
param_config.lane_anno_dir = lane_anno_dir
param_config.use_preprocessed = True
param_config.assign_method = "CLRNet"
param_config.enable_3d = False

# Model
model = L(CLRNet)(
    backbone=L(ResNetWrapper)(
        resnet="resnet34",
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck=L(FPN)(
        in_channels=[128, 256, 512], out_channels=64, num_outs=3, attention=False
    ),
    head=L(CLRHead)(
        num_priors=192,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config,
    ),
)

# Transforms
base_transforms = [
    dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
    dict(
        name="ToNDArray",
        parameters=dict(dtype="float32"),
        p=1.0,
    ),
    dict(
        name="Normalize",
        parameters=dict(mean=img_norm["mean"], std=img_norm["std"]),
        p=1.0,
    ),
]

train_transforms = [
    dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
    dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
    dict(
        name="Affine",
        parameters=dict(
            translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)),
            rotate=(-10, 10),
            scale=(0.8, 1.2),
        ),
        p=0.7,
    ),
    dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
]

train_process = [
    L(OpenLaneGenerate)(transforms=train_transforms, cfg=param_config, training=True),
    L(ToTensor)(
        keys=["img", "lane_line", "seg"],
        collect_keys=["lane_categories", "lane_attributes"],
    ),
]

val_transforms = base_transforms
val_process = [
    L(OpenLaneGenerate)(
        transforms=val_transforms,
        training=False,
        cfg=param_config,
    ),
    L(ToTensor)(keys=["img"]),
]

# Dataloader
dataloader = get_config("config/common/openlane.py").dataloader
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config
dataloader.train.total_batch_size = dynamic_total_batch_size
dataloader.train.num_workers = safe_workers_per_gpu
dataloader.train.prefetch_factor = 4
dataloader.train.pin_memory = True
dataloader.train.persistent_workers = True

dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = int(dynamic_total_batch_size / 2)
dataloader.test.num_workers = safe_workers_per_gpu

# Evaluator
dataloader.evaluator = L(OpenLaneEvaluator)(
    cfg=param_config,
    evaluate_bin_path="/data1/lxy_log/workspace/ms/UnLanedet/tools/exe/openlane_2d_evaluate",
    output_dir="/data1/lxy_log/workspace/ms/UnLanedet/output/clrnet/resnet34_openlane/eval_results/",
    iou_threshold=0.5,
    width=30,
    metric="OpenLane/F1",
)

# Training Config
train = get_config("config/common/train.py").train
epochs = 15
epoch_per_iter = 88880 // dynamic_total_batch_size + 1
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = epoch_per_iter
train.eval_period = epoch_per_iter
train.output_dir = "output/clrnet/resnet34_openlane"

optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 0.6e-3
optimizer.weight_decay = 0.01

lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)
