import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_NUM_THREADS"] = "0"
os.environ["CV_NUM_THREADS"] = "0"

runtime_num_gpus = 1
if "--num-gpus" in sys.argv:
    try:
        idx = sys.argv.index("--num-gpus")
        if idx + 1 < len(sys.argv):
            runtime_num_gpus = int(sys.argv[idx + 1])
    except (ValueError, IndexError):
        pass

# Resource Calc
MAX_TOTAL_WORKERS = 12
TARGET_BATCH_PER_GPU = 60

safe_workers_per_gpu = max(1, MAX_TOTAL_WORKERS // runtime_num_gpus)
dynamic_total_batch_size = TARGET_BATCH_PER_GPU * runtime_num_gpus

print(f"\n[Config] Detected {runtime_num_gpus} GPUs.")
print(
    f"[Config] Auto-adjusting: Workers/GPU={safe_workers_per_gpu}, TotalBatchSize={dynamic_total_batch_size}\n"
)

from ..modelzoo import get_config
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L
from unlanedet.model.LLANet.mobilenetv4_small import MobileNetV4Small
from unlanedet.model.LLANet.llanet_head import LLANetHead
from unlanedet.model.LLANet.gsa_fpn import GSAFPN
from unlanedet.model.LLANet.llanet import LLANet
from unlanedet.data.transform import *
from unlanedet.data.transform.generate_lane_line_openlane import (
    GenerateLaneLineOpenLane,
)
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    CompositeParamScheduler,
    LinearParamScheduler,
)
from ..modelzoo import get_config
from unlanedet.evaluation.openlane_evaluator import OpenLaneEvaluator

# === Loss Weights ===
iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
category_loss_weight = 2.0
attribute_loss_weight = 0.5

num_points = 72
max_lanes = 24
sample_y = range(589, 230, -20)
num_priors = 96
num_lane_categories = 15
num_lr_attributes = 4

test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

ori_img_w = 1920
ori_img_h = 1280
img_w = 800
img_h = 320
cut_height = 270

img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 64
num_classes = 4 + 1
data_root = "/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/"
lane_anno_dir = "lane3d_300/"  # 车道线标注目录，相对于data_root的目录（lane3d_300是小数据集，lane3d_1000为大数据集）

param_config = OmegaConf.create()
param_config.use_preprocessed = True
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
param_config.lane_anno_dir = lane_anno_dir
param_config.ignore_label = ignore_label
param_config.bg_weight = bg_weight
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_classes = num_classes
param_config.num_lane_categories = num_lane_categories
param_config.num_lr_attributes = num_lr_attributes

# Model
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
        num_priors=num_priors,
        refine_layers=3,
        fc_hidden_dim=64,
        sample_points=36,
        cfg=param_config,
        enable_category=True,
        enable_attribute=True,
    ),
)

# Training Config
train = get_config("config/common/train.py").train
epochs = 15

# Dataset size for OpenLane lane3d_300
train_samples = 47533  # 训练集样本数

# Dynamic Batch Size
batch_size = dynamic_total_batch_size
epoch_per_iter = (train_samples + batch_size - 1) // batch_size
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = 500
train.eval_period = epoch_per_iter * 3  # 每3个epoch评估一次，避免训练初期频繁评估
train.output_dir = "./output/llanet/mobilenetv4_small_gsafpn_openlane/"

# Optimizer
optimizer = get_config("config/common/optim.py").AdamW
optimizer.lr = 1e-4
optimizer.weight_decay = 1e-3

lr_multiplier = L(CompositeParamScheduler)(
    schedulers=[
        L(LinearParamScheduler)(start_value=0.01, end_value=1.0),
        L(CosineParamScheduler)(start_value=1.0, end_value=0.001),
    ],
    lengths=[0.2, 0.8],
    interval_scaling=["rescaled", "rescaled"],  # 添加这行
)
# === Transform Definition (Optimized) ===
train_transforms = [
    dict(name="HorizontalFlip", parameters=dict(p=1.0), p=0.5),
    dict(name="ChannelShuffle", parameters=dict(p=1.0), p=0.1),
    dict(
        name="MultiplyAndAddToBrightness",
        parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
        p=0.6,
    ),
    dict(name="AddToHueAndSaturation", parameters=dict(value=(-10, 10)), p=0.7),
    # === REMOVED SLOW BLURS ===
    # dict(name="OneOf", transforms=[...], p=0.2),
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
    L(GenerateLaneLineOpenLane)(
        transforms=train_transforms, cfg=param_config, training=True
    ),
    L(ToTensor)(
        keys=["img", "lane_line", "seg"],
        collect_keys=["lane_categories", "lane_attributes"],
    ),
]

val_process = [
    L(GenerateLaneLineOpenLane)(
        transforms=[
            dict(
                name="Resize",
                parameters=dict(size=dict(height=img_h, width=img_w)),
                p=1.0,
            )
        ],
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
dataloader.train.total_batch_size = batch_size

dataloader.train.num_workers = safe_workers_per_gpu
# === INCREASE PREFETCH ===
dataloader.train.prefetch_factor = 4
dataloader.train.pin_memory = True
dataloader.train.persistent_workers = True

# Eval
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = int(dynamic_total_batch_size / 2)
dataloader.test.num_workers = safe_workers_per_gpu

dataloader.evaluator = L(OpenLaneEvaluator)(
    cfg=param_config,
    evaluate_bin_path="/data1/lxy_log/workspace/ms/UnLanedet/tools/exe/openlane_2d_evaluate",
    output_dir="/data1/lxy_log/workspace/ms/UnLanedet/output/llanet/mobilenetv4_small_gsafpn_openlane/eval_results/",
    iou_threshold=0.5,
    width=30,
    metric="OpenLane/F1",
)

# DDP & AMP
train.ddp = dict(
    broadcast_buffers=False,
    find_unused_parameters=True,
    fp16_compression=False,
)
train.amp = dict(enabled=False)
