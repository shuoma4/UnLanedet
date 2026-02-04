import os
import sys

from unlanedet import config
from unlanedet.model.LLANet import prior

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_NUM_THREADS"] = "0"
os.environ["CV_NUM_THREADS"] = "0"

from ..modelzoo import get_config
from omegaconf import OmegaConf
from unlanedet.config import LazyCall as L
from unlanedet.data.transform import *
from unlanedet.data.transform.generate_lane_line import GenerateLaneLine
from unlanedet.data.culane import CULane
from unlanedet.evaluation import CULaneEvaluator

from .model_factory import create_llanet_model
import numpy as np

sample_ys = np.array(
    [
        319.6830,
        304.4314,
        296.6077,
        289.9666,
        283.9961,
        278.5653,
        273.5956,
        268.9543,
        264.5477,
        260.2969,
        256.1838,
        252.2238,
        248.4223,
        244.7704,
        241.2527,
        237.8515,
        234.5618,
        231.4032,
        228.3351,
        225.3596,
        222.4419,
        219.6044,
        216.8432,
        214.1359,
        211.4870,
        208.8865,
        206.3553,
        203.8630,
        201.4409,
        199.0703,
        196.7593,
        194.5031,
        192.3014,
        190.1433,
        188.0445,
        186.0056,
        183.9941,
        182.0517,
        180.1459,
        178.3078,
        176.4976,
        174.7333,
        173.0130,
        171.3468,
        169.7142,
        168.1124,
        166.5363,
        164.9963,
        163.4869,
        162.0092,
        160.5554,
        159.1370,
        157.7392,
        156.3574,
        154.9977,
        153.6529,
        152.3217,
        150.9902,
        149.6592,
        148.3292,
        146.9880,
        145.6281,
        144.2045,
        142.7456,
        141.2108,
        139.5907,
        137.8038,
        135.7950,
        133.4230,
        130.3301,
        124.8574,
        3.1798,
    ],
    dtype=np.float32,
)

# === Parameters ===
iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
category_loss_weight = 1.0
attribute_loss_weight = 2.0

assign_method_name = "CLRNet"
w_cls = 2.0
w_geom = 4.0
w_iou = 2.0

start_w_cls = 0.5
start_category_loss_weight = 1e-6
start_attribute_loss_weight = 1e-6
warmup_epochs = 15

num_points = 72
max_lanes = 24
num_priors = 96
num_lane_categories = 15
num_lr_attributes = 4

sample_y = sample_ys
test_parameters = dict(conf_threshold=0.2, nms_thres=0.5, nms_topk=max_lanes)

# CULane Original Dims
ori_img_w = 1640
ori_img_h = 590
# Model Input Dims (OpenLane)
img_w = 800
img_h = 320
cut_height = 270  # CULane standard

img_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 64
num_classes = 4 + 1
data_root = "/data0/lxy_data/mslanedet/CULane/"

use_preprocessed = False  # CULane usually not preprocessed like OpenLane
enable_3d = False

param_config = OmegaConf.create()
param_config.dataset_statistics = "/data1/lxy_log/workspace/ms/UnLanedet/source/openlane_statistics/openlane_priors_with_clusters.npz"  # Use OpenLane stats for model
param_config.use_preprocessed = use_preprocessed
param_config.enable_3d = enable_3d
param_config.use_pretrained_backbone = True
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.sample_y = sample_ys.tolist()
param_config.img_w = img_w  # Use OpenLane sample_y for model head
param_config.img_w = img_w
param_config.img_h = img_h
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.category_loss_weight = category_loss_weight
param_config.attribute_loss_weight = attribute_loss_weight
param_config.w_cls = w_cls
param_config.w_geom = w_geom
param_config.w_iou = w_iou
param_config.start_w_cls = start_w_cls
param_config.start_category_loss_weight = start_category_loss_weight
param_config.start_attribute_loss_weight = start_attribute_loss_weight
param_config.warmup_epochs = warmup_epochs
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.test_parameters = test_parameters
param_config.cut_height = cut_height
param_config.img_norm = img_norm
param_config.data_root = data_root
param_config.ignore_label = ignore_label
param_config.bg_weight = bg_weight
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_classes = num_classes
param_config.num_lane_categories = num_lane_categories
param_config.num_lr_attributes = num_lr_attributes
param_config.num_priors = num_priors

# Model Config
param_config.fc_hidden_dim = 64
param_config.epoch_per_iter = 1000  # Dummy
param_config.assign_method = assign_method_name
param_config.pretrained_model_name = "mobilenetv4_conv_small"
param_config.enable_category = True
param_config.enable_attribute = True
param_config.scale_factor = 20.0
param_config.output_dir = "output/llanet/culane"  # Override later
param_config.detailed_loss_logger_config = dict(
    output_dir=param_config.output_dir, filename="detailed_metrics.json"
)

# === Create Model ===
model = create_llanet_model(param_config)

# === Train Config (for checkpointer) ===
train = get_config("config/common/train.py").train
train.init_checkpoint = (
    "output/llanet/mobilenetv4_small_gsafpn_openlane_0131/model_final.pth"
)
train.output_dir = "output/llanet/culane"

# === Transforms ===
base_transforms = [
    dict(name="Resize", parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0)
]

val_transforms = base_transforms
val_process = [
    L(GenerateLaneLine)(
        transforms=val_transforms,
        training=False,
        cfg=param_config,
    ),
    L(ToTensor)(keys=["img"], collect_keys=["img_name", "img_path"]),
]

# === Dataloader ===
dataloader = get_config("config/common/culane.py").dataloader

# Override Test Dataloader
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = 16
dataloader.test.num_workers = 4

# Override Evaluator
dataloader.evaluator.output_basedir = train.output_dir
dataloader.evaluator.ori_img_h = ori_img_h
dataloader.evaluator.ori_img_w = ori_img_w

# Create specific config for evaluator with CULane sample_y
# CULane requires sampling across the image height (590 to 270)
eval_config = param_config.copy()
eval_config.sample_y = list(range(590, 270, -10))
dataloader.evaluator.cfg = eval_config
dataloader.evaluator.data_root = data_root
