from omegaconf import OmegaConf

from unlanedet.config import LazyCall as L
from unlanedet.data.build import build_batch_data_loader
from unlanedet.data.openlane import OpenLane
from unlanedet.data.transform import *
from unlanedet.evaluation import OpenLaneEvaluator
from unlanedet.model import FPN, ResNetWrapper
from unlanedet.model.CLRNet import CLRHead, CLRNet

from ..modelzoo import get_config

# Model Parameters (from CLRNet CULane config)
iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
num_points = 72
max_lanes = 4
sample_y = range(589, 230, -20)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

# Image Parameters
# OpenLane original size
ori_img_w = 1920
ori_img_h = 1280
# Model input size (same as CULane training)
img_w = 800
img_h = 320
# Cut height: Try to keep similar FOV ratio as CULane
# CULane: 590 total, 270 cut (45% cut).
# OpenLane: 1280 total. 45% is ~576.
# Let's try 590 to be safe and close to previous attempts.
cut_height = 800

# Data Norm: CULane training did NOT use Normalize or BGR2RGB.
# It used GenerateLaneLine which outputs BGR 0-1 float.
# So we define dummy norms just to satisfy config structure if needed, but won't use them in transforms.
img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1.0, 1.0, 1.0])
ignore_label = 255
bg_weight = 0.4
featuremap_out_channel = 192
num_classes = 4 + 1

# Data Root
data_root = '/data1/lxy_log/workspace/ms/OpenLane/dataset/raw'

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
param_config.featuremap_out_channel = featuremap_out_channel
param_config.num_classes = num_classes
param_config.lane_decode_mode = 'abs_fixed_y'  # OpenLane might need this? Or maybe 'culane' default?
# CULane config didn't specify lane_decode_mode in the snippet I saw, but it's used in model.

# Model Definition
model = L(CLRNet)(
    backbone=L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck=L(FPN)(in_channels=[128, 256, 512], out_channels=64, num_outs=3, attention=False),
    head=L(CLRHead)(num_priors=192, refine_layers=3, fc_hidden_dim=64, sample_points=36, cfg=param_config),
)

# Training/Eval Config
train = get_config('config/common/train.py').train
train.output_dir = 'output/openlane/clrnet_resnet34_eval'
train.init_checkpoint = 'assets/clrnet_model_best_culane.pth'

# Data Processing - MATCHING CULANE TRAINING PIPELINE
val_process = [
    L(GenerateLaneLine)(
        transforms=[
            dict(name='Resize', parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
        ],
        training=False,
        cfg=param_config,
    ),
    L(ToTensor)(keys=['img'], collect_keys=['img_path', 'origin_img_path']),
]

# Dataloader
dataloader = OmegaConf.create()
dataloader.test = L(build_batch_data_loader)(
    dataset=L(OpenLane)(
        data_root=data_root,
        split='val',
        cut_height=cut_height,
        processes=val_process,
        cfg=param_config,
    ),
    total_batch_size=48,
    num_workers=4,
    drop_last=False,
    shuffle=False,
)

# Evaluator
dataloader.evaluator = L(OpenLaneEvaluator)(
    cfg=param_config,
    evaluate_bin_path='unlanedet/evaluation/openlane/evaluate',
    iou_threshold=0.5,
    width=30,
    metric='F1',
)
