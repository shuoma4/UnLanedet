from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

from config.modelzoo import get_config
from unlanedet.config import LazyCall as L

# import dataset and transform
from unlanedet.data.transform import *
from unlanedet.data.transform.generate_lane_line_openlane import GenerateLaneLineOpenLane
from unlanedet.model import FPN, ResNetWrapper
from unlanedet.model.llanetv2.llanetv2 import LLANetV2, LLANetV2Head

# OpenLane configs
iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
num_points = 72
max_lanes = 12
num_priors = 192
fc_hidden_dim = 64
featuremap_out_channel = 192
lane_decode_mode = 'abs_fixed_y'

# OpenLane specific sample_y (needs adjustment for OpenLane resolution/cut)
ori_img_w = 1920
ori_img_h = 1280
img_w = 800
img_h = 320
cut_height = 590
sample_y = range(1250, 590, -20)
test_parameters = dict(conf_threshold=0.4, nms_thres=50, nms_topk=max_lanes)

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1.0, 1.0, 1.0])
ignore_label = 255
bg_weight = 0.4
num_classes = 1 + 1

# OpenLane paths
data_root = '/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/'

param_config = OmegaConf.create()
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.sample_y = [i for i in range(1250, 590, -20)]
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
param_config.lane_decode_mode = lane_decode_mode

param_config.encoder = L(GenerateLaneLineOpenLane)(cfg=param_config)

model = L(LLANetV2)(
    backbone=L(ResNetWrapper)(
        resnet='resnet34',
        pretrained=True,
        replace_stride_with_dilation=[False, False, False],
        out_conv=False,
    ),
    neck=L(FPN)(in_channels=[128, 256, 512], out_channels=64, num_outs=3, attention=False),
    head=L(LLANetV2Head)(num_priors=192, refine_layers=3, fc_hidden_dim=64, sample_points=36, cfg=param_config),
)

# Training Config
train = get_config('config/common/train.py').train
epochs = 30
batch_size = 96
train_samples = 45903
epoch_per_iter = train_samples // batch_size + 1
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = epoch_per_iter
train.eval_period = epoch_per_iter
train.output_dir = 'output/openlane/llanetv2_resnet34_opt_eval'
# Set checkpoint to specific file
train.init_checkpoint = 'output/openlane/llanetv2_resnet34_opt/model_0008142.pth'

dataloader = get_config('config/common/openlane.py').dataloader
dataloader.train.dataset.cut_height = cut_height
dataloader.test.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config
dataloader.test.dataset.cfg = param_config
dataloader.train.dataset.processes = [L(GenerateLaneLineOpenLane)(cfg=param_config, training=True)]
dataloader.test.dataset.processes = [L(GenerateLaneLineOpenLane)(cfg=param_config, training=False)]
dataloader.train.total_batch_size = batch_size
dataloader.test.total_batch_size = batch_size

optimizer = get_config('config/common/optim.py').AdamW
optimizer.lr = 2.4e-3  # Scaled up for Batch Size 96 (vs 24 in CULane)
optimizer.weight_decay = 0.01

lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)
