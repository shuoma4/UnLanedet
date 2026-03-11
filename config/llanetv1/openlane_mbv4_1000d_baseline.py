from fvcore.common.param_scheduler import CosineParamScheduler
from omegaconf import OmegaConf

from unlanedet.config import LazyCall as L
from unlanedet.data.transform.custom_transforms import BGR2RGB
from unlanedet.data.transform.lane_decoder import LaneDecoder
from unlanedet.data.transform.lane_encoder import LaneEncoder
from unlanedet.data.transform.openlane_generate import OpenLaneGenerate
from unlanedet.data.transform.transforms import ToTensor

from ..modelzoo import get_config
from .model_factory import create_llanetv1_model

iou_loss_weight = 2.0
cls_loss_weight = 2.0
xyt_loss_weight = 0.2
seg_loss_weight = 1.0
category_loss_weight = 1.0
num_points = 72
max_lanes = 12
num_priors = 192
num_lane_categories = 15
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
num_classes = 2
data_root = '/data1/lxy_log/workspace/ms/OpenLane/dataset/raw/lane3d_1000'
dataset_statistics = '/data1/lxy_log/workspace/ms/UnLanedet/source/openlane_statistics/openlane_priors_with_clusters.npz'

param_config = OmegaConf.create()
param_config.backbone_type = 'mobilenetv4'
param_config.backbone_name = 'mobilenetv4_conv_small'
param_config.use_pretrained_backbone = True
param_config.neck_type = 'GSAFPN'
param_config.enable_global_semantic = True
param_config.enable_category_head = True
param_config.category_head_type = 'linear'
param_config.category_scale_factor = 20.0
param_config.use_data_driven_priors = True
param_config.assign_method = 'GeometryAware'
param_config.enable_temporal_model = False
param_config.temporal_loss_weight = 0.0
param_config.distill_cfg = dict(enable=False)
param_config.deploy_cfg = dict(enable_qat=False, enable_onnx_export=False)
param_config.dataset_statistics = dataset_statistics
param_config.iou_loss_weight = iou_loss_weight
param_config.cls_loss_weight = cls_loss_weight
param_config.xyt_loss_weight = xyt_loss_weight
param_config.seg_loss_weight = seg_loss_weight
param_config.category_loss_weight = category_loss_weight
param_config.num_points = num_points
param_config.max_lanes = max_lanes
param_config.num_priors = num_priors
param_config.num_lane_categories = num_lane_categories
param_config.refine_layers = 3
param_config.sample_points = 36
param_config.sample_y = [i for i in range(1279, 270, -20)]
param_config.test_parameters = test_parameters
param_config.ori_img_w = ori_img_w
param_config.ori_img_h = ori_img_h
param_config.img_w = img_w
param_config.img_h = img_h
param_config.cut_height = cut_height
param_config.img_norm = img_norm
param_config.ignore_label = ignore_label
param_config.bg_weight = bg_weight
param_config.featuremap_out_channel = featuremap_out_channel
param_config.fc_hidden_dim = 64
param_config.num_classes = num_classes
param_config.data_root = data_root
param_config.w_cls = 2.0
param_config.w_geom = 4.0
param_config.w_iou = 2.0
param_config.simota_q = 10
param_config.encoder = L(LaneEncoder)(cfg=param_config)
param_config.decoder = L(LaneDecoder)(cfg=param_config)

model = create_llanetv1_model(param_config)

train = get_config('config/common/train.py').train
epochs = 15
batch_size = 48
epoch_per_iter = 142226 // batch_size + 1
total_iter = epoch_per_iter * epochs
train.max_iter = total_iter
train.checkpointer.period = epoch_per_iter
train.eval_period = epoch_per_iter
train.output_dir = 'output/llanetv1/1000d/mbv4_baseline'
param_config.output_dir = train.output_dir
param_config.epoch_per_iter = epoch_per_iter

optimizer = get_config('config/common/optim.py').AdamW
optimizer.lr = 2.6e-4
optimizer.weight_decay = 0.01
lr_multiplier = L(CosineParamScheduler)(start_value=1.0, end_value=0.001)

train_process = [
    L(BGR2RGB)(),
    L(OpenLaneGenerate)(
        transforms=[
            dict(name='Resize', parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness', parameters=dict(mul=(0.85, 1.15), add=(-10, 10)), p=0.6),
            dict(name='AddToHueAndSaturation', parameters=dict(value=(-10, 10)), p=0.7),
            dict(
                name='OneOf',
                transforms=[
                    dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                    dict(name='MedianBlur', parameters=dict(k=(3, 5))),
                ],
                p=0.2,
            ),
            dict(
                name='Affine',
                parameters=dict(
                    translate_percent=dict(x=(-0.1, 0.1), y=(-0.1, 0.1)), rotate=(-10, 10), scale=(0.8, 1.2)
                ),
                p=0.7,
            ),
            dict(name='Resize', parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0),
        ],
        cfg=param_config,
    ),
    L(ToTensor)(keys=['img', 'lane_line', 'sample_xs', 'seg'], collect_keys=['lane_categories']),
]

val_process = [
    L(BGR2RGB)(),
    L(OpenLaneGenerate)(
        transforms=[dict(name='Resize', parameters=dict(size=dict(height=img_h, width=img_w)), p=1.0)],
        training=False,
        cfg=param_config,
    ),
    L(ToTensor)(keys=['img'], collect_keys=['img_path']),
]

dataloader = get_config('config/common/openlane.py').dataloader
dataloader.train.dataset.data_root = data_root
dataloader.train.dataset.processes = train_process
dataloader.train.dataset.cut_height = cut_height
dataloader.train.dataset.cfg = param_config
dataloader.train.total_batch_size = batch_size
dataloader.train.num_workers = 8
dataloader.train.persistent_workers = True
dataloader.train.pin_memory = True
dataloader.train.prefetch_factor = 4

dataloader.test.dataset.data_root = data_root
dataloader.test.dataset.processes = val_process
dataloader.test.dataset.cut_height = cut_height
dataloader.test.dataset.cfg = param_config
dataloader.test.total_batch_size = batch_size
dataloader.test.num_workers = 4

dataloader.evaluator.cfg = param_config
dataloader.evaluator.output_dir = f'{train.output_dir}/eval_results'

train.amp.enabled = True
