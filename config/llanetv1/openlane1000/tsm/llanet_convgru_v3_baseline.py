from ..common import build_config
from unlanedet.config import LazyCall as L
import os

# Import new modules
from unlanedet.data.openlane_temporal import OpenLaneTemporal
from unlanedet.data.transform.openlane_generator_kornia import OpenLaneGeneratorKornia
from unlanedet.data.transform.openlane_generator_temporal import (
    TemporalToTensor,
    TemporalNormalize,
)
from unlanedet.data.transform.custom_transforms import BGR2RGB
from unlanedet.model.llanetv1.temporal_modules import ConvGRUTemporalWrapper
from ..common import TRAIN_TRANSFORMS, VAL_TRANSFORMS

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="tsm/convgru_v3_kornia_baseline",
    backbone_type="resnet",
    backbone_name="resnet34",
    neck_type="GSAFPN",
    enable_category_head=True,
    category_head_type="combined",
    use_data_driven_priors=False,
    assign_method="CLRNet",
    enable_temporal_model=False,  # We override it manually below
    temporal_loss_weight=0.5,
    enable_global_semantic=True,
    # Keep batch size identical to v3 for fair comparison
    batch_size=32,
)

param_config.scm_kernel_size = 9
param_config.category_loss_weight = 5.0
train.amp.enabled = False  # 关闭 AMP，避免时序/GRU 下 NaN
train.float32_precision = "high"  # TF32
train.cudnn_benchmark = True

# Keep dataloader throttling identical to v3
_cpu_count = os.cpu_count() or 8
dataloader.train.num_workers = max(4, min(8, _cpu_count // 2))
dataloader.train.persistent_workers = True

# 1. Override the Datasets
dataloader.train.dataset = L(OpenLaneTemporal)(
    data_root=dataloader.train.dataset.data_root,
    split="train",
    cut_height=dataloader.train.dataset.cut_height,
    seq_len=3,
    cfg=param_config,
)
dataloader.test.dataset = L(OpenLaneTemporal)(
    data_root=dataloader.test.dataset.data_root,
    split="val",
    cut_height=dataloader.test.dataset.cut_height,
    seq_len=3,
    cfg=param_config,
)

# 2. Override the Data Process pipelines
dataloader.train.dataset.processes = [
    L(BGR2RGB)(),
    L(OpenLaneGeneratorKornia)(
        transforms_cfg=TRAIN_TRANSFORMS, cfg=param_config, training=True
    ),
    L(TemporalToTensor)(
        keys=["img", "lane_line", "seg", "lane_vis", "intrinsic", "extrinsic", "pose"],
        collect_keys=[
            "lane_categories",
            "lane_attributes",
            "lane_track_ids",
            "track_id",
            "seq_visibility",
            "seq_xyz",
            "seq_track_id",
        ],
    ),
]

dataloader.test.dataset.processes = [
    L(BGR2RGB)(),
    L(OpenLaneGeneratorKornia)(
        transforms_cfg=VAL_TRANSFORMS, cfg=param_config, training=False
    ),
    L(TemporalToTensor)(
        keys=["img", "lane_vis", "intrinsic", "extrinsic", "pose"],
        collect_keys=[
            "img_path",
            "lane_categories",
            "lane_attributes",
            "lane_track_ids",
            "track_id",
            "seq_visibility",
            "seq_xyz",
            "seq_track_id",
        ],
    ),
]

# 3. Temporal module baseline: ConvGRU
model.temporal_model = L(ConvGRUTemporalWrapper)(
    in_channels=64,
    num_levels=getattr(param_config, "refine_layers", 3),
    cfg=param_config,
)
dataloader.train.total_batch_size = 32
dataloader.test.total_batch_size = 4

# 4. Training I/O
dataloader.train.pin_memory = True
dataloader.train.prefetch_factor = 1
dataloader.test.num_workers = 2
dataloader.test.persistent_workers = False
dataloader.test.pin_memory = True
dataloader.test.prefetch_factor = 2

# Keep temporal loss schedule/gates identical to v3 for fair comparison
param_config.temporal_loss_weight = 0.1
param_config.temporal_warmup_iters = 2000
param_config.temporal_ramp_iters = 4000
param_config.temporal_feature_warmup_iters = 2000
param_config.temporal_huber_delta = 3.0
param_config.temporal_reproj_tau = 8.0
param_config.temporal_geo_scale = 1.0
param_config.temporal_geo_cap = 8.0
param_config.temporal_total_cap = 20.0
param_config.temporal_min_valid_points = 8
param_config.temporal_max_point_error = 150.0
