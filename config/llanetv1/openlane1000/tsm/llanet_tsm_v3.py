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
from unlanedet.model.llanetv1.temporal_modules import TemporalFusionWrapper
from ..common import TRAIN_TRANSFORMS, VAL_TRANSFORMS

model, dataloader, train, optimizer, lr_multiplier, param_config = build_config(
    run_name="tsm/v3_kornia_optimized",
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
    # FP32 + 时序 B*T 叠 batch 过 backbone，显存压力大；32 易 OOM，降到 20
    batch_size=20,
)

param_config.scm_kernel_size = 9
param_config.category_loss_weight = 5.0
# FP32 下单次 B*T 过 backbone 显存峰值高；限制堆叠阈值，强制逐帧 backbone（略慢但更省显存）
param_config.max_stacked_sequence_bt = 40
train.amp.enabled = False  # 关闭 AMP，避免时序分支 fp16 下 NaN/反传不稳定
train.float32_precision = "high"  # TF32（Ada Lovelace 上有额外加速）
train.cudnn_benchmark = True

# 自动按 CPU 核心数限流，避免 worker 过多引发系统卡死
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


# 3. Re-enable temporal fusion with robust temporal loss guards.
model.temporal_model = L(TemporalFusionWrapper)(
    in_channels=64,
    num_levels=getattr(param_config, "refine_layers", 3),
    cfg=param_config,
)
dataloader.train.total_batch_size = 20
# 评估时 batch 小；单卡 seq_len=3 不宜过大
dataloader.test.total_batch_size = 4

# 4. 训练 I/O
dataloader.train.pin_memory = True
# seq_len=3 且样本较大，prefetch 过高会显著放大内存和 CPU 压力
dataloader.train.prefetch_factor = 1
dataloader.test.num_workers = 2
dataloader.test.persistent_workers = False  # eval 结束后立即释放 worker 内存
dataloader.test.pin_memory = True
dataloader.test.prefetch_factor = 2

# Temporal loss integration weight (applied in LLANetV1.forward).
param_config.temporal_loss_weight = 0.1
# Robust temporal schedule / gates.
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
